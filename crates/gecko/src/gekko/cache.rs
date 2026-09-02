//! Blocks decoded once and run until a branch leaves them. Stepping one instruction at a
//! time pays the fetch, the address translation and the walk down the dispatch tree
//! every time; a block pays them once and keeps the handlers. Interrupts are taken at
//! block boundaries, as the JIT takes them, and a block the JIT would call idle skips to
//! the next deadline after one pass. RAM written under a block, by a store or a DMA,
//! reaches here through the same pending lines the JIT invalidates from. A host that can
//! compile a block to its own code (`wasmjit`) hands one over, and the block runs there
//! instead of here.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::gekko::block;
use crate::gekko::idle::{self, IdleClass};
use crate::gekko::instruction::Instruction;
#[cfg(feature = "wasm-jit")]
use crate::gekko::wasmjit::{self, BlockCompiler};
use crate::mmio::{self, Mmio, CODE_LINE_BYTES, CODE_LINE_MASK};
use crate::system::{System, SystemId};

const LOOKUP_SLOTS: usize = 1 << 16;
const LOOKUP_MASK: u32 = LOOKUP_SLOTS as u32 - 1;
const NO_BLOCK: u32 = u32::MAX;

struct Block<const SYSTEM: SystemId> {
    start_pc: u32,
    /// Handler number, word and address of each instruction, in the order they run; an
    /// unconditional forward branch the scanner followed is not among them.
    steps: Vec<(u16, Instruction, u32)>,
    end_pc: u32,
    idle: bool,
    lines: SmallVec<[u32; 4]>,
    /// The host's table slot holding this block compiled, when it has one.
    #[cfg(feature = "wasm-jit")]
    code: Option<u32>,
    #[cfg(feature = "wasm-jit")]
    runs: u32,
}

/// How many times a block runs interpreted before it is worth compiling: most code runs
/// once, at a load, and compiling it would cost more than it saves.
#[cfg(feature = "wasm-jit")]
const HOT: u32 = 16;

pub struct BlockCache<const SYSTEM: SystemId> {
    /// Direct-mapped by pc, ahead of the map, as the JIT's lookup table is.
    lookup: Box<[u32]>,
    blocks: Vec<Option<Block<SYSTEM>>>,
    free: Vec<u32>,
    by_pc: FxHashMap<u32, u32>,
    blocks_by_line: FxHashMap<u32, SmallVec<[u32; 2]>>,
    #[cfg(feature = "wasm-jit")]
    compiler: Option<Box<dyn BlockCompiler>>,
}

impl<const SYSTEM: SystemId> Default for BlockCache<SYSTEM> {
    fn default() -> Self {
        Self {
            lookup: vec![NO_BLOCK; LOOKUP_SLOTS].into_boxed_slice(),
            blocks: Vec::new(),
            free: Vec::new(),
            by_pc: FxHashMap::default(),
            blocks_by_line: FxHashMap::default(),
            #[cfg(feature = "wasm-jit")]
            compiler: None,
        }
    }
}

impl<const SYSTEM: SystemId> BlockCache<SYSTEM> {
    /// Gives new blocks to the host to compile. Blocks already built stay interpreted.
    /// Hooks run only in the interpreter, so a build with them never compiles.
    #[cfg(feature = "wasm-jit")]
    pub fn set_compiler(&mut self, compiler: Option<Box<dyn BlockCompiler>>) {
        self.compiler = if cfg!(feature = "hooks") { None } else { compiler };
    }

    /// Runs the block at the pc, building it first if it is new, and leaves the pc where
    /// control went. True when the block is an idle loop that just closed on itself.
    pub fn run_block(&mut self, sys: &mut System<SYSTEM>) -> bool {
        let pc = sys.gekko.pc;
        let index = self.lookup_or_build(sys, pc);
        #[cfg(feature = "wasm-jit")]
        {
            let block = self.blocks[index as usize].as_mut().unwrap();
            if block.code.is_none() && self.compiler.is_some() {
                block.runs += 1;
                if block.runs == HOT {
                    let compiler = self.compiler.as_mut().unwrap();
                    block.code = compiler.compile(&wasmjit::emit::<SYSTEM>(sys, &block.steps, block.end_pc));
                }
            }
            if let Some(slot) = block.code {
                let run: wasmjit::BlockFn<SYSTEM> = unsafe { core::mem::transmute(slot as usize) };
                if wasmjit::validate::on() {
                    let before = Snapshot::of(sys);
                    wasmjit::validate::begin();
                    sys.gekko.pc = run(sys);
                    let compiled = Snapshot::of(sys);
                    before.restore(sys);
                    wasmjit::validate::undo(sys);
                    let idle = self.interpret(sys, index);
                    let interpreted = Snapshot::of(sys);
                    let stores = wasmjit::validate::stores_that_differ(sys);
                    if compiled != interpreted || !stores.is_empty() {
                        let block = self.blocks[index as usize].as_ref().unwrap();
                        let diff = compiled.describe(&interpreted) + &stores;
                        wasmjit::report_disagreement(block.start_pc, &block.steps, &diff);
                    }
                    return idle;
                }
                let nia = run(sys);
                sys.gekko.pc = nia;
                return block.idle && nia == block.start_pc;
            }
        }

        self.interpret(sys, index)
    }

    fn interpret(&self, sys: &mut System<SYSTEM>, index: u32) -> bool {
        let block = self.blocks[index as usize].as_ref().unwrap();
        let last = block.steps.len().saturating_sub(1);

        for (i, &(step, instr, cia)) in block.steps.iter().enumerate() {
            let expected = if i == last { block.end_pc } else { block.steps[i + 1].2 };
            sys.gekko.cia = cia;
            sys.gekko.nia = expected;

            #[cfg(feature = "hooks")]
            if sys.hook_flags.contains(crate::hooks::HookFlags::CPU_PRE) && sys.hook_filters.cpu_pre.matches(cia) {
                if let Some(mut host) = sys.hook_host.take() {
                    host.on_cpu_pre(sys);
                    sys.sync_pending_hook_state(host.as_mut());
                    sys.hook_host = Some(host);
                }
            }

            crate::gekko::execute::<SYSTEM>(step, sys, instr);

            #[cfg(feature = "hooks")]
            if sys.hook_flags.contains(crate::hooks::HookFlags::CPU_POST) && sys.hook_filters.cpu_post.matches(cia) {
                if let Some(mut host) = sys.hook_host.take() {
                    host.on_cpu_post(sys);
                    sys.sync_pending_hook_state(host.as_mut());
                    sys.hook_host = Some(host);
                }
            }

            if sys.gekko.nia != expected {
                sys.gekko.pc = sys.gekko.nia;
                return block.idle && sys.gekko.nia == block.start_pc;
            }
        }

        sys.gekko.pc = sys.gekko.nia;
        false
    }

    /// Forgets every block, compiled or not.
    pub fn reset(&mut self, mmio: &mut Mmio<SYSTEM>) {
        let pcs: Vec<u32> = self.by_pc.keys().copied().collect();
        for pc in pcs {
            self.forget(mmio, pc);
        }
    }

    /// Forgets every block with an instruction on `line`, because RAM there changed.
    pub fn invalidate_line(&mut self, mmio: &mut Mmio<SYSTEM>, line: u32) {
        let Some(pcs) = self.blocks_by_line.remove(&line) else {
            return;
        };
        for pc in pcs {
            self.forget(mmio, pc);
        }
    }

    fn lookup_or_build(&mut self, sys: &mut System<SYSTEM>, pc: u32) -> u32 {
        let slot = ((pc >> 2) & LOOKUP_MASK) as usize;
        let index = self.lookup[slot];
        if index != NO_BLOCK && self.blocks[index as usize].as_ref().is_some_and(|b| b.start_pc == pc) {
            return index;
        }
        if let Some(&index) = self.by_pc.get(&pc) {
            self.lookup[slot] = index;
            return index;
        }

        let block = Self::build(sys, pc);
        let lines = block.lines.clone();
        let index = match self.free.pop() {
            Some(index) => {
                self.blocks[index as usize] = Some(block);
                index
            }
            None => {
                self.blocks.push(Some(block));
                (self.blocks.len() - 1) as u32
            }
        };
        self.lookup[slot] = index;
        self.by_pc.insert(pc, index);
        for line in lines {
            self.blocks_by_line.entry(line).or_default().push(pc);
            sys.mmio.mark_code(line, CODE_LINE_BYTES);
        }
        index
    }

    fn build(sys: &System<SYSTEM>, pc: u32) -> Block<SYSTEM> {
        let spec = block::discover(sys, pc);
        let steps = spec
            .instrs
            .iter()
            .zip(&spec.pcs)
            .map(|(&word, &at)| {
                let instr = Instruction(word);
                (crate::gekko::resolve::<SYSTEM>(instr), instr, at)
            })
            .collect();
        let idle = matches!(
            idle::classify::<SYSTEM>(&spec, &sys.gekko.gprs),
            IdleClass::BranchToSelf | IdleClass::PollingLoop
        );
        let mut lines = SmallVec::new();
        for &at in &spec.pcs {
            let line = mmio::virt_to_phys(at) & CODE_LINE_MASK;
            if lines.last() != Some(&line) {
                lines.push(line);
            }
        }
        Block {
            start_pc: pc,
            steps,
            end_pc: spec.end_pc(),
            idle,
            lines,
            #[cfg(feature = "wasm-jit")]
            code: None,
            #[cfg(feature = "wasm-jit")]
            runs: 0,
        }
    }

    fn forget(&mut self, mmio: &mut Mmio<SYSTEM>, pc: u32) {
        let Some(index) = self.by_pc.remove(&pc) else {
            return;
        };
        let Some(block) = self.blocks[index as usize].take() else {
            return;
        };
        let slot = ((pc >> 2) & LOOKUP_MASK) as usize;
        if self.lookup[slot] == index {
            self.lookup[slot] = NO_BLOCK;
        }
        self.free.push(index);
        #[cfg(feature = "wasm-jit")]
        if let (Some(slot), Some(compiler)) = (block.code, self.compiler.as_mut()) {
            compiler.release(slot);
        }
        for line in block.lines {
            mmio.unmark_code(line, CODE_LINE_BYTES);
            if let Some(pcs) = self.blocks_by_line.get_mut(&line) {
                pcs.retain(|p| *p != pc);
                if pcs.is_empty() {
                    self.blocks_by_line.remove(&line);
                }
            }
        }
    }
}

/// The registers a block may change, for holding the compiled run against the interpreter.
#[cfg(feature = "wasm-jit")]
#[derive(PartialEq)]
struct Snapshot {
    gprs: [u32; 32],
    fprs: [[u64; 2]; 32],
    pc: u32,
    cr: u32,
    xer: u32,
    lr: u32,
    ctr: u32,
    msr: u32,
    fpscr: u32,
    cycles: u64,
    srr0: u32,
    srr1: u32,
    sprg: [u32; 4],
}

#[cfg(feature = "wasm-jit")]
impl Snapshot {
    fn of<const SYSTEM: SystemId>(sys: &System<SYSTEM>) -> Self {
        let g = &sys.gekko;
        Self {
            gprs: g.gprs,
            fprs: g.fprs.0.map(|pair| [pair[0].to_bits(), pair[1].to_bits()]),
            pc: g.pc,
            cr: g.cr.raw(),
            xer: g.spr.xer.raw(),
            lr: g.spr.lr,
            ctr: g.spr.ctr,
            msr: g.msr.raw(),
            fpscr: g.fpscr.raw(),
            cycles: sys.scheduler.cycles,
            srr0: g.spr.srr0.raw(),
            srr1: g.spr.srr1,
            sprg: [g.spr.sprg0, g.spr.sprg1, g.spr.sprg2, g.spr.sprg3],
        }
    }

    fn restore<const SYSTEM: SystemId>(&self, sys: &mut System<SYSTEM>) {
        let g = &mut sys.gekko;
        g.gprs = self.gprs;
        g.fprs.0 = self.fprs.map(|pair| [f64::from_bits(pair[0]), f64::from_bits(pair[1])]);
        g.pc = self.pc;
        g.cr = crate::gekko::condition::ConditionRegister::from(self.cr);
        g.spr.xer = crate::gekko::spr::Xer::from(self.xer);
        g.spr.lr = self.lr;
        g.spr.ctr = self.ctr;
        g.msr = crate::gekko::msr::Msr::from(self.msr);
        g.fpscr = crate::gekko::fpscr::Fpscr::from(self.fpscr);
        g.spr.srr0 = crate::gekko::spr::Srr0::from(self.srr0);
        g.spr.srr1 = self.srr1;
        [g.spr.sprg0, g.spr.sprg1, g.spr.sprg2, g.spr.sprg3] = self.sprg;
        sys.scheduler.cycles = self.cycles;
    }

    /// What differs between this (the compiled run) and the interpreter's.
    fn describe(&self, other: &Self) -> String {
        let mut out = String::new();
        for i in 0..32 {
            if self.gprs[i] != other.gprs[i] {
                out.push_str(&format!(" r{i}={:08x}/{:08x}", self.gprs[i], other.gprs[i]));
            }
            if self.fprs[i] != other.fprs[i] {
                out.push_str(&format!(
                    " f{i}={:016x}:{:016x}/{:016x}:{:016x}",
                    self.fprs[i][0], self.fprs[i][1], other.fprs[i][0], other.fprs[i][1]
                ));
            }
        }
        for (name, a, b) in [
            ("pc", self.pc, other.pc),
            ("cr", self.cr, other.cr),
            ("xer", self.xer, other.xer),
            ("lr", self.lr, other.lr),
            ("ctr", self.ctr, other.ctr),
            ("msr", self.msr, other.msr),
            ("fpscr", self.fpscr, other.fpscr),
            ("srr0", self.srr0, other.srr0),
            ("srr1", self.srr1, other.srr1),
        ] {
            if a != b {
                out.push_str(&format!(" {name}={a:08x}/{b:08x}"));
            }
        }
        if self.cycles != other.cycles {
            out.push_str(&format!(" cycles={}/{}", self.cycles, other.cycles));
        }
        out
    }
}
