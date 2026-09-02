//! Blocks decoded once and run until a branch leaves them. Stepping one instruction at a
//! time pays the fetch, the address translation and the walk down the dispatch tree
//! every time; a block pays them once and keeps the handlers. Interrupts are taken at
//! block boundaries, as the JIT takes them, and a block the JIT would call idle skips to
//! the next deadline after one pass. RAM written under a block, by a store or a DMA,
//! reaches here through the same pending lines the JIT invalidates from.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::gekko::block;
use crate::gekko::idle::{self, IdleClass};
use crate::gekko::instruction::Instruction;
use crate::mmio::{self, CODE_LINE_BYTES, CODE_LINE_MASK, Mmio};
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
}

pub struct BlockCache<const SYSTEM: SystemId> {
    /// Direct-mapped by pc, ahead of the map, as the JIT's lookup table is.
    lookup: Box<[u32]>,
    blocks: Vec<Option<Block<SYSTEM>>>,
    free: Vec<u32>,
    by_pc: FxHashMap<u32, u32>,
    blocks_by_line: FxHashMap<u32, SmallVec<[u32; 2]>>,
}

impl<const SYSTEM: SystemId> Default for BlockCache<SYSTEM> {
    fn default() -> Self {
        Self {
            lookup: vec![NO_BLOCK; LOOKUP_SLOTS].into_boxed_slice(),
            blocks: Vec::new(),
            free: Vec::new(),
            by_pc: FxHashMap::default(),
            blocks_by_line: FxHashMap::default(),
        }
    }
}

impl<const SYSTEM: SystemId> BlockCache<SYSTEM> {
    /// Runs the block at the pc, building it first if it is new, and leaves the pc where
    /// control went. True when the block is an idle loop that just closed on itself.
    pub fn run_block(&mut self, sys: &mut System<SYSTEM>) -> bool {
        let pc = sys.gekko.pc;
        let index = self.lookup_or_build(sys, pc);
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
