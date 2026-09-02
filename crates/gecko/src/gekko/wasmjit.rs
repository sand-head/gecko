//! Blocks compiled to WebAssembly, for a host that is WebAssembly itself and can run
//! nothing else. Each block becomes a module of one function that imports the host's
//! memory and function table, so it reads the console where the interpreter does and
//! calls what the interpreter calls. The host instantiates it and puts the function in a
//! slot of its own table, and to Rust on wasm32 a slot is a function pointer.
//!
//! The emitter knows an instruction by the handler the resolver picked for it, which the
//! generated tables name by its `OP_*` constant, and translates the ones worth
//! translating: integer arithmetic, rotates,
//! compares, branches, loads and stores against RAM, and the floating-point and
//! paired-single arithmetic a game spends its time in. Everything else calls the
//! interpreter's handler by its own slot, so every instruction means what the interpreter
//! says it means, and a translation is held to the same standard: the interpreter is the
//! oracle, and a block that disagrees with it is a bug here.

use wasm_encoder::{
    BlockType, CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection, ImportSection,
    InstructionSink, MemArg, MemoryType, Module, RefType, TableType, TypeSection, ValType,
};

use crate::gekko::abi;
use crate::gekko::cycles::cycles_for_op;
use crate::gekko::instruction::Instruction;
use crate::gekko::interpreter::{FRC_KEEP_MASK, FRC_ROUND_BIT};
use crate::gekko::lut::*;
use crate::mmio::constants::RAM_END;
use crate::system::{System, SystemId};

/// Debug: which translations are on. Bits: 1 alu, 2 logical, 4 rotate, 8 compare, 16 cr,
/// 32 nop, 64 spr, 128 load, 256 store, 512 fp, 1024 ps, 2048 fp compare, 4096 fp
/// load/store, 8192 psq load, 16384 branch, 32768 psq store.
pub static TRANSLATE: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(u32::MAX);

/// The host's side of the bargain: module bytes in, a slot of its function table out.
pub trait BlockCompiler {
    fn compile(&mut self, module: &[u8]) -> Option<u32>;
    fn release(&mut self, slot: u32);
}

/// What a slot holds: the console in, the address control went to out.
pub type BlockFn<const SYSTEM: SystemId> = extern "C" fn(*mut System<SYSTEM>) -> u32;

/// The module's imports, and the name of the one function it exports.
pub const IMPORT_MODULE: &str = "e";
pub const MEMORY_IMPORT: &str = "m";
pub const TABLE_IMPORT: &str = "t";
pub const BLOCK_EXPORT: &str = "b";

// The module's function types, in the order the type section declares them.
const T_BLOCK: u32 = 0;
const T_HANDLER: u32 = 1;
const T_READ: u32 = 2;
const T_WRITE: u32 = 3;
const T_READ_F64: u32 = 4;
const T_WRITE_F64: u32 = 5;
const T_RAISE: u32 = 6;
const T_FMA: u32 = 7;

// The block function's locals: the console, scratch, and two facts about MSR that hold
// for the whole block because the instructions that change it end one.
const CTX: u32 = 0;
const A: u32 = 1;
const V: u32 = 2;
const T: u32 = 3;
const P: u32 = 4;
const FP_OK: u32 = 5;
const FE: u32 = 6;
const F: u32 = 7;
const G: u32 = 8;
const Q: u32 = 9;
const S: u32 = 10;

const MSR_FP: i32 = 1 << (31 - 18);
const MSR_FE: i32 = (1 << (31 - 20)) | (1 << (31 - 23));
const FPSCR_FEX: i32 = 1 << (31 - 1);
const XER_SO_SHIFT: i32 = 31;
const XER_CA: i32 = 1 << (31 - 2);

/// A compiled block disagreed with the interpreter: each block is reported once, and
/// after a couple of dozen the rest are the same bugs.
pub fn report_disagreement(start_pc: u32, steps: &[(u16, Instruction, u32)], diff: &str) {
    use std::sync::Mutex;
    static REPORTED: Mutex<Vec<u32>> = Mutex::new(Vec::new());
    let mut reported = REPORTED.lock().unwrap();
    if reported.len() < 24 && !reported.contains(&start_pc) {
        reported.push(start_pc);
        let words: Vec<String> = steps.iter().map(|(_, instr, _)| format!("{:08x}", instr.0)).collect();
        tracing::warn!(
            pc = format!("{start_pc:08x}"),
            block = words.join(" "),
            diff,
            "compiled block disagrees"
        );
    }
}

/// Debug: run every compiled block through the interpreter too, from the same registers
/// and the same RAM, and report the first disagreements. A block that writes a device
/// register and reads it back will disagree with itself, which is noise; a block that
/// disagrees about anything else is a translation bug. Blocks compiled while this is
/// on carry the store log call, so turn it on before the console boots.
pub mod validate {
    use core::sync::atomic::{AtomicBool, Ordering};
    use std::cell::RefCell;

    use crate::system::{System, SystemId};

    pub static ON: AtomicBool = AtomicBool::new(false);

    thread_local! {
        static STORES: RefCell<Vec<(u32, u8, u32, u32)>> = const { RefCell::new(Vec::new()) };
    }

    pub fn on() -> bool {
        ON.load(Ordering::Relaxed)
    }

    /// Notes the value about to be overwritten at physical `phys`.
    pub extern "C" fn note_store<const SYSTEM: SystemId>(ctx: *mut System<SYSTEM>, phys: u32, width: u32) {
        let mmio = unsafe { &(*ctx).mmio };
        let old = match width {
            1 => mmio.ram_read_u8(phys) as u32,
            2 => mmio.ram_read_u16(phys) as u32,
            _ => mmio.ram_read_u32(phys),
        };
        STORES.with(|stores| stores.borrow_mut().push((phys, width as u8, old, 0)));
    }

    /// The same for a store the bus made, when it went to RAM through the usual segments.
    pub fn note_bus_store<const SYSTEM: SystemId>(ctx: *mut System<SYSTEM>, ea: u32, width: u32) {
        let phys = crate::mmio::virt_to_phys(ea);
        if (ea >> 28) | 4 == 12 && phys <= crate::mmio::constants::RAM_END - (width - 1) {
            note_store(ctx, phys, width);
        }
    }

    pub fn begin() {
        STORES.with(|stores| stores.borrow_mut().clear());
    }

    fn read<const SYSTEM: SystemId>(sys: &System<SYSTEM>, phys: u32, width: u8) -> u32 {
        match width {
            1 => sys.mmio.ram_read_u8(phys) as u32,
            2 => sys.mmio.ram_read_u16(phys) as u32,
            _ => sys.mmio.ram_read_u32(phys),
        }
    }

    /// Puts RAM back the way it was before the block, remembering what the block left.
    pub fn undo<const SYSTEM: SystemId>(sys: &mut System<SYSTEM>) {
        STORES.with(|stores| {
            let mut stores = stores.borrow_mut();
            for entry in stores.iter_mut() {
                entry.3 = read(sys, entry.0, entry.1);
            }
            for &(phys, width, old, _) in stores.iter().rev() {
                match width {
                    1 => sys.mmio.ram_write_u8(phys, old as u8),
                    2 => sys.mmio.ram_write_u16(phys, old as u16),
                    _ => sys.mmio.ram_write_u32(phys, old),
                }
            }
        });
    }

    /// After the interpreter's run: the RAM the block stored to, where the interpreter
    /// left something else.
    pub fn stores_that_differ<const SYSTEM: SystemId>(sys: &System<SYSTEM>) -> String {
        STORES.with(|stores| {
            let mut out = String::new();
            for &(phys, width, _, compiled) in stores.borrow().iter() {
                let interpreted = read(sys, phys, width);
                if interpreted != compiled {
                    out.push_str(&format!(" [{phys:08x}]{width}={compiled:x}/{interpreted:x}"));
                }
            }
            out
        })
    }
}

// ---- what a block calls -----------------------------------------------------------

extern "C" fn read_u8<const SYSTEM: SystemId>(ctx: *mut System<SYSTEM>, ea: u32) -> u32 {
    unsafe { (*ctx).read_u8(ea) as u32 }
}

extern "C" fn read_u16<const SYSTEM: SystemId>(ctx: *mut System<SYSTEM>, ea: u32) -> u32 {
    unsafe { (*ctx).read_u16(ea) as u32 }
}

extern "C" fn read_u32<const SYSTEM: SystemId>(ctx: *mut System<SYSTEM>, ea: u32) -> u32 {
    unsafe { (*ctx).read_u32(ea) }
}

extern "C" fn write_u8<const SYSTEM: SystemId>(ctx: *mut System<SYSTEM>, ea: u32, value: u32) {
    if validate::on() {
        validate::note_bus_store(ctx, ea, 1);
    }
    unsafe { (*ctx).write_u8(ea, value as u8) }
}

extern "C" fn write_u16<const SYSTEM: SystemId>(ctx: *mut System<SYSTEM>, ea: u32, value: u32) {
    if validate::on() {
        validate::note_bus_store(ctx, ea, 2);
    }
    unsafe { (*ctx).write_u16(ea, value as u16) }
}

extern "C" fn write_u32<const SYSTEM: SystemId>(ctx: *mut System<SYSTEM>, ea: u32, value: u32) {
    if validate::on() {
        validate::note_bus_store(ctx, ea, 4);
    }
    unsafe { (*ctx).write_u32(ea, value) }
}

extern "C" fn read_f64<const SYSTEM: SystemId>(ctx: *mut System<SYSTEM>, ea: u32) -> f64 {
    unsafe { (*ctx).read_f64(ea) }
}

extern "C" fn write_f64<const SYSTEM: SystemId>(ctx: *mut System<SYSTEM>, ea: u32, value: f64) {
    if validate::on() {
        validate::note_bus_store(ctx, ea, 4);
        validate::note_bus_store(ctx, ea.wrapping_add(4), 4);
    }
    unsafe { (*ctx).write_f64(ea, value) }
}

extern "C" fn fp_program_exception<const SYSTEM: SystemId>(ctx: *mut System<SYSTEM>) {
    unsafe { (*ctx).cause_fp_program_exception() }
}

/// WebAssembly has no fused multiply-add; the interpreter's is exact, and so is this.
extern "C" fn fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

/// Where the block finds each of them: on wasm32 a function pointer is an index into the
/// module's function table.
struct Helpers {
    read_u8: i32,
    read_u16: i32,
    read_u32: i32,
    write_u8: i32,
    write_u16: i32,
    write_u32: i32,
    read_f64: i32,
    write_f64: i32,
    raise_fp: i32,
    fma: i32,
    note_store: i32,
}

impl Helpers {
    fn for_system<const SYSTEM: SystemId>() -> Self {
        Self {
            read_u8: read_u8::<SYSTEM> as *const () as usize as i32,
            read_u16: read_u16::<SYSTEM> as *const () as usize as i32,
            read_u32: read_u32::<SYSTEM> as *const () as usize as i32,
            write_u8: write_u8::<SYSTEM> as *const () as usize as i32,
            write_u16: write_u16::<SYSTEM> as *const () as usize as i32,
            write_u32: write_u32::<SYSTEM> as *const () as usize as i32,
            read_f64: read_f64::<SYSTEM> as *const () as usize as i32,
            write_f64: write_f64::<SYSTEM> as *const () as usize as i32,
            raise_fp: fp_program_exception::<SYSTEM> as *const () as usize as i32,
            fma: fma as *const () as usize as i32,
            note_store: validate::note_store::<SYSTEM> as *const () as usize as i32,
        }
    }
}

// ---- the module -------------------------------------------------------------------

/// Emits the module for a block, given each instruction's handler, word and address in
/// the order they run, and the address after the last. `sys` is the console the block
/// will run on: RAM and the code-line counts never move, so their addresses are baked in
/// (truncated to what a 32-bit memory can hold, which on wasm32 they already are).
pub fn emit<const SYSTEM: SystemId>(sys: &System<SYSTEM>, steps: &[(u16, Instruction, u32)], end_pc: u32) -> Vec<u8> {
    emit_with(sys, steps, end_pc, Helpers::for_system::<SYSTEM>())
}

fn emit_with<const SYSTEM: SystemId>(
    sys: &System<SYSTEM>,
    steps: &[(u16, Instruction, u32)],
    end_pc: u32,
    helpers: Helpers,
) -> Vec<u8> {
    let mut body = Function::new([
        (6, ValType::I32),
        (2, ValType::F64),
        (1, ValType::I64),
        (1, ValType::F32),
    ]);
    {
        let mut t = Translator::<SYSTEM> {
            code: body.instructions(),
            helpers,
            ram: sys.mmio.ram_ptr as u32 as u64,
            code_lines: sys.mmio.code_refcount_ptr as u32 as u64,
            pending: 0,
            mask: TRANSLATE.load(core::sync::atomic::Ordering::Relaxed),
        };
        t.prologue();
        let last = steps.len().saturating_sub(1);
        for (i, &(leaf, instr, at)) in steps.iter().enumerate() {
            let expected = if i == last { end_pc } else { steps[i + 1].2 };
            t.step(leaf, instr, at, expected, i == last);
        }
        t.exit_const(end_pc);
        t.code.end();
    }

    let mut types = TypeSection::new();
    let i32s = |n: usize| core::iter::repeat_n(ValType::I32, n);
    types.ty().function(i32s(1), [ValType::I32]);
    types.ty().function(i32s(2), []);
    types.ty().function(i32s(2), [ValType::I32]);
    types.ty().function(i32s(3), []);
    types.ty().function(i32s(2), [ValType::F64]);
    types.ty().function([ValType::I32, ValType::I32, ValType::F64], []);
    types.ty().function(i32s(1), []);
    types
        .ty()
        .function([ValType::F64, ValType::F64, ValType::F64], [ValType::F64]);
    let mut imports = ImportSection::new();
    imports.import(
        IMPORT_MODULE,
        MEMORY_IMPORT,
        EntityType::Memory(MemoryType {
            minimum: 0,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        }),
    );
    imports.import(
        IMPORT_MODULE,
        TABLE_IMPORT,
        EntityType::Table(TableType {
            element_type: RefType::FUNCREF,
            table64: false,
            minimum: 0,
            maximum: None,
            shared: false,
        }),
    );
    let mut functions = FunctionSection::new();
    functions.function(T_BLOCK);
    let mut exports = ExportSection::new();
    exports.export(BLOCK_EXPORT, ExportKind::Func, 0);
    let mut codes = CodeSection::new();
    codes.function(&body);

    let mut module = Module::new();
    module
        .section(&types)
        .section(&imports)
        .section(&functions)
        .section(&exports)
        .section(&codes);
    module.finish()
}

fn field(offset: usize) -> MemArg {
    MemArg {
        offset: offset as u64,
        align: 2,
        memory_index: 0,
    }
}

fn wide(offset: usize) -> MemArg {
    MemArg {
        offset: offset as u64,
        align: 3,
        memory_index: 0,
    }
}

fn at(base: u64, align: u32) -> MemArg {
    MemArg {
        offset: base,
        align,
        memory_index: 0,
    }
}

fn rlw_mask(mb: u32, me: u32) -> u32 {
    let begin = 0xFFFF_FFFFu32 >> mb;
    let end = if me >= 31 { 0 } else { 0xFFFF_FFFFu32 >> (me + 1) };
    if mb <= me {
        begin & !end
    } else {
        begin | !end
    }
}

struct Translator<'a, const SYSTEM: SystemId> {
    code: InstructionSink<'a>,
    helpers: Helpers,
    ram: u64,
    code_lines: u64,
    /// Cycles the translated instructions so far cost, added to the console's count
    /// before anything that could observe it.
    pending: i64,
    mask: u32,
}

macro_rules! translate {
    ($self:ident, $op:ident; $($known:ident => $body:expr;)*) => {
        $(
            if $op == $known {
                $self.pending += cycles_for_op($known);
                $body;
                return true;
            }
        )*
    };
}

impl<const SYSTEM: SystemId> Translator<'_, SYSTEM> {
    // -- the console's fields ---------------------------------------------------------

    fn gpr(r: u8) -> MemArg {
        field(abi::gpr_base_offset::<SYSTEM>() + 4 * r as usize)
    }

    fn fpr(r: u8) -> MemArg {
        wide(abi::fpr_base_offset::<SYSTEM>() + 16 * r as usize)
    }

    fn ps1(r: u8) -> MemArg {
        wide(abi::ps1_base_offset::<SYSTEM>() + 16 * r as usize)
    }

    fn cr() -> MemArg {
        field(abi::cr_offset::<SYSTEM>())
    }

    fn xer() -> MemArg {
        field(abi::xer_offset::<SYSTEM>())
    }

    fn lr() -> MemArg {
        field(abi::lr_offset::<SYSTEM>())
    }

    fn ctr() -> MemArg {
        field(abi::ctr_offset::<SYSTEM>())
    }

    fn cia() -> MemArg {
        field(abi::cia_offset::<SYSTEM>())
    }

    fn nia() -> MemArg {
        field(abi::nia_offset::<SYSTEM>())
    }

    fn prologue(&mut self) {
        self.code
            .local_get(CTX)
            .i32_load(field(abi::msr_offset::<SYSTEM>()))
            .local_tee(T)
            .i32_const(MSR_FP)
            .i32_and()
            .local_set(FP_OK)
            .local_get(T)
            .i32_const(MSR_FE)
            .i32_and()
            .local_set(FE);
    }

    // -- leaving the block --------------------------------------------------------------

    fn flush_cycles(&mut self) {
        if self.pending != 0 {
            let cycles = wide(abi::cycles_offset::<SYSTEM>());
            self.code
                .local_get(CTX)
                .local_get(CTX)
                .i64_load(cycles)
                .i64_const(self.pending)
                .i64_add()
                .i64_store(cycles);
            self.pending = 0;
        }
    }

    /// A call the bus may answer with the clock in hand: the interpreter charges an
    /// instruction before running it, so a device it reaches sees every cycle so far.
    /// The pending ones are lent for the call and taken back after, since the other
    /// path through the block still owes them at its end.
    fn bus_call(&mut self, call: impl FnOnce(&mut Self)) {
        let pending = self.pending;
        if pending != 0 {
            self.charge_now(pending);
        }
        call(self);
        if pending != 0 {
            self.charge_now(-pending);
        }
    }

    /// Charges cycles on this path alone, right now.
    fn charge_now(&mut self, cycles: i64) {
        let field = wide(abi::cycles_offset::<SYSTEM>());
        self.code
            .local_get(CTX)
            .local_get(CTX)
            .i64_load(field)
            .i64_const(cycles)
            .i64_add()
            .i64_store(field);
    }

    /// Adds the pending cycles without forgetting them, for a path that returns while
    /// another still needs them.
    fn flush_cycles_keeping(&mut self) {
        let pending = self.pending;
        self.flush_cycles();
        self.pending = pending;
    }

    fn exit_const(&mut self, pc: u32) {
        self.flush_cycles();
        self.code.i32_const(pc as i32).return_();
    }

    /// Leaves with the address the console's `nia` holds. Like `exit_const`, only for a
    /// path every other path has already left: it settles the pending cycles for good.
    fn exit_nia(&mut self) {
        self.flush_cycles();
        self.code.local_get(CTX).i32_load(Self::nia()).return_();
    }

    /// The interpreter's handler for the instruction, with the addresses it expects set
    /// and, unless the block ends here anyway, a check that it did not send control away.
    fn call_handler(&mut self, leaf: u16, instr: Instruction, at: u32, expected: u32, last: bool) {
        self.flush_cycles();
        self.code
            .local_get(CTX)
            .i32_const(at as i32)
            .i32_store(Self::cia())
            .local_get(CTX)
            .i32_const(expected as i32)
            .i32_store(Self::nia())
            .local_get(CTX)
            .i32_const(instr.0 as i32)
            .i32_const(crate::gekko::handler_address::<SYSTEM>(leaf) as i32)
            .call_indirect(0, T_HANDLER);
        if last {
            self.exit_nia();
        } else {
            self.code
                .local_get(CTX)
                .i32_load(Self::nia())
                .i32_const(expected as i32)
                .i32_ne()
                .if_(BlockType::Empty);
            self.exit_nia();
            self.code.end();
        }
    }

    // -- one instruction ----------------------------------------------------------------

    fn step(&mut self, leaf: u16, instr: Instruction, at: u32, expected: u32, last: bool) {
        let op = crate::gekko::op_of::<SYSTEM>(leaf);
        if last && self.mask & 16384 != 0 && self.terminator(op, instr, at, expected) {
            return;
        }
        if self.integer(op, instr) || self.memory(op, instr) || self.floating(op, leaf, instr, at, expected, last) {
            return;
        }
        self.call_handler(leaf, instr, at, expected, last);
    }

    fn integer(&mut self, op: u32, instr: Instruction) -> bool {
        let rc = instr.rc();
        let mask = self.mask;
        if mask & 1 != 0 {
            translate!(self, op;
                OP_ADDI => self.add_imm(instr, instr.simm());
                OP_ADDIS => self.add_imm(instr, instr.simm() << 16);
                OP_ORI => self.logic_imm(instr, instr.uimm() as i32, false, |c| { c.i32_or(); });
                OP_ORIS => self.logic_imm(instr, (instr.uimm() as i32) << 16, false, |c| { c.i32_or(); });
                OP_XORI => self.logic_imm(instr, instr.uimm() as i32, false, |c| { c.i32_xor(); });
                OP_XORIS => self.logic_imm(instr, (instr.uimm() as i32) << 16, false, |c| { c.i32_xor(); });
                OP_ANDI_DOT => self.logic_imm(instr, instr.uimm() as i32, true, |c| { c.i32_and(); });
                OP_ANDIS_DOT => self.logic_imm(instr, (instr.uimm() as i32) << 16, true, |c| { c.i32_and(); });
                OP_ADDIC => self.add_imm_carry(instr, false);
                OP_ADDIC_DOT => self.add_imm_carry(instr, true);
                OP_SUBFIC => self.subfic(instr);
                OP_MULLI => self.mulli(instr);
            );
        }
        if mask & 1 != 0 && !instr.oe() {
            translate!(self, op;
                OP_ADDX => self.arith(instr, rc, |t| {
                    t.load_gpr(instr.ra());
                    t.load_gpr(instr.rb());
                    t.code.i32_add();
                });
                OP_SUBFX => self.arith(instr, rc, |t| {
                    t.load_gpr(instr.rb());
                    t.load_gpr(instr.ra());
                    t.code.i32_sub();
                });
                OP_NEGX => self.arith(instr, rc, |t| {
                    t.code.i32_const(0);
                    t.load_gpr(instr.ra());
                    t.code.i32_sub();
                });
                OP_MULLWX => self.arith(instr, rc, |t| {
                    t.load_gpr(instr.ra());
                    t.load_gpr(instr.rb());
                    t.code.i32_mul();
                });
                OP_MULHWX => self.arith(instr, rc, |t| {
                    t.load_gpr(instr.ra());
                    t.code.i64_extend_i32_s();
                    t.load_gpr(instr.rb());
                    t.code.i64_extend_i32_s().i64_mul().i64_const(32).i64_shr_s().i32_wrap_i64();
                });
                OP_MULHWUX => self.arith(instr, rc, |t| {
                    t.load_gpr(instr.ra());
                    t.code.i64_extend_i32_u();
                    t.load_gpr(instr.rb());
                    t.code.i64_extend_i32_u().i64_mul().i64_const(32).i64_shr_u().i32_wrap_i64();
                });
            );
        }
        if mask & 2 != 0 {
            translate!(self, op;
                OP_ANDX => self.logic(instr, |c| { c.i32_and(); });
                OP_ORX => self.logic(instr, |c| { c.i32_or(); });
                OP_XORX => self.logic(instr, |c| { c.i32_xor(); });
                OP_NANDX => self.logic(instr, |c| { c.i32_and().i32_const(-1).i32_xor(); });
                OP_NORX => self.logic(instr, |c| { c.i32_or().i32_const(-1).i32_xor(); });
                OP_EQVX => self.logic(instr, |c| { c.i32_xor().i32_const(-1).i32_xor(); });
                OP_ANDCX => self.logic(instr, |c| { c.i32_const(-1).i32_xor().i32_and(); });
                OP_ORCX => self.logic(instr, |c| { c.i32_const(-1).i32_xor().i32_or(); });
                OP_SLWX => self.shift_by_register(instr, |c| { c.i32_shl(); });
                OP_SRWX => self.shift_by_register(instr, |c| { c.i32_shr_u(); });
                OP_SRAWIX => self.srawi(instr);
                OP_CNTLZWX => self.unary(instr, |c| { c.i32_clz(); });
                OP_EXTSHX => self.unary(instr, |c| { c.i32_extend16_s(); });
                OP_EXTSBX => self.unary(instr, |c| { c.i32_extend8_s(); });
            );
        }
        if mask & 4 != 0 {
            translate!(self, op;
                OP_RLWINMX => self.rlwinm(instr);
                OP_RLWIMIX => self.rlwimi(instr);
                OP_RLWNMX => self.rlwnm(instr);
            );
        }
        if mask & 8 != 0 {
            translate!(self, op;
                OP_CMP => self.compare(instr, true, |t| t.load_gpr(instr.rb()));
                OP_CMPL => self.compare(instr, false, |t| t.load_gpr(instr.rb()));
                OP_CMPI => self.compare(instr, true, |t| { t.code.i32_const(instr.simm()); });
                OP_CMPLI => self.compare(instr, false, |t| { t.code.i32_const(instr.uimm() as i32); });
            );
        }
        if mask & 16 != 0 {
            translate!(self, op;
                OP_MCRF => self.mcrf(instr);
                OP_MFCR => self.store_gpr(instr.rd(), |t| { t.code.local_get(CTX).i32_load(Self::cr()); });
                OP_MTCRF => self.mtcrf(instr);
                OP_CRXOR => self.cr_bit(instr, |c| { c.i32_xor(); });
                OP_CROR => self.cr_bit(instr, |c| { c.i32_or(); });
                OP_CRAND => self.cr_bit(instr, |c| { c.i32_and(); });
                OP_CREQV => self.cr_bit(instr, |c| { c.i32_eq(); });
                OP_CRNOR => self.cr_bit(instr, |c| { c.i32_or().i32_eqz(); });
                OP_CRNAND => self.cr_bit(instr, |c| { c.i32_and().i32_eqz(); });
                OP_CRANDC => self.cr_bit(instr, |c| { c.i32_eqz().i32_and(); });
                OP_CRORC => self.cr_bit(instr, |c| { c.i32_eqz().i32_or(); });
            );
        }
        if mask & 32 != 0 {
            translate!(self, op;
                OP_DCBT => ();
                OP_DCBTST => ();
            );
        }
        if mask & 64 == 0 {
            return false;
        }
        match instr.spr_swapped() {
            8 => {
                translate!(self, op;
                OP_MFSPR => self.store_gpr(instr.rd(), |t| { t.code.local_get(CTX).i32_load(Self::lr()); });
                OP_MTSPR => self.store_field(Self::lr(), |t| t.load_gpr(instr.rs()));
                );
            }
            9 => {
                translate!(self, op;
                OP_MFSPR => self.store_gpr(instr.rd(), |t| { t.code.local_get(CTX).i32_load(Self::ctr()); });
                OP_MTSPR => self.store_field(Self::ctr(), |t| t.load_gpr(instr.rs()));
                );
            }
            1 => {
                translate!(self, op;
                OP_MFSPR => self.store_gpr(instr.rd(), |t| { t.code.local_get(CTX).i32_load(Self::xer()); });
                );
            }
            _ => {}
        }
        false
    }

    fn memory(&mut self, op: u32, instr: Instruction) -> bool {
        let mask = self.mask;
        if mask & 128 != 0 {
            translate!(self, op;
                OP_LWZ => self.load(instr, Width::Word, Form::D, false, false);
                OP_LWZU => self.load(instr, Width::Word, Form::D, true, false);
                OP_LWZX => self.load(instr, Width::Word, Form::X, false, false);
                OP_LWZUX => self.load(instr, Width::Word, Form::X, true, false);
                OP_LHZ => self.load(instr, Width::Half, Form::D, false, false);
                OP_LHZU => self.load(instr, Width::Half, Form::D, true, false);
                OP_LHZX => self.load(instr, Width::Half, Form::X, false, false);
                OP_LHZUX => self.load(instr, Width::Half, Form::X, true, false);
                OP_LHA => self.load(instr, Width::Half, Form::D, false, true);
                OP_LHAU => self.load(instr, Width::Half, Form::D, true, true);
                OP_LHAX => self.load(instr, Width::Half, Form::X, false, true);
                OP_LHAUX => self.load(instr, Width::Half, Form::X, true, true);
                OP_LBZ => self.load(instr, Width::Byte, Form::D, false, false);
                OP_LBZU => self.load(instr, Width::Byte, Form::D, true, false);
                OP_LBZX => self.load(instr, Width::Byte, Form::X, false, false);
                OP_LBZUX => self.load(instr, Width::Byte, Form::X, true, false);
            );
        }
        if mask & 256 != 0 {
            translate!(self, op;
                OP_STW => self.store(instr, Width::Word, Form::D, false);
                OP_STWU => self.store(instr, Width::Word, Form::D, true);
                OP_STWX => self.store(instr, Width::Word, Form::X, false);
                OP_STWUX => self.store(instr, Width::Word, Form::X, true);
                OP_STH => self.store(instr, Width::Half, Form::D, false);
                OP_STHU => self.store(instr, Width::Half, Form::D, true);
                OP_STHX => self.store(instr, Width::Half, Form::X, false);
                OP_STHUX => self.store(instr, Width::Half, Form::X, true);
                OP_STB => self.store(instr, Width::Byte, Form::D, false);
                OP_STBU => self.store(instr, Width::Byte, Form::D, true);
                OP_STBX => self.store(instr, Width::Byte, Form::X, false);
                OP_STBUX => self.store(instr, Width::Byte, Form::X, true);
            );
        }
        false
    }

    fn floating(&mut self, op: u32, leaf: u16, instr: Instruction, at: u32, expected: u32, last: bool) -> bool {
        // Cycles are settled before the paths part: a handler adds its own and leaves,
        // and the translation's are charged on its path alone. A quantized access
        // charges its own, because it may still take the handler's path.
        macro_rules! fp {
            ($($known:ident => $body:expr;)*) => {
                $(
                    if op == $known {
                        self.flush_cycles();
                        self.code.local_get(FP_OK).i32_eqz().if_(BlockType::Empty);
                        self.call_handler(leaf, instr, at, expected, true);
                        self.code.end();
                        self.pending += cycles_for_op($known);
                        $body;
                        self.fp_exception_check(at, expected);
                        return true;
                    }
                )*
            };
        }
        macro_rules! quantized {
            ($($known:ident => $body:expr;)*) => {
                $(
                    if op == $known {
                        self.flush_cycles();
                        self.code.local_get(FP_OK).i32_eqz().if_(BlockType::Empty);
                        self.call_handler(leaf, instr, at, expected, true);
                        self.code.end();
                        $body;
                        self.fp_exception_check(at, expected);
                        return true;
                    }
                )*
            };
        }
        let mask = self.mask;
        if !instr.rc() && mask & 512 != 0 {
            fp!(
                OP_FMRX => self.fp_unary(instr, false, |_| {});
                OP_FNEGX => self.fp_unary(instr, false, |c| { c.f64_neg(); });
                OP_FABSX => self.fp_unary(instr, false, |c| { c.f64_abs(); });
                OP_FNABSX => self.fp_unary(instr, false, |c| { c.f64_abs().f64_neg(); });
                OP_FRSPX => self.fp_unary(instr, true, |c| { c.f32_demote_f64().f64_promote_f32(); });
                OP_FSQRTX => self.fp_unary(instr, false, |c| { c.f64_sqrt(); });
                OP_FSQRTSX => self.fp_unary(instr, true, |c| { c.f64_sqrt().f32_demote_f64().f64_promote_f32(); });
                OP_FRSQRTEX => self.fp_unary(instr, false, |c| { c.f64_sqrt().local_set(F).f64_const(1.0.into()).local_get(F).f64_div(); });
                OP_FRESX => self.fp_unary(instr, true, |c| { c.f32_demote_f64().local_set(S).f32_const(1.0.into()).local_get(S).f32_div().f64_promote_f32(); });
                OP_FCTIWZX => self.fp_unary(instr, false, |c| { c.i32_trunc_sat_f64_s().i64_extend_i32_u().f64_reinterpret_i64(); });
                OP_FADDX => self.fp_binary(instr, false, Operand::B, |c| { c.f64_add(); });
                OP_FSUBX => self.fp_binary(instr, false, Operand::B, |c| { c.f64_sub(); });
                OP_FMULX => self.fp_binary(instr, false, Operand::C, |c| { c.f64_mul(); });
                OP_FDIVX => self.fp_binary(instr, false, Operand::B, |c| { c.f64_div(); });
                OP_FADDSX => self.fp_binary(instr, true, Operand::B, |c| { c.f64_add(); });
                OP_FSUBSX => self.fp_binary(instr, true, Operand::B, |c| { c.f64_sub(); });
                OP_FMULSX => self.fp_binary(instr, true, Operand::RoundedC, |c| { c.f64_mul(); });
                OP_FDIVSX => self.fp_binary(instr, true, Operand::B, |c| { c.f64_div(); });
                OP_FMADDX => self.fp_madd(instr, false, false, false);
                OP_FMSUBX => self.fp_madd(instr, false, true, false);
                OP_FNMADDX => self.fp_madd(instr, false, false, true);
                OP_FNMSUBX => self.fp_madd(instr, false, true, true);
                OP_FMADDSX => self.fp_madd(instr, true, false, false);
                OP_FMSUBSX => self.fp_madd(instr, true, true, false);
                OP_FNMADDSX => self.fp_madd(instr, true, false, true);
                OP_FNMSUBSX => self.fp_madd(instr, true, true, true);
                OP_FSELX => self.fsel(instr);
            );
        }
        if !instr.rc() && mask & 1024 != 0 {
            fp!(
                OP_PS_MR => self.ps_unary(instr, |_| {});
                OP_PS_NEG => self.ps_unary(instr, |c| { c.f64_neg(); });
                OP_PS_ABS => self.ps_unary(instr, |c| { c.f64_abs(); });
                OP_PS_NABS => self.ps_unary(instr, |c| { c.f64_abs().f64_neg(); });
                OP_PS_ADD => self.ps_binary(instr, |c| { c.f64_add(); });
                OP_PS_SUB => self.ps_binary(instr, |c| { c.f64_sub(); });
                OP_PS_DIV => self.ps_binary(instr, |c| { c.f64_div(); });
                OP_PS_MUL => self.ps_mul(instr, None);
                OP_PS_MULS0 => self.ps_mul(instr, Some(0));
                OP_PS_MULS1 => self.ps_mul(instr, Some(1));
                OP_PS_MADD => self.ps_madd(instr, None, false, false);
                OP_PS_MSUB => self.ps_madd(instr, None, true, false);
                OP_PS_NMADD => self.ps_madd(instr, None, false, true);
                OP_PS_NMSUB => self.ps_madd(instr, None, true, true);
                OP_PS_MADDS0 => self.ps_madd(instr, Some(0), false, false);
                OP_PS_MADDS1 => self.ps_madd(instr, Some(1), false, false);
                OP_PS_SUM0 => self.ps_sum(instr, 0);
                OP_PS_SUM1 => self.ps_sum(instr, 1);
                OP_PS_MERGE00 => self.ps_merge(instr, 0, 0);
                OP_PS_MERGE01 => self.ps_merge(instr, 0, 1);
                OP_PS_MERGE10 => self.ps_merge(instr, 1, 0);
                OP_PS_MERGE11 => self.ps_merge(instr, 1, 1);
                OP_PS_SEL => self.ps_sel(instr);
                OP_PS_RES => self.ps_single(instr, |c| { c.f32_demote_f64().local_set(S).f32_const(1.0.into()).local_get(S).f32_div().f64_promote_f32(); });
                OP_PS_RSQRTE => self.ps_single(instr, |c| { c.f32_demote_f64().f32_sqrt().local_set(S).f32_const(1.0.into()).local_get(S).f32_div().f64_promote_f32(); });
            );
        }
        if mask & 2048 != 0 {
            fp!(
                OP_FCMPU => self.fp_compare(instr, 0);
                OP_FCMPO => self.fp_compare(instr, 0);
                OP_PS_CMPU0 => self.fp_compare(instr, 0);
                OP_PS_CMPO0 => self.fp_compare(instr, 0);
                OP_PS_CMPU1 => self.fp_compare(instr, 1);
                OP_PS_CMPO1 => self.fp_compare(instr, 1);
            );
        }
        if mask & 4096 != 0 {
            fp!(
                OP_LFS => self.load_single(instr, Form::D, false);
                OP_LFSU => self.load_single(instr, Form::D, true);
                OP_LFSX => self.load_single(instr, Form::X, false);
                OP_LFSUX => self.load_single(instr, Form::X, true);
                OP_STFS => self.store_single(instr, Form::D, false);
                OP_STFSU => self.store_single(instr, Form::D, true);
                OP_STFSX => self.store_single(instr, Form::X, false);
                OP_STFSUX => self.store_single(instr, Form::X, true);
                OP_LFD => self.load_double(instr, Form::D, false);
                OP_LFDU => self.load_double(instr, Form::D, true);
                OP_LFDX => self.load_double(instr, Form::X, false);
                OP_LFDUX => self.load_double(instr, Form::X, true);
                OP_STFD => self.store_double(instr, Form::D, false);
                OP_STFDU => self.store_double(instr, Form::D, true);
                OP_STFDX => self.store_double(instr, Form::X, false);
                OP_STFDUX => self.store_double(instr, Form::X, true);
            );
        }
        if mask & 8192 != 0 {
            quantized!(
                OP_PSQ_L => self.psq_load(instr, Form::D, false, leaf, at, expected, last);
                OP_PSQ_LU => self.psq_load(instr, Form::D, true, leaf, at, expected, last);
                OP_PSQ_LX => self.psq_load(instr, Form::X, false, leaf, at, expected, last);
                OP_PSQ_LUX => self.psq_load(instr, Form::X, true, leaf, at, expected, last);
            );
        }
        if mask & 32768 != 0 {
            quantized!(
                OP_PSQ_ST => self.psq_store(instr, Form::D, false, leaf, at, expected, last);
                OP_PSQ_STU => self.psq_store(instr, Form::D, true, leaf, at, expected, last);
                OP_PSQ_STX => self.psq_store(instr, Form::X, false, leaf, at, expected, last);
                OP_PSQ_STUX => self.psq_store(instr, Form::X, true, leaf, at, expected, last);
            );
        }
        false
    }

    /// The last instruction of a block, when it is a branch: where control goes is the
    /// block's answer, so it is computed rather than written to `nia` and read back.
    fn terminator(&mut self, op: u32, instr: Instruction, at: u32, expected: u32) -> bool {
        if op == OP_BX {
            self.pending += cycles_for_op(OP_BX);
            let target = if instr.aa() {
                instr.li() as u32
            } else {
                at.wrapping_add_signed(instr.li())
            };
            if instr.lk() {
                self.store_field(Self::lr(), |t| {
                    t.code.i32_const(at.wrapping_add(4) as i32);
                });
            }
            self.exit_const(target);
            return true;
        }
        let (counted, target): (bool, Option<u32>) = if op == OP_BCX {
            self.pending += cycles_for_op(OP_BCX);
            (
                true,
                Some(if instr.aa() {
                    instr.bd() as u32
                } else {
                    at.wrapping_add_signed(instr.bd())
                }),
            )
        } else if op == OP_BCLRX {
            self.pending += cycles_for_op(OP_BCLRX);
            (true, None)
        } else if op == OP_BCCTRX {
            self.pending += cycles_for_op(OP_BCCTRX);
            (false, None)
        } else {
            return false;
        };
        let bo = instr.bo();
        // Taken when the counter says so and the condition says so; either may be
        // waived. bcctr never touches the counter.
        self.code.i32_const(1);
        if counted && bo & 0x04 == 0 {
            self.code
                .local_get(CTX)
                .local_get(CTX)
                .i32_load(Self::ctr())
                .i32_const(1)
                .i32_sub()
                .local_tee(T)
                .i32_store(Self::ctr())
                .local_get(T);
            if bo & 0x02 != 0 {
                self.code.i32_eqz();
            } else {
                self.code.i32_const(0).i32_ne();
            }
            self.code.i32_and();
        }
        if bo & 0x10 == 0 {
            self.code
                .local_get(CTX)
                .i32_load(Self::cr())
                .i32_const(31 - instr.bi() as i32)
                .i32_shr_u()
                .i32_const(1)
                .i32_and()
                .i32_const(((bo >> 3) & 1) as i32)
                .i32_eq()
                .i32_and();
        }
        self.code.if_(BlockType::Empty);
        match target {
            Some(target) => {
                self.code.i32_const(target as i32);
            }
            None if counted => {
                self.code.local_get(CTX).i32_load(Self::lr()).i32_const(-4).i32_and();
            }
            None => {
                self.code.local_get(CTX).i32_load(Self::ctr()).i32_const(-4).i32_and();
            }
        }
        self.code.local_set(A);
        if instr.lk() {
            self.store_field(Self::lr(), |t| {
                t.code.i32_const(at.wrapping_add(4) as i32);
            });
        }
        self.flush_cycles_keeping();
        self.code.local_get(A).return_().end();
        self.exit_const(expected);
        true
    }

    // -- registers ------------------------------------------------------------------------

    fn load_gpr(&mut self, r: u8) {
        self.code.local_get(CTX).i32_load(Self::gpr(r));
    }

    /// `rA` where zero means the constant, as an address base does.
    fn load_gpr_or_zero(&mut self, r: u8) {
        if r == 0 {
            self.code.i32_const(0);
        } else {
            self.load_gpr(r);
        }
    }

    fn store_gpr(&mut self, r: u8, value: impl FnOnce(&mut Self)) {
        self.code.local_get(CTX);
        value(self);
        self.code.i32_store(Self::gpr(r));
    }

    fn store_field(&mut self, field: MemArg, value: impl FnOnce(&mut Self)) {
        self.code.local_get(CTX);
        value(self);
        self.code.i32_store(field);
    }

    /// CR0 from the value in `V`, with SO copied from XER as the interpreter copies it.
    fn update_cr0(&mut self) {
        self.code
            .local_get(CTX)
            .local_get(CTX)
            .i32_load(Self::cr())
            .i32_const(0x0FFF_FFFF)
            .i32_and()
            .local_get(V)
            .i32_const(0)
            .i32_lt_s()
            .i32_const(31)
            .i32_shl()
            .i32_or()
            .local_get(V)
            .i32_const(0)
            .i32_gt_s()
            .i32_const(30)
            .i32_shl()
            .i32_or()
            .local_get(V)
            .i32_eqz()
            .i32_const(29)
            .i32_shl()
            .i32_or()
            .local_get(CTX)
            .i32_load(Self::xer())
            .i32_const(XER_SO_SHIFT)
            .i32_shr_u()
            .i32_const(28)
            .i32_shl()
            .i32_or()
            .i32_store(Self::cr());
    }

    /// Writes the nibble in `T` to CR field `crf`.
    fn set_cr_field(&mut self, crf: u8) {
        let shift = 28 - 4 * crf as i32;
        self.code
            .local_get(CTX)
            .local_get(CTX)
            .i32_load(Self::cr())
            .i32_const(!(0xF << shift))
            .i32_and()
            .local_get(T)
            .i32_const(shift)
            .i32_shl()
            .i32_or()
            .i32_store(Self::cr());
    }

    /// XER's carry from the flag in `T`.
    fn set_carry(&mut self) {
        self.code
            .local_get(CTX)
            .local_get(CTX)
            .i32_load(Self::xer())
            .i32_const(!XER_CA)
            .i32_and()
            .local_get(T)
            .i32_const(XER_CA)
            .i32_mul()
            .i32_or()
            .i32_store(Self::xer());
    }

    // -- integer instructions -----------------------------------------------------------------

    fn add_imm(&mut self, instr: Instruction, imm: i32) {
        self.store_gpr(instr.rd(), |t| {
            t.load_gpr_or_zero(instr.ra());
            t.code.i32_const(imm).i32_add();
        });
    }

    fn logic_imm(&mut self, instr: Instruction, imm: i32, rc: bool, op: impl FnOnce(&mut InstructionSink<'_>)) {
        self.store_gpr(instr.ra(), |t| {
            t.load_gpr(instr.rs());
            t.code.i32_const(imm);
            op(&mut t.code);
            if rc {
                t.code.local_tee(V);
            }
        });
        if rc {
            self.update_cr0();
        }
    }

    /// `rA` is read once, into `A`, because `rD` may be the same register.
    fn add_imm_carry(&mut self, instr: Instruction, rc: bool) {
        self.load_gpr(instr.ra());
        self.code.local_set(A);
        self.store_gpr(instr.rd(), |t| {
            t.code.local_get(A).i32_const(instr.simm()).i32_add().local_tee(V);
        });
        self.code.local_get(V).local_get(A).i32_lt_u().local_set(T);
        self.set_carry();
        if rc {
            self.update_cr0();
        }
    }

    fn subfic(&mut self, instr: Instruction) {
        self.load_gpr(instr.ra());
        self.code.local_set(A);
        self.store_gpr(instr.rd(), |t| {
            t.code.i32_const(instr.simm()).local_get(A).i32_sub();
        });
        self.code.local_get(A).i32_const(instr.simm()).i32_le_u().local_set(T);
        self.set_carry();
    }

    fn mulli(&mut self, instr: Instruction) {
        self.store_gpr(instr.rd(), |t| {
            t.load_gpr(instr.ra());
            t.code.i32_const(instr.simm()).i32_mul();
        });
    }

    /// `rD` from a computation the caller pushes, with CR0 when asked.
    fn arith(&mut self, instr: Instruction, rc: bool, value: impl FnOnce(&mut Self)) {
        self.store_gpr(instr.rd(), |t| {
            value(t);
            if rc {
                t.code.local_tee(V);
            }
        });
        if rc {
            self.update_cr0();
        }
    }

    /// `rA` from `rS op rB`, with CR0 when asked.
    fn logic(&mut self, instr: Instruction, op: impl FnOnce(&mut InstructionSink<'_>)) {
        let rc = instr.rc();
        self.store_gpr(instr.ra(), |t| {
            t.load_gpr(instr.rs());
            t.load_gpr(instr.rb());
            op(&mut t.code);
            if rc {
                t.code.local_tee(V);
            }
        });
        if rc {
            self.update_cr0();
        }
    }

    fn unary(&mut self, instr: Instruction, op: impl FnOnce(&mut InstructionSink<'_>)) {
        let rc = instr.rc();
        self.store_gpr(instr.ra(), |t| {
            t.load_gpr(instr.rs());
            op(&mut t.code);
            if rc {
                t.code.local_tee(V);
            }
        });
        if rc {
            self.update_cr0();
        }
    }

    /// A shift by the low six bits of `rB`: WebAssembly wraps the count at 32 and the
    /// machine gives zero from there.
    fn shift_by_register(&mut self, instr: Instruction, op: impl FnOnce(&mut InstructionSink<'_>)) {
        let rc = instr.rc();
        self.store_gpr(instr.ra(), |t| {
            t.load_gpr(instr.rs());
            t.load_gpr(instr.rb());
            t.code.i32_const(0x3F).i32_and().local_tee(T);
            op(&mut t.code);
            t.code.i32_const(0).local_get(T).i32_const(32).i32_lt_u().select();
            if rc {
                t.code.local_tee(V);
            }
        });
        if rc {
            self.update_cr0();
        }
    }

    fn srawi(&mut self, instr: Instruction) {
        let sh = instr.sh() as i32;
        let rc = instr.rc();
        self.load_gpr(instr.rs());
        self.code.local_set(V);
        if sh == 0 {
            self.code.i32_const(0).local_set(T);
        } else {
            // Carry: a negative value that lost a set bit.
            let lost = (1u32 << sh) - 1;
            self.code
                .local_get(V)
                .i32_const(0)
                .i32_lt_s()
                .local_get(V)
                .i32_const(lost as i32)
                .i32_and()
                .i32_const(0)
                .i32_ne()
                .i32_and()
                .local_set(T);
        }
        self.set_carry();
        self.store_gpr(instr.ra(), |t| {
            t.code.local_get(V).i32_const(sh).i32_shr_s();
            if rc {
                t.code.local_tee(V);
            }
        });
        if rc {
            self.update_cr0();
        }
    }

    fn rlwinm(&mut self, instr: Instruction) {
        let mask = rlw_mask(instr.mb() as u32, instr.me() as u32) as i32;
        let sh = instr.sh() as i32;
        self.unary(instr, |c| {
            c.i32_const(sh).i32_rotl().i32_const(mask).i32_and();
        });
    }

    fn rlwimi(&mut self, instr: Instruction) {
        let mask = rlw_mask(instr.mb() as u32, instr.me() as u32) as i32;
        let sh = instr.sh() as i32;
        let rc = instr.rc();
        self.store_gpr(instr.ra(), |t| {
            t.load_gpr(instr.rs());
            t.code.i32_const(sh).i32_rotl().i32_const(mask).i32_and();
            t.load_gpr(instr.ra());
            t.code.i32_const(!mask).i32_and().i32_or();
            if rc {
                t.code.local_tee(V);
            }
        });
        if rc {
            self.update_cr0();
        }
    }

    fn rlwnm(&mut self, instr: Instruction) {
        let mask = rlw_mask(instr.mb() as u32, instr.me() as u32) as i32;
        self.logic(instr, |c| {
            c.i32_rotl().i32_const(mask).i32_and();
        });
    }

    /// A comparison of `rA` with what the caller pushes, into CR field `crfD`, with SO
    /// from XER.
    fn compare(&mut self, instr: Instruction, signed: bool, rhs: impl Fn(&mut Self)) {
        self.load_gpr(instr.ra());
        self.code.local_set(A);
        rhs(self);
        self.code.local_set(V);
        self.code.local_get(A).local_get(V);
        if signed {
            self.code.i32_lt_s();
        } else {
            self.code.i32_lt_u();
        }
        self.code.i32_const(3).i32_shl().local_get(A).local_get(V);
        if signed {
            self.code.i32_gt_s();
        } else {
            self.code.i32_gt_u();
        }
        self.code
            .i32_const(2)
            .i32_shl()
            .i32_or()
            .local_get(A)
            .local_get(V)
            .i32_eq()
            .i32_const(1)
            .i32_shl()
            .i32_or()
            .local_get(CTX)
            .i32_load(Self::xer())
            .i32_const(XER_SO_SHIFT)
            .i32_shr_u()
            .i32_or()
            .local_set(T);
        self.set_cr_field(instr.crfd());
    }

    fn mcrf(&mut self, instr: Instruction) {
        let from = 28 - 4 * instr.crfs() as i32;
        self.code
            .local_get(CTX)
            .i32_load(Self::cr())
            .i32_const(from)
            .i32_shr_u()
            .i32_const(0xF)
            .i32_and()
            .local_set(T);
        self.set_cr_field(instr.crfd());
    }

    fn mtcrf(&mut self, instr: Instruction) {
        let crm = instr.crm();
        let mut mask = 0u32;
        for i in 0..8u8 {
            if crm & (1 << (7 - i)) != 0 {
                mask |= 0xF << ((7 - i) * 4);
            }
        }
        self.store_field(Self::cr(), |t| {
            t.code
                .local_get(CTX)
                .i32_load(Self::cr())
                .i32_const(!mask as i32)
                .i32_and();
            t.load_gpr(instr.rs());
            t.code.i32_const(mask as i32).i32_and().i32_or();
        });
    }

    /// CR bit `crbD` from bits `crbA` and `crbB` combined as the caller says, each a 0 or
    /// 1 on the stack.
    fn cr_bit(&mut self, instr: Instruction, op: impl FnOnce(&mut InstructionSink<'_>)) {
        let bit = |c: &mut InstructionSink<'_>, b: u8| {
            c.local_get(CTX)
                .i32_load(Self::cr())
                .i32_const(31 - b as i32)
                .i32_shr_u()
                .i32_const(1)
                .i32_and();
        };
        let mask = 1i32 << (31 - instr.crbd() as i32);
        self.code
            .local_get(CTX)
            .local_get(CTX)
            .i32_load(Self::cr())
            .i32_const(!mask)
            .i32_and();
        bit(&mut self.code, instr.crba());
        bit(&mut self.code, instr.crbb());
        op(&mut self.code);
        self.code.i32_const(mask).i32_mul().i32_or().i32_store(Self::cr());
    }

    // -- memory ---------------------------------------------------------------------------------

    /// The effective address into `A`.
    fn effective_address(&mut self, instr: Instruction, form: Form) {
        self.load_gpr_or_zero(instr.ra());
        match form {
            Form::D => {
                self.code.i32_const(instr.disp());
            }
            Form::X => self.load_gpr(instr.rb()),
        }
        self.code.i32_add().local_set(A);
    }

    /// Whether `A` is a RAM address reached through the usual segments, leaving the
    /// physical address in `P`: the bus's own fast path, so the block reads RAM directly
    /// and leaves everything else to the bus.
    fn in_ram(&mut self, width: Width) {
        self.code
            .local_get(A)
            .i32_const(0x3FFF_FFFF)
            .i32_and()
            .local_tee(P)
            .i32_const((RAM_END - (width.bytes() - 1)) as i32)
            .i32_le_u()
            .local_get(A)
            .i32_const(28)
            .i32_shr_u()
            .i32_const(4)
            .i32_or()
            .i32_const(12)
            .i32_eq()
            .i32_and();
    }

    /// ...and a store also needs the line not to hold code the cache is running.
    fn in_ram_for_store(&mut self, width: Width) {
        self.in_ram(width);
        self.code
            .local_get(P)
            .i32_const(5)
            .i32_shr_u()
            .i32_load8_u(at(self.code_lines, 0))
            .i32_eqz()
            .i32_and();
    }

    /// The value at `A`, `width` wide and zero-extended, onto the stack.
    fn read(&mut self, width: Width) {
        self.in_ram(width);
        self.code.if_(BlockType::Result(ValType::I32)).local_get(P);
        match width {
            Width::Word => {
                self.code.i32_load(at(self.ram, 0));
                self.byte_swap();
            }
            Width::Half => {
                self.code.i32_load16_u(at(self.ram, 0));
                self.half_swap();
            }
            Width::Byte => {
                self.code.i32_load8_u(at(self.ram, 0));
            }
        }
        let helper = match width {
            Width::Word => self.helpers.read_u32,
            Width::Half => self.helpers.read_u16,
            Width::Byte => self.helpers.read_u8,
        };
        self.code.else_();
        self.bus_call(|t| {
            t.code
                .local_get(CTX)
                .local_get(A)
                .i32_const(helper)
                .call_indirect(0, T_READ);
        });
        self.code.end();
    }

    /// The value in `V` to `A`, `width` wide.
    fn write(&mut self, width: Width) {
        self.in_ram_for_store(width);
        self.code.if_(BlockType::Empty);
        if validate::on() {
            self.code
                .local_get(CTX)
                .local_get(P)
                .i32_const(width.bytes() as i32)
                .i32_const(self.helpers.note_store)
                .call_indirect(0, T_WRITE);
        }
        self.code.local_get(P).local_get(V);
        match width {
            Width::Word => {
                self.byte_swap();
                self.code.i32_store(at(self.ram, 0));
            }
            Width::Half => {
                self.half_swap();
                self.code.i32_store16(at(self.ram, 0));
            }
            Width::Byte => {
                self.code.i32_store8(at(self.ram, 0));
            }
        }
        let helper = match width {
            Width::Word => self.helpers.write_u32,
            Width::Half => self.helpers.write_u16,
            Width::Byte => self.helpers.write_u8,
        };
        self.code.else_();
        self.bus_call(|t| {
            t.code
                .local_get(CTX)
                .local_get(A)
                .local_get(V)
                .i32_const(helper)
                .call_indirect(0, T_WRITE);
        });
        self.code.end();
    }

    /// The word on the stack with its bytes reversed: RAM is big-endian and WebAssembly
    /// is not.
    fn byte_swap(&mut self) {
        self.code
            .local_tee(T)
            .i32_const(0xFF00_FF00u32 as i32)
            .i32_and()
            .i32_const(8)
            .i32_rotl()
            .local_get(T)
            .i32_const(0x00FF_00FF)
            .i32_and()
            .i32_const(8)
            .i32_rotr()
            .i32_or();
    }

    /// The low halfword on the stack with its bytes reversed, whatever the high half held.
    fn half_swap(&mut self) {
        self.code
            .local_tee(T)
            .i32_const(0xFF)
            .i32_and()
            .i32_const(8)
            .i32_shl()
            .local_get(T)
            .i32_const(8)
            .i32_shr_u()
            .i32_const(0xFF)
            .i32_and()
            .i32_or();
    }

    fn load(&mut self, instr: Instruction, width: Width, form: Form, update: bool, signed: bool) {
        self.effective_address(instr, form);
        self.store_gpr(instr.rd(), |t| {
            t.read(width);
            if signed {
                t.code.i32_extend16_s();
            }
        });
        if update {
            self.store_gpr(instr.ra(), |t| {
                t.code.local_get(A);
            });
        }
    }

    fn store(&mut self, instr: Instruction, width: Width, form: Form, update: bool) {
        self.effective_address(instr, form);
        self.load_gpr(instr.rs());
        self.code.local_set(V);
        self.write(width);
        if update {
            self.store_gpr(instr.ra(), |t| {
                t.code.local_get(A);
            });
        }
    }

    // -- floating point -------------------------------------------------------------------------

    /// After a floating-point instruction, the interpreter's program-exception check:
    /// an enabled exception mode and a summary bit in FPSCR.
    fn fp_exception_check(&mut self, at: u32, expected: u32) {
        self.code
            .local_get(FE)
            .if_(BlockType::Empty)
            .local_get(CTX)
            .i32_load(field(abi::fpscr_offset::<SYSTEM>()))
            .i32_const(FPSCR_FEX)
            .i32_and()
            .if_(BlockType::Empty)
            .local_get(CTX)
            .i32_const(at as i32)
            .i32_store(Self::cia())
            .local_get(CTX)
            .i32_const(expected as i32)
            .i32_store(Self::nia())
            .local_get(CTX)
            .i32_const(self.helpers.raise_fp)
            .call_indirect(0, T_RAISE);
        // This path leaves; the cycles it settles are still owed on the other.
        self.flush_cycles_keeping();
        self.code.local_get(CTX).i32_load(Self::nia()).return_().end().end();
    }

    fn load_fpr(&mut self, r: u8) {
        self.code.local_get(CTX).f64_load(Self::fpr(r));
    }

    fn load_ps1(&mut self, r: u8) {
        self.code.local_get(CTX).f64_load(Self::ps1(r));
    }

    /// `fD` from the stack, and `ps1` too when the result is single: a single-precision
    /// result lands in both halves.
    fn store_fp_result(&mut self, r: u8, single: bool) {
        self.code.local_set(F);
        self.code.local_get(CTX).local_get(F).f64_store(Self::fpr(r));
        if single {
            self.code.local_get(CTX).local_get(F).f64_store(Self::ps1(r));
        }
    }

    /// `fD` from `ps1` on the stack and `ps0` in `F`.
    fn store_ps_pair(&mut self, r: u8) {
        self.code.local_set(G);
        self.code.local_get(CTX).local_get(F).f64_store(Self::fpr(r));
        self.code.local_get(CTX).local_get(G).f64_store(Self::ps1(r));
    }

    /// The value on the stack rounded the way the multiplier's `fC` input is.
    fn round_frc(&mut self) {
        self.code
            .i64_reinterpret_f64()
            .local_tee(Q)
            .i64_const(FRC_KEEP_MASK as i64)
            .i64_and()
            .local_get(Q)
            .i64_const(FRC_ROUND_BIT as i64)
            .i64_and()
            .i64_add()
            .f64_reinterpret_i64();
    }

    fn to_single(&mut self) {
        self.code.f32_demote_f64().f64_promote_f32();
    }

    fn fp_unary(&mut self, instr: Instruction, single: bool, op: impl FnOnce(&mut InstructionSink<'_>)) {
        self.load_fpr(instr.rb());
        op(&mut self.code);
        self.store_fp_result(instr.rd(), single);
    }

    fn fp_binary(&mut self, instr: Instruction, single: bool, rhs: Operand, op: impl FnOnce(&mut InstructionSink<'_>)) {
        self.load_fpr(instr.ra());
        match rhs {
            Operand::B => self.load_fpr(instr.rb()),
            Operand::C => self.load_fpr(instr.fc()),
            Operand::RoundedC => {
                self.load_fpr(instr.fc());
                self.round_frc();
            }
        }
        op(&mut self.code);
        if single {
            self.to_single();
        }
        self.store_fp_result(instr.rd(), single);
    }

    /// `fA * fC + fB` fused, with `fB` negated for a subtract and the result negated
    /// for the `n` forms, which leave a NaN alone.
    fn fp_madd(&mut self, instr: Instruction, single: bool, subtract: bool, negate: bool) {
        self.load_fpr(instr.ra());
        self.load_fpr(instr.fc());
        if single {
            self.round_frc();
        }
        self.load_fpr(instr.rb());
        if subtract {
            self.code.f64_neg();
        }
        self.code.i32_const(self.helpers.fma).call_indirect(0, T_FMA);
        if single {
            self.to_single();
        }
        if negate {
            self.negate_unless_nan();
        }
        self.store_fp_result(instr.rd(), single);
    }

    /// Uses `G` as scratch: `F` may hold the first half of a pair by now.
    fn negate_unless_nan(&mut self) {
        self.code
            .local_tee(G)
            .local_get(G)
            .f64_neg()
            .local_get(G)
            .local_get(G)
            .f64_ne()
            .select();
    }

    fn fsel(&mut self, instr: Instruction) {
        self.load_fpr(instr.fc());
        self.load_fpr(instr.rb());
        self.load_fpr(instr.ra());
        self.code.f64_const(0.0.into()).f64_ge().select();
        self.store_fp_result(instr.rd(), false);
    }

    /// Both halves through the same operation on `fB`.
    fn ps_unary(&mut self, instr: Instruction, op: impl Fn(&mut InstructionSink<'_>)) {
        self.load_fpr(instr.rb());
        op(&mut self.code);
        self.code.local_set(F);
        self.load_ps1(instr.rb());
        op(&mut self.code);
        self.store_ps_pair(instr.rd());
    }

    /// Both halves through the same operation on `fA` and `fB`, rounded to single.
    fn ps_binary(&mut self, instr: Instruction, op: impl Fn(&mut InstructionSink<'_>)) {
        self.load_fpr(instr.ra());
        self.load_fpr(instr.rb());
        op(&mut self.code);
        self.to_single();
        self.code.local_set(F);
        self.load_ps1(instr.ra());
        self.load_ps1(instr.rb());
        op(&mut self.code);
        self.to_single();
        self.store_ps_pair(instr.rd());
    }

    /// Both halves through a single-precision operation on `fB`.
    fn ps_single(&mut self, instr: Instruction, op: impl Fn(&mut InstructionSink<'_>)) {
        self.load_fpr(instr.rb());
        op(&mut self.code);
        self.code.local_set(F);
        self.load_ps1(instr.rb());
        op(&mut self.code);
        self.store_ps_pair(instr.rd());
    }

    /// `fA * fC` by halves, or by one half of `fC` for the `s0`/`s1` forms.
    fn ps_mul(&mut self, instr: Instruction, scalar: Option<u8>) {
        self.load_fpr(instr.ra());
        self.load_fc_half(instr.fc(), scalar, 0);
        self.code.f64_mul();
        self.to_single();
        self.code.local_set(F);
        self.load_ps1(instr.ra());
        self.load_fc_half(instr.fc(), scalar, 1);
        self.code.f64_mul();
        self.to_single();
        self.store_ps_pair(instr.rd());
    }

    fn load_fc_half(&mut self, fc: u8, scalar: Option<u8>, half: u8) {
        match scalar.unwrap_or(half) {
            0 => self.load_fpr(fc),
            _ => self.load_ps1(fc),
        }
        self.round_frc();
    }

    fn ps_madd(&mut self, instr: Instruction, scalar: Option<u8>, subtract: bool, negate: bool) {
        for half in 0..2u8 {
            if half == 0 {
                self.load_fpr(instr.ra());
            } else {
                self.load_ps1(instr.ra());
            }
            self.load_fc_half(instr.fc(), scalar, half);
            if half == 0 {
                self.load_fpr(instr.rb());
            } else {
                self.load_ps1(instr.rb());
            }
            if subtract {
                self.code.f64_neg();
            }
            self.code.i32_const(self.helpers.fma).call_indirect(0, T_FMA);
            self.to_single();
            if negate {
                self.negate_unless_nan();
            }
            if half == 0 {
                self.code.local_set(F);
            }
        }
        self.store_ps_pair(instr.rd());
    }

    /// `ps_sum0` puts `fA.ps0 + fB.ps1` in `ps0` and `fC.ps1` in `ps1`; `ps_sum1` puts
    /// `fC.ps0` in `ps0` and the sum in `ps1`.
    fn ps_sum(&mut self, instr: Instruction, into: u8) {
        if into == 0 {
            self.load_fpr(instr.ra());
            self.load_ps1(instr.rb());
            self.code.f64_add();
            self.to_single();
            self.code.local_set(F);
            self.load_ps1(instr.fc());
        } else {
            self.load_fpr(instr.fc());
            self.code.local_set(F);
            self.load_fpr(instr.ra());
            self.load_ps1(instr.rb());
            self.code.f64_add();
            self.to_single();
        }
        self.store_ps_pair(instr.rd());
    }

    fn ps_merge(&mut self, instr: Instruction, a: u8, b: u8) {
        if a == 0 {
            self.load_fpr(instr.ra());
        } else {
            self.load_ps1(instr.ra());
        }
        self.code.local_set(F);
        if b == 0 {
            self.load_fpr(instr.rb());
        } else {
            self.load_ps1(instr.rb());
        }
        self.store_ps_pair(instr.rd());
    }

    fn ps_sel(&mut self, instr: Instruction) {
        self.load_fpr(instr.fc());
        self.load_fpr(instr.rb());
        self.load_fpr(instr.ra());
        self.code.f64_const(0.0.into()).f64_ge().select().local_set(F);
        self.load_ps1(instr.fc());
        self.load_ps1(instr.rb());
        self.load_ps1(instr.ra());
        self.code.f64_const(0.0.into()).f64_ge().select();
        self.store_ps_pair(instr.rd());
    }

    /// `fA` against `fB` into CR field `crfD`: less, greater, equal, or unordered.
    fn fp_compare(&mut self, instr: Instruction, half: u8) {
        if half == 0 {
            self.load_fpr(instr.ra());
            self.code.local_set(F);
            self.load_fpr(instr.rb());
        } else {
            self.load_ps1(instr.ra());
            self.code.local_set(F);
            self.load_ps1(instr.rb());
        }
        self.code
            .local_set(G)
            .local_get(F)
            .local_get(G)
            .f64_lt()
            .i32_const(3)
            .i32_shl()
            .local_get(F)
            .local_get(G)
            .f64_gt()
            .i32_const(2)
            .i32_shl()
            .i32_or()
            .local_get(F)
            .local_get(G)
            .f64_eq()
            .i32_const(1)
            .i32_shl()
            .i32_or()
            .local_get(F)
            .local_get(F)
            .f64_ne()
            .local_get(G)
            .local_get(G)
            .f64_ne()
            .i32_or()
            .i32_or()
            .local_set(T);
        self.set_cr_field(instr.crfd());
    }

    fn load_single(&mut self, instr: Instruction, form: Form, update: bool) {
        self.effective_address(instr, form);
        self.read(Width::Word);
        self.code.f32_reinterpret_i32().f64_promote_f32();
        self.store_fp_result(instr.fd(), true);
        if update {
            self.store_gpr(instr.ra(), |t| {
                t.code.local_get(A);
            });
        }
    }

    fn store_single(&mut self, instr: Instruction, form: Form, update: bool) {
        self.effective_address(instr, form);
        self.load_fpr(instr.fs());
        self.code.f32_demote_f64().i32_reinterpret_f32().local_set(V);
        self.write(Width::Word);
        if update {
            self.store_gpr(instr.ra(), |t| {
                t.code.local_get(A);
            });
        }
    }

    /// A double from RAM, read as the bus reads it: by physical address alone.
    fn load_double(&mut self, instr: Instruction, form: Form, update: bool) {
        self.effective_address(instr, form);
        self.code
            .local_get(A)
            .i32_const(0x3FFF_FFFF)
            .i32_and()
            .local_tee(P)
            .i32_const((RAM_END - 7) as i32)
            .i32_le_u()
            .if_(BlockType::Result(ValType::F64))
            .local_get(P)
            .i32_load(at(self.ram, 0));
        self.byte_swap();
        self.code
            .i64_extend_i32_u()
            .i64_const(32)
            .i64_shl()
            .local_get(P)
            .i32_load(at(self.ram + 4, 0));
        self.byte_swap();
        self.code.i64_extend_i32_u().i64_or().f64_reinterpret_i64().else_();
        self.bus_call(|t| {
            t.code
                .local_get(CTX)
                .local_get(A)
                .i32_const(t.helpers.read_f64)
                .call_indirect(0, T_READ_F64);
        });
        self.code.end();
        self.store_fp_result(instr.fd(), false);
        if update {
            self.store_gpr(instr.ra(), |t| {
                t.code.local_get(A);
            });
        }
    }

    fn store_double(&mut self, instr: Instruction, form: Form, update: bool) {
        self.effective_address(instr, form);
        self.flush_cycles();
        self.code.local_get(CTX).local_get(A);
        self.load_fpr(instr.fs());
        self.code
            .i32_const(self.helpers.write_f64)
            .call_indirect(0, T_WRITE_F64);
        if update {
            self.store_gpr(instr.ra(), |t| {
                t.code.local_get(A);
            });
        }
    }

    /// A quantized load of floats, which is what the GQR nearly always says; any other
    /// type goes to the handler.
    fn psq_load(
        &mut self,
        instr: Instruction,
        form: Form,
        update: bool,
        leaf: u16,
        at: u32,
        expected: u32,
        last: bool,
    ) {
        let (index, w) = match form {
            Form::D => (instr.psq_i(), instr.psq_w()),
            Form::X => (instr.psq_ix(), instr.psq_wx()),
        };
        self.psq_address(instr, form);
        self.code
            .local_get(CTX)
            .i32_load(field(abi::gqr_offset::<SYSTEM>(index)))
            .i32_const(0x0007_0000)
            .i32_and()
            .if_(BlockType::Empty);
        self.call_handler(leaf, instr, at, expected, last);
        self.code.else_();
        self.charge_now(cycles_for_op(OP_PSQ_L));
        self.read(Width::Word);
        self.code.f32_reinterpret_i32().f64_promote_f32().local_set(F);
        if w {
            self.code.f64_const(1.0.into());
        } else {
            self.code
                .local_get(A)
                .local_set(V)
                .local_get(A)
                .i32_const(4)
                .i32_add()
                .local_set(A);
            self.read(Width::Word);
            self.code
                .f32_reinterpret_i32()
                .f64_promote_f32()
                .local_get(V)
                .local_set(A);
        }
        self.store_ps_pair(instr.fd());
        if update {
            self.store_gpr(instr.ra(), |t| {
                t.code.local_get(A);
            });
        }
        self.code.end();
    }

    fn psq_store(
        &mut self,
        instr: Instruction,
        form: Form,
        update: bool,
        leaf: u16,
        at: u32,
        expected: u32,
        last: bool,
    ) {
        let (index, w) = match form {
            Form::D => (instr.psq_i(), instr.psq_w()),
            Form::X => (instr.psq_ix(), instr.psq_wx()),
        };
        self.psq_address(instr, form);
        self.code
            .local_get(CTX)
            .i32_load(field(abi::gqr_offset::<SYSTEM>(index)))
            .i32_const(0x0000_0007)
            .i32_and()
            .if_(BlockType::Empty);
        self.call_handler(leaf, instr, at, expected, last);
        self.code.else_();
        self.charge_now(cycles_for_op(OP_PSQ_ST));
        self.load_fpr(instr.fs());
        self.code.f32_demote_f64().i32_reinterpret_f32().local_set(V);
        self.write(Width::Word);
        if !w {
            self.code.local_get(A).i32_const(4).i32_add().local_set(A);
            self.load_ps1(instr.fs());
            self.code.f32_demote_f64().i32_reinterpret_f32().local_set(V);
            self.write(Width::Word);
        }
        if update {
            self.store_gpr(instr.ra(), |t| {
                t.code.local_get(A);
                if !w {
                    t.code.i32_const(4).i32_sub();
                }
            });
        }
        self.code.end();
    }

    fn psq_address(&mut self, instr: Instruction, form: Form) {
        self.load_gpr_or_zero(instr.ra());
        match form {
            Form::D => {
                self.code.i32_const(instr.disp_psq());
            }
            Form::X => self.load_gpr(instr.rb()),
        }
        self.code.i32_add().local_set(A);
    }
}

#[derive(Clone, Copy)]
enum Width {
    Byte,
    Half,
    Word,
}

impl Width {
    fn bytes(self) -> u32 {
        match self {
            Width::Byte => 1,
            Width::Half => 2,
            Width::Word => 4,
        }
    }
}

#[derive(Clone, Copy)]
enum Form {
    D,
    X,
}

enum Operand {
    B,
    C,
    RoundedC,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::GC;

    fn steps(words: &[u32], start: u32) -> Vec<(u16, Instruction, u32)> {
        words
            .iter()
            .enumerate()
            .map(|(i, &word)| {
                let instr = Instruction(word);
                (crate::gekko::resolve::<GC>(instr), instr, start + 4 * i as u32)
            })
            .collect()
    }

    fn validate(words: &[u32]) {
        let sys = crate::gamecube::GameCube::new(0x8000_3000);
        let steps = steps(words, 0x8000_3000);
        let module = emit::<GC>(&sys, &steps, 0x8000_3000 + 4 * words.len() as u32);
        wasmparser::validate(&module).expect("the emitted module is valid");
    }

    #[test]
    fn a_mixed_block_validates() {
        // lwz r3,0(r4) ; addi r3,r3,1 ; stw r3,0(r4) ; lfs f1,8(r4) ; fmuls f1,f1,f2 ;
        // ps_madd f3,f1,f2,f3 ; psq_l f4,16(r4),0,0 ; cmpwi r3,10 ; bne -32
        validate(&[
            0x8064_0000,
            0x3863_0001,
            0x9064_0000,
            0xC024_0008,
            0xEC21_0072,
            0x1061_10FA,
            0xE084_0010,
            0x2C03_000A,
            0x4082_FFE0,
        ]);
    }

    #[test]
    fn every_translation_validates() {
        // One of each shape the emitter knows: the integer and rotate forms, the
        // compares and CR moves, the loads and stores of each width and form, the
        // single, double and paired-single arithmetic, and each kind of branch.
        validate(&[
            0x3C60_8000,
            0x6063_0001,
            0x6C63_0001,
            0x7063_0001,
            0x3063_0001,
            0x3463_0001,
            0x2063_0001,
            0x1C63_0003,
            0x7C63_2214,
            0x7C63_2050,
            0x7C63_00D0,
            0x7C63_21D6,
            0x7C63_2096,
            0x7C63_2016,
            0x7C63_2039,
            0x7C63_2378,
            0x7C63_2278,
            0x7C63_23B8,
            0x7C63_20F8,
            0x7C63_2238,
            0x7C63_2078,
            0x7C63_2338,
            0x7C63_2030,
            0x7C63_2430,
            0x7C63_0E70,
            0x7C63_0034,
            0x7C63_0734,
            0x7C63_0774,
            0x5463_103A,
            0x5063_103A,
            0x5C63_203E,
            0x7C03_2000,
            0x7C03_2040,
            0x2C03_0001,
            0x2803_0001,
            0x4C80_0000,
            0x7C60_0026,
            0x7C6F_F120,
            0x4C42_1182,
            0x4C42_1382,
            0x4C42_1202,
            0x4C42_1242,
            0x4C42_1042,
            0x4C42_11C2,
            0x4C42_1102,
            0x4C42_1342,
            0x7C00_222C,
            0x7C00_21EC,
            0x7C68_02A6,
            0x7C68_03A6,
            0x7C69_02A6,
            0x7C69_03A6,
            0x7C61_02A6,
            0x8064_0000,
            0x8464_0004,
            0x7C64_282E,
            0x7C64_286E,
            0xA064_0000,
            0xA464_0002,
            0x7C64_2A2E,
            0x7C64_2A6E,
            0xA864_0000,
            0xAC64_0002,
            0x7C64_2AAE,
            0x7C64_2AEE,
            0x8864_0000,
            0x8C64_0001,
            0x7C64_28AE,
            0x7C64_28EE,
            0x9064_0000,
            0x9464_0004,
            0x7C64_292E,
            0x7C64_296E,
            0xB064_0000,
            0xB464_0002,
            0x7C64_2B2E,
            0x7C64_2B6E,
            0x9864_0000,
            0x9C64_0001,
            0x7C64_29AE,
            0x7C64_29EE,
            0xFC20_1090,
            0xFC20_1050,
            0xFC20_1210,
            0xFC20_1110,
            0xFC20_1018,
            0xFC20_102C,
            0xEC20_102C,
            0xFC20_1034,
            0xEC20_1030,
            0xFC20_101E,
            0xFC22_182A,
            0xFC22_1828,
            0xFC22_00F2,
            0xFC22_1824,
            0xEC22_182A,
            0xEC22_1828,
            0xEC22_00F2,
            0xEC22_1824,
            0xFC22_20FA,
            0xFC22_20F8,
            0xFC22_20FE,
            0xFC22_20FC,
            0xEC22_20FA,
            0xEC22_20F8,
            0xEC22_20FE,
            0xEC22_20FC,
            0xFC22_20EE,
            0x1020_1090,
            0x1020_1050,
            0x1020_1210,
            0x1020_1110,
            0x1022_182A,
            0x1022_1828,
            0x1022_1824,
            0x1022_00F2,
            0x1022_0018,
            0x1022_0019,
            0x1022_20FA,
            0x1022_20F8,
            0x1022_20FE,
            0x1022_20FC,
            0x1022_201C,
            0x1022_201D,
            0x1022_2014,
            0x1022_2016,
            0x1022_1C20,
            0x1022_1C60,
            0x1022_1CA0,
            0x1022_1CE0,
            0x1022_20EE,
            0x1020_1030,
            0x1020_1034,
            0xFC01_1000,
            0xFC01_1040,
            0x1001_1000,
            0x1001_1040,
            0x1001_1080,
            0x1001_10C0,
            0xC024_0000,
            0xC424_0004,
            0x7C24_2C2E,
            0x7C24_2C6E,
            0xD024_0000,
            0xD424_0004,
            0x7C24_2D2E,
            0x7C24_2D6E,
            0xC824_0000,
            0xCC24_0008,
            0x7C24_2CAE,
            0x7C24_2CEE,
            0xD824_0000,
            0xDC24_0008,
            0x7C24_2DAE,
            0x7C24_2DEE,
            0xE024_0000,
            0xE424_0008,
            0x1024_200C,
            0x1024_204C,
            0xF024_0000,
            0xF424_0008,
            0x1024_200E,
            0x1024_204E,
            0x4082_0010,
        ]);
        validate(&[0x4800_0010]);
        validate(&[0x4E80_0020]);
        validate(&[0x4E80_0420]);
        validate(&[0x4200_FFFC]);
    }

    /// How many calls the block makes: a translated integer instruction makes none, a
    /// floating-point one carries two it takes only when FP is off or trapping, one that
    /// touches memory has one for the bus, and a handler call is one.
    fn calls(words: &[u32]) -> usize {
        let sys = crate::gamecube::GameCube::new(0x8000_3000);
        let module = emit::<GC>(&sys, &steps(words, 0x8000_3000), 0x8000_3000 + 4 * words.len() as u32);
        let mut calls = 0;
        for payload in wasmparser::Parser::new(0).parse_all(&module) {
            if let wasmparser::Payload::CodeSectionEntry(body) = payload.expect("parses") {
                for op in body.get_operators_reader().expect("operators") {
                    if matches!(op.expect("operator"), wasmparser::Operator::CallIndirect { .. }) {
                        calls += 1;
                    }
                }
            }
        }
        calls
    }

    #[test]
    fn translated_instructions_call_nothing() {
        // addi ; rlwinm ; cmpwi ; add. ; bne
        assert_eq!(
            calls(&[0x3863_0001, 0x5463_103A, 0x2C03_000A, 0x7C63_2215, 0x4082_FFE0]),
            0
        );
        // ps_mul
        assert_eq!(calls(&[0x1022_0032]), 2);
        // lwz
        assert_eq!(calls(&[0x8064_0000]), 1);
        // sc
        assert_eq!(calls(&[0x4400_0002]), 1);
    }

    #[test]
    fn handler_calls_validate() {
        // sc ; rfi
        validate(&[0x4400_0002, 0x4C00_0064]);
    }

    #[test]
    fn an_empty_block_validates() {
        validate(&[]);
    }
}

/// The translations run, in a WebAssembly interpreter, over an image of the console,
/// beside the interpreter's handlers on the console itself, from the same registers.
#[cfg(test)]
mod arithmetic_tests {
    use super::*;
    use crate::gekko::msr::Msr;
    use crate::system::GC;

    const START: u32 = 0x8000_3000;
    const CTX: usize = 64;
    const FMA_SLOT: i32 = 1;

    /// A-form: `op rD, rA, rB, rC` with a five-bit extended opcode.
    fn a_form(op: u32, xo: u32) -> u32 {
        (op << 26) | (4 << 21) | (1 << 16) | (2 << 11) | (3 << 6) | (xo << 1)
    }

    /// X-form: `op rD, rA, rB` with a ten-bit extended opcode.
    fn x_form(op: u32, xo: u32) -> u32 {
        (op << 26) | (4 << 21) | (1 << 16) | (2 << 11) | (xo << 1)
    }

    const INTERESTING: [f64; 14] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.1,
        3.0,
        1.0e-10,
        1.0e10,
        123456.789,
        -2.5,
        0.75,
        16777216.0,
        1.0 / 3.0,
        -6.02e23,
    ];

    fn image_of(sys: &System<GC>, data: &mut [u8]) {
        let msr = u32::from(sys.gekko.msr).to_le_bytes();
        data[CTX + abi::msr_offset::<GC>()..][..4].copy_from_slice(&msr);
        for (i, gpr) in sys.gekko.gprs.iter().enumerate() {
            data[CTX + abi::gpr_base_offset::<GC>() + 4 * i..][..4].copy_from_slice(&gpr.to_le_bytes());
        }
        data[CTX + abi::xer_offset::<GC>()..][..4].copy_from_slice(&sys.gekko.spr.xer.raw().to_le_bytes());
        data[CTX + abi::lr_offset::<GC>()..][..4].copy_from_slice(&sys.gekko.spr.lr.to_le_bytes());
        data[CTX + abi::ctr_offset::<GC>()..][..4].copy_from_slice(&sys.gekko.spr.ctr.to_le_bytes());
        for (i, pair) in sys.gekko.fprs.0.iter().enumerate() {
            let at = CTX + abi::fpr_base_offset::<GC>() + 16 * i;
            data[at..at + 8].copy_from_slice(&pair[0].to_bits().to_le_bytes());
            data[at + 8..at + 16].copy_from_slice(&pair[1].to_bits().to_le_bytes());
        }
        data[CTX + abi::cr_offset::<GC>()..][..4].copy_from_slice(&sys.gekko.cr.raw().to_le_bytes());
    }

    /// Runs `words` both ways from the same registers and returns what disagrees.
    fn disagreement(words: &[u32], fprs: &[[f64; 2]; 8], gprs: &[u32; 8], xer: u32) -> Option<String> {
        let mut sys = crate::gamecube::GameCube::new(START);
        sys.gekko.msr = Msr::from(u32::from(sys.gekko.msr) | MSR_FP as u32);
        for (i, pair) in fprs.iter().enumerate() {
            sys.gekko.fprs.0[i] = *pair;
        }
        for (i, gpr) in gprs.iter().enumerate() {
            sys.gekko.gprs[i] = *gpr;
        }
        sys.gekko.spr.xer = crate::gekko::spr::Xer::from(xer);
        sys.gekko.spr.lr = 0x8000_1234;
        sys.gekko.spr.ctr = 7;

        let engine = wasmi::Engine::default();
        let mut store = wasmi::Store::new(&engine, ());
        let pages = (core::mem::size_of::<System<GC>>() / 65536 + 2) as u32;
        let memory = wasmi::Memory::new(&mut store, wasmi::MemoryType::new(pages, None)).unwrap();
        image_of(&sys, memory.data_mut(&mut store));
        let table = wasmi::Table::new(
            &mut store,
            wasmi::TableType::new(wasmi::core::ValType::FuncRef, 2, None),
            wasmi::Val::FuncRef(wasmi::Ref::Null),
        )
        .unwrap();
        let fma = wasmi::Func::wrap(&mut store, |a: f64, b: f64, c: f64| a.mul_add(b, c));
        table
            .set(&mut store, FMA_SLOT as u64, wasmi::Val::FuncRef(wasmi::Ref::Val(fma)))
            .unwrap();
        let mut linker = <wasmi::Linker<()>>::new(&engine);
        linker.define(IMPORT_MODULE, MEMORY_IMPORT, memory).unwrap();
        linker.define(IMPORT_MODULE, TABLE_IMPORT, table).unwrap();

        let steps: Vec<(u16, Instruction, u32)> = words
            .iter()
            .enumerate()
            .map(|(i, &word)| {
                let instr = Instruction(word);
                (crate::gekko::resolve::<GC>(instr), instr, START + 4 * i as u32)
            })
            .collect();
        let end = START + 4 * words.len() as u32;
        let helpers = Helpers {
            read_u8: 0,
            read_u16: 0,
            read_u32: 0,
            write_u8: 0,
            write_u16: 0,
            write_u32: 0,
            read_f64: 0,
            write_f64: 0,
            raise_fp: 0,
            fma: FMA_SLOT,
            note_store: 0,
        };
        let module = wasmi::Module::new(&engine, emit_with::<GC>(&sys, &steps, end, helpers)).unwrap();
        let instance = linker.instantiate_and_start(&mut store, &module).unwrap();
        let block = instance.get_typed_func::<i32, i32>(&store, BLOCK_EXPORT).unwrap();
        let nia = block.call(&mut store, CTX as i32).unwrap();

        for &(leaf, instr, at) in &steps {
            sys.gekko.cia = at;
            sys.gekko.nia = at.wrapping_add(4);
            crate::gekko::execute::<GC>(leaf, &mut sys, instr);
        }

        let data = memory.data(&store);
        let mut out = String::new();
        for (i, pair) in sys.gekko.fprs.0.iter().enumerate() {
            let at = CTX + abi::fpr_base_offset::<GC>() + 16 * i;
            let ps0 = u64::from_le_bytes(data[at..at + 8].try_into().unwrap());
            let ps1 = u64::from_le_bytes(data[at + 8..at + 16].try_into().unwrap());
            if [ps0, ps1] != [pair[0].to_bits(), pair[1].to_bits()] {
                out.push_str(&format!(
                    " f{i}={ps0:016x}:{ps1:016x} not {:016x}:{:016x}",
                    pair[0].to_bits(),
                    pair[1].to_bits()
                ));
            }
        }
        for (i, gpr) in sys.gekko.gprs.iter().enumerate() {
            let got = u32::from_le_bytes(
                data[CTX + abi::gpr_base_offset::<GC>() + 4 * i..][..4]
                    .try_into()
                    .unwrap(),
            );
            if got != *gpr {
                out.push_str(&format!(" r{i}={got:08x} not {gpr:08x}"));
            }
        }
        let word = |offset: usize| u32::from_le_bytes(data[CTX + offset..][..4].try_into().unwrap());
        for (name, got, want) in [
            ("cr", word(abi::cr_offset::<GC>()), sys.gekko.cr.raw()),
            ("xer", word(abi::xer_offset::<GC>()), sys.gekko.spr.xer.raw()),
            ("lr", word(abi::lr_offset::<GC>()), sys.gekko.spr.lr),
            ("ctr", word(abi::ctr_offset::<GC>()), sys.gekko.spr.ctr),
        ] {
            if got != want {
                out.push_str(&format!(" {name}={got:08x} not {want:08x}"));
            }
        }
        if nia != end as i32 {
            out.push_str(&format!(" nia={nia:08x} not {end:08x}"));
        }
        (!out.is_empty()).then_some(out)
    }

    const INTEGERS: [u32; 16] = [
        0,
        1,
        0xFFFF_FFFF,
        0x7FFF_FFFF,
        0x8000_0000,
        0x1234_5678,
        0xFFFF_0000,
        3,
        100,
        0xDEAD_BEEF,
        31,
        32,
        33,
        64,
        0x0000_8000,
        0xC000_0000,
    ];

    fn check_all(words: &[u32]) {
        let mut seed = 0x2545_F491u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let mut failures = Vec::new();
        for &word in words {
            for _ in 0..16 {
                let mut fprs = [[0.0f64; 2]; 8];
                for pair in fprs.iter_mut() {
                    *pair = [
                        INTERESTING[(next() % INTERESTING.len() as u32) as usize],
                        INTERESTING[(next() % INTERESTING.len() as u32) as usize],
                    ];
                }
                let mut gprs = [0u32; 8];
                for gpr in gprs.iter_mut() {
                    *gpr = INTEGERS[(next() % INTEGERS.len() as u32) as usize];
                }
                let xer = if next() & 1 == 0 { 0 } else { 0xE000_0000 };
                if let Some(diff) = disagreement(&[word], &fprs, &gprs, xer) {
                    failures.push(format!("{word:08x} from {gprs:x?} xer {xer:08x} {fprs:?}:{diff}"));
                    break;
                }
            }
        }
        assert!(
            failures.is_empty(),
            "translations disagree with the interpreter:\n{}",
            failures.join("\n")
        );
    }

    /// D-form: `op rD, rA, imm`.
    fn d_form(op: u32, imm: u32) -> u32 {
        (op << 26) | (4 << 21) | (1 << 16) | (imm & 0xFFFF)
    }

    /// X- and XO-form integer: `op rD/rS, rA, rB` with a ten-bit extended opcode, with
    /// and without the record bit.
    fn integer_x(xo: u32) -> [u32; 2] {
        let word = (31 << 26) | (4 << 21) | (1 << 16) | (2 << 11) | (xo << 1);
        [word, word | 1]
    }

    #[test]
    fn integer_agrees_with_the_interpreter() {
        let mut words = Vec::new();
        for imm in [0x1234, 0x8000, 0xFFFF, 1, 0] {
            for op in [14, 15, 12, 13, 8, 7, 11, 10, 24, 25, 26, 27, 28, 29] {
                words.push(d_form(op, imm));
            }
        }
        for xo in [
            266, 40, 104, 235, 75, 11, 28, 444, 316, 476, 124, 60, 412, 284, 24, 536, 26, 922, 954, 0, 32,
        ] {
            words.extend(integer_x(xo));
        }
        for sh in [0, 1, 7, 31] {
            // srawi rA, rS, sh
            words.extend(integer_x(824).map(|w| (w & !(0x1F << 11)) | (sh << 11)));
            for (mb, me) in [(0, 31), (24, 31), (0, 29), (30, 30), (28, 3)] {
                let fields = (sh << 11) | (mb << 6) | (me << 1);
                words.push((21 << 26) | (4 << 21) | (1 << 16) | fields);
                words.push((21 << 26) | (4 << 21) | (1 << 16) | fields | 1);
                words.push((20 << 26) | (4 << 21) | (1 << 16) | fields);
                words.push((23 << 26) | (4 << 21) | (1 << 16) | (2 << 11) | (mb << 6) | (me << 1));
            }
        }
        // mcrf 2,0 ; mfcr ; mtcrf 0xFF ; mtcrf 0x81 ; crxor/cror/crand/creqv/crnor/crnand/crandc/crorc 3,1,2
        words.extend([0x4D00_0000, 0x7C80_0026, 0x7C8F_F120, 0x7C88_1120]);
        for xo in [193, 449, 257, 289, 33, 225, 129, 417] {
            words.push((19 << 26) | (3 << 21) | (1 << 16) | (2 << 11) | (xo << 1));
        }
        // mflr/mtlr, mfctr/mtctr, mfxer
        words.extend([0x7C88_02A6, 0x7C88_03A6, 0x7C89_02A6, 0x7C89_03A6, 0x7C81_02A6]);
        check_all(&words);
    }

    #[test]
    fn floating_point_agrees_with_the_interpreter() {
        let a: Vec<u32> = [18, 20, 21, 22, 23, 25, 26, 28, 29, 30, 31]
            .iter()
            .map(|&xo| a_form(63, xo))
            .collect();
        let s: Vec<u32> = [18, 20, 21, 22, 24, 25, 28, 29, 30, 31]
            .iter()
            .map(|&xo| a_form(59, xo))
            .collect();
        let x: Vec<u32> = [0, 32, 12, 15, 40, 72, 136, 264]
            .iter()
            .map(|&xo| x_form(63, xo))
            .collect();
        check_all(&[a, s, x].concat());
    }

    #[test]
    fn paired_single_agrees_with_the_interpreter() {
        let a: Vec<u32> = [10, 11, 12, 13, 14, 15, 18, 20, 21, 23, 24, 25, 26, 28, 29, 30, 31]
            .iter()
            .map(|&xo| a_form(4, xo))
            .collect();
        let x: Vec<u32> = [0, 32, 64, 96, 40, 72, 136, 264, 528, 560, 592, 624]
            .iter()
            .map(|&xo| x_form(4, xo))
            .collect();
        check_all(&[a, x].concat());
    }
}
