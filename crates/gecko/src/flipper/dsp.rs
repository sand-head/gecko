pub mod accelerator;
pub mod addr;
pub mod condition;
pub mod core;
#[allow(dead_code, unused_variables, non_upper_case_globals, clippy::all)]
pub mod instruction;
pub mod interpreter;
#[cfg(feature = "jit")]
pub mod jit;
pub mod regs;

#[allow(dead_code, unused_variables, non_upper_case_globals, clippy::all)]
pub mod lut {
    include!(concat!(env!("OUT_DIR"), "/dsp_lut.rs"));
    include!(concat!(env!("OUT_DIR"), "/dsp_resolve.rs"));
}

#[allow(dead_code, unused_variables, non_upper_case_globals, clippy::all)]
pub mod lut_wii {
    include!(concat!(env!("OUT_DIR"), "/dsp_lut_wii.rs"));
    include!(concat!(env!("OUT_DIR"), "/dsp_resolve_wii.rs"));
}

use crate::flipper::dsp::instruction::Instruction;
use crate::mmio::Mmio;
use crate::system::{ExecutionMode, System, SystemId};

#[cfg(feature = "jit")]
pub const DSP_JIT_CHAIN_BUDGET: u32 = 64;

/// One instruction as the interpreter needs it, worked out once. Reaching an instruction
/// used to cost two reads of instruction memory, a word built out of their bytes, one
/// walk of the decode tree to find how long it is and another to find its handler, ending
/// in a call through a table — which in WebAssembly is a checked indirect call. All of
/// that is the same every time the instruction runs, so it is done once and kept here,
/// and running it is a load and a jump.
///
/// `size` is zero for an entry nothing has filled in yet; a real one is one word or two.
#[derive(Clone, Copy, Default, Debug)]
pub struct Decoded {
    pub instr: u32,
    pub leaf: u16,
    pub size: u8,
}

/// Instruction memory is 0x1000 words of IRAM and 0x1000 of IROM, and the cache holds
/// one entry for each.
const DECODED_SLOTS: usize = 0x2000;

pub struct Dsp {
    pub registers: core::Registers,
    /// Instructions already worked out, by where they are. Emptied whenever instruction
    /// memory changes, which is the only thing that can make an entry wrong.
    pub decoded: Box<[Decoded; DECODED_SLOTS]>,

    pub iram: Box<[u8; 0x2000]>,
    pub irom: Box<[u8; 0x2000]>,

    pub dram: Box<[u8; 0x2000]>,
    pub coef: Box<[u8; 0x2000]>,
    pub ifx: Box<[u8; 0x200]>,

    pub aram: Box<[u8; 16 * 1024 * 1024]>,

    pub csr: regs::ControlStatus,
    pub mailbox_to_dsp_hi: regs::MailboxToDspHi,
    pub mailbox_to_dsp_lo: regs::MailboxToDspLo,
    pub mailbox_to_cpu_hi: regs::MailboxToCpuHi,
    pub mailbox_to_cpu_lo: regs::MailboxToCpuLo,
    pub aram_info: regs::AramInfo,
    pub aram_mode: regs::AramMode,
    pub aram_refresh: regs::AramRefresh,
    pub aram_dma_mmio_addr: regs::AramDmaMmioAddr,
    pub aram_dma_aram_addr: regs::AramDmaAramAddr,
    pub aram_dma_control: regs::AramDmaControl,
    pub audio_dma_start_addr: regs::AudioDmaStartAddr,
    pub audio_dma_control: regs::AudioDmaControl,

    pub dma_control: core::regs::DspDmaControl,
    pub dma_length: u16,
    pub dma_dsp_addr: u16,
    pub dma_ram_addr_hi: u16,
    pub dma_ram_addr_lo: u16,

    pub accelerator: accelerator::Accelerator,

    #[cfg(feature = "jit")]
    pub jit: Option<Box<dyn DspJitHandle + Send>>,

    #[cfg(feature = "jit")]
    pub chain_budget: u32,

    #[cfg(feature = "jit")]
    pub instr_count: u32,

    pub wait_table: Box<[u8; 0x10000]>,

    pub scheduler_suspended: bool,
}

impl Dsp {
    pub fn new() -> Self {
        let aram = unsafe { Box::<[u8; 16 * 1024 * 1024]>::new_zeroed().assume_init() };
        let iram = unsafe { Box::<[u8; 0x2000]>::new_zeroed().assume_init() };
        let irom = unsafe { Box::<[u8; 0x2000]>::new_zeroed().assume_init() };
        let dram = unsafe { Box::<[u8; 0x2000]>::new_zeroed().assume_init() };
        let coef = unsafe { Box::<[u8; 0x2000]>::new_zeroed().assume_init() };
        let ifx = unsafe { Box::<[u8; 0x200]>::new_zeroed().assume_init() };

        Dsp {
            registers: core::Registers::default(),
            decoded: vec![Decoded::default(); DECODED_SLOTS]
                .into_boxed_slice()
                .try_into()
                .expect("the decode cache is exactly its own size"),
            iram,
            irom,
            dram,
            coef,
            ifx,
            aram,
            csr: regs::ControlStatus::default(),
            mailbox_to_dsp_hi: regs::MailboxToDspHi::from_raw(0),
            mailbox_to_dsp_lo: regs::MailboxToDspLo::from_raw(0),
            mailbox_to_cpu_hi: regs::MailboxToCpuHi::from_raw(0),
            mailbox_to_cpu_lo: regs::MailboxToCpuLo::from_raw(0),
            aram_info: regs::AramInfo::from_raw(0),
            aram_mode: regs::AramMode::from_raw(0),
            aram_refresh: regs::AramRefresh::from_raw(0),
            aram_dma_mmio_addr: regs::AramDmaMmioAddr::from_raw(0),
            aram_dma_aram_addr: regs::AramDmaAramAddr::from_raw(0),
            aram_dma_control: regs::AramDmaControl::from_raw(0),
            audio_dma_start_addr: regs::AudioDmaStartAddr::from_raw(0),
            audio_dma_control: regs::AudioDmaControl::from_raw(0),
            dma_control: core::regs::DspDmaControl::new(),
            dma_length: 0,
            dma_dsp_addr: 0,
            dma_ram_addr_hi: 0,
            dma_ram_addr_lo: 0,
            accelerator: accelerator::Accelerator::new(),

            #[cfg(feature = "jit")]
            jit: None,
            #[cfg(feature = "jit")]
            chain_budget: 0,
            #[cfg(feature = "jit")]
            instr_count: 0,
            wait_table: unsafe { Box::<[u8; 0x10000]>::new_zeroed().assume_init() },
            scheduler_suspended: false,
        }
    }

    #[inline(always)]
    pub fn is_waiting_for_cpu_mail(&self) -> bool {
        self.wait_table[self.registers.pc as usize] & 1 != 0
    }

    #[inline(always)]
    pub fn is_waiting_for_dsp_mail(&self) -> bool {
        self.wait_table[self.registers.pc as usize] & 2 != 0
    }

    #[inline(always)]
    pub fn parked_in_mailbox_wait(&self) -> bool {
        let cpu_mail_quiet = !self.mailbox_to_dsp_hi.busy();
        let dsp_mail_full = self.mailbox_to_cpu_hi.busy();
        (cpu_mail_quiet && self.is_waiting_for_cpu_mail())
            || (dsp_mail_full && self.is_waiting_for_dsp_mail())
            || self.is_parked_on_dram()
    }

    /// The Zelda ucode idles on its own DRAM rather than on the mailbox — a flag its
    /// mail interrupt handler sets, or a ring buffer's read and write words drawing
    /// level — so it is parked while the words say wait and no interrupt is pending.
    #[inline(always)]
    fn is_parked_on_dram(&self) -> bool {
        let entry = self.wait_table[self.registers.pc as usize];
        if entry & 0b1100 == 0 || self.csr.pi_interrupt() {
            return false;
        }
        let start = self.registers.pc.wrapping_sub((entry >> 4) as u16);
        let word = |at: u16| read_word(&*self.dram, self.read_imem(start.wrapping_add(at)));
        match entry & 0b1100 {
            0b0100 => word(1) == 0,
            _ => word(3) == word(5),
        }
    }

    /// Where in the decode cache an instruction address lives, if it is somewhere the
    /// cache covers.
    #[inline(always)]
    fn decoded_slot(addr: u16) -> Option<usize> {
        match addr {
            0x0000..0x1000 => Some(addr as usize),
            0x8000..0x9000 => Some(0x1000 + (addr - 0x8000) as usize),
            _ => None,
        }
    }

    pub fn rebuild_wait_table(&mut self) {
        // Instruction memory has changed, so everything worked out from it is wrong.
        self.decoded.fill(Decoded::default());
        const SDK_OFFSETS: [i16; 3] = [0, -1, -3];
        const IPL_OFFSETS: [i16; 5] = [0, -1, -2, -3, -5];
        const ZELDA_MAIL_OFFSETS: [i16; 3] = [0, -2, -4];
        const ZELDA_FLAG_OFFSETS: [i16; 3] = [0, -2, -3];
        const ZELDA_RING_OFFSETS: [i16; 6] = [0, -1, -2, -4, -6, -7];

        for pc in 0u32..0x10000 {
            let pc = pc as u16;

            let cpu = SDK_OFFSETS.iter().any(|&o| self.matches_cpu_mail_wait_at(pc, o))
                || IPL_OFFSETS.iter().any(|&o| self.matches_ipl_cpu_mail_wait_at(pc, o))
                || ZELDA_MAIL_OFFSETS
                    .iter()
                    .any(|&o| self.matches_zelda_cpu_mail_wait_at(pc, o));
            let dsp = SDK_OFFSETS.iter().any(|&o| self.matches_dsp_mail_wait_at(pc, o));
            // Bits 2-3 say which DRAM wait, bits 4.. how far back the loop starts, so
            // the addresses it polls can be read from the loads at its head.
            let dram = ZELDA_FLAG_OFFSETS
                .iter()
                .find(|&&o| self.matches_zelda_flag_wait_at(pc, o))
                .map(|&o| 0b0100 | ((-o as u8) << 4))
                .or_else(|| {
                    ZELDA_RING_OFFSETS
                        .iter()
                        .find(|&&o| self.matches_zelda_ring_wait_at(pc, o))
                        .map(|&o| 0b1000 | ((-o as u8) << 4))
                })
                .unwrap_or(0);

            self.wait_table[pc as usize] = (cpu as u8) | ((dsp as u8) << 1) | dram;
        }
    }

    fn matches_cpu_mail_wait_at(&self, pc: u16, offset: i16) -> bool {
        let start = pc.wrapping_add_signed(offset);
        let words = self.read_imem_window::<5>(start);
        let pattern_a = [0x26FE, 0x02C0, 0x8000, 0x029C, start];
        let pattern_b = [0x27FE, 0x03C0, 0x8000, 0x029C, start];
        let pattern_c = [0x26FE, 0x02A0, 0x8000, 0x029D, start];
        let pattern_d = [0x27FE, 0x03A0, 0x8000, 0x029D, start];
        words == pattern_a || words == pattern_b || words == pattern_c || words == pattern_d
    }

    // LR $AC0.M, @CMBH; ANDCF; JLNZ — the same wait as the Zelda ucode spells it.
    fn matches_zelda_cpu_mail_wait_at(&self, pc: u16, offset: i16) -> bool {
        let start = pc.wrapping_add_signed(offset);
        let words = self.read_imem_window::<6>(start);
        words == [0x00DE, 0xFFFE, 0x02C0, 0x8000, 0x029C, start]
    }

    fn matches_ipl_cpu_mail_wait_at(&self, pc: u16, offset: i16) -> bool {
        let start = pc.wrapping_add_signed(offset);
        let words = self.read_imem_window::<7>(start);
        words == [0x8100, 0x8900, 0x26FE, 0x02C0, 0x8000, 0x029C, start]
    }

    // LR $AX0.H, @0x0352; TST $ACC0; JZ — the Zelda ucode waiting for its handler, as
    // Dolphin's analyzer knows it.
    fn matches_zelda_flag_wait_at(&self, pc: u16, offset: i16) -> bool {
        let start = pc.wrapping_add_signed(offset);
        let words = self.read_imem_window::<4>(start);
        words == [0x00DA, 0x0352, 0x8600, 0x0295]
    }

    // CLR $ACC0; CLR $ACC1; LR $AC1.M, @read; LR $AC0.M, @write; CMP; JEQ back — the
    // Zelda ucode waiting for its command ring to fill.
    fn matches_zelda_ring_wait_at(&self, pc: u16, offset: i16) -> bool {
        let start = pc.wrapping_add_signed(offset);
        let words = self.read_imem_window::<9>(start);
        words[0] == 0x8100
            && words[1] == 0x8900
            && words[2] == 0x00DF
            && words[4] == 0x00DE
            && words[6] == 0x8200
            && words[7] == 0x0293
            && words[8] == start
    }

    fn matches_dsp_mail_wait_at(&self, pc: u16, offset: i16) -> bool {
        let start = pc.wrapping_add_signed(offset);
        let words = self.read_imem_window::<5>(start);
        let pattern_a = [0x26FC, 0x02C0, 0x8000, 0x029D, start];
        let pattern_b = [0x27FC, 0x03C0, 0x8000, 0x029D, start];
        let pattern_c = [0x26FC, 0x02A0, 0x8000, 0x029C, start];
        let pattern_d = [0x27FC, 0x03A0, 0x8000, 0x029C, start];
        words == pattern_a || words == pattern_b || words == pattern_c || words == pattern_d
    }

    fn read_imem_window<const N: usize>(&self, start: u16) -> [u16; N] {
        let mut out = [0u16; N];
        for i in 0..N {
            out[i] = self.read_imem(start.wrapping_add(i as u16));
        }
        out
    }

    pub fn process_aram_dma<const SYSTEM: SystemId>(&mut self, mmio: &mut Mmio<SYSTEM>) {
        let ram_addr = (self.aram_dma_mmio_addr.raw() & 0x3FFFFFFF) as usize;
        let aram_addr = self.aram_dma_aram_addr.raw() as usize;
        let count = self.aram_dma_control.count() as usize;

        tracing::debug!(
            ram_addr = format!("{ram_addr:08X}"),
            aram_addr = format!("{aram_addr:08X}"),
            count,
            direction = ?self.aram_dma_control.direction(),
            "ARAM DMA"
        );

        let within_bounds = aram_addr + count <= self.aram.len();
        match self.aram_dma_control.direction() {
            regs::DmaDirection::AramToRam if within_bounds => {
                let src = &self.aram[aram_addr..aram_addr + count];
                let dst = mmio.virt_slice_mut(ram_addr as u32, count);
                dst.copy_from_slice(src);
                mmio.queue_icbi_for_range(crate::mmio::virt_to_phys(ram_addr as u32), count as u32);
            }
            regs::DmaDirection::RamToAram if within_bounds => {
                let src = mmio.virt_slice(ram_addr as u32, count);
                self.aram[aram_addr..aram_addr + count].copy_from_slice(&src);
            }
            _ => tracing::warn!("Ignoring out-of-bounds ARAM DMA transfer"),
        }

        self.aram_dma_control.set_count(0);
        self.csr.set_dma_status(false);
        self.csr = self.csr.with_ar_interrupt(true);
    }

    pub fn process_ucode_upload<const SYSTEM: SystemId>(&mut self, mmio: &mut Mmio<SYSTEM>) {
        const UCODE_ADDR: usize = 0x8100_0000;
        const UCODE_SIZE: usize = 1024;
        let src = mmio.virt_slice(UCODE_ADDR as u32, UCODE_SIZE);
        self.iram[..UCODE_SIZE].copy_from_slice(&src);

        tracing::info!(
            mmio_addr = format!("{UCODE_ADDR:08X}"),
            count = UCODE_SIZE,
            "DSP stub uploaded from RAM to IRAM, executing IRAM"
        );

        self.csr.set_dma_status(false);
        self.csr.set_dsp_interrupt(true);

        self.rebuild_wait_table();

        #[cfg(feature = "jit")]
        if let Some(jit) = self.jit.as_mut() {
            jit.flush();
        }
    }

    pub fn process_dsp_dma<const SYSTEM: SystemId>(&mut self, mmio: &mut Mmio<SYSTEM>) {
        let ram_addr = ((self.dma_ram_addr_hi as u32) << 16) | self.dma_ram_addr_lo as u32;
        let dsp_addr = (self.dma_dsp_addr as usize) * 2;
        let len = self.dma_length as usize;

        tracing::debug!(
            ram_addr = format!("{ram_addr:08X}"),
            dsp_addr = format!("{dsp_addr:04X}"),
            len,
            dir = ?self.dma_control.direction(),
            mem = ?self.dma_control.memory_type(),
            "DSP DMA"
        );

        let memory = match self.dma_control.memory_type() {
            core::regs::DspMemoryType::Data => &mut *self.dram,
            core::regs::DspMemoryType::Instruction => &mut *self.iram,
        };

        let mem_type = self.dma_control.memory_type();
        let direction = self.dma_control.direction();
        match direction {
            core::regs::DspDmaDirection::MainToDsp => {
                let src = mmio.virt_slice(ram_addr, len);
                memory[dsp_addr..dsp_addr + len].copy_from_slice(&src);
            }
            core::regs::DspDmaDirection::DspToMain => {
                let src = &memory[dsp_addr..dsp_addr + len];
                let dst = mmio.virt_slice_mut(ram_addr, len);
                dst.copy_from_slice(src);
                mmio.queue_icbi_for_range(crate::mmio::virt_to_phys(ram_addr), len as u32);
            }
        }

        if matches!(
            (mem_type, direction),
            (
                core::regs::DspMemoryType::Instruction,
                core::regs::DspDmaDirection::MainToDsp
            )
        ) {
            self.rebuild_wait_table();
            #[cfg(feature = "jit")]
            if let Some(jit) = self.jit.as_mut() {
                jit.flush();
            }
        }

        self.dma_length = 0;
    }
}

impl<const SYSTEM: SystemId> System<SYSTEM> {
    #[inline(always)]
    pub fn step_dsp_instruction(&mut self) -> bool {
        if self.dsp.csr.reset() || self.dsp.csr.halt() {
            return false;
        }
        if self.dsp.csr.pi_interrupt() && self.dsp.registers.status.external_interrupt_enable() {
            self.dsp.csr = self.dsp.csr.with_pi_interrupt(false);
            self.dsp.registers.call_stack.push(self.dsp.registers.pc);
            self.dsp.registers.data_stack.push(self.dsp.registers.status.raw());
            self.dsp.registers.status = self.dsp.registers.status.with_external_interrupt_enable(false);
            self.dsp.registers.pc = 0x000E;
        }

        let pc = self.dsp.registers.pc;
        let Decoded { instr, leaf, size } = self.dsp.decode::<SYSTEM>(pc);
        let instr = Instruction(instr);
        self.dsp.registers.cia = pc;
        self.dsp.registers.nia = pc.wrapping_add(size as u16);

        let ext_op = instr.ext_opcode();
        if ext_op.is_some() {
            self.dsp.registers.cache_ext_ac();
        }

        self::execute(self, leaf, instr);

        if let Some(ext) = ext_op {
            self::dispatch_gc_dsp_ext(self, instruction::GcDspExt(ext));
        }

        let at_loop_end =
            !self.dsp.registers.loop_addr.is_empty() && self.dsp.registers.nia == self.dsp.registers.loop_addr.top();
        if at_loop_end {
            let counter = self.dsp.registers.loop_counter.top().wrapping_sub(1);
            if counter != 0 {
                self.dsp.registers.loop_counter.set_top(counter);
                self.dsp.registers.nia = self.dsp.registers.call_stack.top();
            } else {
                self.dsp.registers.loop_counter.pop();
                self.dsp.registers.loop_addr.pop();
                self.dsp.registers.call_stack.pop();
            }
        }

        self.dsp.registers.pc = self.dsp.registers.nia;
        true
    }

    pub fn execute_dsp_batch(&mut self) {
        #[cfg(feature = "jit")]
        if self.execution_mode == ExecutionMode::Jit {
            self.execute_dsp_batch_jit();
            return;
        }
        self.execute_dsp_batch_interp();
    }

    pub fn drain_dsp_synchronous(&mut self, max_steps: u32) {
        #[cfg(feature = "jit")]
        if self.execution_mode == ExecutionMode::Jit {
            self.drain_dsp_synchronous_jit(max_steps);
            return;
        }
        self.drain_dsp_synchronous_interp(max_steps);
    }

    #[cfg(feature = "jit")]
    fn dsp_jit_step(&mut self, iram: &[u8], irom: &[u8]) -> u64 {
        let ctx_ptr = self as *mut crate::system::System<SYSTEM> as *mut ::core::ffi::c_void;

        if self.dsp.csr.pi_interrupt() && self.dsp.registers.status.external_interrupt_enable() {
            self.dsp.csr = self.dsp.csr.with_pi_interrupt(false);
            self.dsp.registers.call_stack.push(self.dsp.registers.pc);
            self.dsp.registers.data_stack.push(self.dsp.registers.status.raw());
            self.dsp.registers.status = self.dsp.registers.status.with_external_interrupt_enable(false);
            self.dsp.registers.pc = 0x000E;
        }

        let start_pc = self.dsp.registers.pc;
        self.dsp.chain_budget = DSP_JIT_CHAIN_BUDGET;
        self.dsp.instr_count = 0;

        let next_pc = self.dsp.jit.as_mut().unwrap().run_block(ctx_ptr, iram, irom, start_pc);
        self.dsp.registers.pc = next_pc;

        #[cfg(feature = "jit-stats")]
        {
            let chain_depth = DSP_JIT_CHAIN_BUDGET - self.dsp.chain_budget;
            self.dsp.jit.as_mut().unwrap().record_chain_depth(chain_depth);
        }

        (self.dsp.instr_count as u64).max(1)
    }

    #[cfg(feature = "jit")]
    #[cfg_attr(feature = "hotpath", hotpath::measure(label = "dsp_batch"))]
    fn execute_dsp_batch_jit(&mut self) {
        #[cfg(feature = "jit-stats")]
        let batch_start = std::time::Instant::now();

        if self.dsp.csr.reset() || self.dsp.csr.halt() {
            self::refresh_interrupts(self);
            return;
        }

        if self.dsp.jit.is_none() {
            self.dsp.jit = Some(Box::new(jit::JitEngine::<SYSTEM>::new()));
        }

        let iram_ptr = self.dsp.iram.as_ptr();
        let irom_ptr = self.dsp.irom.as_ptr();
        let iram_len = self.dsp.iram.len();
        let irom_len = self.dsp.irom.len();
        let iram = unsafe { ::core::slice::from_raw_parts(iram_ptr, iram_len) };
        let irom = unsafe { ::core::slice::from_raw_parts(irom_ptr, irom_len) };

        let mut budget = crate::scheduler::DSP_BATCH_SIZE as u64;
        while budget > 0 {
            if self.dsp.parked_in_mailbox_wait() {
                break;
            }

            if self.dsp.csr.reset() || self.dsp.csr.halt() {
                break;
            }

            let consumed = self.dsp_jit_step(iram, irom);
            budget = budget.saturating_sub(consumed);
        }

        #[cfg(feature = "jit-stats")]
        {
            use std::sync::atomic::Ordering;

            DSP_BATCH_CALLS.fetch_add(1, Ordering::Relaxed);
            DSP_BATCH_STEPS.fetch_add(crate::scheduler::DSP_BATCH_SIZE - budget, Ordering::Relaxed);
            DSP_BATCH_NANOS.fetch_add(batch_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }

        self::refresh_interrupts(self);
    }

    #[inline(always)]
    fn execute_dsp_batch_interp(&mut self) {
        for _ in 0..crate::scheduler::DSP_BATCH_SIZE {
            if !self.step_dsp_instruction() {
                break;
            }
        }
        self::refresh_interrupts(self);
    }

    #[cfg(feature = "jit")]
    #[cfg_attr(feature = "hotpath", hotpath::measure(label = "dsp_drain"))]
    fn drain_dsp_synchronous_jit(&mut self, max_steps: u32) {
        #[cfg(feature = "jit-stats")]
        let drain_start = std::time::Instant::now();

        let already_busy = self.dsp.mailbox_to_cpu_hi.busy();

        if self.dsp.csr.reset() || self.dsp.csr.halt() {
            #[cfg(feature = "jit-stats")]
            {
                DSP_DRAIN_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                DSP_DRAIN_EXIT_HALT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }

            self::refresh_interrupts(self);
            return;
        }

        if self.dsp.jit.is_none() {
            self.dsp.jit = Some(Box::new(jit::JitEngine::<SYSTEM>::new()));
        }

        let iram_ptr = self.dsp.iram.as_ptr();
        let irom_ptr = self.dsp.irom.as_ptr();
        let iram_len = self.dsp.iram.len();
        let irom_len = self.dsp.irom.len();
        let iram = unsafe { ::core::slice::from_raw_parts(iram_ptr, iram_len) };
        let irom = unsafe { ::core::slice::from_raw_parts(irom_ptr, irom_len) };

        let mut budget = max_steps as u64;

        #[cfg(feature = "jit-stats")]
        let mut blocks: u64 = 0;

        while budget > 0 {
            if self.dsp.csr.reset() || self.dsp.csr.halt() {
                break;
            }

            if !already_busy && self.dsp.mailbox_to_cpu_hi.busy() {
                break;
            }

            if self.dsp.parked_in_mailbox_wait() {
                break;
            }

            let consumed = self.dsp_jit_step(iram, irom);
            budget = budget.saturating_sub(consumed);

            #[cfg(feature = "jit-stats")]
            {
                blocks += 1;
            }
        }

        #[cfg(feature = "jit-stats")]
        {
            use std::sync::atomic::Ordering;

            DSP_DRAIN_CALLS.fetch_add(1, Ordering::Relaxed);
            DSP_DRAIN_STEPS.fetch_add(max_steps as u64 - budget, Ordering::Relaxed);
            DSP_DRAIN_BLOCKS.fetch_add(blocks, Ordering::Relaxed);
            DSP_DRAIN_NANOS.fetch_add(drain_start.elapsed().as_nanos() as u64, Ordering::Relaxed);

            let exit = if self.dsp.csr.reset() || self.dsp.csr.halt() {
                &DSP_DRAIN_EXIT_HALT
            } else if !already_busy && self.dsp.mailbox_to_cpu_hi.busy() {
                &DSP_DRAIN_EXIT_ANSWERED
            } else if budget > 0 {
                &DSP_DRAIN_EXIT_WAIT
            } else {
                &DSP_DRAIN_EXIT_BUDGET
            };
            exit.fetch_add(1, Ordering::Relaxed);
        }

        self::refresh_interrupts(self);
    }

    fn drain_dsp_synchronous_interp(&mut self, max_steps: u32) {
        let already_busy = self.dsp.mailbox_to_cpu_hi.busy();

        for _ in 0..max_steps {
            if !self.step_dsp_instruction() {
                break;
            }

            if !already_busy && self.dsp.mailbox_to_cpu_hi.busy() {
                break;
            }

            if self.dsp.parked_in_mailbox_wait() {
                break;
            }
        }

        self::refresh_interrupts(self);
    }
}

crate::mmio_device_dispatch! {
    read = dsp_read,
    write = dsp_write,
    registers = [
        regs::ControlStatus,
        regs::MailboxToDspHi,
        regs::MailboxToDspLo,
        regs::MailboxToCpuHi,
        regs::MailboxToCpuLo,
        regs::AramInfo,
        regs::AramMode,
        regs::AramRefresh,
        regs::AramDmaMmioAddr,
        regs::AramDmaAramAddr,
        regs::AramDmaControl,
        regs::AudioDmaStartAddr,
        regs::AudioDmaControl,
        regs::AudioDmaBlocksLeft,
    ],
}

impl Dsp {
    /// The instruction at `addr`, worked out the first time and remembered after.
    #[inline(always)]
    pub fn decode<const SYSTEM: SystemId>(&mut self, addr: u16) -> Decoded {
        let slot = Self::decoded_slot(addr);
        if let Some(slot) = slot {
            let entry = self.decoded[slot];
            if entry.size != 0 {
                return entry;
            }
        }
        let w0 = self.read_imem(addr);
        let w1 = self.read_imem(addr.wrapping_add(1));
        let instr = Instruction::from_be_bytes(&[(w0 >> 8) as u8, w0 as u8, (w1 >> 8) as u8, w1 as u8]);
        let entry = Decoded {
            instr: instr.0,
            leaf: self::resolve::<SYSTEM>(instr),
            size: if SYSTEM == crate::system::GC {
                lut::instr_size(instr) as u8
            } else {
                lut_wii::instr_size(instr) as u8
            },
        };
        if let Some(slot) = slot {
            self.decoded[slot] = entry;
        }
        entry
    }

    pub fn read_imem(&self, addr: u16) -> u16 {
        match addr {
            0x0000..0x1000 => read_word(&*self.iram, addr),
            0x8000..0x9000 => read_word(&*self.irom, addr - 0x8000),
            _ => 0,
        }
    }

    pub fn load_irom(&mut self, data: &[u8]) {
        let len = data.len().min(self.irom.len());
        self.irom[..len].copy_from_slice(&data[..len]);
        tracing::info!(size = len, "loaded DSP IROM");
        self.rebuild_wait_table();
    }

    pub fn load_coef(&mut self, data: &[u8]) {
        let len = data.len().min(self.coef.len());
        self.coef[..len].copy_from_slice(&data[..len]);
        tracing::info!(size = len, "loaded DSP coefficient ROM");
    }

    #[inline(always)]
    pub fn interrupt_active(&self) -> bool {
        (self.csr.ai_interrupt() && self.csr.ai_interrupt_mask())
            || (self.csr.ar_interrupt() && self.csr.ar_interrupt_mask())
            || (self.csr.dsp_interrupt() && self.csr.dsp_interrupt_mask())
    }
}

#[inline(always)]
pub fn refresh_interrupts<const SYSTEM: SystemId>(sys: &mut System<SYSTEM>) {
    use crate::flipper::pi::InterruptFlag;

    if sys.dsp.interrupt_active() {
        sys.pi.assert_interrupt(InterruptFlag::Dsp);
    } else {
        sys.pi.clear_interrupt(InterruptFlag::Dsp);
    }

    if sys.dsp.csr.pi_interrupt() && sys.dsp.registers.status.external_interrupt_enable() {
        self::wake_dsp_scheduler::<SYSTEM>(sys);
    }
}

#[cfg(feature = "jit-stats")]
pub static DSP_SUSPEND_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "jit-stats")]
pub static DSP_WAKE_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "jit-stats")]
pub static DSP_DRAIN_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "jit-stats")]
pub static DSP_DRAIN_STEPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "jit-stats")]
pub static DSP_DRAIN_BLOCKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "jit-stats")]
pub static DSP_DRAIN_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "jit-stats")]
pub static DSP_DRAIN_EXIT_HALT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "jit-stats")]
pub static DSP_DRAIN_EXIT_ANSWERED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "jit-stats")]
pub static DSP_DRAIN_EXIT_WAIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "jit-stats")]
pub static DSP_DRAIN_EXIT_BUDGET: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "jit-stats")]
pub static DSP_BATCH_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "jit-stats")]
pub static DSP_BATCH_STEPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "jit-stats")]
pub static DSP_BATCH_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[inline]
pub fn wake_dsp_scheduler<const SYSTEM: SystemId>(sys: &mut System<SYSTEM>) {
    if !sys.dsp.scheduler_suspended {
        return;
    }

    if sys.dsp.csr.halt() || sys.dsp.csr.reset() {
        return;
    }

    sys.dsp.scheduler_suspended = false;
    sys.scheduler.schedule_in(
        crate::scheduler::dsp_batch_interval(SYSTEM),
        crate::scheduler::dsp_batch_handler::<SYSTEM>,
    );

    #[cfg(feature = "jit-stats")]
    DSP_WAKE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[inline(always)]
pub fn read_dmem<const SYSTEM: SystemId>(sys: &mut System<SYSTEM>, addr: u16) -> u16 {
    match addr {
        0x0000..0x1000 => read_word(&*sys.dsp.dram, addr),
        0x1000..0x2000 => read_word(&*sys.dsp.coef, addr - 0x1000),
        0xFF00..=0xFFFF => read_ifx(sys, addr),
        _ => 0,
    }
}

#[inline(always)]
pub fn write_dmem<const SYSTEM: SystemId>(sys: &mut System<SYSTEM>, addr: u16, value: u16) {
    match addr {
        0x0000..0x1000 => write_word(&mut *sys.dsp.dram, addr, value),
        0xFF00..=0xFFFF => write_ifx(sys, addr, value),
        _ => {}
    }
}

#[inline(always)]
fn read_ifx<const SYSTEM: SystemId>(sys: &mut System<SYSTEM>, addr: u16) -> u16 {
    match addr {
        // CMBH (CPU Mailbox High): reading returns data + M bit.
        // M is only cleared when CMBL is read.
        addr::IFX_CMBH => sys.dsp.mailbox_to_dsp_hi.raw(),
        // CMBL (CPU Mailbox Low): reading clears CMBH.M (busy)
        addr::IFX_CMBL => {
            sys.dsp.mailbox_to_dsp_hi.set_busy(false);
            sys.dsp.mailbox_to_dsp_lo.raw()
        }
        // DMBH (DSP Mailbox High): DSP reads back what it wrote
        addr::IFX_DMBH => sys.dsp.mailbox_to_cpu_hi.raw(),
        // DMBL (DSP Mailbox Low): DSP reads back what it wrote (no side effects)
        addr::IFX_DMBL => sys.dsp.mailbox_to_cpu_lo.raw(),
        // DSP DMA registers
        addr::IFX_DSCR => sys.dsp.dma_control.raw(),
        addr::IFX_DSBL => sys.dsp.dma_length,
        addr::IFX_DSPA => sys.dsp.dma_dsp_addr,
        addr::IFX_DSMAH => sys.dsp.dma_ram_addr_hi,
        addr::IFX_DSMAL => sys.dsp.dma_ram_addr_lo,
        // Audio sample accelerator
        addr::IFX_FORMAT => sys.dsp.accelerator.format.raw(),
        addr::IFX_ACSAH => (sys.dsp.accelerator.start_addr >> 16) as u16,
        addr::IFX_ACSAL => sys.dsp.accelerator.start_addr as u16,
        addr::IFX_ACEAH => (sys.dsp.accelerator.end_addr >> 16) as u16,
        addr::IFX_ACEAL => sys.dsp.accelerator.end_addr as u16,
        addr::IFX_ACCAH => (sys.dsp.accelerator.current_addr >> 16) as u16,
        addr::IFX_ACCAL => sys.dsp.accelerator.current_addr as u16,
        addr::IFX_PRED_SCALE => sys.dsp.accelerator.pred_scale,
        addr::IFX_YN1 => sys.dsp.accelerator.yn1 as u16,
        addr::IFX_YN2 => sys.dsp.accelerator.yn2 as u16,
        addr::IFX_GAIN => sys.dsp.accelerator.gain as u16,
        addr::IFX_ACIN => sys.dsp.accelerator.input,
        addr::IFX_ACDSAMP => accelerator::read_decoded_sample::<SYSTEM>(&mut sys.dsp, sys.mmio.ram_view()),
        addr::IFX_ACDRAW => accelerator::read_raw::<SYSTEM>(&mut sys.dsp, sys.mmio.ram_view()),
        _ => {
            tracing::debug!(addr = format!("{:04X}", addr), "Read from unknown DSP IFX register");
            read_word(&*sys.dsp.ifx, addr - 0xFF00)
        }
    }
}

#[inline(always)]
fn write_ifx<const SYSTEM: SystemId>(sys: &mut System<SYSTEM>, addr: u16, value: u16) {
    match addr {
        // DMBH (DSP Mailbox High): store data bits (14:0), busy is preserved
        addr::IFX_DMBH => {
            let busy = sys.dsp.mailbox_to_cpu_hi.busy();
            sys.dsp.mailbox_to_cpu_hi = regs::MailboxToCpuHi::from_raw(value & 0x7FFF).with_busy(busy);
        }
        // DMBL (DSP Mailbox Low): writing sets DMBH.M
        addr::IFX_DMBL => {
            sys.dsp.mailbox_to_cpu_lo = regs::MailboxToCpuLo::from_raw(value);
            sys.dsp.mailbox_to_cpu_hi.set_busy(true);
        }
        // DIRQ: DSP explicitly raises interrupt to CPU
        addr::IFX_DIRQ => {
            if value & 1 != 0 {
                tracing::debug!("DSP DIRQ: requesting CPU interrupt");
                sys.dsp.csr.set_dsp_interrupt(true);
            }
        }
        // CMBH/CMBL are read-only from DSP side
        addr::IFX_CMBH | addr::IFX_CMBL => {}

        addr::IFX_DSBL => {
            sys.dsp.dma_length = value;
            sys.dsp.process_dsp_dma(&mut sys.mmio);
        }
        addr::IFX_DSCR => sys.dsp.dma_control = core::regs::DspDmaControl::from_raw(value),
        addr::IFX_DSPA => sys.dsp.dma_dsp_addr = value,
        addr::IFX_DSMAH => sys.dsp.dma_ram_addr_hi = value,
        addr::IFX_DSMAL => sys.dsp.dma_ram_addr_lo = value,
        // Audio sample accelerator
        addr::IFX_FORMAT => sys.dsp.accelerator.format = accelerator::SampleFormat::from_raw(value),
        addr::IFX_ACSAH => {
            let new = ((value as u32) << 16) | (sys.dsp.accelerator.start_addr & 0x0000_FFFF);
            sys.dsp.accelerator.set_start_addr(new);
        }
        addr::IFX_ACSAL => {
            let new = (sys.dsp.accelerator.start_addr & 0xFFFF_0000) | value as u32;
            sys.dsp.accelerator.set_start_addr(new);
        }
        addr::IFX_ACEAH => {
            let new = ((value as u32) << 16) | (sys.dsp.accelerator.end_addr & 0x0000_FFFF);
            sys.dsp.accelerator.set_end_addr(new);
        }
        addr::IFX_ACEAL => {
            let new = (sys.dsp.accelerator.end_addr & 0xFFFF_0000) | value as u32;
            sys.dsp.accelerator.set_end_addr(new);
        }
        addr::IFX_ACCAH => {
            let new = ((value as u32) << 16) | (sys.dsp.accelerator.current_addr & 0x0000_FFFF);
            sys.dsp.accelerator.set_current_addr(new);
        }
        addr::IFX_ACCAL => {
            let new = (sys.dsp.accelerator.current_addr & 0xFFFF_0000) | value as u32;
            sys.dsp.accelerator.set_current_addr(new);
        }
        addr::IFX_PRED_SCALE => sys.dsp.accelerator.set_pred_scale(value),
        addr::IFX_YN1 => sys.dsp.accelerator.yn1 = value as i16,
        addr::IFX_YN2 => sys.dsp.accelerator.set_yn2(value as i16),
        addr::IFX_GAIN => sys.dsp.accelerator.gain = value as i16,
        addr::IFX_ACIN => sys.dsp.accelerator.input = value,
        addr::IFX_ACDRAW => accelerator::write_raw::<SYSTEM>(&mut sys.dsp, sys.mmio.ram_view_mut(), value),
        // ACDSAMP is read-only
        addr::IFX_ACDSAMP => {}
        _ => {
            tracing::debug!(
                addr = format!("{:04X}", addr),
                value = format!("{:04X}", value),
                "Write to unknown DSP IFX register"
            );
            write_word(&mut *sys.dsp.ifx, addr - 0xFF00, value);
        }
    }
}

/// Read a big-endian u16 from a byte slice at a DSP word address.
#[inline(always)]
fn read_word(mem: &[u8], word_addr: u16) -> u16 {
    let off = word_addr as usize * 2;
    u16::from_be_bytes([mem[off], mem[off + 1]])
}

/// Write a big-endian u16 into a byte slice at a DSP word address.
#[inline(always)]
fn write_word(mem: &mut [u8], word_addr: u16, value: u16) {
    let off = word_addr as usize * 2;
    mem[off..off + 2].copy_from_slice(&value.to_be_bytes());
}

#[inline(always)]
pub fn dispatch<const SYSTEM: SystemId>(ctx: &mut System<SYSTEM>, instr: Instruction) {
    if SYSTEM == crate::system::GC {
        let ctx: &mut crate::gamecube::GameCube = unsafe { ::core::mem::transmute(ctx) };
        self::lut::dispatch(ctx, instr);
    } else {
        let ctx: &mut crate::wii::Wii = unsafe { ::core::mem::transmute(ctx) };
        self::lut_wii::dispatch(ctx, instr);
    }
}

/// The number `dispatch` would have walked its tables to reach, so an instruction the
/// decode cache has already seen goes straight to its handler through one jump.
#[inline(always)]
pub fn resolve<const SYSTEM: SystemId>(instr: Instruction) -> u16 {
    if SYSTEM == crate::system::GC {
        self::lut::resolve(instr)
    } else {
        self::lut_wii::resolve(instr)
    }
}

#[inline(always)]
pub fn execute<const SYSTEM: SystemId>(ctx: &mut System<SYSTEM>, leaf: u16, instr: Instruction) {
    if SYSTEM == crate::system::GC {
        let ctx: &mut crate::gamecube::GameCube = unsafe { ::core::mem::transmute(ctx) };
        self::lut::execute(leaf, ctx, instr);
    } else {
        let ctx: &mut crate::wii::Wii = unsafe { ::core::mem::transmute(ctx) };
        self::lut_wii::execute(leaf, ctx, instr);
    }
}

#[inline(always)]
pub fn dispatch_gc_dsp_ext<const SYSTEM: SystemId>(ctx: &mut System<SYSTEM>, instr: instruction::GcDspExt) {
    if SYSTEM == crate::system::GC {
        let ctx: &mut crate::gamecube::GameCube = unsafe { ::core::mem::transmute(ctx) };
        self::lut::dispatch_gc_dsp_ext(ctx, instr);
    } else {
        let ctx: &mut crate::wii::Wii = unsafe { ::core::mem::transmute(ctx) };
        self::lut_wii::dispatch_gc_dsp_ext(ctx, instr);
    }
}

#[cfg(feature = "jit")]
pub trait DspJitHandle {
    fn run_block(&mut self, ctx_ptr: *mut ::core::ffi::c_void, iram: &[u8], irom: &[u8], start_pc: u16) -> u16;
    fn record_chain_depth(&mut self, depth: u32);
    fn flush(&mut self);
    fn dump_hot_blocks(&self, top_k: usize);
    fn dump_hot_blocks_csv(&self, top_k: usize, path: &std::path::Path) -> std::io::Result<()>;
    fn dump_top_clif(&mut self, top_k: usize, iram: &[u8], irom: &[u8]);
    fn cached_blocks(&self) -> Vec<crate::jit::cache::CachedBlockDsp>;
    fn precompile_blocks(
        &mut self,
        iram: &[u8],
        irom: &[u8],
        blocks: &[crate::jit::cache::CachedBlockDsp],
    ) -> (usize, usize);
}

#[cfg(feature = "jit")]
impl<const SYSTEM: SystemId> DspJitHandle for jit::JitEngine<SYSTEM> {
    fn run_block(&mut self, ctx_ptr: *mut ::core::ffi::c_void, iram: &[u8], irom: &[u8], start_pc: u16) -> u16 {
        let entry = match self.lookup_block_fast(start_pc) {
            Some(entry) => entry,
            None => self.lookup_or_compile(iram, irom, start_pc),
        };
        Self::run_block(self, ctx_ptr, entry)
    }

    fn record_chain_depth(&mut self, depth: u32) {
        Self::record_chain_depth(self, depth);
    }

    fn flush(&mut self) {
        Self::flush(self);
    }

    fn dump_hot_blocks(&self, _top_k: usize) {
        #[cfg(feature = "jit-stats")]
        Self::dump_hot_blocks(self, _top_k);
        #[cfg(not(feature = "jit-stats"))]
        tracing::warn!("feature `jit-stats` is not enabled. Rebuild with `--features jit-stats`.");
    }

    fn dump_hot_blocks_csv(&self, _top_k: usize, _path: &std::path::Path) -> std::io::Result<()> {
        #[cfg(feature = "jit-stats")]
        return Self::dump_hot_blocks_csv(self, _top_k, _path);
        #[cfg(not(feature = "jit-stats"))]
        Ok(())
    }

    fn cached_blocks(&self) -> Vec<crate::jit::cache::CachedBlockDsp> {
        Self::cached_blocks(self)
    }

    fn precompile_blocks(
        &mut self,
        iram: &[u8],
        irom: &[u8],
        blocks: &[crate::jit::cache::CachedBlockDsp],
    ) -> (usize, usize) {
        Self::precompile_blocks(self, iram, irom, blocks)
    }

    fn dump_top_clif(&mut self, _top_k: usize, _iram: &[u8], _irom: &[u8]) {
        #[cfg(feature = "jit-stats")]
        {
            let mut pcs: Vec<(u16, u64)> = self.hits.iter().map(|(&pc, &n)| (pc, n)).collect();
            pcs.sort_by(|a, b| b.1.cmp(&a.1));
            for (pc, hits) in pcs.into_iter().take(_top_k) {
                tracing::info!("hits={hits} pc={pc:04X}");
                self.dump_block_clif(pc, _iram, _irom);
            }
        }
        #[cfg(not(feature = "jit-stats"))]
        tracing::warn!("feature `jit-stats` is not enabled. Rebuild with `--features jit-stats`.");
    }
}

/// The decode cache sends an instruction to a handler by number, where `dispatch` walked
/// a tree of tables to reach it. They have to be the same handler, for every instruction
/// there is: the number comes from a table generated beside the one the tree walks, and
/// nothing else checks that the two stayed in step.
#[cfg(test)]
mod decode_tests {
    use super::*;
    use crate::system::GC;

    /// Everything a handler could have touched, cheaply.
    fn fingerprint(dsp: &Dsp) -> Vec<u16> {
        let r = &dsp.registers;
        let mut out = vec![
            r.pc,
            r.nia,
            r.cia,
            r.ac0_high,
            r.ac1_high,
            r.ac0_mid,
            r.ac1_mid,
            r.ac0_low,
            r.ac1_low,
            r.config,
            r.status.raw(),
            r.product_low,
            r.product_mid1,
            r.product_high,
            r.product_mid2,
        ];
        out.extend(r.ar);
        out.extend(r.ix);
        out.extend(r.wr);
        out.extend(r.ax);
        out.extend(r.axh);
        let sum = |bytes: &[u8]| {
            bytes
                .iter()
                .fold(0u16, |a, &b| a.wrapping_mul(31).wrapping_add(b as u16))
        };
        out.push(sum(&dsp.dram[..]));
        out.push(sum(&dsp.ifx[..]));
        out
    }

    /// The same starting point for both runs, with something in every register so a
    /// handler that only moves one has something to move.
    fn seed(dsp: &mut Dsp) {
        dsp.registers = core::Registers::default();
        dsp.registers.pc = 0x0100;
        for i in 0..4 {
            dsp.registers.ar[i] = 0x0010 + i as u16;
            dsp.registers.ix[i] = 1;
            dsp.registers.wr[i] = 0xFFFF;
        }
        dsp.registers.ac0_high = 0x1234;
        dsp.registers.ac1_high = 0x5678;
        dsp.registers.ac0_mid = 0x9ABC;
        dsp.registers.ac1_mid = 0xDEF0;
        dsp.registers.ax[0] = 0x0F0F;
        dsp.registers.ax[1] = 0xF0F0;
        dsp.registers.axh[0] = 0x00FF;
        dsp.registers.axh[1] = 0xFF00;
        for (i, byte) in dsp.dram.iter_mut().enumerate() {
            *byte = i as u8;
        }
        dsp.ifx.fill(0);
    }

    #[test]
    fn every_opcode_reaches_the_handler_dispatch_would() {
        let mut sys = crate::gamecube::GameCube::new(0x8000_3000);
        // An unimplemented opcode panics, in both paths alike; the sweep visits plenty.
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        let mut checked = 0u32;
        let mut differ = Vec::new();
        for top in 0..=0xFFFFu32 {
            let instr = Instruction((top << 16) | 0x00FF);

            seed(&mut sys.dsp);
            let walked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self::dispatch::<GC>(&mut sys, instr);
                fingerprint(&sys.dsp)
            }));

            seed(&mut sys.dsp);
            let numbered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let leaf = self::resolve::<GC>(instr);
                self::execute::<GC>(&mut sys, leaf, instr);
                fingerprint(&sys.dsp)
            }));

            match (walked, numbered) {
                (Ok(a), Ok(b)) if a == b => checked += 1,
                (Err(_), Err(_)) => {}
                _ if differ.len() < 8 => differ.push(top),
                _ => {}
            }
        }
        std::panic::set_hook(hook);

        assert!(differ.is_empty(), "opcodes reaching a different handler: {differ:04x?}");
        assert!(
            checked > 40_000,
            "only {checked} opcodes ran at all; the sweep proves little"
        );
    }
}
