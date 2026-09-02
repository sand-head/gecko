//! Idle loops: a few instructions that read something only an interrupt or a DMA can
//! change, and branch back until it does. Nothing the loop does can end it before the
//! next scheduler deadline, so both execution modes run it once and skip to there.

use crate::gekko::instruction::Instruction;

pub const MAX_IDLE_BODY: usize = 6;

pub fn is_branch_to_self(instr: Instruction, pc: u32) -> bool {
    if instr.primary_opcode() != 18 || instr.lk() {
        return false;
    }
    let target = if instr.aa() {
        instr.li() as u32
    } else {
        pc.wrapping_add_signed(instr.li())
    };
    target == pc
}

pub fn is_idle_loop_terminator(
    instr: crate::gekko::instruction::Instruction,
    branch_pc: u32,
    block_start_pc: u32,
) -> bool {
    if instr.primary_opcode() != 16 {
        return false;
    }

    if instr.lk() {
        return false;
    }

    if instr.bo() & 0b00100 == 0 {
        return false;
    }

    let target = if instr.aa() {
        instr.bd() as u32
    } else {
        branch_pc.wrapping_add_signed(instr.bd())
    };

    target == block_start_pc
}

pub fn validate_idle_loop(body: &[u32]) -> bool {
    let mut write_disallowed: u32 = 0;
    let mut written: u32 = 0;

    for &raw in body {
        let (reads, writes) = match gpr_dataflow(crate::gekko::instruction::Instruction(raw)) {
            Some(p) => p,
            None => return false,
        };

        let externals = reads & !written;
        write_disallowed |= externals;
        if writes & write_disallowed != 0 {
            return false;
        }

        written |= writes;
    }

    true
}

fn gpr_dataflow(instr: crate::gekko::instruction::Instruction) -> Option<(u32, u32)> {
    let rd_or_s = instr.rd() as u32;
    let ra = instr.ra() as u32;
    let rb = instr.rb() as u32;
    let bit = |r: u32| 1u32 << r;
    let read_a_or_zero = if ra == 0 { 0 } else { bit(ra) };

    Some(match instr.primary_opcode() {
        14 | 15 => (read_a_or_zero, bit(rd_or_s)),
        7 | 8 | 12 | 13 => (bit(ra), bit(rd_or_s)),
        10 | 11 => (bit(ra), 0),
        24 | 25 | 26 | 27 | 28 | 29 => (bit(rd_or_s), bit(ra)),
        20 => (bit(rd_or_s) | bit(ra), bit(ra)),
        21 => (bit(rd_or_s), bit(ra)),
        23 => (bit(rd_or_s) | bit(rb), bit(ra)),
        32 | 34 | 40 | 42 => (read_a_or_zero, bit(rd_or_s)),
        33 | 35 | 41 | 43 => (bit(ra), bit(rd_or_s) | bit(ra)),
        31 => return xform_dataflow(instr),
        _ => return None,
    })
}

fn xform_dataflow(instr: crate::gekko::instruction::Instruction) -> Option<(u32, u32)> {
    let rd_or_s = instr.rd() as u32;
    let ra = instr.ra() as u32;
    let rb = instr.rb() as u32;
    let bit = |r: u32| 1u32 << r;
    let read_a_or_zero = if ra == 0 { 0 } else { bit(ra) };

    Some(match instr.xo10() {
        266 | 40 | 10 | 138 | 202 | 234 | 8 | 136 | 200 | 232 | 104 | 235 | 75 | 11 | 491 | 459 | 778 | 552 | 522
        | 650 | 714 | 746 | 520 | 648 | 712 | 744 | 616 | 747 | 1003 | 971 => (bit(ra) | bit(rb), bit(rd_or_s)),
        28 | 60 | 124 | 284 | 316 | 412 | 444 | 476 => (bit(rd_or_s) | bit(rb), bit(ra)),
        26 | 922 | 954 => (bit(rd_or_s), bit(ra)),
        24 | 536 | 792 => (bit(rd_or_s) | bit(rb), bit(ra)),
        824 => (bit(rd_or_s), bit(ra)),
        0 | 32 => (bit(ra) | bit(rb), 0),
        23 | 87 | 279 | 343 | 534 | 790 => (read_a_or_zero | bit(rb), bit(rd_or_s)),
        55 | 119 | 311 | 375 => (bit(ra) | bit(rb), bit(rd_or_s) | bit(ra)),
        _ => return None,
    })
}
