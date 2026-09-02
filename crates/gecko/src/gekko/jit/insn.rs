use crate::gekko::instruction::Instruction;

#[allow(dead_code)]
impl Instruction {
    #[inline(always)]
    pub fn xo5(&self) -> u32 {
        (self.0 >> 1) & 0x1F
    }

    #[inline(always)]
    pub fn xo6(&self) -> u32 {
        (self.0 >> 1) & 0x3F
    }
}
