use std::path::PathBuf;
use std::{env, fs};

fn main() {
    chipi_build::generate_bindings("spec/gekko.bindings.chipi").expect("chipi codegen failed (gekko)");
    chipi_build::generate_bindings("spec/dsp.bindings.chipi").expect("chipi codegen failed (dsp)");
    chipi_build::generate_bindings("spec/wii_gekko.bindings.chipi").expect("chipi codegen failed (wii gekko)");
    chipi_build::generate_bindings("spec/wii_dsp.bindings.chipi").expect("chipi codegen failed (wii dsp)");
    chipi_build::generate_bindings("spec/gekko_jit.bindings.chipi").expect("chipi codegen failed (gekko jit)");
    chipi_build::generate_bindings("spec/wii_gekko_jit.bindings.chipi").expect("chipi codegen failed (wii gekko jit)");
    chipi_build::generate_bindings("spec/dsp_jit.bindings.chipi").expect("chipi codegen failed (dsp jit)");

    // The interpreters' dispatch trees, walked once per instruction by the caches that
    // remember what they found instead of once per execution — the Gekko's block cache
    // and the DSP's decode cache. chipi does not know about either, so they are
    // generated from the tables chipi wrote.
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    const GEKKO: &str = "crate::gekko::instruction::Instruction";
    const DSP: &str = "crate::flipper::dsp::instruction::Instruction";
    for (lut, resolve, instruction) in [
        ("gekko_lut.rs", "gekko_resolve.rs", GEKKO),
        ("gekko_lut_wii.rs", "gekko_resolve_wii.rs", GEKKO),
        ("dsp_lut.rs", "dsp_resolve.rs", DSP),
        ("dsp_lut_wii.rs", "dsp_resolve_wii.rs", DSP),
    ] {
        let source = fs::read_to_string(out.join(lut)).expect(lut);
        fs::write(out.join(resolve), resolvers_for(&source, instruction)).expect(resolve);
    }
}

/// `dispatch` walks a tree of tables to a handler and calls it; `resolve` walks the
/// same tree and returns the handler's number, and `execute` runs a numbered handler
/// through one `match`, so a block decoded once dispatches with a jump rather than a
/// call through a table. Every `_dN` decoder gets an `_rN` resolver over a parallel
/// `_RN` table whose leaves are the numbers.
fn resolvers_for(lut: &str, instruction: &str) -> String {
    let mut out = format!("pub type Resolver = fn({instruction}) -> u16;\n\n");
    let mut leaves: Vec<String> = Vec::new();
    let mut lines = lut.lines();
    while let Some(line) = lines.next() {
        if let Some(rest) = line.strip_prefix("static _T") {
            let (table, rest) = rest.split_once(':').unwrap();
            let count = rest.trim().trim_start_matches("[Handler; ").split(']').next().unwrap();
            out.push_str(&format!("static _R{table}: [Resolver; {count}] = [\n"));
            for entry in lines.by_ref() {
                if entry.starts_with("];") {
                    break;
                }
                let entry = entry.trim().split(", //").next().unwrap().trim_end_matches(',');
                if let Some(decoder) = entry.strip_prefix("_d") {
                    out.push_str(&format!("    _r{decoder},\n"));
                } else {
                    let leaf = match leaves.iter().position(|l| l == entry) {
                        Some(i) => i,
                        None => {
                            leaves.push(entry.to_string());
                            leaves.len() - 1
                        }
                    };
                    out.push_str(&format!("    _leaf{leaf},\n"));
                }
            }
            out.push_str("];\n\n");
        } else if let Some(rest) = line.strip_prefix("fn _d") {
            let decoder: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            let body = lines.next().unwrap();
            let (table, rest) = body.trim().split_once('[').unwrap();
            let index = rest.rsplit_once("](ctx, instr)").unwrap().0;
            let table = table.trim_start_matches("_T");
            out.push_str(&format!(
                "fn _r{decoder}(instr: {instruction}) -> u16 {{\n    _R{table}[{index}](instr)\n}}\n\n"
            ));
            lines.next();
        }
    }
    for i in 0..leaves.len() {
        out.push_str(&format!("fn _leaf{i}(_: {instruction}) -> u16 {{\n    {i}\n}}\n\n"));
    }
    out.push_str(&format!(
        "/// The number of the handler `dispatch` would call for this instruction word.\n#[inline(always)]\npub fn resolve(instr: {instruction}) -> u16 {{\n    _r0(instr)\n}}\n\n"
    ));
    let ctx = lut
        .lines()
        .find_map(|l| l.strip_prefix("pub type Handler = fn(&mut "))
        .and_then(|l| l.split(',').next())
        .expect("Handler type");
    out.push_str(&format!(
        "/// Runs handler number `leaf`, as `resolve` numbered it.\n#[inline(always)]\npub fn execute(leaf: u16, ctx: &mut {ctx}, instr: {instruction}) {{\n    match leaf {{\n"
    ));
    for (i, leaf) in leaves.iter().enumerate() {
        out.push_str(&format!("        {i} => {leaf}(ctx, instr),\n"));
    }
    out.push_str("        _ => _unimpl(ctx, instr),\n    }\n}\n\n");
    out.push_str(
        "/// Handler number `leaf` itself, for a caller that will call it more than once.\n#[inline(always)]\npub fn handler(leaf: u16) -> Handler {\n    match leaf {\n",
    );
    for (i, leaf) in leaves.iter().enumerate() {
        out.push_str(&format!("        {i} => {leaf},\n"));
    }
    out.push_str("        _ => _unimpl,\n    }\n}\n\n");
    out.push_str(
        "/// The `OP_*` constant handler number `leaf` is specialised on, which names the instruction; `u32::MAX` for a handler that has none.\n#[inline(always)]\npub fn op_of(leaf: u16) -> u32 {\n    match leaf {\n",
    );
    for (i, leaf) in leaves.iter().enumerate() {
        if let Some(op) = leaf
            .split("::<{ ")
            .nth(1)
            .and_then(|rest| rest.split(' ').next())
            .filter(|op| op.starts_with("OP_"))
        {
            out.push_str(&format!("        {i} => {op},\n"));
        }
    }
    out.push_str("        _ => u32::MAX,\n    }\n}\n");
    out
}
