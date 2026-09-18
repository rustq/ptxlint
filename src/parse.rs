//! A deliberately forgiving PTX scanner.
//!
//! This is not a full PTX grammar, and that is the point: a CI tool must never
//! hard-fail on an instruction NVIDIA added last month. We tokenise statements,
//! read the directives we care about, and record every instruction as
//! `opcode + qualifiers`. Anything unrecognised is kept verbatim and simply
//! doesn't match any lint.

use std::collections::BTreeMap;

#[derive(Debug, Default)]
pub struct Module {
    pub version: Option<String>,
    pub target: Option<String>,
    pub address_size: Option<u32>,
    /// Module-scope `.shared` variables, in bytes.
    pub global_shared_bytes: u64,
    pub kernels: Vec<Kernel>,
}

#[derive(Debug, Default)]
pub struct Kernel {
    pub name: String,
    pub line: u32,
    pub params: Vec<String>,
    /// Declared virtual registers per type, e.g. `b32 -> 86`.
    pub regs: BTreeMap<String, u32>,
    /// `.local` bytes declared in the body (stack depot).
    pub local_bytes: u64,
    /// `.shared` bytes declared in the body.
    pub shared_bytes: u64,
    /// `.maxntid x, y, z`, if present.
    pub maxntid: Option<(u32, u32, u32)>,
    pub reqntid: Option<(u32, u32, u32)>,
    pub minnctapersm: Option<u32>,
    pub insts: Vec<Inst>,
}

#[derive(Debug, Clone)]
pub struct Inst {
    pub line: u32,
    /// Full opcode as written, e.g. `ld.global.nc.v4.f32`.
    pub op: String,
    /// First component, e.g. `ld`.
    pub base: String,
    /// Remaining dot-separated components, e.g. `["global", "nc", "v4", "f32"]`.
    pub quals: Vec<String>,
    /// True when guarded by `@%p` / `@!%p`.
    pub predicated: bool,
    pub operands: String,
}

impl Inst {
    pub fn has_qual(&self, q: &str) -> bool {
        self.quals.iter().any(|x| x == q)
    }

    /// The type suffix (`f32`, `s64`, `b128`, ...), if the opcode carries one.
    pub fn ty(&self) -> Option<&str> {
        self.quals
            .iter()
            .rev()
            .map(|s| s.as_str())
            .find(|q| is_type(q))
    }

    /// Memory state space for load/store style instructions.
    pub fn space(&self) -> Option<&str> {
        const SPACES: [&str; 7] = [
            "global", "shared", "local", "param", "const", "generic", "tex",
        ];
        self.quals
            .iter()
            .map(|s| s.as_str())
            .find(|q| SPACES.contains(q))
    }

    /// Vector width from a `.v2` / `.v4` / `.v8` qualifier.
    pub fn vector_width(&self) -> u32 {
        for q in &self.quals {
            if let Some(n) = q.strip_prefix('v') {
                if let Ok(n) = n.parse::<u32>() {
                    return n;
                }
            }
        }
        1
    }
}

fn is_type(q: &str) -> bool {
    matches!(
        q,
        "pred"
            | "f16"
            | "f16x2"
            | "bf16"
            | "bf16x2"
            | "tf32"
            | "f32"
            | "f64"
            | "s8"
            | "s16"
            | "s32"
            | "s64"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "b8"
            | "b16"
            | "b32"
            | "b64"
            | "b128"
            | "e4m3"
            | "e5m2"
    )
}

/// Width in bits of a PTX type suffix.
pub fn type_bits(t: &str) -> u32 {
    match t {
        "pred" => 1,
        "f16" | "bf16" | "s16" | "u16" | "b16" | "e4m3" | "e5m2" => 16,
        "s8" | "u8" | "b8" => 8,
        "f64" | "s64" | "u64" | "b64" => 64,
        "b128" => 128,
        _ => 32,
    }
}

/// Strip `//` and `/* */` comments, preserving byte count so line numbers survive.
fn strip_comments(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                out.push(' ');
                i += 1;
            }
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            while i < b.len() && !(b[i] == b'*' && i + 1 < b.len() && b[i + 1] == b'/') {
                out.push(if b[i] == b'\n' { '\n' } else { ' ' });
                i += 1;
            }
            i = (i + 2).min(b.len());
            out.push_str("  ");
        } else if b[i] == b'"' {
            // string literal: copy verbatim
            out.push('"');
            i += 1;
            while i < b.len() && b[i] != b'"' {
                out.push(b[i] as char);
                i += 1;
            }
            if i < b.len() {
                out.push('"');
                i += 1;
            }
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

struct Chunk {
    text: String,
    line: u32,
    /// `;`, `{`, `}` or `\0` at end of input.
    delim: u8,
}

fn chunks(src: &str) -> Vec<Chunk> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut line = 1u32;
    let mut start = 1u32;
    // Vector operands are also written with braces: `ld.global.v4.f32 {%f1, %f2,
    // %f3, %f4}, [%rd1];`. Those must not be mistaken for a function body.
    let mut operand_braces = 0usize;
    for c in src.chars() {
        match c {
            '{' if !(cur.trim().is_empty() || cur.contains(".entry") || cur.contains(".func")) => {
                operand_braces += 1;
                cur.push('{');
            }
            '}' if operand_braces > 0 => {
                operand_braces -= 1;
                cur.push('}');
            }
            ';' | '{' | '}' => {
                out.push(Chunk {
                    text: cur.trim().to_string(),
                    line: start,
                    delim: c as u8,
                });
                cur.clear();
                operand_braces = 0;
                start = line;
            }
            '\n' => {
                line += 1;
                cur.push(' ');
                if cur.trim().is_empty() {
                    start = line;
                }
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(Chunk {
            text: cur.trim().to_string(),
            line: start,
            delim: 0,
        });
    }
    out
}

/// Parse `[N]` array size and `.bN` element type into a byte count.
fn var_bytes(text: &str) -> u64 {
    let elem_bits = text
        .split_whitespace()
        .filter_map(|t| t.strip_prefix('.'))
        .find(|t| is_type(t))
        .map(type_bits)
        .unwrap_or(8) as u64;
    let count = match (text.find('['), text.find(']')) {
        (Some(a), Some(b)) if b > a + 1 => text[a + 1..b].trim().parse::<u64>().unwrap_or(1),
        _ => 1,
    };
    count * elem_bits.div_ceil(8)
}

/// `.reg .b32 %r<86>;` -> ("b32", 86); `.reg .b64 %SP;` -> ("b64", 1)
fn reg_decl(text: &str) -> Option<(String, u32)> {
    let ty = text
        .split_whitespace()
        .filter_map(|t| t.strip_prefix('.'))
        .find(|t| is_type(t))?
        .to_string();
    let n = match (text.find('<'), text.find('>')) {
        (Some(a), Some(b)) if b > a + 1 => text[a + 1..b].trim().parse::<u32>().unwrap_or(1),
        _ => 1,
    };
    Some((ty, n))
}

fn ntid(text: &str) -> Option<(u32, u32, u32)> {
    let nums: Vec<u32> = text
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse().ok())
        .collect();
    match nums.len() {
        0 => None,
        1 => Some((nums[0], 1, 1)),
        2 => Some((nums[0], nums[1], 1)),
        _ => Some((nums[0], nums[1], nums[2])),
    }
}

fn parse_inst(text: &str, line: u32) -> Option<Inst> {
    // Drop any leading labels: `$L__BB0_2: add.s32 ...`
    let mut t = text.trim();
    while let Some(colon) = t.find(':') {
        let (head, rest) = t.split_at(colon);
        if head.contains(char::is_whitespace) || head.is_empty() {
            break;
        }
        t = rest[1..].trim_start();
    }
    if t.is_empty() || t.starts_with('.') {
        return None;
    }
    let predicated = t.starts_with('@');
    if predicated {
        t = t
            .split_once(char::is_whitespace)
            .map(|(_, r)| r.trim_start())?;
    }
    let (op, operands) = match t.split_once(char::is_whitespace) {
        Some((o, r)) => (o, r.trim()),
        None => (t, ""),
    };
    if op.is_empty() || !op.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') {
        return None;
    }
    let mut parts = op.split('.');
    let base = parts.next().unwrap_or_default().to_string();
    Some(Inst {
        line,
        op: op.to_string(),
        base,
        quals: parts.map(str::to_string).collect(),
        predicated,
        operands: operands.to_string(),
    })
}

pub fn parse(src: &str) -> Module {
    let cleaned = strip_comments(src);
    let mut m = Module::default();
    // `.version` / `.target` / `.address_size` are terminated by a newline
    // rather than a semicolon, so they are read line-wise.
    for raw in cleaned.lines() {
        let t = raw.trim();
        // Only the first token matters: `.target sm_52, debug` -> `sm_52,`.
        let first = |v: &str| v.split_whitespace().next().unwrap_or("").to_string();
        if let Some(v) = t.strip_prefix(".version ") {
            m.version = Some(first(v));
        } else if let Some(v) = t.strip_prefix(".target ") {
            m.target = Some(first(v));
        } else if let Some(v) = t.strip_prefix(".address_size ") {
            m.address_size = first(v).parse().ok();
        }
    }
    let chunks = chunks(&cleaned);
    let mut depth = 0usize;
    let mut cur: Option<Kernel> = None;

    for ch in chunks {
        let text = ch.text.trim().to_string();
        match ch.delim {
            b'{' => {
                if depth == 0 && (text.contains(".entry") || text.contains(".func")) {
                    if text.contains(".entry") {
                        cur = Some(parse_header(&text, ch.line));
                    } else {
                        cur = None; // device function: not a launchable kernel
                    }
                }
                depth += 1;
            }
            b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    if let Some(k) = cur.take() {
                        m.kernels.push(k);
                    }
                }
            }
            _ => {
                if depth == 0 {
                    // .version/.target/.address_size were read line-wise above.
                    if text.contains(".shared") {
                        m.global_shared_bytes += var_bytes(&text);
                    } else if text.contains(".entry") {
                        // Entry declared without a body (prototype); ignore.
                    }
                    continue;
                }
                let Some(k) = cur.as_mut() else { continue };
                if text.starts_with(".reg") {
                    if let Some((ty, n)) = reg_decl(&text) {
                        *k.regs.entry(ty).or_insert(0) += n;
                    }
                } else if text.starts_with(".local") {
                    k.local_bytes += var_bytes(&text);
                } else if text.starts_with(".shared") {
                    k.shared_bytes += var_bytes(&text);
                } else if text.starts_with(".maxntid") {
                    k.maxntid = ntid(&text);
                } else if text.starts_with(".reqntid") {
                    k.reqntid = ntid(&text);
                } else if text.starts_with(".minnctapersm") {
                    k.minnctapersm = ntid(&text).map(|t| t.0);
                } else if !text.starts_with('.') {
                    if let Some(i) = parse_inst(&text, ch.line) {
                        k.insts.push(i);
                    }
                }
            }
        }
    }
    // A kernel whose closing brace was missing.
    if let Some(k) = cur {
        m.kernels.push(k);
    }
    m
}

fn parse_header(text: &str, line: u32) -> Kernel {
    let mut k = Kernel {
        line,
        ..Default::default()
    };
    let after = text.split(".entry").nth(1).unwrap_or("").trim_start();
    let name_end = after.find(['(', ' ', '\t']).unwrap_or(after.len());
    k.name = after[..name_end].trim().to_string();
    if let (Some(a), Some(b)) = (after.find('('), after.rfind(')')) {
        if b > a {
            k.params = after[a + 1..b]
                .split(',')
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
                .collect();
        }
    }
    // Header directives such as `.maxntid 256, 1, 1` live after the parameter list.
    let tail = text;
    if let Some(idx) = tail.find(".maxntid") {
        k.maxntid = ntid(&tail[idx + 8..]);
    }
    if let Some(idx) = tail.find(".reqntid") {
        k.reqntid = ntid(&tail[idx + 8..]);
    }
    if let Some(idx) = tail.find(".minnctapersm") {
        k.minnctapersm = ntid(&tail[idx + 13..]).map(|t| t.0);
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = r#"
.version 8.5
.target sm_90
.address_size 64

// a comment with ; and { }
.visible .entry demo(
	.param .u64 demo_param_0,
	.param .u32 demo_param_1
)
.maxntid 256, 1, 1
{
	.reg .pred 	%p<3>;
	.reg .b32 	%r<12>;
	.reg .f64 	%fd<4>;
	.local .align 8 .b8 	__local_depot0[128];
	.shared .align 4 .b8 	smem[2048];

	ld.param.u64 	%rd1, [demo_param_0];
	@%p1 bra 	$L__BB0_2;
	ld.global.nc.v4.f32 	{%f1, %f2, %f3, %f4}, [%rd2];
	add.f64 	%fd1, %fd2, %fd3;
$L__BB0_2:
	st.local.b32 	[%rd3], %r1;
	ret;
}
"#;

    #[test]
    fn parses_module_and_kernel() {
        let m = parse(SRC);
        assert_eq!(m.target.as_deref(), Some("sm_90"));
        assert_eq!(m.address_size, Some(64));
        assert_eq!(m.kernels.len(), 1);
        let k = &m.kernels[0];
        assert_eq!(k.name, "demo");
        assert_eq!(k.params.len(), 2);
        assert_eq!(k.regs["b32"], 12);
        assert_eq!(k.regs["f64"], 4);
        assert_eq!(k.local_bytes, 128);
        assert_eq!(k.shared_bytes, 2048);
        assert_eq!(k.maxntid, Some((256, 1, 1)));
    }

    #[test]
    fn parses_instructions() {
        let m = parse(SRC);
        let k = &m.kernels[0];
        let ops: Vec<&str> = k.insts.iter().map(|i| i.op.as_str()).collect();
        assert_eq!(
            ops,
            [
                "ld.param.u64",
                "bra",
                "ld.global.nc.v4.f32",
                "add.f64",
                "st.local.b32",
                "ret"
            ]
        );
        let v4 = k.insts.iter().find(|i| i.op.contains("v4")).unwrap();
        assert_eq!(v4.vector_width(), 4);
        assert_eq!(v4.space(), Some("global"));
        assert_eq!(v4.ty(), Some("f32"));
        assert!(k.insts.iter().find(|i| i.base == "bra").unwrap().predicated);
    }

    #[test]
    fn label_prefixed_statement_is_an_instruction() {
        let m = parse(SRC);
        let st = m.kernels[0].insts.iter().find(|i| i.base == "st").unwrap();
        assert_eq!(st.op, "st.local.b32");
    }

    #[test]
    fn survives_unknown_instructions() {
        let src = ".visible .entry k() { .reg .b32 %r<2>; \
                   some.future.instruction.sm_999 %r1, [%r2], {%r3}; ret; }";
        let m = parse(src);
        assert_eq!(m.kernels.len(), 1);
        assert_eq!(m.kernels[0].insts[0].base, "some");
    }

    #[test]
    fn device_functions_are_not_kernels() {
        let src = ".visible .func (.param .b32 r) helper(.param .b32 a) { ret; } \
                   .visible .entry k() { ret; }";
        let m = parse(src);
        assert_eq!(m.kernels.len(), 1);
        assert_eq!(m.kernels[0].name, "k");
    }
}
