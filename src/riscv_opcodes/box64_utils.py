"""Generate the instruction decoder fragments used by Box64's RV64 DynaRec."""

from .constants import arg_lut
from .resources import resource_root
from .shared_utils import create_inst_dict

PSEUDO_OPS = ("prefetch_i", "prefetch_r", "prefetch_w", "rdcycle", "rdtime", "rdinstret")
DIRECT_BRANCHES = {"jal", "beq", "bne", "blt", "bge", "bltu", "bgeu"}
FP_EXTENSIONS = {"rv_f", "rv_d", "rv64_f", "rv64_d"}
SKIP_FIELDS = {"imm12hi", "bimm12hi", "zimm6hi"}
SPLIT_FIELDS = {
    "imm12lo": ("(FX(opcode, 31, 25) << 5) | FX(opcode, 11, 7)", 12),
    "bimm12lo": ("(BX(opcode, 31) << 12) | (BX(opcode, 7) << 11) | (FX(opcode, 30, 25) << 5) | (FX(opcode, 11, 8) << 1)", 13),
    "jimm20": ("(BX(opcode, 31) << 20) | (FX(opcode, 19, 12) << 12) | (BX(opcode, 20) << 11) | (FX(opcode, 30, 21) << 1)", 21),
    "zimm6lo": ("(BX(opcode, 26) << 5) | FX(opcode, 19, 15)", 0),
}
VECTOR_FIELDS = {
    "vd": 0, "rd": 0, "vs3": 0, "vs2": 1, "vs1": 2, "rs1": 2,
    "simm5": 2, "zimm5": 2, "zimm6lo": 2, "vm": 3, "nf": 4,
}


def register_bank(name, field, extension):
    if field.startswith("v"):
        return "vpr"
    if name in {"th_flrd", "th_fmv_hw_x"}:
        return "fpr" if field == "rd" else "gpr"
    if name == "th_fmv_x_hw":
        return "fpr" if field == "rs1" else "gpr"
    if extension == "rv_v":
        return "fpr" if name == "vfmv_f_s" or (field == "rs1" and name.endswith(("_vf", "_vfm", "_s_f", "_v_f"))) else "gpr"
    if extension not in FP_EXTENSIONS:
        return "gpr"
    if name.startswith(("fclass", "flt", "feq", "fle")):
        return "gpr" if field == "rd" else "fpr"
    if name in {"fld", "flw", "fsd", "fsw"}:
        return "gpr" if field == "rs1" else "fpr"
    if name.startswith(("fcvt_", "fmv_")):
        return "fpr" if name.split("_")[1 if field == "rd" else 2] in {"s", "d"} else "gpr"
    return "fpr"


def operand(name, field, extension):
    msb, lsb = arg_lut[field]
    expr = f"FX(opcode, {msb}, {lsb})"
    if field in {"rd", "rs1", "rs2", "rs3", "vd", "vs1", "vs2", "vs3"}:
        return field, expr, "%s", [f"{register_bank(name, field, extension)}[a.{field}]"]
    if field in {"aq", "rl", "rm", "vm", "nf"}:
        return field, expr, "%s", [f"{field}[a.{field}]"]
    if field == "csr":
        return field, expr, "0x%x(%d)", ["(unsigned)a.csr", "a.csr"]
    if field in SPLIT_FIELDS:
        expr, width = SPLIT_FIELDS[field]
    elif name.startswith("prefetch_") and field == "imm12hi":
        expr, width = f"{expr} << 5", 12
    elif field in {"imm12", "simm5", "zthimm5"}:
        width = msb - lsb + 1
    elif field.startswith(("shamt", "imm", "zimm", "zthimm")):
        width = 0
    else:
        raise ValueError(f"No Box64 operand formatting for {field} in {name}")
    if width:
        expr = f"SIGN_EXTEND(({expr}), {width})"
    dest = "imm2" if field in {"zthimm6_", "zthimm2"} else "imm"
    return dest, expr, "0x%x(%d)", [f"(unsigned)a.{dest}", f"a.{dest}"]


def print_vset(name):
    immediate = name == "vsetivli"
    print(f"    a.imm = FX(opcode, {29 if immediate else 30}, 20);")
    print("    a.rd = FX(opcode, 11, 7);")
    if immediate:
        print("    a.imm2 = FX(opcode, 19, 15);")
    else:
        print("    a.rs1 = FX(opcode, 19, 15);")
    print('    static const char* const lmul[] = {"m1", "m2", "m4", "m8", "reserved", "mf8", "mf4", "mf2"};')
    print('    static const char* const sew[] = {"e8", "e16", "e32", "e64", "reserved", "reserved", "reserved", "reserved"};')
    avl = "a.imm2" if immediate else "gpr[a.rs1]"
    fmt = "%d" if immediate else "%s"
    print(f'    snprintf(buff, sizeof(buff), "%-15s %s, {fmt}, %s, %s, %s, %s", "{name.upper()}", gpr[a.rd], {avl},')
    print('             sew[(a.imm >> 3) & 7], lmul[a.imm & 7], a.imm & 64 ? "ta" : "tu", a.imm & 128 ? "ma" : "mu");')


def print_instruction(name, instr):
    extension = min(instr["extension"])
    mnemonic = name.upper().replace("_", ".")
    print(f"// {extension:>7}, {mnemonic}")
    print(f"if ((opcode & {instr['mask']}) == {instr['match']}) {{")
    if name in {"vsetvli", "vsetivli"}:
        print_vset(name)
    else:
        fields = instr["variable_fields"]
        if name in {"fence", "fence_i"}:
            fields = []
        elif name == "vsetvl":
            fields = ["rd", "rs1", "rs2"]
        elif extension == "rv_v" and "rs1" in fields and (int(instr["match"], 16) & 0x7f) in {0x07, 0x27}:
            order = ["vd", "vs3", "rs1", "vs2", "rs2", "vm", "nf"]
            fields = sorted(fields, key=order.index)
        elif extension == "rv_v" or extension.startswith("rv_zv"):
            fields = sorted(fields, key=lambda f: VECTOR_FIELDS[f] if f not in SKIP_FIELDS else -1)
        elif "imm12lo" in fields:
            fields = ["rs2", "rs1", "imm12lo"]
        elif "csr" in fields:
            fields = ["rd", "csr", "rs1" if "rs1" in fields else "zimm5"]
        formats, args = [], []
        for field in fields:
            if field in SKIP_FIELDS and not name.startswith("prefetch_"):
                continue
            dest, expr, fmt, values = operand(name, field, extension)
            print(f"    a.{dest} = {expr};")
            formats.append(fmt)
            args.extend(values)
        fmt = "%-15s" + (" " + ", ".join(formats) if formats else "")
        operands = ", " + ", ".join(args) if args else ""
        direct = name in DIRECT_BRANCHES
        result = "int len = " if direct else ""
        print(f'    {result}snprintf(buff, sizeof(buff), "{fmt}", "{mnemonic}"{operands});')
        if direct:
            print('    snprintf(buff + len, sizeof(buff) - len, " # %+di(0x%lx)", a.imm >> 2, addr + (uintptr_t)a.imm);')
    print("    return buff;\n}\n")


def print_decoder(extensions):
    if not extensions:
        raise SystemExit("Usage: parse.py -box64 EXTENSION [EXTENSION ...]")
    for extension in extensions:
        if not (resource_root() / "extensions" / extension).is_file():
            raise SystemExit(f"Unknown RISC-V extension: {extension}")
    instructions = create_inst_dict(extensions, include_pseudo_ops=list(PSEUDO_OPS))
    for name in sorted(instructions, key=lambda name: (name not in PSEUDO_OPS, name)):
        print_instruction(name, instructions[name])
