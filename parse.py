#!/usr/bin/env python3
"""Entry point for the Box64 printer and the upstream opcode tools."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if len(sys.argv) > 1 and sys.argv[1] == "-box64":
    from riscv_opcodes.box64_utils import print_decoder

    print_decoder(sys.argv[2:])
else:
    from riscv_opcodes.parse import main

    main()
