#!/usr/bin/env python3
"""Regression test for _mmss() in app/server.py (Nov 4 2026).

A cut timestamp is written as MM:SS.mmm in the cinema prompt. The original
implementation floored the minutes from the raw value and only THEN rounded
the seconds, so a value whose seconds part rounds up past 59.999 produced an
impossible "60" in the seconds field: _mmss(59.9996) == "00:60.000" instead
of "01:00.000" (and _mmss(119.9999) == "01:60.000" instead of "02:00.000").

The fix rounds to whole milliseconds first, then splits, so the carry into the
minutes field always happens. This test extracts _mmss straight from
server.py (AST, no import -- the server module needs comfy-sdk and does
module-level setup) so it pins the real code, not a copy.

Usage: python tools/tests/test_mmss.py [path/to/server.py]
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "app", "server.py")
SERVER = os.path.normpath(SERVER)


def load_mmss(path):
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_mmss":
            mod = ast.Module(body=[node], type_ignores=[])
            ns = {}
            exec(compile(mod, path, "exec"), ns)
            return ns["_mmss"]
    raise AssertionError(f"no top-level _mmss defined in {path}")


def main():
    mmss = load_mmss(SERVER)

    # (value in seconds, expected MM:SS.mmm)
    cases = [
        (59.9996, "01:00.000"),   # the reported defect: carry into the minutes
        (119.9999, "02:00.000"),  # same defect one minute later
        (59.9994, "00:59.999"),   # just under the boundary: must NOT carry
        (60.0, "01:00.000"),
        (120.0, "02:00.000"),
        (0.0, "00:00.000"),
        (1.2, "00:01.200"),
        (1.2345, "00:01.234"),
        (59.999, "00:59.999"),
        (89.9996, "01:30.000"),   # carry within the seconds field
        (179.9996, "03:00.000"),  # carry two minutes up
        (3599.9996, "60:00.000"),  # the old bug shaped this one as "59:60.000"
        (3600.0, "60:00.000"),
    ]
    failures = []
    for t, want in cases:
        got = mmss(t)
        status = "ok  " if got == want else "FAIL"
        print(f"  {status} _mmss({t}) = {got!r}" + ("" if got == want else f"  (expected {want!r})"))
        if got != want:
            failures.append((t, want, got))

    # Structural invariants for a dense sweep: the seconds field must never
    # reach 60, and the result must be a valid MM:SS.mmm.
    for i in range(0, 2000):
        t = i / 1000 * 59.99996 + 0.0004  # sweep across many minute boundaries
        s = mmss(t)
        mins, secs = s.split(":")
        sec_val = float(secs)
        if sec_val >= 60.0:
            failures.append((t, "seconds field < 60", s))
            break
        if int(mins) != int(round(t * 1000) // 60000):
            failures.append((t, f"minutes {int(round(t * 1000) // 60000)}", s))
            break

    if failures:
        print(f"\nFAIL {len(failures)} case(s) for _mmss in {SERVER}")
        return 1
    print(f"\nPASS all {len(cases) + 1} checks for _mmss in {SERVER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
