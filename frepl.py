#!/usr/bin/env python3
"""
Flux REPL (frepl.py)
Interactive read-eval-print loop powered by the Flux VM.

Copyright (C) 2026 Karac V. Thweatt
"""

import sys
import os
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Provide a minimal llvmlite stub so that fast.py / ftypesys.py can be imported
# without a full LLVM installation.  The FVM codegen path never calls any LLVM
# IR methods, so a catch-all sentinel is sufficient.
try:
    import llvmlite  # noqa: F401
except ImportError:
    class _IrStubModule(types.ModuleType):
        """Stub for llvmlite.ir: returns a no-op sentinel for every attribute."""
        class _S:
            def __init__(self, *a, **kw): pass
            def __call__(self, *a, **kw): return type(self)()
            def __getattr__(self, n): return type(self)()
            def __class_getitem__(cls, i): return cls
        def __getattr__(self, name):
            return self._S

    _ir_stub       = _IrStubModule('llvmlite.ir')
    _llvmlite_stub = types.ModuleType('llvmlite')
    _llvmlite_stub.ir = _ir_stub
    sys.modules['llvmlite']                 = _llvmlite_stub
    sys.modules['llvmlite.ir']              = _ir_stub
    sys.modules['llvmlite.ir.instructions'] = types.ModuleType('llvmlite.ir.instructions')

from fvm import FluxVM, Val, TTag, Op, VMError
from fvmcodegen import FVMCodegen, FVMCodegenError
from flexer import FluxLexer
from fparser import FluxParser
from ferrors import FluxParseError, ParseError
from fast import ComptimeBlock


# ---------------------------------------------------------------------------
# Version / banner
# ---------------------------------------------------------------------------

_VERSION = '0.1.0'

_BANNER = f"""\
Flux REPL {_VERSION}  (FVM-powered)
Type Flux comptime code and press Enter.  Multi-line input is collected until
braces balance.  Type :help for commands, :quit to exit.
"""

_HELP = """\
Commands:
  :quit  :q        Exit the REPL
  :reset           Clear all session state (VM, variables, functions)
  :stack           Dump the current VM stack
  :locals          Dump last_locals (variables from the most recent block)
  :funcs           List comptime functions defined so far
  :help            Show this message

Any other input is treated as the body of a comptime block and executed.
Variables and functions defined in one input survive into the next.
"""


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

class ReplSession:
    """Holds all mutable state for one REPL session."""

    def __init__(self):
        self.vm              = FluxVM()
        self.known_functions = {}   # name -> List[Instr], accumulated across inputs
        self.known_layouts   = {}   # struct name -> StructLayout
        self.captured_scope  = {}   # name -> Val, live variable bindings
        self._locals_map     = {}   # name -> slot, from most recent codegen
        self._emit_cursor    = 0    # index into vm.emit_results already shown

    def reset(self):
        self.__init__()


# ---------------------------------------------------------------------------
# Input collection
# ---------------------------------------------------------------------------

def _brace_depth(text: str) -> int:
    """Return net open-brace count, ignoring braces inside string literals."""
    depth   = 0
    in_str  = False
    str_ch  = ''
    i       = 0
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == '\\':
                i += 2
                continue
            if ch == str_ch:
                in_str = False
        else:
            if ch in ('"', "'"):
                in_str = True
                str_ch = ch
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
        i += 1
    return depth


def _is_complete(lines: list, depth: int) -> bool:
    """
    Return True when the accumulated input is ready to execute.

    Rules:
      - If braces are still open (depth > 0), always keep collecting.
      - If braces are balanced (depth == 0) AND the last non-empty line ends
        with ';', the statement is self-contained -- execute immediately.
      - Otherwise (depth == 0 but no trailing ';', e.g. 'namespace Test')
        keep collecting until the user submits a blank line.
    """
    if depth > 0:
        return False
    last = next((l for l in reversed(lines) if l.strip()), '')
    return last.rstrip().endswith(';')


def collect_input(first_line: str) -> str:
    """
    Given the first line already read, collect further lines until the input
    is complete.  Completion is determined by _is_complete():
      - Balanced braces + trailing semicolon  -> execute immediately.
      - Balanced braces but no trailing ';'   -> wait for a blank line.
      - Unbalanced braces                     -> keep collecting regardless.
    Returns the complete input text (blank continuation line not included).
    """
    lines = [first_line]
    depth = _brace_depth(first_line)
    while not _is_complete(lines, depth):
        try:
            line = input('... ')
        except EOFError:
            break
        # A blank line while braces are balanced signals end-of-input
        if line.strip() == '' and depth == 0:
            break
        lines.append(line)
        depth += _brace_depth(line)
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------

def _wrap_comptime(source: str) -> str:
    """Wrap bare user input in a comptime { ... }; block."""
    return f'comptime\n{{\n{source}\n}};'


def _parse_comptime_blocks(source: str):
    """
    Lex and parse source, return all ComptimeBlock nodes found.
    Raises FluxParseError / ParseError on failure.
    """
    lexer   = FluxLexer(source)
    tokens  = lexer.tokenize()
    parser  = FluxParser(tokens)
    program = parser.parse()
    blocks  = [s for s in program.statements if isinstance(s, ComptimeBlock)]
    return blocks


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _run_block(session: ReplSession, block: ComptimeBlock):
    """
    Compile and execute one ComptimeBlock against the session.
    Returns the top-of-stack Val after execution, or None.
    Raises FVMCodegenError or VMError on failure.
    """
    cg = FVMCodegen(
        known_functions=dict(session.known_functions),
        known_struct_layouts=dict(session.known_layouts),
    )
    bc = cg.compile(block, captured_scope=dict(session.captured_scope))

    # Register every function the codegen produced before executing
    for name, instrs in cg.compiled_functions.items():
        session.vm.register_function(name, instrs)
    session.known_functions.update(cg.compiled_functions)

    # Accumulate any new struct layouts
    session.known_layouts.update(cg._struct_layouts)
    session.vm.struct_layouts.update(cg._struct_layouts)

    result = session.vm.execute(bc.instructions, bc.local_count)

    # Capture live variables into the persistent scope using the codegen's locals map
    for name, slot in cg._locals.items():
        if slot < len(session.vm.last_locals) and session.vm.last_locals[slot] is not None:
            session.captured_scope[name] = session.vm.last_locals[slot]

    session._locals_map = dict(cg._locals)
    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _val_repr(v: Val) -> str:
    if v is None:
        return 'void'
    if v.tag == TTag.BOOL:
        return 'true' if v.data else 'false'
    if v.tag == TTag.BYTES and isinstance(v.data, (bytes, bytearray)):
        try:
            return repr(v.data.decode('utf-8'))
        except Exception:
            return repr(v.data)
    if v.tag == TTag.PTR:
        struct_type = v.meta.get('struct_type')
        if struct_type:
            return f'<{struct_type} @ 0x{v.data:x}>'
        elem_type = v.meta.get('elem_type')
        count     = v.meta.get('count')
        if elem_type and count is not None:
            return f'<array[{elem_type}; {count}] @ 0x{v.data:x}>'
        return f'<ptr 0x{v.data:x}>'
    if v.tag in (TTag.FLOAT, TTag.DOUBLE):
        return repr(float(v.data))
    return str(v.data)


def _print_result(v: Val):
    if v is None or v.tag == TTag.VOID:
        return
    print(f'=> {_val_repr(v)} : {v.tag.value}')


def _print_new_emissions(session: ReplSession):
    results = session.vm.emit_results
    while session._emit_cursor < len(results):
        entry = results[session._emit_cursor]
        kind  = entry[0]
        if kind == 'const':
            _, val = entry
            print(f'[emit:const]  {_val_repr(val)} : {val.tag.value}')
        elif kind == 'global':
            _, data, meta = entry
            print(f'[emit:global] {data.hex()}  meta={meta}')
        elif kind == 'type':
            _, val = entry
            label = val.data.decode() if isinstance(val.data, (bytes, bytearray)) else str(val.data)
            print(f'[emit:type]   {label}')
        elif kind == 'flux':
            _, text = entry
            print(f'[emit:flux]   {text}')
        else:
            print(f'[emit:{kind}]  {entry[1:]}')
        session._emit_cursor += 1


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _cmd_stack(session: ReplSession):
    if not session.vm.stack:
        print('(stack is empty)')
        return
    for i, v in enumerate(session.vm.stack):
        print(f'  [{i}]  {_val_repr(v)} : {v.tag.value}')


def _cmd_locals(session: ReplSession):
    if not session.captured_scope:
        print('(no variables in scope)')
        return
    for name, val in sorted(session.captured_scope.items()):
        print(f'  {name} = {_val_repr(val)} : {val.tag.value}')


def _cmd_funcs(session: ReplSession):
    if not session.known_functions:
        print('(no comptime functions defined)')
        return
    for name in sorted(session.known_functions):
        n = len(session.known_functions[name])
        print(f'  {name}  ({n} instructions)')


# ---------------------------------------------------------------------------
# Main REPL loop
# ---------------------------------------------------------------------------

def repl():
    print(_BANNER)
    session = ReplSession()

    while True:
        # --- prompt ---
        try:
            first = input('fx> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not first:
            continue

        # --- built-in commands ---
        if first in (':quit', ':q'):
            break
        if first == ':reset':
            session.reset()
            print('Session reset.')
            continue
        if first == ':help':
            print(_HELP)
            continue
        if first == ':stack':
            _cmd_stack(session)
            continue
        if first == ':locals':
            _cmd_locals(session)
            continue
        if first == ':funcs':
            _cmd_funcs(session)
            continue

        # --- collect multi-line input ---
        try:
            source = collect_input(first)
        except KeyboardInterrupt:
            print()
            continue

        # --- wrap and parse ---
        wrapped = _wrap_comptime(source)
        try:
            blocks = _parse_comptime_blocks(wrapped)
        except (FluxParseError, ParseError) as e:
            print(f'Parse error: {e}')
            continue
        except Exception as e:
            print(f'Parse error: {e}')
            continue

        if not blocks:
            print('(nothing to execute)')
            continue

        # --- execute each block ---
        for block in blocks:
            try:
                result = _run_block(session, block)
            except FVMCodegenError as e:
                print(f'Codegen error: {e}')
                break
            except VMError as e:
                print(f'VM error: {e}')
                break
            except Exception as e:
                print(f'Error: {e}')
                break
            else:
                _print_new_emissions(session)
                _print_result(result)

    print('Goodbye.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    repl()