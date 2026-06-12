#!/usr/bin/env python3
"""
Flux VM Codegen (fvmcodegen.py)
Compiles Flux AST nodes inside comptime blocks to fvm.py bytecode.

This mirrors fcodegen.py's visitor pattern but targets the FluxVM
instead of LLVM IR. It does not use llvmlite, ir.IRBuilder, or
ir.Module in any way.

Copyright (C) 2026 Karac V. Thweatt
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from fvm import Op, TTag, Val, Instr
from fast import (
    ASTNode,
    ComptimeBlock, EmitFlux,
    VariableDeclaration,
    Assignment, CompoundAssignment,
    BinaryOp, UnaryOp,
    Literal, Identifier, StringLiteral, FStringLiteral,
    FunctionCall, MethodCall, MemberAccess,
    ArrayLiteral,
    AddressOf, CastExpression, TypeConvertExpression,
    IfStatement,
    WhileLoop, DoLoop, DoWhileLoop,
    ForLoop,
    ReturnStatement,
    BreakStatement, ContinueStatement,
    ExpressionStatement,
    Block,
    ArrayAccess,
    SwitchStatement, Case,
    FunctionDef, FunctionDefStatement,
)
from ftypesys import DataType, Operator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _datatype_to_ttag(dt: DataType) -> TTag:
    """Map a Flux DataType to the closest VM TTag."""
    _map = {
        DataType.SINT:    TTag.INT,
        DataType.UINT:   TTag.UINT,
        DataType.SLONG:   TTag.LONG,
        DataType.ULONG:  TTag.ULONG,
        DataType.FLOAT:  TTag.FLOAT,
        DataType.DOUBLE: TTag.DOUBLE,
        DataType.BOOL:   TTag.BOOL,
        DataType.BYTE:   TTag.BYTE,
        DataType.CHAR:   TTag.CHAR,
    }
    return _map.get(dt, TTag.INT)


def _instr(op: Op, *operands) -> Instr:
    return Instr(op, list(operands))


# ---------------------------------------------------------------------------
# Codegen result
# ---------------------------------------------------------------------------

class ComptimeBytecode:
    """Holds the flat instruction list and local count produced for a comptime block."""
    def __init__(self, instructions: List[Instr], local_count: int):
        self.instructions = instructions
        self.local_count   = local_count


# ---------------------------------------------------------------------------
# FVMCodegen visitor
# ---------------------------------------------------------------------------

class FVMCodegen:
    """
    Visits a ComptimeBlock AST node and emits FluxVM bytecode.

    Usage:
        cg = FVMCodegen()
        bc = cg.compile(node, captured_scope)
        vm.execute(bc.instructions, bc.local_count)
    """

    def __init__(self):
        self._instructions: List[Instr] = []
        # Maps variable name -> local slot index
        self._locals: Dict[str, int] = {}
        self._local_count: int = 0
        # Loop-level break/continue patch lists (list of instruction indices)
        self._loop_patches: List[Tuple[int, str]] = []
        # Compiled comptime functions: name -> List[Instr]
        self.compiled_functions: Dict[str, List[Instr]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(
        self,
        node: ComptimeBlock,
        captured_scope: Dict[str, Val] = None,
    ) -> ComptimeBytecode:
        """
        Compile a ComptimeBlock to VM bytecode.

        captured_scope: variables from the enclosing Flux scope, captured by
        value (read-only).  Each entry is pre-allocated as a local slot and
        initialised with a PUSH / LOCAL_SET pair.
        """
        captured_scope = captured_scope or {}

        # Pre-allocate captured scope variables as read-only locals
        for name, val in captured_scope.items():
            slot = self._alloc_local(name)
            self._emit(_instr(Op.PUSH, val))
            self._emit(_instr(Op.LOCAL_SET, slot))

        # Compile the body
        self._visit_body(node.body)

        # Terminate with HALT
        self._emit(_instr(Op.HALT))

        return ComptimeBytecode(self._instructions, self._local_count)

    # ------------------------------------------------------------------
    # Emit helpers
    # ------------------------------------------------------------------

    def _emit(self, instr: Instr) -> int:
        """Append an instruction, return its index."""
        idx = len(self._instructions)
        self._instructions.append(instr)
        return idx

    def _current_ip(self) -> int:
        return len(self._instructions)

    def _alloc_local(self, name: str) -> int:
        if name not in self._locals:
            self._locals[name] = self._local_count
            self._local_count += 1
        return self._locals[name]

    def _patch_at(self, idx: int, new_op: Op, addr: int):
        """Replace the instruction at idx with new_op targeting addr."""
        self._instructions[idx] = _instr(new_op, addr)

    # ------------------------------------------------------------------
    # Body / statement dispatch
    # ------------------------------------------------------------------

    def _visit_body(self, stmts: list):
        for stmt in stmts:
            self._visit_stmt(stmt)

    def _visit_stmt(self, node: ASTNode):
        if node is None:
            return
        t = type(node)
        if t is VariableDeclaration:     self._visit_var_decl(node)
        elif t is Assignment:            self._visit_assignment(node)
        elif t is CompoundAssignment:    self._visit_compound_assign(node)
        elif t is ExpressionStatement:   self._visit_expr_stmt(node)
        elif t is IfStatement:           self._visit_if(node)
        elif t is ForLoop:               self._visit_for(node)
        elif t is WhileLoop:             self._visit_while(node)
        elif t is DoLoop:                self._visit_do(node)
        elif t is DoWhileLoop:           self._visit_do_while(node)
        elif t is ReturnStatement:       self._visit_return(node)
        elif t is BreakStatement:        self._visit_break(node)
        elif t is ContinueStatement:     self._visit_continue(node)
        elif t is Block:                 self._visit_body(node.statements)
        elif t is SwitchStatement:       self._visit_switch(node)
        elif t is FunctionDef:           self._visit_function_def(node)
        elif t is FunctionDefStatement:  self._visit_function_def(node.function_def)
        elif t is EmitFlux:              self._visit_emitflux(node)
        elif t is ComptimeBlock:         self._visit_body(node.body)
        elif isinstance(node, list):     self._visit_body(node)
        else:
            raise NotImplementedError(
                f'fvmcodegen: unsupported node in comptime: {type(node).__name__}'
            )

    # ------------------------------------------------------------------
    # Variable declaration
    # ------------------------------------------------------------------

    def _visit_var_decl(self, node: VariableDeclaration):
        slot = self._alloc_local(node.name)
        if node.initial_value is not None:
            self._visit_expr(node.initial_value)
        else:
            # Zero-initialise (all Flux variables are zero-init by default)
            ttag = _datatype_to_ttag(
                node.type_spec.base_type if node.type_spec else DataType.SINT
            )
            self._emit(_instr(Op.PUSH, Val(ttag, 0)))
        self._emit(_instr(Op.LOCAL_SET, slot))

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def _visit_assignment(self, node: Assignment):
        if isinstance(node.target, Identifier):
            self._visit_expr(node.value)
            slot = self._alloc_local(node.target.name)
            self._emit(_instr(Op.LOCAL_SET, slot))
        else:
            raise NotImplementedError(
                'fvmcodegen: only simple identifier assignment supported in comptime'
            )

    def _visit_compound_assign(self, node: CompoundAssignment):
        """Desugar compound assignment: x op= val  ->  x = x op val"""
        if not isinstance(node.target, Identifier):
            raise NotImplementedError(
                'fvmcodegen: only simple identifier compound-assignment in comptime'
            )
        name = node.target.name
        slot = self._alloc_local(name)
        # Push LHS
        self._emit(_instr(Op.LOCAL_GET, slot))
        # Push RHS
        self._visit_expr(node.value)
        # Emit the base operator
        self._emit_binary_op_for_compound(node.op_token)
        # Store result
        self._emit(_instr(Op.LOCAL_SET, slot))

    def _emit_binary_op_for_compound(self, op_token):
        """Map a compound-assignment operator token to its base binary opcode."""
        from flexer import TokenType
        _map = {
            TokenType.PLUS_ASSIGN:    Op.ADD,
            TokenType.MINUS_ASSIGN:   Op.SUB,
            TokenType.STAR_ASSIGN:    Op.MUL,
            TokenType.SLASH_ASSIGN:   Op.DIV,
            TokenType.PERCENT_ASSIGN: Op.MOD,
        }
        oc = _map.get(op_token)
        if oc:
            self._emit(_instr(oc))
        else:
            raise NotImplementedError(
                f'fvmcodegen: compound assign op {op_token!r} not supported'
            )

    # ------------------------------------------------------------------
    # Expression statement
    # ------------------------------------------------------------------

    def _visit_expr_stmt(self, node: ExpressionStatement):
        pushes = self._visit_expr(node.expression)
        # Only pop if the expression left a value on the stack
        if pushes:
            self._emit(_instr(Op.POP))

    # ------------------------------------------------------------------
    # Expressions
    # ------------------------------------------------------------------

    def _visit_expr(self, node: ASTNode) -> bool:
        """
        Compile an expression.
        Returns True if a value was left on the VM stack, False if not
        (e.g. void compiler.io calls that consume args and push nothing).
        """
        t = type(node)
        if t is Literal:               return self._visit_literal(node)
        elif t is StringLiteral:       return self._visit_string_literal(node)
        elif t is FStringLiteral:      return self._visit_fstring_literal(node)
        elif t is ArrayLiteral:        return self._visit_array_literal(node)
        elif t is Identifier:          return self._visit_identifier(node)
        elif t is BinaryOp:            return self._visit_binary_op(node)
        elif t is UnaryOp:             return self._visit_unary_op(node)
        elif t is FunctionCall:        return self._visit_function_call(node)
        elif t is MethodCall:          return self._visit_method_call(node)
        elif t is ArrayAccess:         return self._visit_array_access(node)
        elif t is AddressOf:           return self._visit_address_of(node)
        elif t is CastExpression:      return self._visit_cast(node)
        elif t is TypeConvertExpression: return self._visit_cast(node)
        else:
            raise NotImplementedError(
                f'fvmcodegen: unsupported expression in comptime: {type(node).__name__}'
            )

    def _visit_literal(self, node: Literal):
        dt = node.type
        ttag = _datatype_to_ttag(dt)
        raw = node.value
        if ttag in (TTag.INT, TTag.UINT, TTag.LONG, TTag.ULONG, TTag.BYTE, TTag.CHAR):
            data = int(raw) if not isinstance(raw, int) else raw
        elif ttag in (TTag.FLOAT, TTag.DOUBLE):
            data = float(raw) if not isinstance(raw, float) else raw
        elif ttag == TTag.BOOL:
            data = int(bool(raw))
        else:
            data = raw
        self._emit(_instr(Op.PUSH, Val(ttag, data)))
        return True

    def _visit_string_literal(self, node: StringLiteral):
        self._emit(_instr(Op.PUSH, Val(TTag.BYTES, node.value.encode('utf-8'))))
        return True

    def _visit_fstring_literal(self, node: FStringLiteral) -> bool:
        """
        f-string: each part is either a plain str or an Expression.
        Strategy: push each part as a BYTES value (converting expressions via
        INT_TO_STR), then STR_CAT them pairwise into a single string.
        An empty f-string pushes an empty BYTES value.
        """
        parts = node.parts
        if not parts:
            self._emit(_instr(Op.PUSH, Val(TTag.BYTES, b'')))
            return True

        # Strip f" prefix from first string part and closing " from last string part
        # when parse_f_string was called from primary_expression without pre-stripping.
        if parts and isinstance(parts[0], str) and parts[0].startswith('f"'):
            parts = list(parts)
            parts[0] = parts[0][2:]  # strip f"
        if parts and isinstance(parts[-1], str) and parts[-1].endswith('"'):
            parts = list(parts)
            parts[-1] = parts[-1][:-1]  # strip closing "

        def _emit_part(part):
            if isinstance(part, str):
                self._emit(_instr(Op.PUSH, Val(TTag.BYTES, part.encode('utf-8'))))
            else:
                self._visit_expr(part)
                self._emit(_instr(Op.INT_TO_STR))

        _emit_part(parts[0])
        for part in parts[1:]:
            _emit_part(part)
            self._emit(_instr(Op.STR_CAT))
        return True

    def _visit_array_literal(self, node: ArrayLiteral):
        """
        String literals (ArrayLiteral with is_string=True) are pushed as
        Val(TTag.BYTES, bytes).  _read_vm_string() in fvm.py accepts BYTES
        directly, so compiler.io.console.print and friends work without
        needing heap allocation at codegen time.
        """
        if node.is_string and node.string_value is not None:
            raw = node.string_value.encode('utf-8')
            self._emit(_instr(Op.PUSH, Val(TTag.BYTES, raw)))
            return True
        # Non-string array literal: allocate on VM heap at runtime
        elem_type = 'int'
        if node.element_type is not None:
            from ftypesys import DataType as _DT
            _dt_to_name = {
                _DT.INT: 'int', _DT.UINT: 'uint',
                _DT.LONG: 'long', _DT.ULONG: 'ulong',
                _DT.FLOAT: 'float', _DT.DOUBLE: 'double',
                _DT.BOOL: 'bool', _DT.BYTE: 'byte', _DT.CHAR: 'char',
            }
            elem_type = _dt_to_name.get(node.element_type.base_type, 'int')
        count = len(node.elements)
        self._emit(_instr(Op.ARRAY_NEW, elem_type, count))
        arr_slot = self._alloc_local('__arr_tmp__')
        self._emit(_instr(Op.LOCAL_SET, arr_slot))
        for idx, elem in enumerate(node.elements):
            self._emit(_instr(Op.LOCAL_GET, arr_slot))
            self._emit(_instr(Op.PUSH, Val(TTag.INT, idx)))
            self._visit_expr(elem)
            self._emit(_instr(Op.INDEX_SET))
        self._emit(_instr(Op.LOCAL_GET, arr_slot))
        return True

    def _visit_identifier(self, node: Identifier):
        name = node.name
        if name in self._locals:
            self._emit(_instr(Op.LOCAL_GET, self._locals[name]))
            return True
        raise NameError(f'fvmcodegen: undefined comptime variable {name!r}')

    def _visit_binary_op(self, node: BinaryOp):
        op = node.operator
        # Short-circuit logical operators as branching sequences
        if op == Operator.AND:
            self._visit_expr(node.left)
            self._emit(_instr(Op.DUP))
            jnf_idx = self._emit(_instr(Op.JNF, 0))
            self._emit(_instr(Op.POP))
            self._visit_expr(node.right)
            self._patch_at(jnf_idx, Op.JNF, self._current_ip())
            return True
        if op == Operator.OR:
            self._visit_expr(node.left)
            self._emit(_instr(Op.DUP))
            jif_idx = self._emit(_instr(Op.JIF, 0))
            self._emit(_instr(Op.POP))
            self._visit_expr(node.right)
            self._patch_at(jif_idx, Op.JIF, self._current_ip())
            return True
        self._visit_expr(node.left)
        self._visit_expr(node.right)
        _op_map = {
            Operator.ADD:            Op.ADD,
            Operator.SUB:            Op.SUB,
            Operator.MUL:            Op.MUL,
            Operator.DIV:            Op.DIV,
            Operator.MOD:            Op.MOD,
            Operator.POWER:          Op.POW,
            Operator.BITAND:         Op.BAND,
            Operator.BITOR:          Op.BOR,
            Operator.BITXOR:         Op.BXOR,
            Operator.BITSHIFT_LEFT:  Op.SHL,
            Operator.BITSHIFT_RIGHT: Op.SHR,
            Operator.EQUAL:          Op.CMP_EQ,
            Operator.NOT_EQUAL:      Op.CMP_NE,
            Operator.LESS_THAN:      Op.CMP_LT,
            Operator.LESS_EQUAL:     Op.CMP_LE,
            Operator.GREATER_THAN:   Op.CMP_GT,
            Operator.GREATER_EQUAL:  Op.CMP_GE,
        }
        vm_op = _op_map.get(op)
        if vm_op is None:
            raise NotImplementedError(
                f'fvmcodegen: binary operator {op!r} not supported in comptime'
            )
        self._emit(_instr(vm_op))
        return True

    def _visit_unary_op(self, node: UnaryOp):
        op = node.operator
        if op in (Operator.INCREMENT, Operator.DECREMENT):
            if not isinstance(node.operand, Identifier):
                raise NotImplementedError(
                    'fvmcodegen: ++/-- only on simple identifiers in comptime'
                )
            name = node.operand.name
            slot = self._alloc_local(name)
            if node.is_postfix:
                self._emit(_instr(Op.LOCAL_GET, slot))
                self._emit(_instr(Op.LOCAL_GET, slot))
                self._emit(_instr(Op.PUSH, Val(TTag.INT, 1)))
                self._emit(_instr(Op.ADD if op == Operator.INCREMENT else Op.SUB))
                self._emit(_instr(Op.LOCAL_SET, slot))
            else:
                self._emit(_instr(Op.LOCAL_GET, slot))
                self._emit(_instr(Op.PUSH, Val(TTag.INT, 1)))
                self._emit(_instr(Op.ADD if op == Operator.INCREMENT else Op.SUB))
                self._emit(_instr(Op.DUP))
                self._emit(_instr(Op.LOCAL_SET, slot))
            return True
        self._visit_expr(node.operand)
        _op_map = {
            Operator.SUB:    Op.NEG,
            Operator.NOT:    Op.NOT,
            Operator.BITNOT: Op.BNOT,
        }
        vm_op = _op_map.get(op)
        if vm_op is None:
            raise NotImplementedError(
                f'fvmcodegen: unary operator {op!r} not supported in comptime'
            )
        self._emit(_instr(vm_op))
        return True

    def _visit_address_of(self, node: AddressOf) -> bool:
        """
        @x in comptime: push Val(TTag.PTR, slot_index).
        The slot index serves as the VM-level pointer to the local variable.
        Only simple identifier operands are supported.
        """
        if not isinstance(node.expression, Identifier):
            raise NotImplementedError(
                'fvmcodegen: address-of only supported on simple identifiers in comptime'
            )
        name = node.expression.name
        if name not in self._locals:
            raise NameError(f'fvmcodegen: undefined comptime variable {name!r}')
        slot = self._locals[name]
        self._emit(_instr(Op.PUSH, Val(TTag.PTR, slot)))
        return True

    def _visit_cast(self, node) -> bool:
        """
        Type cast / type conversion expression: ulong(x), int(y), float(z), etc.
        Evaluates the inner expression then emits a CAST opcode to coerce the
        top-of-stack value to the target TTag.
        """
        self._visit_expr(node.expression)
        from ftypesys import DataType as _DT
        _dt_to_ttag = {
            _DT.SINT:   TTag.INT,
            _DT.UINT:   TTag.UINT,
            _DT.SLONG:  TTag.LONG,
            _DT.ULONG:  TTag.ULONG,
            _DT.FLOAT:  TTag.FLOAT,
            _DT.DOUBLE: TTag.DOUBLE,
            _DT.BOOL:   TTag.BOOL,
            _DT.BYTE:   TTag.BYTE,
            _DT.CHAR:   TTag.CHAR,
        }
        ts = node.target_type
        if ts.base_type is not None and ts.base_type in _dt_to_ttag:
            target_ttag = _dt_to_ttag[ts.base_type]
        else:
            # Pointer cast or unknown — treat as PTR
            target_ttag = TTag.PTR
        self._emit(_instr(Op.CAST, target_ttag))
        return True

    def _flatten_dotted_name(self, node) -> Optional[str]:
        """
        Flatten a chain of MethodCall / MemberAccess / Identifier nodes into a
        dot-separated name string, e.g. 'compiler.io.console.print'.
        Returns None if the chain contains non-name nodes.
        """
        if isinstance(node, Identifier):
            return node.name
        if isinstance(node, MemberAccess):
            base = self._flatten_dotted_name(node.object)
            return f'{base}.{node.member}' if base is not None else None
        if isinstance(node, MethodCall):
            base = self._flatten_dotted_name(node.object)
            return f'{base}.{node.method_name}' if base is not None else None
        return None

    def _visit_function_call(self, node: FunctionCall):
        _COMPILER_IO = {
            'compiler__io__console__print':  Op.COMPILER_PRINT,
            'compiler.io.console.print':     Op.COMPILER_PRINT,
            'compiler__io__console__input':  Op.COMPILER_INPUT,
            'compiler.io.console.input':     Op.COMPILER_INPUT,
            'compiler__io__readfile':        Op.COMPILER_READFILE,
            'compiler.io.readfile':          Op.COMPILER_READFILE,
            'compiler__io__writefile':       Op.COMPILER_WRITEFILE,
            'compiler.io.writefile':         Op.COMPILER_WRITEFILE,
        }
        # Opcodes that push a return value onto the stack
        _PUSHES = {Op.COMPILER_INPUT, Op.COMPILER_READFILE}
        name = node.name
        opcode = _COMPILER_IO.get(name)
        if opcode is not None:
            if opcode != Op.COMPILER_INPUT:
                for arg in node.arguments:
                    self._visit_expr(arg)
            self._emit(_instr(opcode))
            return opcode in _PUSHES
        for arg in node.arguments:
            self._visit_expr(arg)
        self._emit(_instr(Op.CALL, name, len(node.arguments)))
        return True  # CALL always pushes a return value

    def _visit_method_call(self, node: MethodCall):
        """
        Handle chained namespace/method calls inside comptime blocks.
        Flattens the receiver chain into a dotted name and maps compiler.io.*
        to dedicated VM opcodes; falls back to a generic CALL for others.
        """
        _COMPILER_IO_OPCODES = {
            'compiler.io.console.print':  Op.COMPILER_PRINT,
            'compiler.io.console.input':  Op.COMPILER_INPUT,
            'compiler.io.readfile':       Op.COMPILER_READFILE,
            'compiler.io.writefile':      Op.COMPILER_WRITEFILE,
        }
        _PUSHES = {Op.COMPILER_INPUT, Op.COMPILER_READFILE}

        full_name = self._flatten_dotted_name(node)

        if full_name in _COMPILER_IO_OPCODES:
            opcode = _COMPILER_IO_OPCODES[full_name]
            if opcode != Op.COMPILER_INPUT:
                for arg in node.arguments:
                    self._visit_expr(arg)
            self._emit(_instr(opcode))
            return opcode in _PUSHES

        if full_name is not None:
            for arg in node.arguments:
                self._visit_expr(arg)
            self._emit(_instr(Op.CALL, full_name, len(node.arguments)))
            return True

        raise NotImplementedError(
            f'fvmcodegen: method call on complex receiver not supported in comptime: {node!r}'
        )

    def _visit_array_access(self, node):
        self._visit_expr(node.array)
        self._visit_expr(node.index)
        self._emit(_instr(Op.INDEX_GET))
        return True

    # ------------------------------------------------------------------
    # Control flow
    # ------------------------------------------------------------------

    def _visit_if(self, node: IfStatement):
        end_patches = []

        # Main if
        self._visit_expr(node.condition)
        jnf_idx = self._emit(_instr(Op.JNF, 0))
        self._visit_body(
            node.then_block.statements if isinstance(node.then_block, Block)
            else [node.then_block]
        )
        end_patches.append(self._emit(_instr(Op.JMP, 0)))
        self._patch_at(jnf_idx, Op.JNF, self._current_ip())

        # elif chains
        for cond, blk in (node.elif_blocks or []):
            self._visit_expr(cond)
            jnf_idx = self._emit(_instr(Op.JNF, 0))
            self._visit_body(blk.statements if isinstance(blk, Block) else [blk])
            end_patches.append(self._emit(_instr(Op.JMP, 0)))
            self._patch_at(jnf_idx, Op.JNF, self._current_ip())

        # else block
        if node.else_block is not None:
            self._visit_body(
                node.else_block.statements if isinstance(node.else_block, Block)
                else [node.else_block]
            )

        end_ip = self._current_ip()
        for idx in end_patches:
            self._patch_at(idx, Op.JMP, end_ip)

    def _visit_function_def(self, node: FunctionDef):
        """
        Compile a comptime function definition and store its bytecode in
        self.compiled_functions so the caller (visit_ComptimeBlock in fcodegen.py)
        can register it with the VM before execute().
        Parameters are pre-loaded into local slots 0..N-1 by the VM _call() mechanism.
        """
        fn_cg = FVMCodegen()
        # Allocate a slot for each parameter so LOCAL_GET/SET work by name
        for param in node.parameters:
            fn_cg._alloc_local(param.name)
        # Compile the body
        body_stmts = (
            node.body.statements if isinstance(node.body, Block) else [node.body]
        )
        fn_cg._visit_body(body_stmts)
        # Ensure every path ends with a RET
        if not fn_cg._instructions or fn_cg._instructions[-1].op != Op.RET:
            from fvm import TTag as _TTag
            fn_cg._emit(_instr(Op.PUSH, Val(_TTag.VOID, 0)))
            fn_cg._emit(_instr(Op.RET))
        # Propagate nested function definitions upward
        self.compiled_functions.update(fn_cg.compiled_functions)
        self.compiled_functions[node.name] = fn_cg._instructions

    def _visit_switch(self, node: SwitchStatement):
        """
        Compile a switch statement.

        Strategy: evaluate the subject once into a temp slot, then for each
        non-default case emit CMP_EQ + JNF to skip the body.  The default
        case (Case.value is None) is emitted last and always falls through.
        break inside a case body jumps past the entire switch.
        """
        # Evaluate subject expression and store in a temp local
        self._visit_expr(node.expression)
        subject_slot = self._alloc_local('__switch_subject__')
        self._emit(_instr(Op.LOCAL_SET, subject_slot))

        break_patches: List[int] = []
        default_case: Optional[Case] = None

        for case in node.cases:
            if case.value is None:
                # defer default to end
                default_case = case
                continue

            # Push subject and case value, compare
            self._emit(_instr(Op.LOCAL_GET, subject_slot))
            self._visit_expr(case.value)
            self._emit(_instr(Op.CMP_EQ))
            skip_patch = self._emit(_instr(Op.JNF, 0))

            # Emit case body; intercept break
            body_stmts = (
                case.body.statements if isinstance(case.body, Block) else [case.body]
            )
            for stmt in body_stmts:
                if stmt is None:
                    continue
                if type(stmt) is BreakStatement:
                    break_patches.append(self._emit(_instr(Op.JMP, 0)))
                else:
                    self._visit_stmt(stmt)

            # Jump past the switch after a matched (non-breaking) case body
            break_patches.append(self._emit(_instr(Op.JMP, 0)))
            self._patch_at(skip_patch, Op.JNF, self._current_ip())

        # Emit default case body if present
        if default_case is not None:
            body_stmts = (
                default_case.body.statements
                if isinstance(default_case.body, Block)
                else [default_case.body]
            )
            for stmt in body_stmts:
                if stmt is None:
                    continue
                if type(stmt) is BreakStatement:
                    break_patches.append(self._emit(_instr(Op.JMP, 0)))
                else:
                    self._visit_stmt(stmt)

        after_switch = self._current_ip()
        for idx in break_patches:
            self._patch_at(idx, Op.JMP, after_switch)

    def _visit_for(self, node: ForLoop):
        if node.init is not None:
            self._visit_stmt(node.init)

        loop_start = self._current_ip()

        exit_patch = None
        if node.condition is not None:
            self._visit_expr(node.condition)
            exit_patch = self._emit(_instr(Op.JNF, 0))

        break_patches:    List[int] = []
        continue_patches: List[int] = []

        body_stmts = (
            node.body.statements if isinstance(node.body, Block) else [node.body]
        )
        for stmt in body_stmts:
            if stmt is None:
                continue
            if type(stmt) is BreakStatement:
                break_patches.append(self._emit(_instr(Op.JMP, 0)))
            elif type(stmt) is ContinueStatement:
                continue_patches.append(self._emit(_instr(Op.JMP, 0)))
            else:
                self._visit_stmt(stmt)

        update_ip = self._current_ip()
        if node.update is not None:
            self._visit_stmt(node.update)

        self._emit(_instr(Op.JMP, loop_start))
        after_loop = self._current_ip()

        if exit_patch is not None:
            self._patch_at(exit_patch, Op.JNF, after_loop)
        for idx in continue_patches:
            self._patch_at(idx, Op.JMP, update_ip)
        for idx in break_patches:
            self._patch_at(idx, Op.JMP, after_loop)

    def _visit_while(self, node: WhileLoop):
        loop_start = self._current_ip()
        self._visit_expr(node.condition)
        exit_patch = self._emit(_instr(Op.JNF, 0))

        break_patches:    List[int] = []
        continue_patches: List[int] = []
        body_stmts = (
            node.body.statements if isinstance(node.body, Block) else [node.body]
        )
        for stmt in body_stmts:
            if stmt is None:
                continue
            if type(stmt) is BreakStatement:
                break_patches.append(self._emit(_instr(Op.JMP, 0)))
            elif type(stmt) is ContinueStatement:
                continue_patches.append(self._emit(_instr(Op.JMP, 0)))
            else:
                self._visit_stmt(stmt)

        self._emit(_instr(Op.JMP, loop_start))
        after_loop = self._current_ip()
        self._patch_at(exit_patch, Op.JNF, after_loop)
        for idx in break_patches:
            self._patch_at(idx, Op.JMP, after_loop)
        for idx in continue_patches:
            self._patch_at(idx, Op.JMP, loop_start)

    def _visit_do(self, node: DoLoop):
        """Plain do loop: body executes once."""
        self._visit_body(
            node.body.statements if isinstance(node.body, Block) else [node.body]
        )

    def _visit_do_while(self, node: DoWhileLoop):
        loop_start = self._current_ip()
        break_patches:    List[int] = []
        continue_patches: List[int] = []
        body_stmts = (
            node.body.statements if isinstance(node.body, Block) else [node.body]
        )
        for stmt in body_stmts:
            if stmt is None:
                continue
            if type(stmt) is BreakStatement:
                break_patches.append(self._emit(_instr(Op.JMP, 0)))
            elif type(stmt) is ContinueStatement:
                continue_patches.append(self._emit(_instr(Op.JMP, 0)))
            else:
                self._visit_stmt(stmt)
        cond_ip = self._current_ip()
        self._visit_expr(node.condition)
        self._emit(_instr(Op.JIF, loop_start))
        after_loop = self._current_ip()
        for idx in break_patches:
            self._patch_at(idx, Op.JMP, after_loop)
        for idx in continue_patches:
            self._patch_at(idx, Op.JMP, cond_ip)

    def _visit_return(self, node: ReturnStatement):
        if node.value is not None:
            self._visit_expr(node.value)
        else:
            self._emit(_instr(Op.PUSH, Val(TTag.VOID, 0)))
        self._emit(_instr(Op.RET))

    def _visit_break(self, node: BreakStatement):
        self._emit(_instr(Op.JMP, 0))  # patched by enclosing loop visitor

    def _visit_continue(self, node: ContinueStatement):
        self._emit(_instr(Op.JMP, 0))  # patched by enclosing loop visitor

    # ------------------------------------------------------------------
    # emitflux
    # ------------------------------------------------------------------

    def _visit_emitflux(self, node: EmitFlux):
        """
        Emit an EMITFLUX opcode.  Operands:
          [0] source_text : str              -- raw Flux text with placeholder names
          [1] var_names   : [(name, slot)]   -- snapshot of current locals

        At VM execution time the actual values in those slots are used for
        variable substitution in _op_emitflux().
        """
        var_snapshot = list(self._locals.items())  # [(name, slot), ...]
        self._emit(_instr(Op.EMITFLUX, node.source_text, var_snapshot))