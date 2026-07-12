#!/usr/bin/env python3
"""
fbc.py -- Flux Borrow Checker

Copyright (C) 2026 Karac V. Thweatt

Performs a call-graph walk from the program entry point (FRTStartup by default)
and checks for pointer aliasing violations, scope escapes, and optionally
thread safety and heap leaks.

Usage:
    python fbc.py <file_or_dir> [options]

Options:
    --entry NAME      Entry point function name (default: FRTStartup)
    --warn            Treat violations as warnings instead of errors
    --threads         Also check thread safety (pointer escape across spawns)
    --leaks           Also check for heap leaks (unmatched fmalloc/ffree)
    --json            Machine-readable JSON output (for CI)
    --no-color        Disable ANSI color output

Exit codes:
    0   No violations (or --warn mode)
    1   One or more violations found (in default error mode)
"""

import sys
import os
import argparse
from pathlib import Path

# fbc modules
from fbc_alias import AliasMap, Violation
from fbc_report import print_violations, print_summary

# Flux compiler modules
# fbc.py lives at the project root alongside fxc.py.
# Compiler source is in src/compiler/ -- add that to the path.
_root = Path(__file__).parent
for _candidate in [
    _root / 'src' / 'compiler',
    _root / 'compiler',
    _root,
]:
    if (_candidate / 'fparser.py').exists():
        sys.path.insert(0, str(_candidate))
        break

try:
    from fparser import FluxParser
    from flexer import FluxLexer
    from fpreprocess import FXPreprocessor
    from fmacros import build_compiler_macros
    import fast
    from ftypesys import TypeSystem, Operator
except ImportError as e:
    print(f"[FBC] Error: could not import Flux compiler modules: {e}")
    print("      Looked for fparser.py in src/compiler/, compiler/, and project root.")
    sys.exit(2)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_pointer_type(type_spec) -> bool:
    """Return True if a TypeSystem represents a pointer type."""
    if type_spec is None:
        return False
    return getattr(type_spec, 'is_pointer', False) or getattr(type_spec, 'pointer_depth', 0) > 0

def _node_file(node) -> str:
    return getattr(node, 'source_file', '<unknown>')

def _node_line(node) -> int:
    return getattr(node, 'source_line', 0)

def _make_stack_site(func_name: str, var_name: str, line: int) -> str:
    return f"stack:{func_name}:{var_name}@{line}"

def _make_heap_site(func_name: str, line: int) -> str:
    return f"heap:{func_name}:{line}"

def _make_param_site(func_name: str, param_name: str) -> str:
    return f"param:{func_name}:{param_name}"

def _is_fmalloc(node) -> bool:
    if isinstance(node, fast.FunctionCall):
        return node.name == 'fmalloc'
    return False

def _is_ffree(node) -> bool:
    if isinstance(node, fast.FunctionCall):
        return node.name == 'ffree'
    return False

def _is_thread_spawn(node) -> bool:
    """Heuristic: function calls that look like thread spawns."""
    if isinstance(node, fast.FunctionCall):
        return 'spawn' in node.name.lower() or 'thread' in node.name.lower()
    if isinstance(node, fast.MethodCall):
        name = getattr(node, 'name', '') or ''
        return 'spawn' in name.lower() or 'thread' in name.lower()
    return False

# ---------------------------------------------------------------------------
# Call graph builder
# ---------------------------------------------------------------------------

class CallGraph:
    """Collects all FunctionDef nodes keyed by name for call resolution."""

    def __init__(self):
        # name -> FunctionDef
        self.funcs: dict[str, fast.FunctionDef] = {}
        # short name -> mangled name, for intra-namespace call resolution
        self.short_names: dict[str, str] = {}
        # base name -> list of mangled names (for overload resolution at call sites)
        self.overloads: dict[str, list] = {}
        # functions already walked from a non-thread (main) context
        self.visited_main: set[str] = set()
        # functions already walked from a thread context
        self.visited_thread: set[str] = set()
        # how many times each function has been passed to a thread spawn
        # name -> int
        self.spawn_count: dict[str, int] = {}

    # kept as a property so existing code that reads cg.visited still compiles
    @property
    def visited(self) -> set[str]:
        return self.visited_main

    def collect(self, program: fast.Program):
        """Walk the top-level program and collect all function definitions."""
        for node in program.statements:
            self._collect_node(node)
        # build short name index and overload index
        short_count: dict[str, int] = {}
        for mangled in self.funcs:
            short = mangled.split('__')[-1]
            short_count[short] = short_count.get(short, 0) + 1
        for mangled in self.funcs:
            short = mangled.split('__')[-1]
            if short_count[short] == 1:
                self.short_names[short] = mangled
            # overload index: base name is everything before the __N__ arity segment
            # split on __ and find the first all-digit segment; base is parts before it
            # e.g. main__0__ret_intE1 -> parts=['main','0','ret_intE1'] -> base='main'
            parts = mangled.split('__')
            base = mangled
            for idx, part in enumerate(parts):
                if part.isdigit():
                    base = '__'.join(parts[:idx])
                    break
            self.overloads.setdefault(base, []).append(mangled)

    def _collect_node(self, node, ns_prefix: str = ''):
        if isinstance(node, fast.FunctionDef):
            if not node.is_prototype:
                self.funcs[node.name] = node
        elif isinstance(node, fast.FunctionDefStatement):
            self._collect_node(node.function_def, ns_prefix)
        elif isinstance(node, fast.NamespaceDefStatement):
            self._collect_node(node.namespace_def, ns_prefix)
        elif isinstance(node, fast.NamespaceDef):
            prefix = f"{ns_prefix}__{node.name}" if ns_prefix else node.name
            for func in node.functions:
                if not func.is_prototype:
                    mangled = f"{prefix}__{func.name}"
                    self.funcs[mangled] = func
            for obj in node.objects:
                self._collect_object(obj, prefix=prefix)
            for ns in node.nested_namespaces:
                self._collect_node(ns, ns_prefix=prefix)
        elif isinstance(node, fast.ObjectDef):
            self._collect_object(node, prefix=ns_prefix)
        elif isinstance(node, fast.ObjectDefStatement):
            self._collect_object(node.object_def, prefix=ns_prefix)
        elif isinstance(node, fast.StructDef):
            pass
        elif isinstance(node, fast.StructDefStatement):
            pass
        elif isinstance(node, fast.ExternBlock):
            pass
        elif isinstance(node, fast.ExportBlock):
            for func in node.definitions:
                if not func.is_prototype:
                    self.funcs[func.name] = func

    def _collect_object(self, obj, prefix: str):
        # guard: only ObjectDef has methods we can walk; TraitDef/InterfaceDef do not
        if not isinstance(obj, fast.ObjectDef):
            return
        for method in obj.methods:
            if method.name and method.body:
                mangled = f"{prefix}__{obj.name}__{method.name}" if prefix else f"{obj.name}__{method.name}"
                self.funcs[mangled] = method
        for nested in obj.nested_objects:
            self._collect_object(nested, prefix=f"{prefix}__{obj.name}" if prefix else obj.name)

# ---------------------------------------------------------------------------
# AST Walker
# ---------------------------------------------------------------------------

class BorrowChecker:

    def __init__(self, call_graph: CallGraph, entry: str,
                 check_threads: bool = False,
                 check_leaks: bool = False,
                 line_map: list = None):
        self.cg = call_graph
        self.entry = entry
        self.check_threads = check_threads
        self.check_leaks = check_leaks
        self.line_map = line_map or []
        self.alias = AliasMap()
        self.violations: list[Violation] = []
        self.funcs_checked = 0
        # True while walking a function that was invoked from a thread spawn
        self._in_thread_context: bool = False

    def _remap(self, merged_line: int) -> tuple:
        """Map a merged-source line number back to (original_file, original_line)."""
        idx = merged_line - 1
        if self.line_map and 0 <= idx < len(self.line_map):
            fname, orig_line = self.line_map[idx]
            return fname, orig_line
        return '<unknown>', merged_line

    def run(self):
        """Start walk from entry point."""
        if self.entry not in self.cg.funcs:
            print(f"[FBC] Warning: entry point '{self.entry}' not found as a non-prototype function.")
            print(f"      This usually means the entry point body is inside an unresolved #ifdef block.")
            print(f"      Checking all reachable functions instead.")
            for name, func in self.cg.funcs.items():
                if name not in self.cg.visited:
                    self._check_function(func)
        else:
            self._check_function(self.cg.funcs[self.entry])

    # Functions whose job is to allocate and return -- fmalloc inside them
    # that reaches a return statement transfers ownership to the caller, not a leak.
    _ALLOCATOR_FUNCS = {
        'fmalloc', 'ffree',
    }

    def _func_returns_allocation(self, func) -> bool:
        """True if this function is a known allocator wrapper -- fmalloc result returned directly."""
        if func.name in self._ALLOCATOR_FUNCS:
            return True
        if not func.body:
            return False
        # heuristic: single-statement body that returns an fmalloc call or calls fmalloc
        stmts = func.body.statements
        if len(stmts) == 1:
            s = stmts[0]
            if isinstance(s, fast.ReturnStatement) and s.value:
                if _is_fmalloc(s.value):
                    return True
                if isinstance(s.value, fast.CastExpression) and _is_fmalloc(s.value.expression):
                    return True
                if isinstance(s.value, fast.FunctionCall) and isinstance(s.value.name, str):
                    if 'malloc' in s.value.name.lower() or 'alloc' in s.value.name.lower():
                        return True
            if isinstance(s, fast.ExpressionStatement):
                expr = s.expression
                if isinstance(expr, fast.ReturnStatement) and expr.value:
                    if _is_fmalloc(expr.value):
                        return True
        return False

    def _check_function(self, func: fast.FunctionDef, thread_context: bool = False):
        visited_set = self.cg.visited_thread if thread_context else self.cg.visited_main
        if func.name in visited_set:
            return
        visited_set.add(func.name)
        self.funcs_checked += 1

        prev_func = self.alias.func_name
        prev_thread_ctx = self._in_thread_context
        self.alias.func_name = func.name
        self._in_thread_context = thread_context

        self.alias.push_scope()

        # register pointer parameters as borrowed sites
        for param in (func.parameters or []):
            if param.name and _is_pointer_type(param.type_spec):
                site = _make_param_site(func.name, param.name)
                self.alias.declare_ptr(
                    var_name=param.name,
                    site=site,
                    mutable=True,  # conservative: treat all pointer params as mutable
                    file=_node_file(func),
                    line=_node_line(func),
                )

        if func.body:
            self._walk_block(func.body)

        if self.check_leaks and not self._func_returns_allocation(func):
            self.alias.check_heap_leaks()
        else:
            self.alias.heap_sites.clear()

        self.alias.pop_scope()
        self.alias.func_name = prev_func
        self._in_thread_context = prev_thread_ctx

        # collect violations -- remap merged line numbers to original source locations
        for v in self.alias.violations:
            fname, orig_line = self._remap(v.line)
            v.file = fname
            v.line = orig_line
            # remap detail lines that contain "at <unknown>:N" patterns
            remapped_detail = []
            for d in v.detail:
                parts = d.split(' at ')
                if len(parts) == 2:
                    loc = parts[1]
                    colon = loc.rfind(':')
                    if colon != -1:
                        try:
                            merged = int(loc[colon+1:])
                            df, dl = self._remap(merged)
                            d = parts[0] + f' at {df}:{dl}'
                        except ValueError:
                            pass
                remapped_detail.append(d)
            v.detail = remapped_detail
        seen = {self._violation_key(v) for v in self.violations}
        for v in self.alias.violations:
            if self._violation_key(v) not in seen:
                self.violations.append(v)
                seen.add(self._violation_key(v))
        self.alias.violations.clear()

    def _violation_key(self, v) -> tuple:
        return (v.kind, v.file, v.line, v.message)

    def _walk_block(self, block: fast.Block):
        self.alias.push_scope()
        for stmt in block.statements:
            if stmt is not None:
                self._walk_stmt(stmt)
        self.alias.pop_scope(check_leaks=self.check_leaks)

    def _walk_stmt(self, node):
        file = _node_file(node)
        line = _node_line(node)

        if isinstance(node, fast.VariableDeclaration):
            self._on_var_decl(node)

        elif isinstance(node, fast.Assignment):
            self._on_assign(node)

        elif isinstance(node, fast.CompoundAssignment):
            self._walk_expr(node.value)

        elif isinstance(node, fast.ExpressionStatement):
            self._walk_expr(node.expression)

        elif isinstance(node, fast.Block):
            self._walk_block(node)

        elif isinstance(node, fast.IfStatement):
            self._walk_expr(node.condition)
            if node.then_block:
                self._walk_block(node.then_block)
            if node.else_block:
                self._walk_block(node.else_block)

        elif isinstance(node, fast.WhileLoop):
            self._walk_expr(node.condition)
            if node.body:
                self._walk_block(node.body)

        elif isinstance(node, fast.DoWhileLoop):
            if node.body:
                self._walk_block(node.body)
            self._walk_expr(node.condition)

        elif isinstance(node, fast.ForLoop):
            if node.init:
                self._walk_stmt(node.init)
            if node.condition:
                self._walk_expr(node.condition)
            if node.update:
                self._walk_stmt(node.update)
            if node.body:
                self._walk_block(node.body)

        elif isinstance(node, fast.ForInLoop):
            if node.body:
                self._walk_block(node.body)

        elif isinstance(node, fast.SwitchStatement):
            self._walk_expr(node.expression)
            for case in node.cases:
                if case.body:
                    self._walk_block(case.body)

        elif isinstance(node, fast.ReturnStatement):
            if node.value:
                self._walk_expr(node.value)

        elif isinstance(node, fast.DeferStatement):
            if node.expression:
                self._walk_expr(node.expression)
            if node.body:
                for s in node.body:
                    self._walk_stmt(s)

        elif isinstance(node, fast.TryBlock):
            if node.try_body:
                self._walk_block(node.try_body)
            for _, _, catch_body in (node.catch_blocks or []):
                self._walk_block(catch_body)

        elif isinstance(node, fast.FunctionDefStatement):
            # nested function def -- walk it under current context
            self._check_function(node.function_def, thread_context=self._in_thread_context)

        # other statement types (break, continue, label, goto, assert etc.)
        # don't introduce pointers so we skip them

    def _on_var_decl(self, node: fast.VariableDeclaration):
        file = _node_file(node)
        line = _node_line(node)
        is_ptr = _is_pointer_type(node.type_spec)

        # Always register the stack site for this variable so its address
        # can be tracked if taken later
        site = _make_stack_site(self.alias.func_name, node.name, line)

        if is_ptr:
            # it's a pointer variable -- what does it point to?
            ptr_site = self._resolve_site(node.initial_value, node.name, file, line)
            self.alias.declare_ptr(
                var_name=node.name,
                site=ptr_site,
                mutable=True,
                file=file,
                line=line,
                is_stack_owner=False,
            )
        else:
            # non-pointer stack variable -- register its site so @node.name can be resolved
            if self.alias.frames:
                frame = self.alias.frames[-1]
                frame.owned_sites.add(site)
                frame.var_sites[node.name] = site

        if node.initial_value:
            self._walk_expr(node.initial_value)

    def _on_assign(self, node: fast.Assignment):
        file = _node_file(node)
        line = _node_line(node)
        self._walk_expr(node.value)

        # is the target a pointer variable?
        if isinstance(node.target, fast.Identifier):
            var_name = node.target.name
            # check if we know this var as a pointer
            existing = self.alias._find_ptr(var_name)
            if existing is not None:
                new_site = self._resolve_site(node.value, var_name, file, line)
                self.alias.assign_ptr(var_name, new_site, True, file, line)

    _SPAWN_IMPL_FUNCS = {'thread_create', 'thread_create_stack', 'pthread_create', 'CreateThread'}

    def _in_spawn_impl(self) -> bool:
        return any(impl in self.alias.func_name for impl in self._SPAWN_IMPL_FUNCS)

    def _check_ident_escape(self, name: str, file: str, line: int):
        """Check if using this identifier touches a thread-escaped site."""
        if self.check_threads and not self._in_spawn_impl():
            self.alias.check_use_after_escape(name, file, line)

    def _unwrap_ptr_ident(self, expr) -> str:
        """Unwrap casts/address-of to find the underlying pointer variable name, if any."""
        if isinstance(expr, fast.Identifier):
            return expr.name
        if isinstance(expr, (fast.CastExpression, fast.TypeConvertExpression)):
            return self._unwrap_ptr_ident(expr.expression)
        if isinstance(expr, fast.AddressOf):
            return self._unwrap_ptr_ident(expr.expression)
        if isinstance(expr, fast.FunctionCall) and expr.arguments:
            return self._unwrap_ptr_ident(expr.arguments[0])
        return None

    def _resolve_ffree_arg(self, arg) -> str:
        """
        Resolve the argument to ffree() to a heap site string.
        Handles: plain Identifier, @x (AddressOf), cast(x), long(x), u64(x).
        ffree takes a numeric address -- the pointer is often cast before passing.
        """
        # plain identifier: ffree(ptr)
        if isinstance(arg, fast.Identifier):
            info = self.alias._find_ptr(arg.name)
            return info.site if info else 'unknown'
        # cast: ffree(long(argv)) or ffree((u64)argv)
        if isinstance(arg, fast.CastExpression):
            return self._resolve_ffree_arg(arg.expression)
        # function call cast like long(argv) -- FunctionCall with type name
        if isinstance(arg, fast.FunctionCall) and isinstance(arg.name, str):
            if arg.arguments:
                return self._resolve_ffree_arg(arg.arguments[0])
        # address-of: ffree(@argv) -- @argv gives address of the pointer variable
        # the pointer itself is what was malloc'd
        if isinstance(arg, fast.AddressOf):
            inner = arg.expression
            if isinstance(inner, fast.Identifier):
                info = self.alias._find_ptr(inner.name)
                return info.site if info else 'unknown'
        # type convert: long(argv), u64(argv) etc.
        if isinstance(arg, fast.TypeConvertExpression):
            return self._resolve_ffree_arg(arg.expression)
        return 'unknown'

    def _resolve_site(self, expr, var_name: str, file: str, line: int) -> str:
        """
        Try to determine what allocation site an expression refers to.
        Returns a site identity string.
        """
        if expr is None:
            return 'unknown'

        # @someVar or @someArray[i] -- address of a stack variable or element
        if isinstance(expr, fast.AddressOf):
            inner = expr.expression
            if isinstance(inner, fast.Identifier):
                target_name = inner.name
                for frame in reversed(self.alias.frames):
                    if target_name in frame.var_sites:
                        return frame.var_sites[target_name]
                return _make_stack_site(self.alias.func_name, target_name, line)
            # @arr[i] -- treat as address of the base array
            if isinstance(inner, fast.ArrayAccess):
                base = inner.array
                if isinstance(base, fast.Identifier):
                    target_name = base.name
                    for frame in reversed(self.alias.frames):
                        if target_name in frame.var_sites:
                            return frame.var_sites[target_name]
                    return _make_stack_site(self.alias.func_name, target_name, line)
            return 'unknown'

        # fmalloc() call
        if _is_fmalloc(expr):
            return _make_heap_site(self.alias.func_name, _node_line(expr))

        # ffree() -- site resolution returns unknown; side effect handled in _walk_expr
        if _is_ffree(expr):
            return 'unknown'

        # pointer arithmetic -- derived from base
        if isinstance(expr, fast.BinaryOp) and expr.operator in (Operator.ADD, Operator.SUB):
            base_site = self._resolve_site(expr.left, var_name, file, line)
            if base_site != 'unknown':
                return f"derived:{base_site}"
            return self._resolve_site(expr.right, var_name, file, line)

        # identifier -- copy the site from the existing pointer
        if isinstance(expr, fast.Identifier):
            info = self.alias._find_ptr(expr.name)
            if info:
                return info.site
            return 'unknown'

        # cast / type convert -- mark as derived to avoid false alias with original
        if isinstance(expr, fast.CastExpression):
            inner = self._resolve_site(expr.expression, var_name, file, line)
            return f"derived:{inner}" if inner != 'unknown' else 'unknown'
        if isinstance(expr, fast.TypeConvertExpression):
            inner = self._resolve_site(expr.expression, var_name, file, line)
            return f"derived:{inner}" if inner != 'unknown' else 'unknown'

        # deref -- we can't easily track through double-deref without full type info
        if isinstance(expr, fast.PointerDeref):
            return 'unknown'

        return 'unknown'

    def _walk_expr(self, node):
        """Walk an expression for side effects (calls, thread spawns, ffree)."""
        if node is None:
            return

        file = _node_file(node)
        line = _node_line(node)

        if _is_fmalloc(node) and self.check_leaks:
            site = _make_heap_site(self.alias.func_name, _node_line(node))
            self.alias.record_malloc(site, _node_file(node), _node_line(node))

        if _is_ffree(node) and self.check_leaks:
            if isinstance(node, fast.FunctionCall) and node.arguments:
                arg = node.arguments[0]
                # resolve the argument to a site -- handles Identifier, @x, cast(x), etc.
                freed_site = self._resolve_ffree_arg(arg)
                if freed_site and freed_site != 'unknown':
                    self.alias.record_free(freed_site)

        if self.check_threads and _is_thread_spawn(node):
            # look for pointer arguments being passed -- unwrap casts to find identifiers
            args = getattr(node, 'arguments', []) or []

            # find the function-pointer argument -- typically the first arg
            # that resolves to a known function name
            spawned_func_name = None
            for arg in args:
                cand = self._unwrap_ptr_ident(arg)
                if cand and cand in self.cg.funcs:
                    spawned_func_name = cand
                    break
                if cand and cand in self.cg.short_names:
                    spawned_func_name = self.cg.short_names[cand]
                    break

            for arg in args:
                name = self._unwrap_ptr_ident(arg)
                if name:
                    self.alias.check_thread_escape(name, file, line)

            if spawned_func_name and spawned_func_name in self.cg.funcs:
                prev_count = self.cg.spawn_count.get(spawned_func_name, 0)
                self.cg.spawn_count[spawned_func_name] = prev_count + 1

                # walk spawned function under thread context if not done yet
                if spawned_func_name not in self.cg.visited_thread:
                    self._check_function(self.cg.funcs[spawned_func_name], thread_context=True)

                # if this function is spawned more than once, its mutable pointer
                # params can race with themselves across concurrent instances
                if prev_count >= 1:
                    spawned_func = self.cg.funcs[spawned_func_name]
                    mutable_ptr_params = [
                        p for p in (spawned_func.parameters or [])
                        if p.name and _is_pointer_type(p.type_spec)
                    ]
                    if mutable_ptr_params:
                        param_names = ', '.join(p.name for p in mutable_ptr_params)
                        self.violations.append(Violation(
                            kind='thread_self_race',
                            message=(
                                f"function '{spawned_func_name}' is spawned as a thread "
                                f"{prev_count + 1} times -- concurrent instances share "
                                f"mutable pointer parameter(s): {param_names}"
                            ),
                            file=file,
                            line=line,
                            detail=[
                                f"each spawn receives a separate pointer value but if "
                                f"those values alias the same allocation site, a data "
                                f"race exists between the concurrent instances"
                            ]
                        ))

        # check any identifier use against thread-escaped sites
        if self.check_threads and isinstance(node, fast.Identifier):
            self._check_ident_escape(node.name, file, line)

        # resolve calls into the call graph
        if isinstance(node, fast.FunctionCall):
            callee_name = node.name if isinstance(node.name, str) else None
            if callee_name:
                cur_ctx = self._in_thread_context
                visited_set = self.cg.visited_thread if cur_ctx else self.cg.visited_main
                resolved = callee_name if callee_name in self.cg.funcs else self.cg.short_names.get(callee_name)
                if resolved and resolved not in visited_set:
                    self._check_function(self.cg.funcs[resolved], thread_context=cur_ctx)
                elif not resolved:
                    # try overload index -- call site uses base name, codegen adds arity suffix
                    for mangled in self.cg.overloads.get(callee_name, []):
                        if mangled not in visited_set:
                            self._check_function(self.cg.funcs[mangled], thread_context=cur_ctx)
                    # DEBUG
                    #if not self.cg.overloads.get(callee_name):
                    #    print(f"[FBC DEBUG] unresolved call in {self.alias.func_name!r}: {callee_name!r}")
                # DEBUG
                #elif not resolved:
                #    print(f"[FBC DEBUG] unresolved call: {callee_name!r} -- likely extern or intra-namespace (caller scope not tracked)")
            for arg in (node.arguments or []):
                self._walk_expr(arg)

        elif isinstance(node, fast.MethodCall):
            self._walk_expr(node.object)
            for arg in (node.arguments or []):
                self._walk_expr(arg)

        elif isinstance(node, fast.BinaryOp):
            self._walk_expr(node.left)
            self._walk_expr(node.right)

        elif isinstance(node, fast.UnaryOp):
            self._walk_expr(node.operand)

        elif isinstance(node, fast.AddressOf):
            self._walk_expr(node.expression)

        elif isinstance(node, fast.PointerDeref):
            # check for use-after-thread-escape
            if self.check_threads and isinstance(node.pointer, fast.Identifier):
                self.alias.check_use_after_escape(node.pointer.name, file, line)
            self._walk_expr(node.pointer)

        elif isinstance(node, fast.MemberAccess):
            self._walk_expr(node.object)

        elif isinstance(node, fast.ArrayAccess):
            # check for access to thread-escaped pointer
            if self.check_threads and isinstance(node.array, fast.Identifier):
                self.alias.check_use_after_escape(node.array.name, file, line)
            self._walk_expr(node.array)
            self._walk_expr(node.index)

        elif isinstance(node, fast.CastExpression):
            self._walk_expr(node.expression)

        elif isinstance(node, fast.TypeConvertExpression):
            self._walk_expr(node.expression)

        elif isinstance(node, fast.IfExpression):
            self._walk_expr(node.condition)
            self._walk_expr(node.then_expr)
            self._walk_expr(node.else_expr)

        elif isinstance(node, fast.TernaryOp):
            self._walk_expr(node.condition)
            self._walk_expr(node.true_expr)
            self._walk_expr(node.false_expr)

        elif isinstance(node, fast.Assignment):
            self._walk_expr(node.value)
            # check lvalue target for escaped site access
            if self.check_threads:
                if isinstance(node.target, fast.Identifier):
                    self._check_ident_escape(node.target.name, file, line)
                elif isinstance(node.target, fast.ArrayAccess):
                    arr = node.target.array
                    if isinstance(arr, fast.Identifier):
                        self._check_ident_escape(arr.name, file, line)
            # if target is a known pointer, update its site
            if isinstance(node.target, fast.Identifier):
                existing = self.alias._find_ptr(node.target.name)
                if existing is not None:
                    new_site = self._resolve_site(node.value, node.target.name,
                                                  _node_file(node), _node_line(node))
                    self.alias.assign_ptr(node.target.name, new_site, True,
                                          _node_file(node), _node_line(node))

        elif isinstance(node, fast.CompoundAssignment):
            self._walk_expr(node.value)

# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def parse_file(path: str):
    """Returns (Program, line_map) where line_map[i] = (filename, orig_line) for merged line i+1."""
    platform_macros = build_compiler_macros()
    preprocessor = FXPreprocessor(path, compiler_constants=platform_macros)
    source = preprocessor.process()
    line_map = preprocessor.line_map  # list of (filename, orig_line_1based)
    lexer = FluxLexer(source)
    tokens = lexer.tokenize()
    parser = FluxParser(tokens)
    return parser.parse(), line_map

def collect_fx_files(path: str) -> list[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    return sorted(str(f) for f in p.rglob('*.fx'))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Flux Borrow Checker -- static pointer safety analysis'
    )
    parser.add_argument('target', help='File or directory to check')
    parser.add_argument('--entry',    default='FRTStartup',
                        help='Entry point function name (default: FRTStartup)')
    parser.add_argument('--warn',     action='store_true',
                        help='Treat violations as warnings instead of errors')
    parser.add_argument('--threads',  action='store_true',
                        help='Check thread safety (pointer escape across spawns)')
    parser.add_argument('--leaks',    action='store_true',
                        help='Check for heap leaks (unmatched fmalloc/ffree)')
    parser.add_argument('--json',     action='store_true',
                        help='Machine-readable JSON output')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable ANSI color output')
    args = parser.parse_args()

    use_color = not args.no_color and sys.stderr.isatty()

    # --- collect and parse files ---
    fx_files = collect_fx_files(args.target)
    if not fx_files:
        print(f"[FBC] No .fx files found in: {args.target}")
        sys.exit(2)

    if not args.json:
        print(f"[FBC] Checking {len(fx_files)} file(s), entry: {args.entry}")

    programs = []
    line_maps = []
    parse_errors = []
    for fx in fx_files:
        try:
            prog, line_map = parse_file(fx)
            programs.append((fx, prog))
            line_maps.append(line_map)
        except Exception as e:
            parse_errors.append((fx, str(e)))
            if not args.json:
                print(f"[FBC] Parse error in {fx}: {e}", file=sys.stderr)

    if not programs:
        print("[FBC] No files could be parsed.")
        sys.exit(2)

    # merge line maps -- since we parse one file at a time (preprocessor merges imports)
    # we only have one line_map per input file
    combined_line_map = line_maps[0] if line_maps else []

    # --- build call graph ---
    cg = CallGraph()
    for fx, prog in programs:
        cg.collect(prog)

    if not args.json:
        print(f"[FBC] {len(cg.funcs)} function(s) found")

    # --- run borrow checker ---
    checker = BorrowChecker(
        call_graph=cg,
        entry=args.entry,
        check_threads=args.threads,
        check_leaks=args.leaks,
        line_map=combined_line_map,
    )
    checker.run()

    all_violations = checker.violations

    # --- report ---
    mode = 'warn' if args.warn else 'error'
    print_violations(all_violations, mode=mode, use_color=use_color,
                     json_out=args.json)
    print_summary(all_violations, files_checked=len(fx_files),
                  funcs_checked=checker.funcs_checked,
                  use_color=use_color, json_out=args.json)

    if all_violations and not args.warn:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()