# Copyright (c) Karac Von Thweatt. All rights reserved.

"""
cft.py - C to Flux translation layer
Uses libclang to parse C headers/sources and emits equivalent Flux declarations
and function bodies.

Usage:
    python cft.py <input.c|h> [output.fx]
    python cft.py <input.c|h>        # prints to stdout
"""

import sys
import os
import configparser
import clang.cindex as cx
from clang.cindex import CursorKind, TypeKind

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# cft.cfg is looked for next to cft.py, then in the current working directory.
# Example cft.cfg:
#
#   [cft]
#   # Path to libclang shared library. If omitted, cft will attempt to locate
#   # it automatically based on the current platform:
#   #   Windows : libclang.dll  (checked in D:\LLVM\bin, C:\Program Files\LLVM\bin, PATH)
#   #   macOS   : libclang.dylib (checked in Homebrew LLVM, Xcode CommandLineTools)
#   #   Linux   : libclang.so   (checked in /usr/lib/llvm-*, distro paths, LD_LIBRARY_PATH)
#   # Set this explicitly if auto-detection fails.
#   libclang = D:\LLVM\bin\libclang.dll
#
#   [cstdlib]
#   # Root directory where translated .fx files for system headers are written.
#   output_root = C:\flux\stdlib\c
#   # Semicolon-separated list of system include roots to mirror (optional).
#   # When empty, cft infers roots from the include paths libclang reports.
#   include_roots =
#
#   [clang]
#   # Extra flags passed to libclang, space-separated.
#   args = -x c -std=c11

_CFT_CFG_NAME = "cft.cfg"

def _find_config():
    """Return path to cft.cfg if found, else None."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), _CFT_CFG_NAME),
        os.path.join(os.getcwd(), _CFT_CFG_NAME),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

def _find_libclang():
    """Return the path to libclang shared library for the current platform,
    or None if it cannot be located automatically."""
    import platform
    system = platform.system()

    if system == "Windows":
        candidates = [
            r"D:\LLVM\bin\libclang.dll",
            r"C:\Program Files\LLVM\bin\libclang.dll",
        ]
        # Also check PATH entries for libclang.dll
        for entry in os.environ.get("PATH", "").split(os.pathsep):
            candidates.append(os.path.join(entry, "libclang.dll"))
    elif system == "Darwin":
        candidates = [
            "/usr/local/opt/llvm/lib/libclang.dylib",
            "/opt/homebrew/opt/llvm/lib/libclang.dylib",
            "/Library/Developer/CommandLineTools/usr/lib/libclang.dylib",
        ]
        # Homebrew installs versioned dylibs; glob for them
        import glob
        candidates += glob.glob("/usr/local/opt/llvm*/lib/libclang.dylib")
        candidates += glob.glob("/opt/homebrew/opt/llvm*/lib/libclang.dylib")
    else:
        # Linux and other Unix-likes
        candidates = [
            "/usr/lib/llvm/libclang.so",
            "/usr/lib/libclang.so",
            "/usr/lib64/libclang.so",
        ]
        import glob
        # Versioned .so files (libclang-18.so.1, libclang.so.1, etc.)
        candidates += glob.glob("/usr/lib/llvm-*/lib/libclang-*.so*")
        candidates += glob.glob("/usr/lib/llvm-*/lib/libclang.so*")
        candidates += glob.glob("/usr/lib/x86_64-linux-gnu/libclang-*.so*")
        candidates += glob.glob("/usr/lib/x86_64-linux-gnu/libclang.so*")
        candidates += glob.glob("/usr/lib/aarch64-linux-gnu/libclang-*.so*")
        candidates += glob.glob("/usr/lib64/libclang*.so*")
        # Also check LD_LIBRARY_PATH
        for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
            if entry:
                candidates += glob.glob(os.path.join(entry, "libclang*.so*"))

    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def _load_config():
    """Load and return a CftConfig instance."""
    cfg_path = _find_config()
    cp = configparser.ConfigParser()
    if cfg_path:
        cp.read(cfg_path)

    # [cft] section
    libclang = cp.get("cft", "libclang", fallback=None)
    if libclang:
        cx.Config.set_library_file(libclang)
    else:
        found = _find_libclang()
        if found:
            cx.Config.set_library_file(found)

    # [cstdlib] section
    cstdlib_root = cp.get("cstdlib", "output_root", fallback=None)
    raw_roots = cp.get("cstdlib", "include_roots", fallback="")
    include_roots = [r.strip() for r in raw_roots.split(";") if r.strip()]

    # [clang] section
    raw_args = cp.get("clang", "args", fallback="")
    default_args = raw_args.split() if raw_args.strip() else ["-x", "c", "-std=c11"]

    return _CftConfig(
        cstdlib_root=cstdlib_root,
        include_roots=include_roots,
        default_clang_args=default_args,
    )

class _CftConfig:
    def __init__(self, cstdlib_root, include_roots, default_clang_args):
        self.cstdlib_root = cstdlib_root            # str or None
        self.include_roots = include_roots          # list[str]
        self.default_clang_args = default_clang_args  # list[str]

# Loaded once at import time so all helpers can reference it.
CFT_CONFIG = _load_config()

# ---------------------------------------------------------------------------
# Type mapping: C canonical type -> Flux type string
# ---------------------------------------------------------------------------

_TYPEKIND_MAP = {
    TypeKind.VOID:      "void",
    TypeKind.BOOL:      "bool",
    TypeKind.CHAR_U:    "byte",
    TypeKind.UCHAR:     "byte",
    TypeKind.CHAR_S:    "byte",
    TypeKind.SCHAR:     "byte",
    TypeKind.SHORT:     "int",
    TypeKind.USHORT:    "uint",
    TypeKind.INT:       "int",
    TypeKind.UINT:      "uint",
    TypeKind.LONG:      "long",
    TypeKind.ULONG:     "ulong",
    TypeKind.LONGLONG:  "long",
    TypeKind.ULONGLONG: "ulong",
    TypeKind.FLOAT:     "float",
    TypeKind.DOUBLE:    "double",
    TypeKind.LONGDOUBLE:"double",   # Flux has no 80-bit float; map to double
}

_TYPEDEF_MAP = {
    "size_t":    "ulong",
    "ssize_t":   "long",
    "ptrdiff_t": "long",
    "intptr_t":  "long",
    "uintptr_t": "ulong",
    "int8_t":    "byte",
    "uint8_t":   "byte",
    "int16_t":   "int",
    "uint16_t":  "uint",
    "int32_t":   "int",
    "uint32_t":  "uint",
    "int64_t":   "long",
    "uint64_t":  "ulong",
    "wchar_t":   "uint",
    "char16_t":  "uint",
    "char32_t":  "uint",
}

# C binary operator token -> Flux operator
_BINOP_MAP = {
    "+": "+", "-": "-", "*": "*", "/": "/", "%": "%",
    "==": "==", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">=",
    "&&": "&", "||": "|", "!": "!",
    "&": "`&", "|": "`|", "^": "`^^", "~": "`!",
    "<<": "<<", ">>": ">>",
    "=": "=", "+=": "+=", "-=": "-=", "*=": "*=", "/=": "/=", "%=": "%=",
    "&=": "`&=", "|=": "`|=", "^=": "`^^=", "<<=": "<<=", ">>=": ">>=",
}


# Flux reserved keywords that are valid C identifiers -- rename on translation
_RESERVED_RENAME = {
    "data": "dat",
    "from": "frm",
}

def _rename(name):
    """Rename C identifiers that are reserved keywords in Flux."""
    return _RESERVED_RENAME.get(name, name)


def _best_relative(abs_path, roots):
    """Return the relative path of abs_path under the longest matching root,
    or None if no root matches.

    roots is a list of absolute directory paths.
    """
    best_rel = None
    best_len = -1
    norm = os.path.normcase(abs_path)
    for root in roots:
        norm_root = os.path.normcase(root)
        if not norm_root.endswith(os.sep):
            norm_root += os.sep
        if norm.startswith(norm_root) and len(norm_root) > best_len:
            best_rel = abs_path[len(norm_root):]
            best_len = len(norm_root)
    return best_rel


class CTranslator:

    def __init__(self, filepath, args=None):
        self.filepath = filepath
        self.lines = []
        self._emitted_types = set()
        self._pending_typedefs = {}
        self._need_i128 = False
        self._need_u128 = False
        self._indent = 0
        self._switch_depth = 0  # incremented inside switch body
        self._loop_depth_in_switch_stack = []  # per-switch loop nesting depth
        self._index = cx.Index.create()
        clang_args = args or CFT_CONFIG.default_clang_args
        self.tu = self._index.parse(filepath, args=clang_args,
                                    options=cx.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
        with open(filepath, 'rb') as _f:
            self._src_bytes = _f.read()

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def _emit_toplevel_comments(self):
        """Emit any block or line comments that appear before the first declaration
        in the main file, converted to Flux comment syntax."""
        # Find the offset of the first cursor from the main file so we know
        # how far into the token stream to look.
        first_offset = None
        for cursor in self.tu.cursor.get_children():
            if self._is_from_main_file(cursor):
                first_offset = cursor.extent.start.offset
                break

        main_abs = os.path.abspath(self.filepath)
        for tok in self.tu.get_tokens(extent=self.tu.cursor.extent):
            if tok.kind != cx.TokenKind.COMMENT:
                continue
            tok_file = tok.location.file
            if tok_file is None:
                continue
            if os.path.abspath(tok_file.name) != main_abs:
                continue
            # Only include comments that precede the first declaration
            if first_offset is not None and tok.extent.start.offset >= first_offset:
                break
            text = tok.spelling
            if text.startswith("/*"):
                # Convert /* ... */ to Flux /// ... ///
                inner = text[2:]
                if inner.endswith("*/"):
                    inner = inner[:-2]
                self._emit("///")
                for line in inner.splitlines():
                    # Strip the leading " * " or " " that C block comments use
                    stripped = line.rstrip()
                    if stripped.startswith("   "):
                        stripped = stripped[3:]
                    elif stripped.startswith(" "):
                        stripped = stripped[1:]
                    self._emit(stripped)
                self._emit("///")
                self._emit("")
            elif text.startswith("//"):
                self._emit(text)
                self._emit("")

    def translate(self):
        self._emit(f"// Auto-generated by cft from: {os.path.basename(self.filepath)}")
        self._emit("// May require manual edits.")
        self._emit("")

        self._emit_toplevel_comments()
        self._collect_typedef_names(self.tu.cursor)

        for cursor in self.tu.cursor.get_children():
            if not self._is_from_main_file(cursor):
                continue
            self._visit_top(cursor)

        prefix = []
        if self._need_i128:
            prefix += [
                "// __int128 emulated as two paired 64-bit halves (mirrors GCC/Clang ABI)",
                "struct __int128_t { long lo; long hi; };",
                "",
            ]
        if self._need_u128:
            prefix += [
                "// __uint128_t emulated as two paired 64-bit halves (mirrors GCC/Clang ABI)",
                "struct __uint128_t { ulong lo; ulong hi; };",
                "",
            ]
        if prefix:
            self.lines = prefix + self.lines

        return "\n".join(self.lines)

    # -----------------------------------------------------------------------
    # Top-level dispatcher
    # -----------------------------------------------------------------------

    def _visit_top(self, cursor):
        kind = cursor.kind

        if kind == CursorKind.STRUCT_DECL:
            self._emit_struct(cursor)
        elif kind == CursorKind.UNION_DECL:
            self._emit_union(cursor)
        elif kind == CursorKind.ENUM_DECL:
            self._emit_enum(cursor)
        elif kind == CursorKind.TYPEDEF_DECL:
            self._emit_typedef(cursor)
        elif kind == CursorKind.FUNCTION_DECL:
            self._emit_function(cursor)
        elif kind == CursorKind.VAR_DECL:
            self._emit_global_var(cursor)
        elif kind == CursorKind.MACRO_DEFINITION:
            self._emit_macro(cursor)

    # -----------------------------------------------------------------------
    # Struct
    # -----------------------------------------------------------------------

    def _emit_struct(self, cursor):
        name = self._struct_name(cursor)
        if not name or name in self._emitted_types:
            return
        fields = list(cursor.get_children())
        if not fields:
            self._emit(f"struct {name};")
            self._emitted_types.add(name)
            return
        self._emitted_types.add(name)
        self._emit(f"struct {name}")
        self._emit("{")
        for field in fields:
            if field.kind == CursorKind.FIELD_DECL:
                ftype = self._flux_type(field.type, field.spelling)
                # Bitfield
                if field.is_bitfield():
                    width = field.get_bitfield_width()
                    # Emit as data{N} -- signed/unsigned inferred from underlying type
                    canon = field.type.get_canonical()
                    signed = canon.kind in (TypeKind.CHAR_S, TypeKind.SCHAR, TypeKind.SHORT,
                                            TypeKind.INT, TypeKind.LONG, TypeKind.LONGLONG)
                    prefix = "signed " if signed else ""
                    self._emit(f"    {prefix}data{{{width}}} {_rename(field.spelling)};")
                else:
                    # Check for _Alignas attribute
                    align_val = self._get_field_align(field)
                    if align_val is not None:
                        bits = field.type.get_size() * 8
                        canon = field.type.get_canonical()
                        signed = canon.kind in (TypeKind.CHAR_S, TypeKind.SCHAR, TypeKind.SHORT,
                                                TypeKind.INT, TypeKind.LONG, TypeKind.LONGLONG)
                        sign = "signed " if signed else ""
                        self._emit(f"    {sign}data{{{bits}}} {_rename(field.spelling)}; // _Alignas({align_val})")
                    else:
                        self._emit(f"    {ftype} {_rename(field.spelling)};")
        self._emit("};")
        self._emit("")

    # -----------------------------------------------------------------------
    # Union
    # -----------------------------------------------------------------------

    def _emit_union(self, cursor):
        name = cursor.spelling or self._pending_typedefs.get(cursor.hash)
        if not name or name in self._emitted_types:
            return
        fields = list(cursor.get_children())
        if not fields:
            return
        self._emitted_types.add(name)
        self._emit(f"union {name}")
        self._emit("{")
        for field in fields:
            if field.kind == CursorKind.FIELD_DECL:
                ftype = self._flux_type(field.type, field.spelling)
                self._emit(f"    {ftype} {_rename(field.spelling)};")
        self._emit("};")
        self._emit("")

    # -----------------------------------------------------------------------
    # Enum
    # -----------------------------------------------------------------------

    def _emit_enum(self, cursor):
        name = cursor.spelling or self._pending_typedefs.get(cursor.hash)
        enumerators = [c for c in cursor.get_children() if c.kind == CursorKind.ENUM_CONSTANT_DECL]
        if not enumerators:
            return

        # Anonymous enum used purely as integer constants -- emit each as a scalar
        if not name or '(unnamed' in name:
            for e in enumerators:
                val = e.enum_value
                if val < 0:
                    ftype = "long" if val < -(2**31) else "int"
                else:
                    ftype = "ulong" if val > 0xFFFFFFFF else "uint"
                self._emit(f"{ftype} {e.spelling} = {val};")
            self._emit("")
            return

        if name in self._emitted_types:
            return
        self._emitted_types.add(name)

        sequential = all(e.enum_value == i for i, e in enumerate(enumerators))

        self._emit(f"enum {name}")
        self._emit("{")
        parts = []
        for e in enumerators:
            if sequential:
                parts.append(f"    {e.spelling}")
            else:
                parts.append(f"    {e.spelling} = {e.enum_value}")
        self._emit(",\n".join(parts))
        self._emit("};")
        self._emit("")

    # -----------------------------------------------------------------------
    # Typedef
    # -----------------------------------------------------------------------

    def _emit_typedef(self, cursor):
        typedef_name = cursor.spelling
        underlying = cursor.underlying_typedef_type
        canon = underlying.get_canonical()

        if canon.kind in (TypeKind.RECORD, TypeKind.ELABORATED):
            decl = canon.get_declaration()
            if decl and decl.spelling == typedef_name:
                return
            if decl and decl.kind == CursorKind.STRUCT_DECL:
                self._emit_struct(decl)
            elif decl and decl.kind == CursorKind.ENUM_DECL:
                self._emit_enum(decl)
            elif decl and decl.kind == CursorKind.UNION_DECL:
                self._emit_union(decl)
            if decl and decl.spelling and decl.spelling != typedef_name:
                self._emit(f"// typedef alias: {typedef_name} -> {decl.spelling}")
                self._emit("")
            return

        flux = _TYPEDEF_MAP.get(typedef_name)
        if flux:
            return

        # Function pointer typedef -> emit as named cdecl{}* declaration
        if canon.kind in (TypeKind.FUNCTIONPROTO, TypeKind.FUNCTIONNOPROTO):
            ret_str = self._flux_type(canon.get_result(), "")
            params = []
            for arg_type in canon.argument_types():
                params.append(self._flux_type(arg_type, ""))
            if canon.kind == TypeKind.FUNCTIONPROTO and canon.is_function_variadic():
                params.append("...")
            self._emit(f"cdecl{{}}* {typedef_name}({', '.join(params)}) -> {ret_str};")
            self._emit("")
            return

        # Function pointer typedef via pointer-to-proto, or pointer-to-record alias
        if canon.kind == TypeKind.POINTER:
            pointee = canon.get_pointee()
            # Pointer to struct/interface -- emit as `PointeeName* as TypedefName;`
            if pointee.kind in (TypeKind.RECORD, TypeKind.ELABORATED):
                decl = pointee.get_declaration()
                pointee_name = decl.spelling if decl and decl.spelling else pointee.spelling
                if pointee_name and pointee_name != typedef_name:
                    self._emit(f"{pointee_name}* as {typedef_name};")
                    self._emit("")
                    return
            if pointee.kind in (TypeKind.FUNCTIONPROTO, TypeKind.FUNCTIONNOPROTO):
                ret_str = self._flux_type(pointee.get_result(), "")
                # Collect param names from cursor children (PARM_DECL)
                parm_names = [_rename(c.spelling) for c in cursor.get_children()
                              if c.kind == CursorKind.PARM_DECL]
                params = []
                for i, arg_type in enumerate(pointee.argument_types()):
                    ptype = self._flux_type(arg_type, "")
                    pname = parm_names[i] if i < len(parm_names) else ""
                    params.append(f"{ptype} {pname}".strip())
                if pointee.kind == TypeKind.FUNCTIONPROTO and pointee.is_function_variadic():
                    params.append("...")
                self._emit(f"cdecl{{}}* {typedef_name}({', '.join(params)}) -> {ret_str};")
                self._emit("")
                return

        inner = self._flux_type(underlying, typedef_name)
        self._emit(f"// typedef: {typedef_name} = {inner}")

    # -----------------------------------------------------------------------
    # Function (prototype or full definition)
    # -----------------------------------------------------------------------

    def _emit_function(self, cursor):
        name = cursor.spelling
        if not name or name in self._emitted_types:
            return
        self._emitted_types.add(name)

        ft = cursor.type
        ret = ft.get_result()
        noreturn = cursor.is_noreturn_function() if hasattr(cursor, "is_noreturn_function") else False
        ret_str = "void" if noreturn else self._flux_type(ret, "")

        params = []
        for arg in cursor.get_arguments():
            ptype = self._flux_type(arg.type, arg.spelling)
            pname = _rename(arg.spelling) if arg.spelling else ""
            params.append(f"{ptype} {pname}".strip())

        if ft.kind == TypeKind.FUNCTIONPROTO and ft.is_function_variadic():
            params.append("...")

        param_str = ", ".join(params)
        noreturn_comment = " // noreturn" if noreturn else ""

        # Check if this has a body
        body = None
        for child in cursor.get_children():
            if child.kind == CursorKind.COMPOUND_STMT:
                body = child
                break

        if body is None:
            self._emit(f"cdecl {name}({param_str}) -> {ret_str};{noreturn_comment}")
        else:
            self._emit(f"cdecl {name}({param_str}) -> {ret_str}{noreturn_comment}")
            self._emit_compound(body)
            self._emit("")

    # -----------------------------------------------------------------------
    # Global variable
    # -----------------------------------------------------------------------

    def _emit_global_var(self, cursor):
        name = cursor.spelling
        if not name:
            return
        ftype = self._flux_type(cursor.type, name)
        # Check for initializer
        children = list(cursor.get_children())
        init = children[0] if children else None
        if init:
            val = self._emit_expr(init)
            self._emit(f"{ftype} {name} = {val};")
        else:
            self._emit(f"extern {ftype} {name};")

    def _emit_codegen_macro(self, name, params, body_toks):
        """Translate a ##-pasting code-generating macro into a comptime/emitflux block.
        Returns True if successfully translated, False if the pattern is unrecognized."""

        def tok_to_flux(t, next_t=None):
            """Convert a single token, handling ## paste and & address-of."""
            if t.spelling == '##':
                return None  # handled by caller
            if t.spelling == '&':
                return '@'
            return t.spelling

        # Reconstruct token list replacing prefix##param with {param} interpolation
        def rebuild_toks(toks):
            result = []
            i = 0
            while i < len(toks):
                tok = toks[i]
                s = tok.spelling
                if tok.kind == cx.TokenKind.COMMENT:
                    converted = s.replace('/*', '///').replace('*/', '///')
                    result.append(converted)
                    i += 1
                    continue
                if i + 2 < len(toks) and toks[i+1].spelling == '##':
                    # prefix##param or ##param (prefix may be empty string token)
                    param = toks[i+2].spelling
                    result.append(f"{s}{{{param}}}")
                    i += 3
                elif s == '##' and i + 1 < len(toks):
                    # standalone ## param (no prefix)
                    param = toks[i+1].spelling
                    result.append(f"{{{param}}}")
                    i += 2
                elif s == '&':
                    result.append('@')
                    i += 1
                else:
                    result.append(s)
                    i += 1
            return result

        # Split body tokens into function definitions at top-level { } boundaries
        # Each function is: [qualifiers] ret_type name ( params ) { body }
        functions = []
        i = 0
        toks = body_toks
        while i < len(toks):
            # Skip comments and qualifiers like ATTRIBUTE_PURE, static
            # Find: ret_type name ( param_list ) { body }
            # Look for '(' to identify start of param list
            fn_start = i
            # Scan forward to find '{'
            brace_start = None
            j = i
            while j < len(toks):
                if toks[j].spelling == '{':
                    brace_start = j
                    break
                j += 1
            if brace_start is None:
                break
            # Find matching '}'
            depth = 0
            brace_end = None
            for k in range(brace_start, len(toks)):
                if toks[k].spelling == '{':
                    depth += 1
                elif toks[k].spelling == '}':
                    depth -= 1
                    if depth == 0:
                        brace_end = k
                        break
            if brace_end is None:
                break

            signature_toks = toks[fn_start:brace_start]
            body_inner_toks = toks[brace_start+1:brace_end]
            functions.append((signature_toks, body_inner_toks))
            i = brace_end + 1

        if not functions:
            return False

        # Build emitflux string lines
        lines = []
        for sig_toks, body_inner_toks in functions:
            sig_parts = rebuild_toks(sig_toks)
            body_parts = rebuild_toks(body_inner_toks)

            # Parse signature: find '(' and ')' to split ret+name from params
            sig_str = " ".join(sig_parts)
            # Find the param '(' -- last one before end of sig
            paren_depth = 0
            paren_start = None
            paren_end = None
            for idx, ch in enumerate(sig_str):
                if ch == '(':
                    if paren_depth == 0:
                        paren_start = idx
                    paren_depth += 1
                elif ch == ')':
                    paren_depth -= 1
                    if paren_depth == 0:
                        paren_end = idx
                        break
            if paren_start is None:
                continue

            ret_and_name = sig_str[:paren_start].strip()
            fn_params = sig_str[paren_start+1:paren_end].strip()
            # Strip qualifiers: static, ATTRIBUTE_PURE, inline, etc.
            for qual in ('static ', 'ATTRIBUTE_PURE ', 'inline ', 'extern '):
                ret_and_name = ret_and_name.replace(qual, '')
            ret_and_name = ret_and_name.strip()

            # Split ret type from function name: last word is the name
            parts = ret_and_name.rsplit(None, 1)
            if len(parts) == 2:
                ret_type, fn_name = parts
            else:
                ret_type = 'void'
                fn_name = parts[0]

            body_str = " ".join(body_parts).strip()
            # Strip leading/trailing semicolons from body statements for clean emit
            fn_line = f"        cdecl {fn_name}({fn_params}) -> {ret_type} {{ {body_str} }};"
            lines.append(fn_line)

        if not lines:
            return False

        param_str = ", ".join(params)
        self._emit(f"// code-generating macro: call with ({param_str})")
        self._emit(f"comptime")
        self._emit("{")
        self._emit(f"    emitflux")
        self._emit(f"    {{")
        self._emit(f"~$f\"")
        for line in lines:
            self._emit(line)
        self._emit(f"\";")
        self._emit(f"    }};")
        self._emit("};")
        self._emit("")
        return True

    def _translate_macro_body(self, toks):
        """Translate a sequence of macro body tokens to Flux syntax.
        Key transforms: unary & -> @, binary & -> `&, ## paste -> {param}, /* */ -> /// ///"""
        result = []
        i = 0
        while i < len(toks):
            tok = toks[i]
            s = tok.spelling
            if tok.kind == cx.TokenKind.COMMENT:
                result.append(s.replace('/*', '///').replace('*/', '///'))
                i += 1
                continue
            if s == '##':
                # ## param: merge previous token with next as {next}
                if result and i + 1 < len(toks):
                    prev = result.pop()
                    param = toks[i + 1].spelling
                    result.append(f"{prev}{{{param}}}")
                    i += 2
                    continue
            if s == '&':
                prev = result[-1].strip() if result else ''
                if not prev or prev in ('(', ',', '=', '+', '-', '*', '/', '%',
                                        '!', '~', '|', '^', '<', '>', '&', '?', ':'):
                    result.append('@')
                else:
                    result.append('`&')
                i += 1
                continue
            if s == '~':
                result.append('`!')
                i += 1
                continue
            result.append(s)
            i += 1
        return " ".join(result).strip()

    # -----------------------------------------------------------------------
    # Macro
    # -----------------------------------------------------------------------

    def _emit_macro(self, cursor):
        name = cursor.spelling
        if not name or name.startswith("_"):
            return
        tokens = list(cursor.get_tokens())
        body_tokens = tokens[1:]
        if not body_tokens:
            return
        body = " ".join(t.spelling for t in body_tokens).strip()
        if not body:
            return

        # Detect function-like macros: '(' immediately follows the name with no gap
        is_functionlike = (len(tokens) >= 2 and
                           tokens[1].spelling == '(' and
                           tokens[0].extent.end.offset == tokens[1].extent.start.offset)

        if is_functionlike:
            # Parse: NAME ( param1, param2, ... ) body...
            # tokens[0] is the macro name, tokens[1] is '('
            params = []
            i = 2  # skip name and '('
            while i < len(tokens) and tokens[i].spelling != ')':
                if tokens[i].spelling != ',':
                    params.append(tokens[i].spelling)
                i += 1
            i += 1  # skip ')'
            body_toks = tokens[i:]
            body = self._translate_macro_body(body_toks)
            if not body:
                return
            # Multi-statement macros: check if it's a code-generating ## paste macro
            if ';' in body or (body.startswith('do') and 'while' in body):
                if '##' in body:
                    result = self._emit_codegen_macro(name, params, body_toks)
                    if result:
                        return
                self._emit(f"// macro (multi-statement, manual translation needed): {name}({', '.join(params)}) {body}")
                return
            param_str = ", ".join(params)
            self._emit(f"macro {name}({param_str})")
            self._emit("{")
            self._emit(f"    {body}")
            self._emit("};")
            self._emit("")
            return

        try:
            val = int(body, 0)
            if val < 0:
                if val >= -(2**31):
                    self._emit(f"int {name} = {val};")
                else:
                    self._emit(f"long {name} = {val};")
            else:
                if val <= 0xFFFFFFFF:
                    self._emit(f"uint {name} = {val};")
                else:
                    self._emit(f"ulong {name} = {val};")
            return
        except ValueError:
            pass

        try:
            float(body)
            self._emit(f"double {name} = {body};")
            return
        except ValueError:
            pass

        if body.startswith('"'):
            self._emit(f"byte* {name} = {body};")
            return

        # Multi-statement object-like macro -- comment
        if ';' in body or (body.startswith('do') and 'while' in body):
            self._emit(f"// macro (multi-statement, manual translation needed): {name} {body}")
            return

        # Expression object-like macro -- emit as parameterless macro block
        translated = self._translate_macro_body(body_tokens)
        self._emit(f"macro {name}")
        self._emit("{")
        self._emit(f"    {translated}")
        self._emit("};")
        self._emit("")

    # -----------------------------------------------------------------------
    # Statement emission
    # -----------------------------------------------------------------------

    def _emit_compound(self, cursor):
        pad = "    " * self._indent
        self._emit(f"{pad}{{")
        self._indent += 1
        for child in cursor.get_children():
            self._emit_stmt(child)
        self._indent -= 1
        self._emit(f"{pad}}};")

    def _emit_stmt(self, cursor):
        pad = "    " * self._indent
        kind = cursor.kind

        if kind == CursorKind.COMPOUND_STMT:
            self._emit_compound(cursor)

        elif kind == CursorKind.RETURN_STMT:
            children = list(cursor.get_children())
            if children:
                val = self._emit_expr(children[0])
                self._emit(f"{pad}return {val};")
            else:
                self._emit(f"{pad}return void;")

        elif kind == CursorKind.DECL_STMT:
            for child in cursor.get_children():
                if child.kind == CursorKind.VAR_DECL:
                    self._emit_local_var(child)

        elif kind == CursorKind.IF_STMT:
            self._emit_if(cursor)

        elif kind == CursorKind.FOR_STMT:
            self._emit_for(cursor)

        elif kind == CursorKind.WHILE_STMT:
            self._emit_while(cursor)

        elif kind == CursorKind.DO_STMT:
            self._emit_do_while(cursor)

        elif kind == CursorKind.SWITCH_STMT:
            self._emit_switch(cursor)

        elif kind == CursorKind.CASE_STMT:
            # Should be handled by _emit_switch_body_flat; fallback if encountered standalone
            self._emit_case(cursor, [])

        elif kind == CursorKind.DEFAULT_STMT:
            # Should be handled by _emit_switch_body_flat; fallback if encountered standalone
            self._emit_default(cursor, [])

        elif kind == CursorKind.BREAK_STMT:
            # In Flux, plain break exits loops. To exit a switch, use 'break switch;'.
            if self._switch_depth == 0 or (self._loop_depth_in_switch_stack and self._loop_depth_in_switch_stack[-1] > 0):
                self._emit(f"{pad}break;")
            else:
                self._emit(f"{pad}break switch;")

        elif kind == CursorKind.CONTINUE_STMT:
            self._emit(f"{pad}continue;")

        elif kind == CursorKind.GOTO_STMT:
            children = list(cursor.get_children())
            label = children[0].spelling if children else "?"
            self._emit(f"{pad}goto {label};")

        elif kind == CursorKind.LABEL_STMT:
            children = list(cursor.get_children())
            self._emit(f"{pad}label {cursor.spelling}:")
            if children:
                self._emit_stmt(children[0])

        elif kind == CursorKind.NULL_STMT:
            pass

        else:
            # Expression statement
            expr = self._emit_expr(cursor)
            if expr:
                self._emit(f"{pad}{expr};")

    def _emit_local_var(self, cursor):
        pad = "    " * self._indent
        ftype = self._flux_type(cursor.type, cursor.spelling)
        children = list(cursor.get_children())
        init = None
        for child in children:
            if child.kind not in (CursorKind.TYPE_REF, CursorKind.TEMPLATE_REF):
                init = child
                break
        vname = _rename(cursor.spelling)
        if init:
            val = self._emit_expr(init)
            self._emit(f"{pad}{ftype} {vname} = {val};")
        else:
            self._emit(f"{pad}{ftype} {vname};")

    def _emit_if(self, cursor):
        pad = "    " * self._indent
        children = list(cursor.get_children())
        # children: cond, then, [else]
        cond = self._emit_expr(children[0])
        self._emit(f"{pad}if ({cond})")
        if children[1].kind == CursorKind.COMPOUND_STMT:
            self._emit_compound(children[1])
        else:
            self._indent += 1
            self._emit_stmt(children[1])
            self._indent -= 1
        if len(children) > 2:
            else_child = children[2]
            # elif chain
            if else_child.kind == CursorKind.IF_STMT:
                # rewrite the closing }; as } and emit elif
                # pop the trailing }; we just emitted and replace with elif
                if self.lines and self.lines[-1].rstrip().endswith("};"):
                    self.lines[-1] = self.lines[-1].rstrip()[:-1]  # strip trailing ;
                self._emit(f"{pad}elif ({self._emit_expr(list(else_child.get_children())[0])})")
                sub = list(else_child.get_children())
                if sub[1].kind == CursorKind.COMPOUND_STMT:
                    self._emit_compound(sub[1])
                else:
                    self._indent += 1
                    self._emit_stmt(sub[1])
                    self._indent -= 1
                if len(sub) > 2:
                    self._emit(f"{pad}else")
                    if sub[2].kind == CursorKind.COMPOUND_STMT:
                        self._emit_compound(sub[2])
                    else:
                        self._indent += 1
                        self._emit_stmt(sub[2])
                        self._indent -= 1
            else:
                if self.lines and self.lines[-1].rstrip().endswith("};"):
                    self.lines[-1] = self.lines[-1].rstrip()[:-1]
                self._emit(f"{pad}else")
                if else_child.kind == CursorKind.COMPOUND_STMT:
                    self._emit_compound(else_child)
                else:
                    self._indent += 1
                    self._emit_stmt(else_child)
                    self._indent -= 1

    def _emit_for(self, cursor):
        pad = "    " * self._indent
        children = list(cursor.get_children())
        # For stmt children in clang: init, cond, inc, body (any may be absent)
        # libclang exposes them positionally; absent parts are NullStmt or missing
        # Use token-based reconstruction for the header since absent parts are hard to detect
        init_str = cond_str = inc_str = ""
        body = None
        parts = []
        for child in children:
            if child.kind == CursorKind.COMPOUND_STMT:
                body = child
            else:
                parts.append(child)

        if len(parts) >= 1 and parts[0].kind != CursorKind.NULL_STMT:
            if parts[0].kind == CursorKind.DECL_STMT:
                decl = list(parts[0].get_children())[0]
                ftype = self._flux_type(decl.type, decl.spelling)
                decl_children = [c for c in decl.get_children()
                                 if c.kind not in (CursorKind.TYPE_REF, CursorKind.TEMPLATE_REF)]
                if decl_children:
                    init_str = f"{ftype} {decl.spelling} = {self._emit_expr(decl_children[0])}"
                else:
                    init_str = f"{ftype} {decl.spelling}"
            else:
                init_str = self._emit_expr(parts[0])
        if len(parts) >= 2 and parts[1].kind != CursorKind.NULL_STMT:
            cond_str = self._emit_expr(parts[1])
        if len(parts) >= 3 and parts[2].kind != CursorKind.NULL_STMT:
            inc_str = self._emit_expr(parts[2])

        self._emit(f"{pad}for ({init_str}; {cond_str}; {inc_str})")
        if self._loop_depth_in_switch_stack: self._loop_depth_in_switch_stack[-1] += 1
        if body:
            self._emit_compound(body)
        else:
            self._emit(f"{pad}{{}};")
        if self._loop_depth_in_switch_stack: self._loop_depth_in_switch_stack[-1] -= 1

    def _emit_while(self, cursor):
        pad = "    " * self._indent
        children = list(cursor.get_children())
        cond_cursor = children[0]
        # while(1) / while(true) -- libclang folds 'true' to integer 1
        if cond_cursor.kind == CursorKind.INTEGER_LITERAL:
            toks = list(cond_cursor.get_tokens())
            raw = toks[0].spelling if toks else "0"
            cond = "true" if raw == "1" else raw
        else:
            cond = self._emit_expr(cond_cursor)
        self._emit(f"{pad}while ({cond})")
        if self._loop_depth_in_switch_stack: self._loop_depth_in_switch_stack[-1] += 1
        if len(children) > 1:
            if children[1].kind == CursorKind.COMPOUND_STMT:
                self._emit_compound(children[1])
            else:
                self._indent += 1
                self._emit_stmt(children[1])
                self._indent -= 1
        else:
            self._emit(f"{pad}{{}};")
        if self._loop_depth_in_switch_stack: self._loop_depth_in_switch_stack[-1] -= 1

    def _emit_do_while(self, cursor):
        pad = "    " * self._indent
        children = list(cursor.get_children())
        body = children[0]
        cond = self._emit_expr(children[1])
        self._emit(f"{pad}do")
        if self._loop_depth_in_switch_stack: self._loop_depth_in_switch_stack[-1] += 1
        if body.kind == CursorKind.COMPOUND_STMT:
            self._emit_compound(body)
        else:
            self._indent += 1
            self._emit_stmt(body)
            self._indent -= 1
        if self._loop_depth_in_switch_stack: self._loop_depth_in_switch_stack[-1] -= 1
        # Replace trailing }; with } for do/while continuation
        if self.lines and self.lines[-1].rstrip().endswith("};"):
            self.lines[-1] = self.lines[-1].rstrip()[:-1]
        self._emit(f"{pad}while ({cond});")

    def _emit_switch(self, cursor):
        pad = "    " * self._indent
        children = list(cursor.get_children())
        cond = self._emit_expr(children[0])

        self._switch_depth += 1
        self._loop_depth_in_switch_stack.append(0)

        self._emit(f"{pad}switch ({cond})")
        self._emit(f"{pad}{{")
        self._indent += 1
        if len(children) > 1:
            self._emit_switch_body_flat(children[1])
        self._indent -= 1
        self._emit(f"{pad}}};")

        self._loop_depth_in_switch_stack.pop()
        self._switch_depth -= 1

    def _emit_case(self, cursor, body_stmts):
        """Emit a case block. body_stmts is the list of sibling statements
        that belong to this case (collected by _emit_switch_body_flat)."""
        pad = "    " * self._indent
        children = list(cursor.get_children())
        val = self._emit_expr(children[0])
        self._emit(f"{pad}case ({val})")
        self._emit(f"{pad}{{")
        self._indent += 1
        # child[1] of the CASE_STMT itself is the first inline stmt (if any)
        if len(children) > 1:
            first = children[1]
            if first.kind not in (CursorKind.CASE_STMT, CursorKind.DEFAULT_STMT,
                                  CursorKind.BREAK_STMT, CursorKind.NULL_STMT):
                self._emit_stmt(first)
        for stmt in body_stmts:
            self._emit_stmt(stmt)
        self._indent -= 1
        self._emit(f"{pad}}}")

    def _emit_default(self, cursor, body_stmts):
        pad = "    " * self._indent
        self._emit(f"{pad}default")
        self._emit(f"{pad}{{")
        self._indent += 1
        for stmt in body_stmts:
            self._emit_stmt(stmt)
        self._indent -= 1
        self._emit(f"{pad}}};")

    def _emit_switch_body_flat(self, body_cursor):
        """Iterate the flat list of children inside a switch compound body,
        grouping non-case siblings into the preceding case's body."""
        if body_cursor.kind != CursorKind.COMPOUND_STMT:
            self._emit_stmt(body_cursor)
            return

        children = list(body_cursor.get_children())

        # Group into (case_cursor, [body_stmts]) entries
        groups = []  # list of (cursor, [stmts], is_break)
        current_case = None
        current_body = []
        current_break = False

        for child in children:
            if child.kind in (CursorKind.CASE_STMT, CursorKind.DEFAULT_STMT):
                if current_case is not None:
                    groups.append((current_case, current_body, current_break))
                current_case = child
                current_body = []
                current_break = False
            elif child.kind == CursorKind.BREAK_STMT:
                current_break = True
                # close current case
                if current_case is not None:
                    groups.append((current_case, current_body, True))
                    current_case = None
                    current_body = []
                    current_break = False
            else:
                if current_case is not None:
                    current_body.append(child)
                # else: orphan stmt before first case -- emit directly
                else:
                    self._emit_stmt(child)

        # flush last group (no trailing break -- default often has none)
        if current_case is not None:
            groups.append((current_case, current_body, current_break))

        for case_cursor, body_stmts, has_break in groups:
            if case_cursor.kind == CursorKind.DEFAULT_STMT:
                self._emit_default(case_cursor, body_stmts)
            else:
                pad = "    " * self._indent
                children_c = list(case_cursor.get_children())
                val = self._emit_expr(children_c[0])
                self._emit(f"{pad}case ({val})")
                self._emit(f"{pad}{{")
                self._indent += 1
                # child[1] of CASE_STMT is the first inline body statement
                if len(children_c) > 1:
                    first = children_c[1]
                    if first.kind not in (CursorKind.CASE_STMT, CursorKind.DEFAULT_STMT,
                                          CursorKind.BREAK_STMT, CursorKind.NULL_STMT):
                        self._emit_stmt(first)
                for stmt in body_stmts:
                    self._emit_stmt(stmt)
                if has_break:
                    self._emit(f"{'    ' * self._indent}break switch;")
                self._indent -= 1
                self._emit(f"{pad}}}")

    def _norm(self, s):
        """Normalise a raw C expression string to Flux conventions."""
        s = s.replace("->", ".")
        # Bitwise NOT: ~ -> `!  (must be preceded by non-identifier, i.e. operator context)
        import re as _re
        s = _re.sub(r'(?<![a-zA-Z0-9_])~', '`!', s)
        # Uppercase hex literals
        s = _re.sub(r'0x([0-9a-fA-F]+)', lambda m: "0x" + m.group(1).upper(), s)
        return s

    def _src_slice(self, cursor):
        """Return the raw source text for a cursor's extent, or None if
        the extent is not from the main file."""
        ext = cursor.extent
        start_file = ext.start.file
        if not start_file:
            return None
        if os.path.abspath(start_file.name) != os.path.abspath(self.filepath):
            return None
        raw = self._src_bytes[ext.start.offset:ext.end.offset].decode('utf-8', errors='replace').strip()
        # Flux uses . for all member access including pointer dereference
        return self._norm(raw)

    def _macro_source_spelling(self, cursor):
        """If the cursor's extent in the source file is a single macro invocation
        (i.e. the raw tokens spell a macro name followed by optional parens),
        return the raw source text. Otherwise return None."""
        tokens = list(cursor.get_tokens())
        if not tokens:
            return None
        first = tokens[0]
        if first.kind != cx.TokenKind.IDENTIFIER:
            return None
        raw = " ".join(t.spelling for t in tokens)
        return raw

    # -----------------------------------------------------------------------
    # Expression emission -- returns a string
    # -----------------------------------------------------------------------

    def _emit_expr(self, cursor):
        kind = cursor.kind

        if kind == CursorKind.INTEGER_LITERAL:
            tokens = list(cursor.get_tokens())
            spelling = tokens[0].spelling if tokens else "0"
            # Uppercase hex literals: 0xdeadbeef -> 0xDEADBEEF
            if spelling.lower().startswith("0x"):
                spelling = "0x" + spelling[2:].upper()
            return spelling

        if kind == CursorKind.FLOATING_LITERAL:
            tokens = list(cursor.get_tokens())
            return tokens[0].spelling if tokens else "0.0"

        if kind == CursorKind.STRING_LITERAL:
            tokens = list(cursor.get_tokens())
            return tokens[0].spelling if tokens else '""'

        if kind == CursorKind.CHARACTER_LITERAL:
            tokens = list(cursor.get_tokens())
            return tokens[0].spelling if tokens else "'\\0'"

        if kind == CursorKind.DECL_REF_EXPR:
            return _rename(cursor.spelling)

        if kind == CursorKind.MEMBER_REF_EXPR:
            children = list(cursor.get_children())
            base = self._emit_expr(children[0]) if children else "?"
            return f"{base}.{cursor.spelling}"

        if kind == CursorKind.CALL_EXPR:
            children = list(cursor.get_children())
            if not children:
                return f"{cursor.spelling}()"
            fn = self._emit_expr(children[0])
            args = [self._emit_expr(c) for c in children[1:]]
            return f"{fn}({', '.join(args)})"

        if kind == CursorKind.UNARY_OPERATOR:
            children = list(cursor.get_children())
            operand = self._emit_expr(children[0]) if children else "?"
            tokens = list(cursor.get_tokens())
            # Find the operator token (not part of the operand)
            op = tokens[0].spelling if tokens else "?"
            # Postfix ops: ++ and -- can be postfix
            # Detect by checking if op token comes after operand tokens
            operand_tokens = list(children[0].get_tokens()) if children else []
            op_spellings = [t.spelling for t in tokens]
            operand_spellings = [t.spelling for t in operand_tokens]
            # Operator is whichever token is not in the operand
            diff = [s for s in op_spellings if s not in operand_spellings]
            op = diff[0] if diff else "?"
            postfix = op in ("++", "--") and op_spellings and op_spellings[-1] == op
            if op == "*":
                return f"*{operand}"
            if op == "&":
                return f"@{operand}"
            if op == "!":
                return f"!{operand}"
            if op == "-":
                return f"-{operand}"
            if op == "~":
                return f"`!{operand}"
            if postfix:
                return f"{operand}{op}"
            return f"{op}{operand}"

        if kind == CursorKind.BINARY_OPERATOR:
            children = list(cursor.get_children())
            lhs = self._emit_expr(children[0]) if len(children) > 0 else "?"
            rhs_cursor = children[1] if len(children) > 1 else None
            if rhs_cursor is not None:
                rhs = self._src_slice(rhs_cursor) or self._emit_expr(rhs_cursor)
            else:
                rhs = "?"
            c_op = self._extract_binary_op(cursor, children)
            flux_op = _BINOP_MAP.get(c_op, c_op)
            return f"{lhs} {flux_op} {rhs}"

        if kind == CursorKind.COMPOUND_ASSIGNMENT_OPERATOR:
            children = list(cursor.get_children())
            lhs = self._emit_expr(children[0]) if len(children) > 0 else "?"
            rhs_cursor = children[1] if len(children) > 1 else None
            if rhs_cursor is not None:
                rhs = self._src_slice(rhs_cursor) or self._emit_expr(rhs_cursor)
            else:
                rhs = "?"
            c_op = self._extract_binary_op(cursor, children)
            flux_op = _BINOP_MAP.get(c_op, c_op)
            return f"{lhs} {flux_op} {rhs}"

        if kind == CursorKind.CONDITIONAL_OPERATOR:
            children = list(cursor.get_children())
            raw = self._src_slice(cursor)
            if raw:
                return raw
            cond = self._emit_expr(children[0])
            then = self._emit_expr(children[1])
            else_ = self._emit_expr(children[2])
            return f"{cond} ? {then} : {else_}"

        if kind == CursorKind.CSTYLE_CAST_EXPR:
            children = list(cursor.get_children())
            ftype = self._flux_type(cursor.type, "")
            operand = self._emit_expr(children[-1]) if children else "?"
            return f"({ftype}){operand}"

        if kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
            children = list(cursor.get_children())
            base = self._emit_expr(children[0])
            idx = self._emit_expr(children[1])
            return f"{base}[{idx}]"

        if kind == CursorKind.PAREN_EXPR:
            children = list(cursor.get_children())
            inner = self._emit_expr(children[0]) if children else "?"
            return f"({inner})"

        if kind == CursorKind.INIT_LIST_EXPR:
            children = list(cursor.get_children())
            parts = [self._emit_expr(c) for c in children]
            return "{" + ", ".join(parts) + "}"

        if kind == CursorKind.NULL_STMT:
            return ""

        if kind in (CursorKind.UNEXPOSED_EXPR, CursorKind.UNEXPOSED_STMT):
            children = list(cursor.get_children())
            if children:
                return self._emit_expr(children[0])
            if cursor.spelling:
                return cursor.spelling
            tokens = list(cursor.get_tokens())
            if tokens:
                return " ".join(t.spelling for t in tokens)
            return "?"

        if kind == CursorKind.ADDR_LABEL_EXPR:
            return f"@{cursor.spelling}"

        if kind == CursorKind.CXX_UNARY_EXPR:
            tokens = list(cursor.get_tokens())
            raw = " ".join(t.spelling for t in tokens)
            # Flux sizeof returns bits, C sizeof returns bytes -- divide by 8
            return f"({raw} / 8)"

        # Fallback: reconstruct from tokens
        tokens = list(cursor.get_tokens())
        if tokens:
            return self._norm(" ".join(t.spelling for t in tokens))
        return f"/* untranslated expr: {kind.name} */"

    # -----------------------------------------------------------------------
    # Type conversion
    # -----------------------------------------------------------------------

    def _flux_type(self, ctype, hint=""):
        kind = ctype.kind

        # _Alignas / aligned types: only applies to scalar integer types
        _SCALAR_KINDS = (TypeKind.CHAR_U, TypeKind.UCHAR, TypeKind.CHAR_S, TypeKind.SCHAR,
                         TypeKind.SHORT, TypeKind.USHORT, TypeKind.INT, TypeKind.UINT,
                         TypeKind.LONG, TypeKind.ULONG, TypeKind.LONGLONG, TypeKind.ULONGLONG)
        if kind in _SCALAR_KINDS:
            align = ctype.get_align()
            natural = self._natural_align(ctype)
            if align > 0 and natural > 0 and align != natural:
                bits = ctype.get_size() * 8
                signed_kinds = (TypeKind.CHAR_S, TypeKind.SCHAR, TypeKind.SHORT,
                                TypeKind.INT, TypeKind.LONG, TypeKind.LONGLONG)
                sign = "signed " if kind in signed_kinds else ""
                return f"{sign}data{{{bits}}}"

        if kind == TypeKind.POINTER:
            pointee = ctype.get_pointee()
            if pointee.kind in (TypeKind.FUNCTIONPROTO, TypeKind.FUNCTIONNOPROTO):
                return self._flux_funcptr_type(pointee)
            inner = self._flux_type(pointee, "")
            return f"{inner}*"

        if kind == TypeKind.CONSTANTARRAY:
            elem = self._flux_type(ctype.element_type, "")
            count = ctype.element_count
            return f"{elem}[{count}]"

        if kind == TypeKind.INCOMPLETEARRAY:
            elem = self._flux_type(ctype.element_type, "")
            return f"{elem}*"

        if kind == TypeKind.RECORD:
            decl = ctype.get_declaration()
            name = self._struct_name(decl)
            return name or "void*"

        if kind == TypeKind.ENUM:
            decl = ctype.get_declaration()
            return decl.spelling or "uint"

        if kind in (TypeKind.ELABORATED, TypeKind.TYPEDEF):
            spelling = ctype.spelling
            base = spelling.replace("const ", "").replace("volatile ", "").replace("struct ", "").strip()
            if base in _TYPEDEF_MAP:
                return _TYPEDEF_MAP[base]
            canon = ctype.get_canonical()
            if canon.kind != kind:
                return self._flux_type(canon, hint)
            return base or "void*"

        if kind == TypeKind.VOID:
            return "void"

        if kind == TypeKind.INT128:
            self._need_i128 = True
            return "__int128_t"

        if kind == TypeKind.UINT128:
            self._need_u128 = True
            return "__uint128_t"

        if kind in _TYPEKIND_MAP:
            return _TYPEKIND_MAP[kind]

        if kind == TypeKind.UNEXPOSED:
            canon = ctype.get_canonical()
            if canon.kind != TypeKind.UNEXPOSED:
                return self._flux_type(canon, hint)

        return f"void* /* untranslated: {ctype.spelling} */"

    def _natural_align(self, ctype):
        """Return the natural alignment in bytes for a type, or 0 if unknown."""
        try:
            kind = ctype.get_canonical().kind
            size = ctype.get_size()
            if size > 0:
                return min(size, 8)
        except Exception:
            pass
        return 0

    def _flux_funcptr_type(self, proto):
        ret = self._flux_type(proto.get_result(), "")
        params = []
        if proto.kind == TypeKind.FUNCTIONPROTO:
            for arg_type in proto.argument_types():
                params.append(self._flux_type(arg_type, ""))
            if proto.is_function_variadic():
                params.append("...")
        return f"def{{}}*({', '.join(params)}) -> {ret}"

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _extract_binary_op(self, cursor, children):
        """Find the binary operator token by locating the token that falls
        between the end of the lhs extent and the start of the rhs extent."""
        if len(children) < 2:
            return "?"
        lhs_end = children[0].extent.end.offset
        rhs_start = children[1].extent.start.offset
        for tok in cursor.get_tokens():
            off = tok.extent.start.offset
            if lhs_end <= off < rhs_start:
                return tok.spelling
        return "?"

    def _get_field_align(self, field_cursor):
        """Return the _Alignas value in bytes if present, else None."""
        for child in field_cursor.get_children():
            if child.kind == CursorKind.ALIGNED_ATTR:
                tokens = list(field_cursor.get_tokens())
                # Tokens look like: _Alignas ( 16 ) <type> <name>
                for i, t in enumerate(tokens):
                    if t.spelling == "_Alignas" and i + 2 < len(tokens):
                        try:
                            return int(tokens[i + 2].spelling)
                        except ValueError:
                            pass
        return None

    def _struct_name(self, cursor):
        if cursor.spelling:
            return cursor.spelling
        return self._pending_typedefs.get(cursor.hash)

    def _collect_typedef_names(self, root):
        for cursor in root.get_children():
            if cursor.kind == CursorKind.TYPEDEF_DECL:
                underlying = cursor.underlying_typedef_type.get_canonical()
                if underlying.kind == TypeKind.RECORD:
                    decl = underlying.get_declaration()
                    if decl and not decl.spelling:
                        self._pending_typedefs[decl.hash] = cursor.spelling

    def _is_from_main_file(self, cursor):
        loc = cursor.location
        if not loc or not loc.file:
            return False
        return os.path.abspath(loc.file.name) == os.path.abspath(self.filepath)

    # -----------------------------------------------------------------------
    # Include walking
    # -----------------------------------------------------------------------

    def translate_includes(self, already_translated=None, clang_args=None):
        """Walk all headers included (transitively) by this TU and emit each
        as a .fx file under CFT_CONFIG.cstdlib_root, mirroring the relative
        path from whichever include root the file belongs to.

        already_translated: set of abs paths already written this session.
            Modified in-place so recursive calls skip duplicates.

        Returns the set of abs paths that were written.
        """
        if CFT_CONFIG.cstdlib_root is None:
            print("cft: include walking skipped -- cstdlib.output_root not set in cft.cfg",
                  file=sys.stderr)
            return set()

        if already_translated is None:
            already_translated = set()

        written = set()
        main_abs = os.path.abspath(self.filepath)

        # Collect unique included file paths from the TU (depth-first order).
        seen_includes = []
        seen_set = set()
        for inc in self.tu.get_includes():
            if not inc.include:
                continue
            inc_abs = os.path.abspath(inc.include.name)
            if inc_abs == main_abs:
                continue
            if inc_abs in seen_set:
                continue
            seen_set.add(inc_abs)
            seen_includes.append(inc_abs)

        # Determine which include roots to use for relative-path mirroring.
        # Priority: explicitly configured roots first, then roots inferred from
        # the actual paths of included files (longest matching prefix wins).
        configured_roots = [os.path.abspath(r) for r in CFT_CONFIG.include_roots]

        for inc_abs in seen_includes:
            if inc_abs in already_translated:
                continue

            # Find the best-matching root for this file.
            rel_path = _best_relative(inc_abs, configured_roots)
            if rel_path is None:
                # No configured root matched -- skip files that live outside
                # known include roots (e.g. project-local headers that the
                # caller will handle via normal translate_file paths).
                continue

            # Build the output path: cstdlib_root / relative / path.fx
            rel_no_ext = os.path.splitext(rel_path)[0]
            out_path = os.path.join(CFT_CONFIG.cstdlib_root,
                                    rel_no_ext + ".fx")
            out_dir = os.path.dirname(out_path)
            os.makedirs(out_dir, exist_ok=True)

            already_translated.add(inc_abs)

            try:
                t = CTranslator(inc_abs, args=clang_args)
                content = t.translate()
                with open(out_path, "w") as f:
                    f.write(content)
                print(f"cft: include -> {out_path}", file=sys.stderr)
                written.add(inc_abs)

                # Recurse: translate includes pulled in by this header too.
                sub = t.translate_includes(already_translated=already_translated,
                                           clang_args=clang_args)
                written |= sub
            except Exception as exc:
                print(f"cft: error translating include {inc_abs}: {exc}",
                      file=sys.stderr)

        return written

    def _emit(self, line):
        self.lines.append(line)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def translate_file(input_path, clang_args=None, walk_includes=True,
                   already_translated=None):
    """Translate a single C file to a Flux string.

    If walk_includes is True and CFT_CONFIG.cstdlib_root is set, each header
    included by the file is also translated and written to the cstdlib output
    tree, mirroring the include-root-relative path.

    already_translated: shared set passed across recursive calls to avoid
        re-translating the same header multiple times in one session.
    """
    t = CTranslator(input_path, args=clang_args)
    result = t.translate()
    if walk_includes:
        t.translate_includes(already_translated=already_translated,
                             clang_args=clang_args)
    return result


def translate_pair(c_path, h_path, out_path, clang_args=None,
                   already_translated=None):
    """Translate a .c/.h pair (either may be None) into a single .fx file."""
    # If both exist, translate .c (which will pull in the .h via includes naturally)
    # and also translate the .h separately, merging unique declarations.
    # Simplest correct approach: translate whichever files exist and concatenate,
    # letting _emitted_types dedup across both by sharing a single CTranslator instance
    # with multiple parse passes -- but libclang parses one TU at a time.
    # Instead: translate .c if present (it includes the .h), else translate .h alone.
    # If only .h exists, translate it directly.
    if c_path and os.path.isfile(c_path):
        primary = c_path
    else:
        primary = h_path

    result = translate_file(primary, clang_args=clang_args,
                            already_translated=already_translated)

    with open(out_path, "w") as f:
        f.write(result)
    print(f"cft: wrote {out_path}", file=sys.stderr)


def translate_directory(dir_path, out_dir=None, clang_args=None):
    """
    Walk a directory, pairing .c and .h files by stem, and emit one .fx per pair.
    If out_dir is None, .fx files are written alongside the source files.
    Includes encountered across all pairs are deduplicated via a shared set.
    """
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Collect all .c and .h files, grouped by stem
    stems = {}
    for fname in os.listdir(dir_path):
        base, ext = os.path.splitext(fname)
        if ext.lower() == ".c":
            stems.setdefault(base, {})["c"] = os.path.join(dir_path, fname)
        elif ext.lower() == ".h":
            stems.setdefault(base, {})["h"] = os.path.join(dir_path, fname)

    if not stems:
        print(f"cft: no .c or .h files found in {dir_path}", file=sys.stderr)
        return

    # Share one already_translated set so headers included by multiple source
    # files are only emitted once.
    already_translated = set()

    for stem, files in sorted(stems.items()):
        c_path = files.get("c")
        h_path = files.get("h")
        out_name = stem + ".fx"
        out_path = os.path.join(out_dir if out_dir else dir_path, out_name)
        try:
            translate_pair(c_path, h_path, out_path, clang_args=clang_args,
                           already_translated=already_translated)
        except Exception as e:
            print(f"cft: error translating {stem}: {e}", file=sys.stderr)


def main():
    print("C -> Flux Translation Utility\n\n"
          "\tConvert C source files individually or in batch mode.\n\n"
          "\t*.c & *.h pairs will convert to a singular .fx file.\n")

    cfg_path = _find_config()
    if cfg_path:
        print(f"cft: using config: {cfg_path}", file=sys.stderr)
    else:
        print("cft: no cft.cfg found -- using built-in defaults", file=sys.stderr)

    if CFT_CONFIG.cstdlib_root:
        print(f"cft: cstdlib output root: {CFT_CONFIG.cstdlib_root}", file=sys.stderr)
    else:
        print("cft: include walking disabled (cstdlib.output_root not configured)",
              file=sys.stderr)

    if len(sys.argv) < 2:
        print("Usage:", file=sys.stderr)
        print("  cft.py <input.c|h> [output.fx]", file=sys.stderr)
        print("  cft.py <directory> [output_directory]", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]

    if os.path.isdir(input_path):
        out_dir = sys.argv[2] if len(sys.argv) >= 3 else None
        translate_directory(input_path, out_dir=out_dir)
        return

    if not os.path.isfile(input_path):
        print(f"cft: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    already_translated = set()
    result = translate_file(input_path, already_translated=already_translated)

    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
        with open(out_path, "w") as f:
            f.write(result)
        print(f"cft: wrote {out_path}", file=sys.stderr)
    else:
        print(result)


if __name__ == "__main__":
    main()