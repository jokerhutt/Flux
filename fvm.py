#!/usr/bin/env python3
"""
Flux Virtual Machine (fvm.py)
Experimental comptime execution backend.

Copyright (C) 2026 Karac V. Thweatt
"""

import struct
import ctypes
import ctypes.util
import os
import sys
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Opcodes
# ---------------------------------------------------------------------------

class Op(Enum):
    # Stack
    PUSH        = auto()
    POP         = auto()
    DUP         = auto()
    SWAP        = auto()
    ROT         = auto()   # ROT  - a b c -> b c a
    OVER        = auto()   # OVER - a b   -> a b a

    # Arithmetic
    ADD         = auto()
    SUB         = auto()
    MUL         = auto()
    DIV         = auto()
    MOD         = auto()
    NEG         = auto()
    POW         = auto()
    ABS         = auto()   # ABS  - absolute value
    MIN         = auto()   # MIN  - pop two, push min
    MAX         = auto()   # MAX  - pop two, push max
    CLAMP       = auto()   # CLAMP - pop (val, lo, hi), push clamped

    # Bitwise
    BAND        = auto()
    BOR         = auto()
    BXOR        = auto()
    BNOT        = auto()
    SHL         = auto()
    SHR         = auto()
    ROTL        = auto()   # ROTL <width>  - rotate left by width bits
    ROTR        = auto()   # ROTR <width>  - rotate right by width bits
    BITREV      = auto()   # BITREV <width> - reverse bits within width
    POPCOUNT    = auto()   # POPCOUNT      - count set bits
    CLZ         = auto()   # CLZ <width>   - count leading zeros within width
    CTZ         = auto()   # CTZ <width>   - count trailing zeros

    # Comparison
    CMP_EQ      = auto()
    CMP_NE      = auto()
    CMP_LT      = auto()
    CMP_LE      = auto()
    CMP_GT      = auto()
    CMP_GE      = auto()

    # Logic
    AND         = auto()
    OR          = auto()
    NOT         = auto()

    # Control flow
    JMP         = auto()   # JMP <addr>
    JIF         = auto()   # JIF <addr>  - jump if truthy
    JNF         = auto()   # JNF <addr>  - jump if falsy
    JTABLE      = auto()   # JTABLE <default_addr> [addr0, addr1, ...] - jump table dispatch
    CALL        = auto()   # CALL <name> <argc>
    RET         = auto()
    HALT        = auto()

    # Locals
    LOCAL_GET   = auto()   # LOCAL_GET <slot>
    LOCAL_SET   = auto()   # LOCAL_SET <slot>

    # Memory
    ALLOC       = auto()   # ALLOC - pop size, push pointer
    FREE        = auto()   # FREE  - pop pointer
    LOAD        = auto()   # LOAD <type_tag> <byte_size>
    STORE       = auto()   # STORE <type_tag> <byte_size>
    OFFSET      = auto()   # OFFSET - pop (ptr, n), push ptr+n

    # Structs / arrays
    STRUCT_NEW  = auto()   # STRUCT_NEW <type_name>
    ARRAY_NEW   = auto()   # ARRAY_NEW <type_name> <count>
    FIELD_GET   = auto()   # FIELD_GET <field_name>
    FIELD_SET   = auto()   # FIELD_SET <field_name>
    INDEX_GET   = auto()
    INDEX_SET   = auto()

    # Type introspection
    SIZEOF      = auto()   # SIZEOF <type_name>  - pushes size in bits
    TYPEOF      = auto()   # TYPEOF              - pushes type tag of top
    ALIGNOF     = auto()   # ALIGNOF <type_name>
    ENDIANOF    = auto()   # ENDIANOF <type_name>

    # IO
    IO_OPEN     = auto()   # (path, mode) -> handle
    IO_READ     = auto()   # (handle, size) -> bytes ptr
    IO_WRITE    = auto()   # (handle, bytes ptr, size)
    IO_CLOSE    = auto()   # (handle)

    # FFI
    FFI_LOAD    = auto()   # FFI_LOAD <lib_path> -> lib handle
    FFI_SYM     = auto()   # FFI_SYM <sym_name>  -> callable
    FFI_CALL    = auto()   # FFI_CALL <argc> <ret_type_tag>
    FFI_FREE    = auto()   # FFI_FREE            - unload lib

    # compiler.io built-ins
    COMPILER_PRINT      = auto()   # compiler.io.console.print(byte*)
    COMPILER_INPUT      = auto()   # compiler.io.console.input() -> byte*
    COMPILER_READFILE   = auto()   # compiler.io.readfile(path: byte*) -> byte*
    COMPILER_WRITEFILE  = auto()   # compiler.io.writefile(path: byte*, content: byte*, flags: byte*)

    # String ops
    STR_LEN     = auto()   # STR_LEN     - push length of top string
    STR_CAT     = auto()   # STR_CAT     - pop two strings, push concatenated
    STR_SLICE   = auto()   # STR_SLICE   - pop (str, start, len), push substring
    STR_EQ      = auto()   # STR_EQ      - pop two strings, push 1 if equal
    STR_FIND    = auto()   # STR_FIND    - pop (haystack, needle), push offset or -1
    INT_TO_STR  = auto()   # INT_TO_STR  - pop integer, push string representation
    STR_TO_INT  = auto()   # STR_TO_INT  - pop string, push integer

    # Type conversion
    CAST        = auto()   # CAST <ttag>    - convert top value to target type
    BITCAST     = auto()   # BITCAST <ttag> - reinterpret bytes as target type

    # Diagnostics
    ASSERT      = auto()   # ASSERT      - pop value; if falsy abort with message
    WARN        = auto()   # WARN        - pop string; emit compiler warning
    PANIC       = auto()   # PANIC       - pop string; unconditional abort

    # Boundary crossing
    EMIT_CONST  = auto()   # pop value  -> ir.Constant
    EMIT_GLOBAL = auto()   # pop ptr    -> ir.GlobalVariable (static data)
    EMIT_TYPE   = auto()   # pop type tag -> LLVM type reference
    EMITFLUX    = auto()   # EMITFLUX <source_text> <var_names>
                           # Substitutes comptime locals into source_text and appends
                           # ('flux', substituted_text) to emit_results.


# ---------------------------------------------------------------------------
# Type tags
# ---------------------------------------------------------------------------

class TTag(Enum):
    INT      = 'int'
    UINT     = 'uint'
    LONG     = 'long'
    ULONG    = 'ulong'
    FLOAT    = 'float'
    DOUBLE   = 'double'
    BOOL     = 'bool'
    BYTE     = 'byte'
    CHAR     = 'char'
    DATA     = 'data'
    PTR      = 'ptr'
    VOID     = 'void'
    STRUCT   = 'struct'
    ARRAY    = 'array'
    BYTES    = 'bytes'    # raw VM heap bytes (IO read results)
    FFI_LIB  = 'ffi_lib'
    FFI_SYM  = 'ffi_sym'
    FILE     = 'file'


# ---------------------------------------------------------------------------
# Stack value
# ---------------------------------------------------------------------------

@dataclass
class Val:
    """
    A value on the VM stack.
    tag  : TTag       - the Flux type of this value
    data : Any        - Python representation:
                        int/bool/float for primitives,
                        int for PTR (offset into VM heap),
                        str for type-name tags (STRUCT, ARRAY),
                        bytes for BYTES,
                        ctypes handle for FFI_LIB/FFI_SYM,
                        file object for FILE.
    meta : dict       - optional extra (e.g. struct type name, element type,
                        bit width for DATA, endianness for be/le types).
    """
    tag:  TTag
    data: Any
    meta: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f'Val({self.tag.value}, {self.data!r})'


# ---------------------------------------------------------------------------
# Instruction
# ---------------------------------------------------------------------------

@dataclass
class Instr:
    op:      Op
    operands: List[Any] = field(default_factory=list)

    def __repr__(self):
        if self.operands:
            return f'{self.op.name} {self.operands}'
        return self.op.name


# ---------------------------------------------------------------------------
# Call frame
# ---------------------------------------------------------------------------

@dataclass
class CallFrame:
    func_name: str
    instructions: List[Instr]
    ip:      int = 0                              # instruction pointer
    locals:  List[Optional[Val]] = field(default_factory=list)
    ret_val: Optional[Val] = None


# ---------------------------------------------------------------------------
# VM heap
# ---------------------------------------------------------------------------

class VMHeap:
    """
    Simple bump allocator backed by a bytearray.
    Keeps a type-tag map so TYPEOF and bounds checks work.
    Free list is maintained for reuse.
    """
    DEFAULT_SIZE = 16 * 1024 * 1024  # 16 MB default

    def __init__(self, size: int = DEFAULT_SIZE):
        self._mem   = bytearray(size)
        self._size  = size
        self._bump  = 8              # start at offset 8, 0 is the null pointer
        self._allocs: Dict[int, int] = {}   # ptr -> byte_size
        self._tags:   Dict[int, TTag] = {}  # ptr -> TTag
        self._free_list: List[Tuple[int, int]] = []  # (ptr, size) freed blocks

    def alloc(self, byte_size: int, tag: TTag = TTag.PTR) -> int:
        """Allocate byte_size bytes, return pointer (offset into heap)."""
        if byte_size <= 0:
            raise VMError(f'VMHeap.alloc: invalid size {byte_size}')
        # Try free list first (first-fit)
        for i, (ptr, sz) in enumerate(self._free_list):
            if sz >= byte_size:
                self._free_list.pop(i)
                self._allocs[ptr] = byte_size
                self._tags[ptr]   = tag
                self._mem[ptr:ptr + byte_size] = bytearray(byte_size)
                return ptr
        # Bump allocate
        ptr = self._bump
        if ptr + byte_size > self._size:
            raise VMError(f'VMHeap: out of memory (requested {byte_size}, available {self._size - self._bump})')
        self._bump += byte_size
        # Align next alloc to 8 bytes
        self._bump = (self._bump + 7) & ~7
        self._allocs[ptr] = byte_size
        self._tags[ptr]   = tag
        return ptr

    def free(self, ptr: int):
        """Return allocation to free list."""
        if ptr not in self._allocs:
            raise VMError(f'VMHeap.free: invalid pointer {ptr}')
        sz = self._allocs.pop(ptr)
        self._tags.pop(ptr, None)
        self._free_list.append((ptr, sz))

    def read(self, ptr: int, byte_size: int) -> bytes:
        """Read byte_size bytes from ptr."""
        self._check(ptr, byte_size)
        return bytes(self._mem[ptr:ptr + byte_size])

    def write(self, ptr: int, data: bytes):
        """Write bytes to ptr."""
        self._check(ptr, len(data))
        self._mem[ptr:ptr + len(data)] = data

    def type_of(self, ptr: int) -> Optional[TTag]:
        return self._tags.get(ptr)

    def _check(self, ptr: int, size: int):
        if ptr < 8 or ptr + size > self._size:
            raise VMError(f'VMHeap: out-of-bounds access at ptr={ptr} size={size}')

    def snapshot(self, ptr: int, byte_size: int) -> bytes:
        """Snapshot a region for EMIT_GLOBAL serialization."""
        return bytes(self._mem[ptr:ptr + byte_size])


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class VMError(Exception):
    pass


# ---------------------------------------------------------------------------
# Struct layout cache
# ---------------------------------------------------------------------------

@dataclass
class StructLayout:
    name:       str
    fields:     List[Tuple[str, TTag, int, int]]  # (name, tag, byte_offset, byte_size)
    total_size: int
    endian:     str = 'little'


# ---------------------------------------------------------------------------
# FluxVM
# ---------------------------------------------------------------------------

class FluxVM:
    """
    Flux comptime virtual machine.

    Usage:
        vm = FluxVM(struct_layouts, type_sizes)
        result = vm.execute(instructions, functions)
    """

    def __init__(
        self,
        struct_layouts: Dict[str, StructLayout] = None,
        type_sizes:     Dict[str, int]          = None,
        heap_size:      int                     = VMHeap.DEFAULT_SIZE,
    ):
        self.heap            = VMHeap(heap_size)
        self.stack:  List[Val]       = []
        self.frames: List[CallFrame] = []
        self.struct_layouts  = struct_layouts or {}
        self.type_sizes      = type_sizes     or {}   # type name -> byte size
        self._ffi_libs:  Dict[str, ctypes.CDLL] = {}
        self._io_handles: Dict[int, Any]         = {}
        self._io_next_handle: int                = 1
        # Registered comptime functions: name -> List[Instr]
        self._functions: Dict[str, List[Instr]]  = {}
        # Pending LLVM emission results accumulated during execute()
        self.emit_results: List[Any] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_function(self, name: str, instructions: List[Instr]):
        """Register a comptime function by name."""
        self._functions[name] = instructions

    def execute(self, instructions: List[Instr], local_count: int = 0) -> Optional[Val]:
        """
        Execute a flat instruction list as the top-level comptime block.
        Returns the top-of-stack value after HALT, or None.
        """
        frame = CallFrame(
            func_name='<comptime>',
            instructions=instructions,
            locals=[None] * local_count,
        )
        self.frames.append(frame)
        result = self._run()
        return result

    # ------------------------------------------------------------------
    # Execution loop
    # ------------------------------------------------------------------

    def _run(self) -> Optional[Val]:
        while self.frames:
            frame = self.frames[-1]
            if frame.ip >= len(frame.instructions):
                # Implicit return from frame
                self.frames.pop()
                continue
            instr = frame.instructions[frame.ip]
            frame.ip += 1
            self._dispatch(instr)
        return self.stack[-1] if self.stack else None

    def _dispatch(self, instr: Instr):
        op = instr.op
        o  = instr.operands

        # Stack
        if   op == Op.PUSH:       self._push(o[0])
        elif op == Op.POP:        self._pop()
        elif op == Op.DUP:        self.stack.append(self.stack[-1])
        elif op == Op.SWAP:       self.stack[-1], self.stack[-2] = self.stack[-2], self.stack[-1]
        elif op == Op.ROT:        self._op_rot()
        elif op == Op.OVER:       self._push(self.stack[-2])

        # Arithmetic
        elif op == Op.ADD:        self._binop(lambda a, b: a + b)
        elif op == Op.SUB:        self._binop(lambda a, b: a - b)
        elif op == Op.MUL:        self._binop(lambda a, b: a * b)
        elif op == Op.DIV:        self._binop_div()
        elif op == Op.MOD:        self._binop(lambda a, b: a % b)
        elif op == Op.NEG:        self._unop(lambda a: -a)
        elif op == Op.POW:        self._binop(lambda a, b: a ** b)
        elif op == Op.ABS:        self._unop(lambda a: abs(a))
        elif op == Op.MIN:        self._binop(lambda a, b: a if a < b else b)
        elif op == Op.MAX:        self._binop(lambda a, b: a if a > b else b)
        elif op == Op.CLAMP:      self._op_clamp()

        # Bitwise
        elif op == Op.BAND:       self._binop(lambda a, b: a & b)
        elif op == Op.BOR:        self._binop(lambda a, b: a | b)
        elif op == Op.BXOR:       self._binop(lambda a, b: a ^ b)
        elif op == Op.BNOT:       self._unop(lambda a: ~a)
        elif op == Op.SHL:        self._binop(lambda a, b: a << b)
        elif op == Op.SHR:        self._binop(lambda a, b: a >> b)
        elif op == Op.ROTL:       self._op_rotl(o[0])
        elif op == Op.ROTR:       self._op_rotr(o[0])
        elif op == Op.BITREV:     self._op_bitrev(o[0])
        elif op == Op.POPCOUNT:   self._unop(lambda a: bin(a & ((1 << 64) - 1)).count('1'))
        elif op == Op.CLZ:        self._op_clz(o[0])
        elif op == Op.CTZ:        self._op_ctz(o[0])

        # Comparison
        elif op == Op.CMP_EQ:     self._cmp(lambda a, b: a == b)
        elif op == Op.CMP_NE:     self._cmp(lambda a, b: a != b)
        elif op == Op.CMP_LT:     self._cmp(lambda a, b: a <  b)
        elif op == Op.CMP_LE:     self._cmp(lambda a, b: a <= b)
        elif op == Op.CMP_GT:     self._cmp(lambda a, b: a >  b)
        elif op == Op.CMP_GE:     self._cmp(lambda a, b: a >= b)

        # Logic
        elif op == Op.AND:        self._binop(lambda a, b: int(bool(a) and bool(b)))
        elif op == Op.OR:         self._binop(lambda a, b: int(bool(a) or  bool(b)))
        elif op == Op.NOT:        self._unop(lambda a: int(not bool(a)))

        # Control flow
        elif op == Op.JMP:        self.frames[-1].ip = o[0]
        elif op == Op.JIF:        self._jif(o[0], truthy=True)
        elif op == Op.JNF:        self._jif(o[0], truthy=False)
        elif op == Op.JTABLE:     self._op_jtable(o[0], o[1])
        elif op == Op.CALL:       self._call(o[0], o[1])
        elif op == Op.RET:        self._ret()
        elif op == Op.HALT:       self.frames.clear()

        # Locals
        elif op == Op.LOCAL_GET:  self._push(self.frames[-1].locals[o[0]])
        elif op == Op.LOCAL_SET:  self.frames[-1].locals[o[0]] = self._pop()

        # Memory
        elif op == Op.ALLOC:      self._op_alloc()
        elif op == Op.FREE:       self._op_free()
        elif op == Op.LOAD:       self._op_load(o[0], o[1])
        elif op == Op.STORE:      self._op_store(o[0], o[1])
        elif op == Op.OFFSET:     self._op_offset()

        # Structs / arrays
        elif op == Op.STRUCT_NEW: self._op_struct_new(o[0])
        elif op == Op.ARRAY_NEW:  self._op_array_new(o[0], o[1])
        elif op == Op.FIELD_GET:  self._op_field_get(o[0])
        elif op == Op.FIELD_SET:  self._op_field_set(o[0])
        elif op == Op.INDEX_GET:  self._op_index_get()
        elif op == Op.INDEX_SET:  self._op_index_set()

        # Type introspection
        elif op == Op.SIZEOF:     self._op_sizeof(o[0])
        elif op == Op.TYPEOF:     self._op_typeof()
        elif op == Op.ALIGNOF:    self._op_alignof(o[0])
        elif op == Op.ENDIANOF:   self._op_endianof(o[0])

        # IO
        elif op == Op.IO_OPEN:    self._op_io_open()
        elif op == Op.IO_READ:    self._op_io_read()
        elif op == Op.IO_WRITE:   self._op_io_write()
        elif op == Op.IO_CLOSE:   self._op_io_close()

        # FFI
        elif op == Op.FFI_LOAD:   self._op_ffi_load(o[0])
        elif op == Op.FFI_SYM:    self._op_ffi_sym(o[0])
        elif op == Op.FFI_CALL:   self._op_ffi_call(o[0], o[1])
        elif op == Op.FFI_FREE:   self._op_ffi_free()

        # compiler.io built-ins
        elif op == Op.COMPILER_PRINT:     self._op_compiler_print()
        elif op == Op.COMPILER_INPUT:     self._op_compiler_input()
        elif op == Op.COMPILER_READFILE:  self._op_compiler_readfile()
        elif op == Op.COMPILER_WRITEFILE: self._op_compiler_writefile()

        # String ops
        elif op == Op.STR_LEN:    self._op_str_len()
        elif op == Op.STR_CAT:    self._op_str_cat()
        elif op == Op.STR_SLICE:  self._op_str_slice()
        elif op == Op.STR_EQ:     self._op_str_eq()
        elif op == Op.STR_FIND:   self._op_str_find()
        elif op == Op.INT_TO_STR: self._op_int_to_str()
        elif op == Op.STR_TO_INT: self._op_str_to_int()

        # Type conversion
        elif op == Op.CAST:       self._op_cast(o[0])
        elif op == Op.BITCAST:    self._op_bitcast(o[0])

        # Diagnostics
        elif op == Op.ASSERT:     self._op_assert()
        elif op == Op.WARN:       self._op_warn()
        elif op == Op.PANIC:      self._op_panic()

        # Boundary crossing
        elif op == Op.EMIT_CONST:  self._op_emit_const()
        elif op == Op.EMIT_GLOBAL: self._op_emit_global(o[0])
        elif op == Op.EMIT_TYPE:   self._op_emit_type()
        elif op == Op.EMITFLUX:    self._op_emitflux(o[0], o[1])

        else:
            raise VMError(f'Unknown opcode: {op}')

    # ------------------------------------------------------------------
    # Stack helpers
    # ------------------------------------------------------------------

    def _push(self, v: Val):
        self.stack.append(v)

    def _pop(self) -> Val:
        if not self.stack:
            raise VMError('Stack underflow')
        return self.stack.pop()

    def _peek(self) -> Val:
        if not self.stack:
            raise VMError('Stack underflow')
        return self.stack[-1]

    def _binop(self, fn):
        b = self._pop()
        a = self._pop()
        tag = a.tag
        self._push(Val(tag, fn(a.data, b.data)))

    def _binop_div(self):
        b = self._pop()
        a = self._pop()
        if b.data == 0:
            raise VMError('Division by zero in comptime')
        if a.tag in (TTag.FLOAT, TTag.DOUBLE):
            self._push(Val(a.tag, a.data / b.data))
        else:
            self._push(Val(a.tag, int(a.data / b.data)))

    def _unop(self, fn):
        a = self._pop()
        self._push(Val(a.tag, fn(a.data)))

    def _cmp(self, fn):
        b = self._pop()
        a = self._pop()
        self._push(Val(TTag.BOOL, int(fn(a.data, b.data))))

    def _jif(self, addr: int, truthy: bool):
        v = self._pop()
        if bool(v.data) == truthy:
            self.frames[-1].ip = addr

    # ------------------------------------------------------------------
    # Control flow
    # ------------------------------------------------------------------

    def _call(self, name: str, argc: int):
        if name not in self._functions:
            raise VMError(f'Unknown comptime function: {name!r}')
        args = [self._pop() for _ in range(argc)]
        args.reverse()
        instrs = self._functions[name]
        frame  = CallFrame(
            func_name=name,
            instructions=instrs,
            locals=list(args) + [None] * max(0, 32 - len(args)),
        )
        self.frames.append(frame)

    def _ret(self):
        frame = self.frames.pop()
        # Return value stays on the stack (already pushed by callee)

    # ------------------------------------------------------------------
    # Stack ops
    # ------------------------------------------------------------------

    def _op_rot(self):
        # a b c -> b c a
        if len(self.stack) < 3:
            raise VMError('ROT: stack underflow')
        c = self.stack.pop()
        b = self.stack.pop()
        a = self.stack.pop()
        self.stack.append(b)
        self.stack.append(c)
        self.stack.append(a)

    # ------------------------------------------------------------------
    # Arithmetic ops
    # ------------------------------------------------------------------

    def _op_clamp(self):
        hi  = self._pop()
        lo  = self._pop()
        val = self._pop()
        tag = val.tag
        v = val.data
        lo_v = lo.data
        hi_v = hi.data
        if v < lo_v:
            v = lo_v
        elif v > hi_v:
            v = hi_v
        self._push(Val(tag, v))

    # ------------------------------------------------------------------
    # Bitwise ops
    # ------------------------------------------------------------------

    def _op_rotl(self, width: int):
        shift_val = self._pop()
        val       = self._pop()
        n     = int(val.data)   & ((1 << width) - 1)
        shift = int(shift_val.data) % width
        result = ((n << shift) | (n >> (width - shift))) & ((1 << width) - 1)
        self._push(Val(val.tag, result))

    def _op_rotr(self, width: int):
        shift_val = self._pop()
        val       = self._pop()
        n     = int(val.data)   & ((1 << width) - 1)
        shift = int(shift_val.data) % width
        result = ((n >> shift) | (n << (width - shift))) & ((1 << width) - 1)
        self._push(Val(val.tag, result))

    def _op_bitrev(self, width: int):
        val    = self._pop()
        n      = int(val.data) & ((1 << width) - 1)
        result = 0
        for _ in range(width):
            result = (result << 1) | (n & 1)
            n >>= 1
        self._push(Val(val.tag, result))

    def _op_clz(self, width: int):
        val = self._pop()
        n   = int(val.data) & ((1 << width) - 1)
        if n == 0:
            self._push(Val(TTag.UINT, width))
            return
        count = 0
        mask  = 1 << (width - 1)
        while mask and not (n & mask):
            count += 1
            mask >>= 1
        self._push(Val(TTag.UINT, count))

    def _op_ctz(self, width: int):
        val = self._pop()
        n   = int(val.data) & ((1 << width) - 1)
        if n == 0:
            self._push(Val(TTag.UINT, width))
            return
        count = 0
        while not (n & 1):
            count += 1
            n >>= 1
        self._push(Val(TTag.UINT, count))

    # ------------------------------------------------------------------
    # Control flow ops
    # ------------------------------------------------------------------

    def _op_jtable(self, default_addr: int, table: list):
        idx_val = self._pop()
        idx     = int(idx_val.data)
        if 0 <= idx < len(table):
            self.frames[-1].ip = table[idx]
        else:
            self.frames[-1].ip = default_addr

    # ------------------------------------------------------------------
    # Memory ops
    # ------------------------------------------------------------------

    def _op_alloc(self):
        size_val = self._pop()
        ptr = self.heap.alloc(int(size_val.data), TTag.PTR)
        self._push(Val(TTag.PTR, ptr))

    def _op_free(self):
        ptr_val = self._pop()
        self.heap.free(int(ptr_val.data))

    def _op_load(self, tag: TTag, byte_size: int):
        ptr_val = self._pop()
        ptr     = int(ptr_val.data)
        raw     = self.heap.read(ptr, byte_size)
        data    = self._bytes_to_val(raw, tag, byte_size)
        self._push(Val(tag, data))

    def _op_store(self, tag: TTag, byte_size: int):
        val     = self._pop()
        ptr_val = self._pop()
        ptr     = int(ptr_val.data)
        raw     = self._val_to_bytes(val, byte_size)
        self.heap.write(ptr, raw)

    def _op_offset(self):
        n       = self._pop()
        ptr_val = self._pop()
        self._push(Val(TTag.PTR, int(ptr_val.data) + int(n.data)))

    # ------------------------------------------------------------------
    # Struct / array ops
    # ------------------------------------------------------------------

    def _op_struct_new(self, type_name: str):
        layout = self._get_layout(type_name)
        ptr    = self.heap.alloc(layout.total_size, TTag.STRUCT)
        self.heap._tags[ptr] = TTag.STRUCT
        self._push(Val(TTag.PTR, ptr, meta={'struct_type': type_name}))

    def _op_array_new(self, type_name: str, count: int):
        elem_size = self._type_byte_size(type_name)
        total     = elem_size * count
        ptr       = self.heap.alloc(total, TTag.ARRAY)
        self._push(Val(TTag.PTR, ptr, meta={'elem_type': type_name, 'count': count, 'elem_size': elem_size}))

    def _op_field_get(self, field_name: str):
        ptr_val   = self._pop()
        ptr       = int(ptr_val.data)
        type_name = ptr_val.meta.get('struct_type')
        if not type_name:
            raise VMError(f'FIELD_GET: pointer has no struct_type meta')
        layout    = self._get_layout(type_name)
        f         = self._find_field(layout, field_name)
        field_ptr = ptr + f[2]
        raw       = self.heap.read(field_ptr, f[3])
        data      = self._bytes_to_val(raw, f[1], f[3])
        self._push(Val(f[1], data))

    def _op_field_set(self, field_name: str):
        val       = self._pop()
        ptr_val   = self._pop()
        ptr       = int(ptr_val.data)
        type_name = ptr_val.meta.get('struct_type')
        if not type_name:
            raise VMError(f'FIELD_SET: pointer has no struct_type meta')
        layout    = self._get_layout(type_name)
        f         = self._find_field(layout, field_name)
        field_ptr = ptr + f[2]
        raw       = self._val_to_bytes(val, f[3])
        self.heap.write(field_ptr, raw)

    def _op_index_get(self):
        idx_val = self._pop()
        ptr_val = self._pop()
        ptr     = int(ptr_val.data)
        idx     = int(idx_val.data)
        esz     = ptr_val.meta.get('elem_size', 1)
        etag    = _str_to_ttag(ptr_val.meta.get('elem_type', 'byte'))
        raw     = self.heap.read(ptr + idx * esz, esz)
        data    = self._bytes_to_val(raw, etag, esz)
        self._push(Val(etag, data))

    def _op_index_set(self):
        val     = self._pop()
        idx_val = self._pop()
        ptr_val = self._pop()
        ptr     = int(ptr_val.data)
        idx     = int(idx_val.data)
        esz     = ptr_val.meta.get('elem_size', 1)
        raw     = self._val_to_bytes(val, esz)
        self.heap.write(ptr + idx * esz, raw)

    # ------------------------------------------------------------------
    # Type introspection
    # ------------------------------------------------------------------

    def _op_sizeof(self, type_name: str):
        bits = self._type_byte_size(type_name) * 8
        self._push(Val(TTag.UINT, bits))

    def _op_typeof(self):
        v = self._peek()
        self._push(Val(TTag.BYTES, v.tag.value.encode()))

    def _op_alignof(self, type_name: str):
        # Alignment mirrors byte size for primitives; use layout for structs
        if type_name in self.struct_layouts:
            layout = self.struct_layouts[type_name]
            # Alignment is the max field alignment (largest field size)
            align = max((f[3] for f in layout.fields), default=1)
        else:
            align = self._type_byte_size(type_name)
        self._push(Val(TTag.UINT, align))

    def _op_endianof(self, type_name: str):
        if type_name in self.struct_layouts:
            endian = self.struct_layouts[type_name].endian
        elif type_name.startswith('be'):
            endian = 'big'
        elif type_name.startswith('le'):
            endian = 'little'
        else:
            endian = 'little'   # Flux default
        self._push(Val(TTag.BYTES, endian.encode()))

    # ------------------------------------------------------------------
    # IO ops
    # ------------------------------------------------------------------

    def _op_io_open(self):
        mode_val = self._pop()
        path_val = self._pop()
        path     = self._read_vm_string(path_val)
        mode     = self._read_vm_string(mode_val)
        try:
            fh = open(path, mode)
        except OSError as e:
            raise VMError(f'IO_OPEN: {e}')
        handle = self._io_next_handle
        self._io_next_handle += 1
        self._io_handles[handle] = fh
        self._push(Val(TTag.FILE, handle))

    def _op_io_read(self):
        size_val   = self._pop()
        handle_val = self._pop()
        fh         = self._get_io_handle(int(handle_val.data))
        size       = int(size_val.data)
        raw        = fh.read(size) if size > 0 else fh.read()
        if isinstance(raw, str):
            raw = raw.encode()
        # Allocate on VM heap and return pointer
        ptr = self.heap.alloc(len(raw) + 1, TTag.BYTES)
        self.heap.write(ptr, raw + b'\x00')
        self._push(Val(TTag.PTR, ptr, meta={'elem_type': 'byte', 'count': len(raw), 'elem_size': 1}))

    def _op_io_write(self):
        size_val   = self._pop()
        ptr_val    = self._pop()
        handle_val = self._pop()
        fh         = self._get_io_handle(int(handle_val.data))
        size       = int(size_val.data)
        ptr        = int(ptr_val.data)
        raw        = self.heap.read(ptr, size)
        fh.write(raw)

    def _op_io_close(self):
        handle_val = self._pop()
        fh         = self._get_io_handle(int(handle_val.data))
        fh.close()
        del self._io_handles[int(handle_val.data)]

    def _get_io_handle(self, handle: int):
        if handle not in self._io_handles:
            raise VMError(f'IO: invalid file handle {handle}')
        return self._io_handles[handle]

    # ------------------------------------------------------------------
    # FFI ops
    # ------------------------------------------------------------------

    def _op_ffi_load(self, lib_path: str):
        try:
            lib = ctypes.CDLL(lib_path)
        except OSError as e:
            raise VMError(f'FFI_LOAD: {e}')
        self._ffi_libs[lib_path] = lib
        self._push(Val(TTag.FFI_LIB, lib_path))

    def _op_ffi_sym(self, sym_name: str):
        lib_val  = self._pop()
        lib_path = lib_val.data
        if lib_path not in self._ffi_libs:
            raise VMError(f'FFI_SYM: library {lib_path!r} not loaded')
        lib = self._ffi_libs[lib_path]
        try:
            sym = getattr(lib, sym_name)
        except AttributeError:
            raise VMError(f'FFI_SYM: symbol {sym_name!r} not found in {lib_path!r}')
        self._push(Val(TTag.FFI_SYM, sym, meta={'sym_name': sym_name}))

    def _op_ffi_call(self, argc: int, ret_tag: TTag):
        args    = [self._pop() for _ in range(argc)]
        args.reverse()
        sym_val = self._pop()
        sym     = sym_val.data
        # Marshal arguments to ctypes
        c_args  = [self._val_to_ctype(a) for a in args]
        result  = sym(*c_args)
        self._push(Val(ret_tag, result))

    def _op_ffi_free(self):
        lib_val  = self._pop()
        lib_path = lib_val.data
        if lib_path in self._ffi_libs:
            del self._ffi_libs[lib_path]

    # ------------------------------------------------------------------
    # compiler.io built-in ops
    # ------------------------------------------------------------------

    def _op_compiler_print(self):
        val = self._pop()
        text = self._read_vm_string(val)
        sys.stdout.write(text)
        sys.stdout.flush()

    def _op_compiler_input(self):
        line = sys.stdin.readline()
        raw  = line.encode('utf-8')
        ptr  = self.heap.alloc(len(raw) + 1, TTag.BYTES)
        self.heap.write(ptr, raw + b'\x00')
        self._push(Val(TTag.PTR, ptr, meta={'elem_type': 'byte', 'count': len(raw), 'elem_size': 1}))

    def _op_compiler_readfile(self):
        path_val = self._pop()
        path     = self._read_vm_string(path_val)
        try:
            with open(path, 'rb') as fh:
                raw = fh.read()
        except OSError as e:
            raise VMError(f'compiler.io.readfile: {e}')
        ptr = self.heap.alloc(len(raw) + 1, TTag.BYTES)
        self.heap.write(ptr, raw + b'\x00')
        self._push(Val(TTag.PTR, ptr, meta={'elem_type': 'byte', 'count': len(raw), 'elem_size': 1}))

    def _op_compiler_writefile(self):
        flags_val   = self._pop()
        content_val = self._pop()
        path_val    = self._pop()
        path    = self._read_vm_string(path_val)
        flags   = self._read_vm_string(flags_val)
        # Determine open mode from flags: r/w/a combined with t (text) or default binary
        mode_map = {
            'r':  'rb',  'rt': 'r',
            'w':  'wb',  'wt': 'w',
            'a':  'ab',  'at': 'a',
            'rw': 'r+b', 'rwt': 'r+',
        }
        mode = mode_map.get(flags.replace(' ', ''), 'wb')
        # Read content from VM heap
        if content_val.tag == TTag.PTR:
            ptr   = int(content_val.data)
            count = content_val.meta.get('count', 0)
            if count == 0:
                # Read until null terminator
                buf = bytearray()
                while True:
                    b = self.heap.read(ptr, 1)[0]
                    if b == 0:
                        break
                    buf.append(b)
                    ptr += 1
                raw = bytes(buf)
            else:
                raw = self.heap.read(ptr, count)
        elif content_val.tag == TTag.BYTES:
            raw = content_val.data if isinstance(content_val.data, bytes) else content_val.data.encode()
        else:
            raw = str(content_val.data).encode()
        try:
            with open(path, mode) as fh:
                fh.write(raw if 'b' in mode else raw.decode('utf-8', errors='replace'))
        except OSError as e:
            raise VMError(f'compiler.io.writefile: {e}')

    # ------------------------------------------------------------------
    # String ops
    # ------------------------------------------------------------------

    def _op_str_len(self):
        val = self._pop()
        s   = self._read_vm_string(val)
        self._push(Val(TTag.UINT, len(s)))

    def _op_str_cat(self):
        b_val = self._pop()
        a_val = self._pop()
        a = self._read_vm_string(a_val)
        b = self._read_vm_string(b_val)
        result = (a + b).encode('utf-8')
        ptr = self.heap.alloc(len(result) + 1, TTag.BYTES)
        self.heap.write(ptr, result + b'\x00')
        self._push(Val(TTag.PTR, ptr, meta={'elem_type': 'byte', 'count': len(result), 'elem_size': 1}))

    def _op_str_slice(self):
        length_val = self._pop()
        start_val  = self._pop()
        str_val    = self._pop()
        s      = self._read_vm_string(str_val)
        start  = int(start_val.data)
        length = int(length_val.data)
        result = s[start:start + length].encode('utf-8')
        ptr = self.heap.alloc(len(result) + 1, TTag.BYTES)
        self.heap.write(ptr, result + b'\x00')
        self._push(Val(TTag.PTR, ptr, meta={'elem_type': 'byte', 'count': len(result), 'elem_size': 1}))

    def _op_str_eq(self):
        b_val = self._pop()
        a_val = self._pop()
        a = self._read_vm_string(a_val)
        b = self._read_vm_string(b_val)
        self._push(Val(TTag.BOOL, int(a == b)))

    def _op_str_find(self):
        needle_val   = self._pop()
        haystack_val = self._pop()
        haystack = self._read_vm_string(haystack_val)
        needle   = self._read_vm_string(needle_val)
        idx = haystack.find(needle)
        self._push(Val(TTag.INT, idx))

    def _op_int_to_str(self):
        val    = self._pop()
        result = str(int(val.data)).encode('utf-8')
        ptr = self.heap.alloc(len(result) + 1, TTag.BYTES)
        self.heap.write(ptr, result + b'\x00')
        self._push(Val(TTag.PTR, ptr, meta={'elem_type': 'byte', 'count': len(result), 'elem_size': 1}))

    def _op_str_to_int(self):
        val = self._pop()
        s   = self._read_vm_string(val)
        try:
            n = int(s.strip(), 0)
        except ValueError:
            raise VMError(f'STR_TO_INT: cannot convert {s!r} to integer')
        self._push(Val(TTag.INT, n))

    # ------------------------------------------------------------------
    # Type conversion ops
    # ------------------------------------------------------------------

    def _op_cast(self, target: TTag):
        val = self._pop()
        src = val.tag
        d   = val.data
        if target in (TTag.INT, TTag.UINT, TTag.LONG, TTag.ULONG, TTag.BYTE, TTag.CHAR, TTag.DATA):
            self._push(Val(target, int(d)))
        elif target == TTag.FLOAT:
            self._push(Val(target, float(d)))
        elif target == TTag.DOUBLE:
            self._push(Val(target, float(d)))
        elif target == TTag.BOOL:
            self._push(Val(target, int(bool(d))))
        else:
            self._push(Val(target, d))

    def _op_bitcast(self, target: TTag):
        val     = self._pop()
        raw     = self._val_to_bytes(val, self._ttag_byte_size(val.tag))
        result  = self._bytes_to_val(raw, target, self._ttag_byte_size(target))
        self._push(Val(target, result))

    def _ttag_byte_size(self, tag: TTag) -> int:
        _sizes = {
            TTag.BOOL: 1, TTag.BYTE: 1, TTag.CHAR: 1,
            TTag.INT: 4,  TTag.UINT: 4,
            TTag.LONG: 8, TTag.ULONG: 8,
            TTag.FLOAT: 4, TTag.DOUBLE: 8,
            TTag.PTR: 8,
        }
        return _sizes.get(tag, 4)

    # ------------------------------------------------------------------
    # Diagnostic ops
    # ------------------------------------------------------------------

    def _op_assert(self):
        msg_val  = self._pop()
        cond_val = self._pop()
        if not bool(cond_val.data):
            msg = self._read_vm_string(msg_val) if msg_val.tag in (TTag.BYTES, TTag.PTR) else str(msg_val.data)
            raise VMError(f'comptime assertion failed: {msg}')

    def _op_warn(self):
        val = self._pop()
        msg = self._read_vm_string(val) if val.tag in (TTag.BYTES, TTag.PTR) else str(val.data)
        sys.stderr.write(f'comptime warning: {msg}\n')
        sys.stderr.flush()

    def _op_panic(self):
        val = self._pop()
        msg = self._read_vm_string(val) if val.tag in (TTag.BYTES, TTag.PTR) else str(val.data)
        raise VMError(f'comptime panic: {msg}')

    # ------------------------------------------------------------------
    # Boundary crossing ops
    # ------------------------------------------------------------------

    def _op_emit_const(self):
        val = self._pop()
        self.emit_results.append(('const', val))

    def _op_emit_global(self, byte_size: int):
        ptr_val = self._pop()
        ptr     = int(ptr_val.data)
        data    = self.heap.snapshot(ptr, byte_size)
        self.emit_results.append(('global', data, ptr_val.meta))

    def _op_emit_type(self):
        val = self._pop()
        self.emit_results.append(('type', val))

    def _op_emitflux(self, source_text: str, var_names: list):
        """
        Perform variable substitution on source_text using the current comptime
        locals (passed as a snapshot via var_names: list of (name, slot) pairs),
        then append ('flux', substituted_text) to emit_results.

        Substitution rules:
          - Plain identifiers: replace each occurrence of `name` that appears
            as a whole word with the string representation of its value.
          - ~$f"..." format strings in the text are expanded: the embedded
            {name} placeholders are replaced with the value, then the entire
            ~$f"..." expression is replaced with the resulting identifier text.
        """
        frame = self.frames[-1] if self.frames else None

        def val_to_str(v: Val) -> str:
            if v is None:
                return '0'
            if v.tag == TTag.BOOL:
                return 'true' if v.data else 'false'
            if v.tag in (TTag.FLOAT, TTag.DOUBLE):
                return repr(float(v.data))
            if v.tag == TTag.BYTES and isinstance(v.data, (bytes, bytearray)):
                return v.data.decode('utf-8', errors='replace')
            return str(v.data)

        # Build name -> value-string mapping from the provided snapshot
        subst: Dict[str, str] = {}
        for name, slot in var_names:
            if frame is not None and slot < len(frame.locals):
                v = frame.locals[slot]
                subst[name] = val_to_str(v)
            else:
                subst[name] = '0'

        import re

        # Expand ~$f"..." expressions first
        def expand_codify_fstr(m):
            fstr_body = m.group(1)
            # Replace {name} placeholders inside the f-string body
            def replace_placeholder(pm):
                pname = pm.group(1)
                return subst.get(pname, pname)
            expanded = re.sub(r'\{(\w+)\}', replace_placeholder, fstr_body)
            return expanded  # result is the bare identifier text

        text = re.sub(r'~\$f"([^"]*)"', expand_codify_fstr, source_text)

        # Replace whole-word variable names with their values
        for name, value_str in subst.items():
            text = re.sub(r'\b' + re.escape(name) + r'\b', value_str, text)

        self.emit_results.append(('flux', text))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_layout(self, type_name: str) -> StructLayout:
        if type_name not in self.struct_layouts:
            raise VMError(f'Unknown struct/object type: {type_name!r}')
        return self.struct_layouts[type_name]

    def _find_field(self, layout: StructLayout, name: str):
        for f in layout.fields:
            if f[0] == name:
                return f
        raise VMError(f'Field {name!r} not found in {layout.name!r}')

    def _type_byte_size(self, type_name: str) -> int:
        if type_name in self.type_sizes:
            return self.type_sizes[type_name]
        if type_name in self.struct_layouts:
            return self.struct_layouts[type_name].total_size
        _defaults = {
            'bool': 1, 'byte': 1, 'char': 1,
            'int': 4,  'uint': 4,
            'long': 8, 'ulong': 8,
            'float': 4, 'double': 8,
        }
        if type_name in _defaults:
            return _defaults[type_name]
        # data{N} / i8..i64 / u8..u64
        if type_name.startswith('i') or type_name.startswith('u'):
            try:
                bits = int(type_name[1:])
                return bits // 8
            except ValueError:
                pass
        if type_name.startswith('be') or type_name.startswith('le'):
            try:
                bits = int(type_name[2:])
                return bits // 8
            except ValueError:
                pass
        raise VMError(f'Unknown type size for: {type_name!r}')

    def _bytes_to_val(self, raw: bytes, tag: TTag, byte_size: int) -> Any:
        if tag in (TTag.INT, TTag.LONG):
            return int.from_bytes(raw[:byte_size], 'little', signed=True)
        if tag in (TTag.UINT, TTag.ULONG, TTag.BYTE, TTag.CHAR, TTag.BOOL, TTag.DATA):
            return int.from_bytes(raw[:byte_size], 'little', signed=False)
        if tag == TTag.FLOAT:
            return struct.unpack('<f', raw[:4])[0]
        if tag == TTag.DOUBLE:
            return struct.unpack('<d', raw[:8])[0]
        if tag == TTag.PTR:
            return int.from_bytes(raw[:byte_size], 'little', signed=False)
        return raw

    def _val_to_bytes(self, val: Val, byte_size: int) -> bytes:
        tag = val.tag
        d   = val.data
        if tag in (TTag.INT, TTag.LONG):
            return d.to_bytes(byte_size, 'little', signed=True)
        if tag in (TTag.UINT, TTag.ULONG, TTag.BYTE, TTag.CHAR, TTag.BOOL, TTag.DATA, TTag.PTR):
            return d.to_bytes(byte_size, 'little', signed=False)
        if tag == TTag.FLOAT:
            return struct.pack('<f', d)
        if tag == TTag.DOUBLE:
            return struct.pack('<d', d)
        if isinstance(d, (bytes, bytearray)):
            return bytes(d)[:byte_size].ljust(byte_size, b'\x00')
        raise VMError(f'_val_to_bytes: unhandled tag {tag}')

    def _read_vm_string(self, val: Val) -> str:
        if isinstance(val.data, str):
            return val.data
        if val.tag == TTag.BYTES and isinstance(val.data, (bytes, bytearray)):
            return val.data.decode('utf-8', errors='replace')
        if val.tag == TTag.PTR:
            ptr = int(val.data)
            # Read null-terminated string from heap
            result = bytearray()
            while True:
                b = self.heap.read(ptr, 1)[0]
                if b == 0:
                    break
                result.append(b)
                ptr += 1
            return result.decode('utf-8', errors='replace')
        raise VMError(f'_read_vm_string: cannot read string from {val}')

    def _val_to_ctype(self, val: Val):
        tag = val.tag
        if tag in (TTag.INT,):            return ctypes.c_int(int(val.data))
        if tag in (TTag.UINT,):           return ctypes.c_uint(int(val.data))
        if tag in (TTag.LONG,):           return ctypes.c_long(int(val.data))
        if tag in (TTag.ULONG,):          return ctypes.c_ulong(int(val.data))
        if tag == TTag.FLOAT:             return ctypes.c_float(float(val.data))
        if tag == TTag.DOUBLE:            return ctypes.c_double(float(val.data))
        if tag == TTag.BOOL:              return ctypes.c_bool(bool(val.data))
        if tag == TTag.BYTE:              return ctypes.c_uint8(int(val.data))
        if tag == TTag.PTR:               return ctypes.c_void_p(int(val.data))
        raise VMError(f'_val_to_ctype: unhandled tag {tag}')


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _str_to_ttag(type_name: str) -> TTag:
    _map = {
        'int': TTag.INT, 'uint': TTag.UINT,
        'long': TTag.LONG, 'ulong': TTag.ULONG,
        'float': TTag.FLOAT, 'double': TTag.DOUBLE,
        'bool': TTag.BOOL, 'byte': TTag.BYTE, 'char': TTag.CHAR,
        'data': TTag.DATA, 'ptr': TTag.PTR, 'void': TTag.VOID,
    }
    return _map.get(type_name, TTag.PTR)