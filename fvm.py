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
    CALL_PTR    = auto()   # CALL_PTR <argc> - pop BYTES(func_name), call it
    RET         = auto()
    TAIL_SELF   = auto()   # TAIL_SELF <argc> - reset ip=0, reload args into slots 0..N-1, reuse frame
    HALT        = auto()

    # Locals
    LOCAL_GET      = auto()   # LOCAL_GET <slot>
    LOCAL_SET      = auto()   # LOCAL_SET <slot>
    LOCAL_DEREF    = auto()   # LOCAL_DEREF    - pop PTR(slot), push locals[slot]
    LOCAL_DEREF_SET = auto()  # LOCAL_DEREF_SET - pop val, pop PTR(slot), locals[slot]=val

    # Memory
    ALLOC       = auto()   # ALLOC - pop size, push pointer
    FREE        = auto()   # FREE  - pop pointer
    LOAD        = auto()   # LOAD <type_tag> <byte_size>
    STORE       = auto()   # STORE <type_tag> <byte_size>
    OFFSET      = auto()   # OFFSET - pop (ptr, n), push ptr+n

    # Structs / arrays
    STRUCT_NEW  = auto()   # STRUCT_NEW <type_name>
    ARRAY_NEW   = auto()   # ARRAY_NEW <type_name> <count>
    ARRAY_LEN   = auto()   # ARRAY_LEN  - pop array PTR, push element count
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
    COMPILER_FVM_DUMP   = auto()   # compiler.fvm.dump(path: byte*) - serialise current comptime to .fvm

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
    total_bits: int = 0  # True bit-packed size (sum of raw field bit widths, no padding)


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
        # Locals snapshot from the most recently executed top-level block
        self.last_locals: List[Any] = []
        # Accumulated instruction log across all execute() calls, for compiler.fvm.dump
        self._comptime_log: List[List[Instr]] = []   # one entry per execute() call
        self._comptime_log_locals: int   = 0   # high-water local count across all blocks

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
        if local_count > self._comptime_log_locals:
            self._comptime_log_locals = local_count
        frame = CallFrame(
            func_name='<comptime>',
            instructions=instructions,
            locals=[None] * local_count,
        )
        self.frames.append(frame)
        result = self._run()
        block = [instr for instr in instructions if instr.op != Op.HALT]
        self._comptime_log.append(block)
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
        elif op == Op.CALL_PTR:   self._call_ptr(o[0])
        elif op == Op.RET:        self._ret()
        elif op == Op.TAIL_SELF:  self._tail_self(o[0])
        elif op == Op.HALT:
            if self.frames:
                self.last_locals = list(self.frames[0].locals)
            self.frames.clear()

        # Locals
        elif op == Op.LOCAL_GET:      self._push(self.frames[-1].locals[o[0]])
        elif op == Op.LOCAL_SET:      self.frames[-1].locals[o[0]] = self._pop()
        elif op == Op.LOCAL_DEREF:    self._op_local_deref()
        elif op == Op.LOCAL_DEREF_SET: self._op_local_deref_set()

        # Memory
        elif op == Op.ALLOC:      self._op_alloc()
        elif op == Op.FREE:       self._op_free()
        elif op == Op.LOAD:       self._op_load(o[0], o[1])
        elif op == Op.STORE:      self._op_store(o[0], o[1])
        elif op == Op.OFFSET:     self._op_offset()

        # Structs / arrays
        elif op == Op.STRUCT_NEW: self._op_struct_new(o[0])
        elif op == Op.ARRAY_NEW:  self._op_array_new(o[0], o[1])
        elif op == Op.ARRAY_LEN:  self._op_array_len()
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
        elif op == Op.COMPILER_FVM_DUMP:  self._op_compiler_fvm_dump()

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

    def _call_ptr(self, argc: int):
        """Pop BYTES(func_name) then call that function with argc args."""
        name_val = self._pop()
        name = (name_val.data.decode('utf-8')
                if isinstance(name_val.data, (bytes, bytearray))
                else str(name_val.data))
        self._call(name, argc)

    def _ret(self):
        frame = self.frames.pop()
        # Return value stays on the stack (already pushed by callee)

    def _tail_self(self, argc: int):
        """TAIL_SELF <argc> - zero-stack-growth self tail-call.
        Pop argc args, reload them into slots 0..argc-1 of the current frame,
        then reset ip to 0. The current frame is reused entirely.
        """
        args = [self._pop() for _ in range(argc)]
        args.reverse()
        frame = self.frames[-1]
        for i, arg in enumerate(args):
            frame.locals[i] = arg
        frame.ip = 0

    # ------------------------------------------------------------------
    # Stack ops
    # ------------------------------------------------------------------

    def _op_local_deref(self):
        """Pop a PTR(slot) and push the value at that local slot."""
        ptr_val = self._pop()
        slot = int(ptr_val.data)
        val = self.frames[-1].locals[slot]
        if val is None:
            val = Val(TTag.INT, 0)
        self._push(val)

    def _op_local_deref_set(self):
        """Pop val and PTR(slot), store val into locals[slot]."""
        val     = self._pop()
        ptr_val = self._pop()
        slot    = int(ptr_val.data)
        self.frames[-1].locals[slot] = val

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

    def _op_array_len(self):
        ptr_val = self._pop()
        count = ptr_val.meta.get('count', 0)
        self._push(Val(TTag.INT, count))

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
        if ptr_val.meta.get('stack_slot'):
            # Stack-slot pointer: ptr is a locals index, value is an integer
            # whose bits are packed LSB-first matching the struct layout.
            src_val   = self.frames[-1].locals[ptr]
            raw_int   = int(src_val.data) if src_val is not None else 0
            bit_offset = ptr_val.meta.get('field_bit_offsets', {}).get(f[0], 0)
            _precise  = {TTag.BOOL: 1, TTag.BYTE: 8, TTag.CHAR: 8,
                         TTag.INT: 32, TTag.UINT: 32,
                         TTag.LONG: 64, TTag.ULONG: 64,
                         TTag.FLOAT: 32, TTag.DOUBLE: 64}
            fbit_width = _precise.get(f[1], f[3] * 8)
            mask      = (1 << fbit_width) - 1
            field_int = (raw_int >> bit_offset) & mask
            if f[1] == TTag.BOOL:
                self._push(Val(TTag.BOOL, bool(field_int)))
            else:
                self._push(Val(f[1], field_int))
            return
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
        if ptr_val.meta.get('stack_slot'):
            # Stack-slot pointer: pack field bits back into the integer in the slot.
            src_val   = self.frames[-1].locals[ptr]
            raw_int   = int(src_val.data) if src_val is not None else 0
            bit_offset = ptr_val.meta.get('field_bit_offsets', {}).get(f[0], 0)
            _precise  = {TTag.BOOL: 1, TTag.BYTE: 8, TTag.CHAR: 8,
                         TTag.INT: 32, TTag.UINT: 32,
                         TTag.LONG: 64, TTag.ULONG: 64,
                         TTag.FLOAT: 32, TTag.DOUBLE: 64}
            fbit_width = _precise.get(f[1], f[3] * 8)
            mask      = (1 << fbit_width) - 1
            field_int = int(val.data) & mask
            raw_int   = (raw_int & ~(mask << bit_offset)) | (field_int << bit_offset)
            self.frames[-1].locals[ptr] = Val(src_val.tag if src_val else TTag.INT, raw_int)
            return
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
        self._push(Val(TTag.BYTES, raw))

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
        if val.tag in (TTag.BYTES, TTag.PTR, TTag.CHAR) or isinstance(val.data, (str, bytes, bytearray)):
            text = self._read_vm_string(val)
        else:
            text = str(val.data)
        sys.stdout.write(text)
        sys.stdout.flush()

    def _op_compiler_input(self):
        line = sys.stdin.readline()
        raw  = line.encode('utf-8')
        self._push(Val(TTag.BYTES, raw))

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


    def _op_compiler_fvm_dump(self):
        path_val = self._pop()
        path     = self._read_vm_string(path_val)
        path     = path.replace('\\', '/')
        current_ip     = self.frames[0].ip if self.frames else 0
        current_instrs = self.frames[0].instructions if self.frames else []
        before_dump = [
            instr for instr in current_instrs[:current_ip - 2]
            if instr.op != Op.HALT
        ]
        # Rebase each logged block's jump targets to global indices
        all_instrs = []
        for block in self._comptime_log:
            rebased = _rebase_block(block, len(all_instrs))
            all_instrs.extend(rebased)
        # Rebase the current block's pre-dump instructions
        if before_dump:
            before_rebased = _rebase_block(before_dump, len(all_instrs))
            all_instrs.extend(before_rebased)
        local_count = self._comptime_log_locals
        text = serialise_fvm(all_instrs, local_count, self._functions, self.struct_layouts)
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(text)
        except OSError as e:
            raise VMError(f'compiler.fvm.dump: {e}')

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
        self._push(Val(TTag.BYTES, (a + b).encode('utf-8')))

    def _op_str_slice(self):
        length_val = self._pop()
        start_val  = self._pop()
        str_val    = self._pop()
        s      = self._read_vm_string(str_val)
        start  = int(start_val.data)
        length = int(length_val.data)
        self._push(Val(TTag.BYTES, s[start:start + length].encode('utf-8')))

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
        val = self._pop()
        if val.tag in (TTag.BYTES,) or isinstance(val.data, (bytes, bytearray)):
            result = val.data if isinstance(val.data, (bytes, bytearray)) else val.data.encode('utf-8')
        elif val.tag == TTag.PTR:
            result = self._read_vm_string(val).encode('utf-8')
        elif isinstance(val.data, str):
            result = val.data.encode('utf-8')
        elif val.tag == TTag.CHAR:
            result = chr(int(val.data)).encode('utf-8')
        else:
            result = str(int(val.data)).encode('utf-8')
        self._push(Val(TTag.BYTES, result))

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
        # Coerce single-character strings to their ordinal for numeric casts
        if isinstance(d, str) and len(d) == 1:
            d = ord(d)
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
        if val.tag == TTag.CHAR:
            return chr(int(val.data))
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

# ---------------------------------------------------------------------------
# .fvm serialiser
# ---------------------------------------------------------------------------

def _serialise_val(v) -> str:
    """Render a Val as a type:value token suitable for a PUSH operand.

    For PTR values carrying struct meta (stack_slot pointer), the meta is
    encoded as a bracketed suffix so it can be round-tripped through .fvm:

        ptr:29[struct_type=TB,stack_slot,a:0,b:1,c:2]

    Items in the bracket:
        struct_type=NAME      -- the Flux struct/type name
        stack_slot            -- flag: pointer addresses a frame local slot
        NAME:OFFSET           -- field_bit_offsets entries (name:bit_offset)
    """
    tag = v.tag
    d   = v.data
    if tag == TTag.BYTES:
        if isinstance(d, (bytes, bytearray)):
            text = d.decode('utf-8', errors='replace')
        else:
            text = str(d)
        escaped = (text.replace('\\', '\\\\')
                       .replace('"',  '\\"')
                       .replace('\n', '\\n')
                       .replace('\r', '\\r')
                       .replace('\t', '\\t'))
        return f'bytes:"{escaped}"'
    if tag == TTag.BOOL:
        return f'bool:{"true" if d else "false"}'
    if tag in (TTag.FLOAT, TTag.DOUBLE):
        return f'{tag.value}:{repr(float(d))}'
    if tag == TTag.VOID:
        return 'void:0'
    if tag == TTag.PTR and v.meta:
        parts = []
        if 'struct_type' in v.meta:
            parts.append(f'struct_type={v.meta["struct_type"]}')
        if v.meta.get('stack_slot'):
            parts.append('stack_slot')
        for fname, foffset in v.meta.get('field_bit_offsets', {}).items():
            parts.append(f'{fname}:{foffset}')
        if parts:
            return 'ptr:' + str(d) + '[' + ','.join(parts) + ']'
    return f'{tag.value}:{d}'


def _serialise_instr(instr) -> str:
    """Render one Instr as a .fvm source line."""
    op = instr.op
    o  = instr.operands

    # Zero-operand
    _zero = {
        Op.POP, Op.DUP, Op.SWAP, Op.ROT, Op.OVER,
        Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD, Op.NEG, Op.POW,
        Op.ABS, Op.MIN, Op.MAX, Op.CLAMP,
        Op.BAND, Op.BOR, Op.BXOR, Op.BNOT, Op.SHL, Op.SHR, Op.POPCOUNT,
        Op.CMP_EQ, Op.CMP_NE, Op.CMP_LT, Op.CMP_LE, Op.CMP_GT, Op.CMP_GE,
        Op.AND, Op.OR, Op.NOT,
        Op.RET, Op.HALT,
        Op.LOCAL_DEREF, Op.LOCAL_DEREF_SET,
        Op.ALLOC, Op.FREE, Op.OFFSET,
        Op.ARRAY_LEN, Op.INDEX_GET, Op.INDEX_SET,
        Op.TYPEOF,
        Op.IO_OPEN, Op.IO_READ, Op.IO_WRITE, Op.IO_CLOSE,
        Op.FFI_FREE,
        Op.COMPILER_PRINT, Op.COMPILER_INPUT,
        Op.COMPILER_READFILE, Op.COMPILER_WRITEFILE,
        Op.STR_LEN, Op.STR_CAT, Op.STR_SLICE, Op.STR_EQ, Op.STR_FIND,
        Op.INT_TO_STR, Op.STR_TO_INT,
        Op.ASSERT, Op.WARN, Op.PANIC,
        Op.EMIT_CONST, Op.EMIT_TYPE,
    }
    if op in _zero:
        return op.name

    if op == Op.PUSH:
        return f'PUSH {_serialise_val(o[0])}'

    if op == Op.CALL_PTR:
        return f'CALL_PTR {o[0]}'

    if op in (Op.JMP, Op.JIF, Op.JNF):
        return f'{op.name} {o[0]}'

    if op == Op.JTABLE:
        table_str = ' '.join(str(a) for a in o[1])
        return f'JTABLE {o[0]} {table_str}'

    if op == Op.CALL:
        return f'CALL {o[0]} {o[1]}'

    if op == Op.TAIL_SELF:
        return f'TAIL_SELF {o[0]}'

    if op in (Op.LOCAL_GET, Op.LOCAL_SET):
        return f'{op.name} {o[0]}'

    if op in (Op.ROTL, Op.ROTR, Op.BITREV, Op.CLZ, Op.CTZ):
        return f'{op.name} {o[0]}'

    if op in (Op.SIZEOF, Op.ALIGNOF, Op.ENDIANOF, Op.STRUCT_NEW):
        return f'{op.name} {o[0]}'

    if op in (Op.FIELD_GET, Op.FIELD_SET):
        return f'{op.name} {o[0]}'

    if op == Op.ARRAY_NEW:
        return f'ARRAY_NEW {o[0]} {o[1]}'

    if op in (Op.LOAD, Op.STORE):
        return f'{op.name} {o[0].value} {o[1]}'

    if op in (Op.CAST, Op.BITCAST):
        return f'{op.name} {o[0].value}'

    if op == Op.EMIT_GLOBAL:
        return f'EMIT_GLOBAL {o[0]}'

    if op == Op.FFI_LOAD:
        return f'FFI_LOAD {o[0]}'

    if op == Op.FFI_SYM:
        return f'FFI_SYM {o[0]}'

    if op == Op.FFI_CALL:
        return f'FFI_CALL {o[0]} {o[1].value}'

    if op == Op.EMITFLUX:
        src_text = o[0].replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        bindings = ' '.join(f'{n}:{s}' for n, s in o[1])
        return f'EMITFLUX "{src_text}" {bindings}'.rstrip()

    if op == Op.COMPILER_FVM_DUMP:
        return 'COMPILER_FVM_DUMP'

    # Fallback: emit op name only (unknown operand shape)
    return op.name


def _rebase_block(instrs, offset):
    """Return a copy of instrs with all jump targets shifted by offset."""
    result = []
    for instr in instrs:
        op = instr.op
        if op in (Op.JMP, Op.JIF, Op.JNF):
            result.append(Instr(op, [instr.operands[0] + offset]))
        elif op == Op.JTABLE:
            new_table = [t + offset for t in instr.operands[1]]
            result.append(Instr(op, [instr.operands[0] + offset, new_table]))
        else:
            result.append(instr)
    return result


def serialise_fvm(
    instructions:    'List[Instr]',
    local_count:     int,
    functions:       'Dict[str, List[Instr]]',
    struct_layouts:  'Dict[str, StructLayout]' = None,
) -> str:
    """
    Serialise a set of VM instructions and function definitions to .fvm text.
    This is the inverse of parse_fvm_source().
    """
    lines = ['# Generated by compiler.fvm.dump', '']

    # Struct layouts
    for layout in (struct_layouts or {}).values():
        lines.append(f'struct {layout.name} {layout.endian} {layout.total_size} {layout.total_bits}')
        for fname, fttag, foff, fsz in layout.fields:
            lines.append(f'    field {fname} {fttag.value} {foff} {fsz}')
        lines.append('endstruct')
        lines.append('')

    # Functions
    for name, instrs in functions.items():
        lines.append(f'func {name}')
        for instr in instrs:
            lines.append(f'    {_serialise_instr(instr)}')
        lines.append('endfunc')
        lines.append('')

    # Main block
    if local_count:
        lines.append(f'locals {local_count}')
    for instr in instructions:
        lines.append(_serialise_instr(instr))

    # Always terminate the main block with HALT so the file is self-contained
    if not lines or lines[-1].strip() != 'HALT':
        lines.append('HALT')
    return '\n'.join(lines) + '\n'




# ---------------------------------------------------------------------------
# .fvm source file parser
# ---------------------------------------------------------------------------

def _parse_fvm_val(token: str) -> Val:
    """
    Parse a typed literal token into a Val.

    Syntax:
        type:value        e.g.  int:42   float:3.14   bool:1   bytes:"hello"
        "string"          shorthand for bytes:"string"  (BYTES tag, UTF-8)
        ptr:N[meta]       PTR with meta dict encoded as a bracketed suffix

    PTR meta bracket format: [struct_type=NAME,stack_slot,field:offset,...]
    """
    # Bare quoted string -> BYTES
    if token.startswith('"') and token.endswith('"') and len(token) >= 2:
        raw = token[1:-1].encode('utf-8').decode('unicode_escape').encode('utf-8')
        return Val(TTag.BYTES, raw)

    if ':' not in token:
        raise VMError(f'.fvm: cannot parse value token {token!r} (expected type:value)')

    type_str, _, rest = token.partition(':')
    type_str = type_str.strip().lower()

    # Strip and parse optional [meta] bracket from the value portion
    meta = {}
    raw  = rest
    if '[' in rest:
        bracket_start = rest.index('[')
        raw = rest[:bracket_start]
        bracket_content = rest[bracket_start + 1:].rstrip(']')
        for item in bracket_content.split(','):
            item = item.strip()
            if not item:
                continue
            if item == 'stack_slot':
                meta['stack_slot'] = True
            elif item.startswith('struct_type='):
                meta['struct_type'] = item[len('struct_type='):]
            elif ':' in item:
                fname, _, foffset = item.partition(':')
                if 'field_bit_offsets' not in meta:
                    meta['field_bit_offsets'] = {}
                meta['field_bit_offsets'][fname] = int(foffset)

    if type_str == 'bytes':
        if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
            raw = raw[1:-1].encode('utf-8').decode('unicode_escape').encode('utf-8')
        else:
            raw = raw.encode('utf-8')
        return Val(TTag.BYTES, raw)

    tag_map = {
        'int':    TTag.INT,    'uint':   TTag.UINT,
        'long':   TTag.LONG,   'ulong':  TTag.ULONG,
        'float':  TTag.FLOAT,  'double': TTag.DOUBLE,
        'bool':   TTag.BOOL,   'byte':   TTag.BYTE,
        'char':   TTag.CHAR,   'ptr':    TTag.PTR,
        'void':   TTag.VOID,
    }
    if type_str not in tag_map:
        raise VMError(f'.fvm: unknown type tag {type_str!r} in {token!r}')

    tag = tag_map[type_str]
    if tag in (TTag.FLOAT, TTag.DOUBLE):
        return Val(tag, float(raw))
    elif tag == TTag.VOID:
        return Val(tag, 0)
    elif tag == TTag.BOOL:
        if raw.lower() in ('true', '1'):
            return Val(tag, 1)
        elif raw.lower() in ('false', '0'):
            return Val(tag, 0)
        raise VMError(f'.fvm: invalid bool value {raw!r}')
    else:
        return Val(tag, int(raw, 0), meta)


def _parse_ttag(token: str) -> TTag:
    """Parse a bare type-tag name into a TTag."""
    _map = {t.value: t for t in TTag}
    if token not in _map:
        raise VMError(f'.fvm: unknown TTag {token!r}')
    return _map[token]


class FVMParseError(Exception):
    def __init__(self, message: str, lineno: int):
        super().__init__(f'line {lineno}: {message}')
        self.lineno = lineno


def _fvm_split(line: str) -> list:
    """
    Split a .fvm source line into tokens, keeping quoted strings and
    bracketed meta suffixes intact.

    Token forms:
      - bare word
      - "quoted string" (may contain spaces)
      - word:"quoted string"   (e.g. bytes:"hello world")
      - type:value[meta...]    (e.g. ptr:29[struct_type=TB,stack_slot,a:0])
        The [...] portion is part of the same token.
    """
    tokens = []
    i = 0
    n = len(line)
    while i < n:
        if line[i].isspace():
            i += 1
            continue
        j = i
        # Consume non-whitespace prefix up to a quote or end
        while j < n and not line[j].isspace() and line[j] != '"':
            j += 1
        prefix = line[i:j]
        if j < n and line[j] == '"':
            # Quoted section: absorb into current token
            j += 1
            while j < n:
                if line[j] == '\\':
                    j += 2
                    continue
                if line[j] == '"':
                    j += 1
                    break
                j += 1
            tokens.append(line[i:j])
        else:
            if prefix:
                tokens.append(prefix)
        i = j
    return tokens


def parse_fvm_source(source: str):
    """
    Parse a .fvm text file into (instructions, local_count, functions, struct_layouts).
    """
    instructions: List[Instr] = []
    functions:    Dict[str, List[Instr]] = {}
    struct_layouts: Dict[str, StructLayout] = {}
    local_count   = 0
    in_func       = False
    func_name     = ''
    func_instrs:  List[Instr] = []
    in_struct     = False
    struct_name   = ''
    struct_endian = 'little'
    struct_total_size = 0
    struct_total_bits = 0
    struct_fields: list = []

    op_by_name: Dict[str, Op] = {o.name: o for o in Op}
    lines_src = source.splitlines()

    for lineno, raw_line in enumerate(lines_src, start=1):
        line = raw_line.split('#', 1)[0].strip()
        if not line:
            continue

        tokens = _fvm_split(line)
        if not tokens:
            continue
        directive = tokens[0].upper()

        if directive == 'LOCALS':
            if len(tokens) != 2:
                raise FVMParseError("'locals' requires exactly one argument", lineno)
            if not in_func:
                local_count = int(tokens[1], 0)
            continue

        if directive == 'STRUCT':
            if in_struct:
                raise FVMParseError("nested 'struct' is not allowed", lineno)
            if len(tokens) < 2:
                raise FVMParseError("'struct' requires a name", lineno)
            in_struct         = True
            struct_name       = tokens[1]
            struct_endian     = tokens[2] if len(tokens) > 2 else 'little'
            struct_total_size = int(tokens[3], 0) if len(tokens) > 3 else 0
            struct_total_bits = int(tokens[4], 0) if len(tokens) > 4 else 0
            struct_fields     = []
            continue

        if directive == 'FIELD':
            if not in_struct:
                raise FVMParseError("'field' outside 'struct' block", lineno)
            if len(tokens) < 5:
                raise FVMParseError("'field' requires fname ttag byte_offset byte_size", lineno)
            fname   = tokens[1]
            fttag   = _parse_ttag(tokens[2])
            foff    = int(tokens[3], 0)
            fsz     = int(tokens[4], 0)
            struct_fields.append((fname, fttag, foff, fsz))
            continue

        if directive == 'ENDSTRUCT':
            if not in_struct:
                raise FVMParseError("'endstruct' without matching 'struct'", lineno)
            layout = StructLayout(
                name=struct_name,
                fields=struct_fields,
                total_size=struct_total_size,
                endian=struct_endian,
                total_bits=struct_total_bits,
            )
            struct_layouts[struct_name] = layout
            in_struct = False
            continue

        if directive == 'FUNC':
            if in_func:
                raise FVMParseError("nested 'func' is not allowed", lineno)
            if len(tokens) < 2:
                raise FVMParseError("'func' requires a name", lineno)
            in_func     = True
            func_name   = tokens[1]
            func_instrs = []
            continue

        if directive == 'ENDFUNC':
            if not in_func:
                raise FVMParseError("'endfunc' without matching 'func'", lineno)
            functions[func_name] = func_instrs
            in_func   = False
            func_name = ''
            continue

        op_name = tokens[0].upper()
        if op_name not in op_by_name:
            raise FVMParseError(f'unknown opcode {tokens[0]!r}', lineno)
        op = op_by_name[op_name]

        try:
            operands = _parse_operands(op, tokens[1:], lineno)
        except FVMParseError:
            raise
        except Exception as e:
            raise FVMParseError(str(e), lineno)

        instr = Instr(op, operands)
        if in_func:
            func_instrs.append(instr)
        else:
            instructions.append(instr)

    if in_func:
        raise FVMParseError(f"unterminated 'func {func_name}' at end of file", len(lines_src))
    if in_struct:
        raise FVMParseError(f"unterminated 'struct {struct_name}' at end of file", len(lines_src))

    return instructions, local_count, functions, struct_layouts

def _parse_operand_token(token: str):
    if (token.startswith('"') and token.endswith('"')) or ':' in token:
        return _parse_fvm_val(token)
    try:
        return int(token, 0)
    except ValueError:
        return token


def _parse_operands(op: Op, raw_tokens: List[str], lineno: int) -> list:
    _zero_op = {
        Op.POP, Op.DUP, Op.SWAP, Op.ROT, Op.OVER,
        Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD, Op.NEG, Op.POW,
        Op.ABS, Op.MIN, Op.MAX, Op.CLAMP,
        Op.BAND, Op.BOR, Op.BXOR, Op.BNOT, Op.SHL, Op.SHR,
        Op.POPCOUNT,
        Op.CMP_EQ, Op.CMP_NE, Op.CMP_LT, Op.CMP_LE, Op.CMP_GT, Op.CMP_GE,
        Op.AND, Op.OR, Op.NOT,
        Op.RET, Op.HALT,
        Op.LOCAL_DEREF, Op.LOCAL_DEREF_SET,
        Op.ALLOC, Op.FREE, Op.OFFSET,
        Op.ARRAY_LEN, Op.INDEX_GET, Op.INDEX_SET,
        Op.TYPEOF,
        Op.IO_OPEN, Op.IO_READ, Op.IO_WRITE, Op.IO_CLOSE,
        Op.FFI_FREE,
        Op.COMPILER_PRINT, Op.COMPILER_INPUT,
        Op.COMPILER_READFILE, Op.COMPILER_WRITEFILE,
        Op.COMPILER_FVM_DUMP,
        Op.STR_LEN, Op.STR_CAT, Op.STR_SLICE, Op.STR_EQ, Op.STR_FIND,
        Op.INT_TO_STR, Op.STR_TO_INT,
        Op.ASSERT, Op.WARN, Op.PANIC,
        Op.EMIT_CONST, Op.EMIT_TYPE,
    }
    if op in _zero_op:
        return []

    if op == Op.PUSH:
        if not raw_tokens:
            raise FVMParseError('PUSH requires a value operand', lineno)
        return [_parse_fvm_val(raw_tokens[0])]

    if op == Op.CALL_PTR:
        if not raw_tokens:
            raise FVMParseError('CALL_PTR requires an argc operand', lineno)
        return [int(raw_tokens[0], 0)]

    _single_int_op = {
        Op.JMP, Op.JIF, Op.JNF,
        Op.LOCAL_GET, Op.LOCAL_SET,
        Op.TAIL_SELF,
        Op.EMIT_GLOBAL,
    }
    if op in _single_int_op:
        if not raw_tokens:
            raise FVMParseError(f'{op.name} requires an integer operand', lineno)
        return [int(raw_tokens[0], 0)]

    _width_op = {Op.ROTL, Op.ROTR, Op.BITREV, Op.CLZ, Op.CTZ}
    if op in _width_op:
        if not raw_tokens:
            raise FVMParseError(f'{op.name} requires a width operand', lineno)
        return [int(raw_tokens[0], 0)]

    _type_name_op = {Op.SIZEOF, Op.ALIGNOF, Op.ENDIANOF, Op.STRUCT_NEW}
    if op in _type_name_op:
        if not raw_tokens:
            raise FVMParseError(f'{op.name} requires a type name operand', lineno)
        return [raw_tokens[0]]

    if op in (Op.FIELD_GET, Op.FIELD_SET):
        if not raw_tokens:
            raise FVMParseError(f'{op.name} requires a field name operand', lineno)
        return [raw_tokens[0]]

    if op == Op.ARRAY_NEW:
        if len(raw_tokens) < 2:
            raise FVMParseError('ARRAY_NEW requires type_name and count', lineno)
        return [raw_tokens[0], int(raw_tokens[1], 0)]

    if op in (Op.LOAD, Op.STORE):
        if len(raw_tokens) < 2:
            raise FVMParseError(f'{op.name} requires ttag and byte_size', lineno)
        return [_parse_ttag(raw_tokens[0]), int(raw_tokens[1], 0)]

    if op in (Op.CAST, Op.BITCAST):
        if not raw_tokens:
            raise FVMParseError(f'{op.name} requires a type tag operand', lineno)
        return [_parse_ttag(raw_tokens[0])]

    if op == Op.CALL:
        if len(raw_tokens) < 2:
            raise FVMParseError('CALL requires name and argc', lineno)
        return [raw_tokens[0], int(raw_tokens[1], 0)]

    if op == Op.FFI_LOAD:
        if not raw_tokens:
            raise FVMParseError('FFI_LOAD requires a library path', lineno)
        return [raw_tokens[0].strip('"')]

    if op == Op.FFI_SYM:
        if not raw_tokens:
            raise FVMParseError('FFI_SYM requires a symbol name', lineno)
        return [raw_tokens[0]]

    if op == Op.FFI_CALL:
        if len(raw_tokens) < 2:
            raise FVMParseError('FFI_CALL requires argc and ret_type_tag', lineno)
        return [int(raw_tokens[0], 0), _parse_ttag(raw_tokens[1])]

    if op == Op.JTABLE:
        if len(raw_tokens) < 1:
            raise FVMParseError('JTABLE requires at least a default address', lineno)
        default = int(raw_tokens[0], 0)
        table   = [int(t, 0) for t in raw_tokens[1:]]
        return [default, table]

    if op == Op.EMITFLUX:
        if not raw_tokens:
            raise FVMParseError('EMITFLUX requires a source text operand', lineno)
        src_text = raw_tokens[0].strip('"').encode('utf-8').decode('unicode_escape')
        var_names = []
        for tok in raw_tokens[1:]:
            if ':' in tok:
                n, _, s = tok.partition(':')
                var_names.append((n, int(s, 0)))
        return [src_text, var_names]

    if op == Op.STR_SLICE:
        return []

    raise FVMParseError(f'unhandled operand encoding for {op.name}', lineno)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _fvm_main():
    import sys as _sys

    if len(_sys.argv) < 2:
        print('Flux VM')
        print('Usage: python fvm.py <program.fvm>')
        print()
        print('Runs a .fvm opcode file on the Flux Virtual Machine.')
        _sys.exit(0)

    path = _sys.argv[1]
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            source = fh.read()
    except OSError as e:
        print(f'fvm: cannot open {path!r}: {e}', file=_sys.stderr)
        _sys.exit(1)

    try:
        instructions, local_count, functions, struct_layouts = parse_fvm_source(source)
    except FVMParseError as e:
        print(f'fvm: parse error in {path}: {e}', file=_sys.stderr)
        _sys.exit(1)

    vm = FluxVM(struct_layouts=struct_layouts)
    for name, instrs in functions.items():
        vm.register_function(name, instrs)

    try:
        result = vm.execute(instructions, local_count)
    except VMError as e:
        print(f'fvm: runtime error: {e}', file=_sys.stderr)
        _sys.exit(1)

    for entry in vm.emit_results:
        kind = entry[0]
        if kind == 'const':
            print(f'[emit:const] {entry[1]}')
        elif kind == 'global':
            print(f'[emit:global] {entry[1].hex()}')
        elif kind == 'type':
            print(f'[emit:type] {entry[1]}')
        elif kind == 'flux':
            print(f'[emit:flux] {entry[1]}')


if __name__ == '__main__':
    _fvm_main()