// Author: Karac V. Thweatt

///
Flux Shadow Stack — Windows x64

A software shadow stack that mirrors return addresses for each call frame.
On entry to a guarded function, the real return address is pushed onto the
shadow stack. On exit, the saved address is compared against the live return
address on the hardware stack; a mismatch triggers an immediate abort.

Usage:
    #def FLUX_SHADOW_STACK 1;
    #import "standard.fx";      // pulls in runtime.fx -> shadowstack.fx

Public API (called by compiler-generated prologues / epilogues):
    shadowstack::push(void* ret_addr) -> void
    shadowstack::pop(void* ret_addr)  -> void   // aborts on mismatch
    shadowstack::init()               -> void   // called by FRTStartup
    shadowstack::teardown()           -> void   // called after main() returns

Internal layout (per-thread, heap-allocated):
    void*[]  — array of saved return addresses
    uint     — stack pointer (index of next free slot)
    uint     — capacity (in slots)
///

#ifndef FLUX_SHADOW_STACK
#stop "shadowstack.fx imported without FLUX_SHADOW_STACK defined.";
#endif;

#ifndef FLUX_STANDARD_MEMORY
#import "memory.fx";
#endif;

using standard::memory::allocators::stdheap;

namespace shadowstack
{

// -----------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------

#def SHADOW_STACK_INIT_CAPACITY  4096;   // initial slots
#def SHADOW_STACK_GROW_FACTOR    2;      // double on overflow

// -----------------------------------------------------------------------
// Internal state  (one per thread — single-thread runtime for now)
// -----------------------------------------------------------------------

// Plain pointer array: _stack[n] holds the nth saved return address.
global void** _stack = void;
global uint   _sp    = 0;
global uint   _cap   = 0;

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

def _grow() -> void
{
    uint new_cap = _cap * (uint)SHADOW_STACK_GROW_FACTOR;
    void** new_stack = (void**)fmalloc((u64)new_cap * (u64)8);

    // Bulk-copy existing slots via memcpy
    memcpy((void*)new_stack, (void*)_stack, (u64)_sp * (u64)8);

    ffree(long(_stack));
    _stack = new_stack;
    _cap   = new_cap;
};

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

def init() -> void
{
    _stack = (void**)fmalloc((u64)(uint)SHADOW_STACK_INIT_CAPACITY * (u64)8);
    _sp    = 0;
    _cap   = (uint)SHADOW_STACK_INIT_CAPACITY;
};

def teardown() -> void
{
    ffree(long(_stack));
    _stack = void;
    _sp    = 0;
    _cap   = 0;
};

///
Push a return address onto the shadow stack.
Called in the prologue of every guarded function.
///
def push(void* ret_addr) -> void
{
    if (_sp >= _cap)
    {
        _grow();
    };
    _stack[_sp] = ret_addr;
    _sp++;
};

///
Pop and verify a return address from the shadow stack.
Called in the epilogue of every guarded function, just before ret.
Aborts the process if the live return address does not match the saved one.
///
def pop(void* ret_addr) -> void
{
    if (_sp == 0)
    {
        standard::io::console::print("SHADOW STACK UNDERFLOW\n\0");
        abort();
        noreturn;
    };

    _sp--;
    void* saved = _stack[_sp];

    if (saved != ret_addr)
    {
        standard::io::console::print("SHADOW STACK VIOLATION: return address corrupted\n\0");
        abort();
        noreturn;
    };
};

}; // namespace shadowstack
