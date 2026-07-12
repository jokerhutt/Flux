"""
fbc_alias.py -- Alias map and violation detection for the Flux Borrow Checker.

Copyright (C) 2026 Karac V. Thweatt

The alias map tracks every live pointer variable and what allocation site it
refers to. On each assignment, declaration, and scope exit it is updated.
Violations are collected (never raised) so the full program is always walked.

Allocation site identity strings:
  "stack:<func>:<varname>@<line>"   -- stack-allocated variable
  "heap:<func>:<line>"              -- fmalloc call site
  "param:<func>:<paramname>"        -- pointer parameter (borrowed, no owned site)
  "derived:<base_site>"             -- pointer arithmetic result
  "unknown"                         -- could not resolve (extern, etc.)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PtrInfo:
    """Tracks one live pointer variable."""
    var_name: str
    site: str           # allocation site identity
    mutable: bool       # is this a mutable view of the site
    scope_depth: int    # scope depth at which this pointer was declared
    func_name: str      # enclosing function name
    file: str
    line: int


@dataclass
class Violation:
    kind: str           # 'mutable_alias' | 'scope_escape' | 'use_after_scope' | 'heap_leak'
    message: str
    file: str
    line: int
    detail: list = field(default_factory=list)  # extra context lines

    def format(self) -> str:
        lines = [f"[FBC] {self.kind}"]
        lines.append(f"  {self.file}:{self.line}  {self.message}")
        for d in self.detail:
            lines.append(f"  {d}")
        return '\n'.join(lines)


class ScopeFrame:
    """One scope level's worth of pointer tracking."""
    def __init__(self, depth: int, func_name: str):
        self.depth = depth
        self.func_name = func_name
        # var_name -> PtrInfo
        self.ptrs: dict[str, PtrInfo] = {}
        # allocation sites that are owned at this scope (stack vars)
        self.owned_sites: set[str] = set()
        # var_name -> site string for all stack variables (for address-of resolution)
        self.var_sites: dict[str, str] = {}

    def declare_stack_site(self, var_name: str, site: str, line: int, file: str):
        """Register a new stack allocation site owned by this scope."""
        self.owned_sites.add(site)

    def add_ptr(self, info: PtrInfo):
        self.ptrs[info.var_name] = info

    def remove_ptr(self, var_name: str):
        self.ptrs.pop(var_name, None)

    def all_ptrs(self) -> list[PtrInfo]:
        return list(self.ptrs.values())


class AliasMap:
    """
    Maintains the full stack of scope frames for the current execution path.
    Collects all violations without stopping.
    """

    def __init__(self):
        self.frames: list[ScopeFrame] = []
        self.violations: list[Violation] = []
        self.func_name = '<global>'
        self.file = '<unknown>'
        # heap allocation sites that have been fmalloc'd but not ffree'd
        # site -> (file, line)
        self.heap_sites: dict[str, tuple[str, int]] = {}
        # sites that have been passed to a spawned thread -- any subsequent
        # mutable access to these sites is a race condition
        # site -> (thread_file, thread_line)
        self.thread_escaped_sites: dict[str, tuple[str, int]] = {}

    # ------------------------------------------------------------------
    # Scope management
    # ------------------------------------------------------------------

    def push_scope(self):
        depth = len(self.frames)
        self.frames.append(ScopeFrame(depth, self.func_name))

    def pop_scope(self, check_leaks: bool = False):
        if not self.frames:
            return
        frame = self.frames.pop()
        # any pointer at this scope depth that pointed into this scope's
        # owned sites now dangles -- check all surviving frames for them
        if frame.owned_sites:
            self._check_escape_after_pop(frame.owned_sites, frame)
        # remove ptrs declared at this depth from all frames
        for f in self.frames:
            dead = [n for n, p in f.ptrs.items() if p is not None and p.scope_depth == frame.depth]
            for n in dead:
                f.ptrs.pop(n)

    def _check_escape_after_pop(self, dead_sites: set[str], dead_frame: ScopeFrame):
        """
        After a scope exits, check if any surviving pointer still references
        one of the now-dead stack sites.
        """
        for frame in self.frames:
            for ptr in frame.all_ptrs():
                if ptr.site in dead_sites:
                    self.violations.append(Violation(
                        kind='scope_escape',
                        message=(
                            f"pointer '{ptr.var_name}' outlives its stack allocation "
                            f"'{ptr.site}' (allocation scope exited)"
                        ),
                        file=ptr.file,
                        line=ptr.line,
                    ))

    # ------------------------------------------------------------------
    # Pointer declaration / assignment
    # ------------------------------------------------------------------

    def declare_ptr(self, var_name: str, site: str, mutable: bool,
                    file: str, line: int, is_stack_owner: bool = False):
        """
        Register a new pointer variable in the current scope.
        is_stack_owner=True means this var IS the stack allocation (its address
        may be taken later -- we track the site now).
        """
        if not self.frames:
            return
        frame = self.frames[-1]
        if is_stack_owner:
            frame.declare_stack_site(var_name, site, line, file)

        info = PtrInfo(
            var_name=var_name,
            site=site,
            mutable=mutable,
            scope_depth=frame.depth,
            func_name=self.func_name,
            file=file,
            line=line,
        )
        frame.add_ptr(info)
        self._check_mutable_alias(info)

    def assign_ptr(self, var_name: str, site: str, mutable: bool,
                   file: str, line: int):
        """Update an existing pointer variable to point to a new site."""
        info = self._find_ptr(var_name)
        if info is None:
            # new pointer introduced via assignment without prior declaration
            self.declare_ptr(var_name, site, mutable, file, line)
            return
        old_site = info.site
        info.site = site
        info.mutable = mutable
        info.file = file
        info.line = line
        self._check_mutable_alias(info)

    def _find_ptr(self, var_name: str) -> Optional[PtrInfo]:
        """Search all frames top-down for a pointer variable."""
        for frame in reversed(self.frames):
            if var_name in frame.ptrs:
                return frame.ptrs[var_name]
        return None

    # ------------------------------------------------------------------
    # Heap tracking
    # ------------------------------------------------------------------

    def record_malloc(self, site: str, file: str, line: int):
        self.heap_sites[site] = (file, line)

    def record_free(self, site: str):
        self.heap_sites.pop(site, None)

    def check_heap_leaks(self):
        """Call at function exit to report unfreed heap allocations."""
        for site, (file, line) in self.heap_sites.items():
            self.violations.append(Violation(
                kind='heap_leak',
                message=f"heap allocation '{site}' has no reachable ffree on all paths",
                file=file,
                line=line,
            ))
        self.heap_sites.clear()

    # ------------------------------------------------------------------
    # Alias checking
    # ------------------------------------------------------------------

    def _all_live_ptrs(self) -> list[PtrInfo]:
        result = []
        for frame in self.frames:
            result.extend(frame.all_ptrs())
        return result

    def _check_mutable_alias(self, new_ptr: PtrInfo):
        """
        Check if new_ptr creates a mutable aliasing violation with any
        currently live pointer pointing to the same site.
        """
        if new_ptr.site == 'unknown' or new_ptr.site.startswith('derived:'):
            return
        for existing in self._all_live_ptrs():
            if existing.var_name == new_ptr.var_name:
                continue
            if existing.site != new_ptr.site:
                continue
            # same site -- violation if either is mutable
            if new_ptr.mutable or existing.mutable:
                self.violations.append(Violation(
                    kind='mutable_alias',
                    message=(
                        f"mutable alias violation on site '{new_ptr.site}'"
                    ),
                    file=new_ptr.file,
                    line=new_ptr.line,
                    detail=[
                        f"'{new_ptr.var_name}' -> {new_ptr.site} "
                        f"({'mutable' if new_ptr.mutable else 'immutable'}) "
                        f"at {new_ptr.file}:{new_ptr.line}",
                        f"'{existing.var_name}' -> {existing.site} "
                        f"({'mutable' if existing.mutable else 'immutable'}) "
                        f"at {existing.file}:{existing.line}",
                    ]
                ))

    # ------------------------------------------------------------------
    # Thread safety
    # ------------------------------------------------------------------

    def check_thread_escape(self, ptr_var: str, file: str, line: int):
        """
        Called when a pointer is passed to a spawned thread.
        Phase 1: check for existing mutable aliases at spawn time.
        Phase 2: mark the site as thread-escaped so any subsequent
                 mutable access in the caller is flagged as a race.
        """
        info = self._find_ptr(ptr_var)
        if info is None:
            return

        # Phase 1 -- alias already exists at spawn time
        for existing in self._all_live_ptrs():
            if existing.var_name == ptr_var:
                continue
            if existing.site != info.site:
                continue
            if existing.site == 'unknown' or existing.site.startswith('derived:'):
                continue
            if existing.mutable or info.mutable:
                self.violations.append(Violation(
                    kind='thread_escape',
                    message=(
                        f"pointer '{ptr_var}' passed to thread while mutable "
                        f"alias '{existing.var_name}' exists on site '{info.site}'"
                    ),
                    file=file,
                    line=line,
                    detail=[
                        f"'{existing.var_name}' -> {existing.site} "
                        f"at {existing.file}:{existing.line}",
                    ]
                ))

        # Phase 2 -- mark site as escaped so future accesses are flagged
        # skip unknown sites to avoid false positives from unresolved pointers
        if info.mutable and info.site != 'unknown' and not info.site.startswith('derived:'):
            self.thread_escaped_sites[info.site] = (file, line)

    def check_use_after_escape(self, ptr_var: str, file: str, line: int):
        """
        Call this when a pointer is read or written after a spawn.
        Flags if the pointer's site was passed to a thread.
        """
        info = self._find_ptr(ptr_var)
        if info is None:
            return
        if info.site in self.thread_escaped_sites:
            spawn_file, spawn_line = self.thread_escaped_sites[info.site]
            # skip if this is the same line as the escape (the spawn call itself)
            if spawn_line == line:
                return
            self.violations.append(Violation(
                kind='thread_race',
                message=(
                    f"pointer '{ptr_var}' accessed after its site '{info.site}' "
                    f"was passed to a thread (potential data race)"
                ),
                file=file,
                line=line,
                detail=[
                    f"site escaped to thread at {spawn_file}:{spawn_line}",
                    f"'{ptr_var}' -> {info.site} (mutable) at {file}:{line}",
                ]
            ))