import io
import json
import os
import sys
from pathlib import Path
from typing import Set, Dict, Optional, List

# Reconfigure stdout/stderr to UTF-8 so Unicode characters in diagnostics
# (arrows, checkmarks, box-drawing, etc.) don't crash on Windows consoles
# that default to a narrow codepage like CP1252.
for _stream_name in ('stdout', 'stderr'):
    _stream = getattr(sys, _stream_name)
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')
    elif hasattr(_stream, 'buffer'):
        setattr(sys, _stream_name,
                io.TextIOWrapper(_stream.buffer, encoding='utf-8', errors='replace'))

FLUXC_SRCDIR = Path(os.environ.get("FLUXC_SRCDIR", Path(__file__).parent)).resolve()

UTF8_BOM = '\ufeff'

def _read_source_file(path) -> str:
    """Read a source file as UTF-8, stripping any BOM and warning on
    invalid byte sequences rather than crashing hard."""
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:   # utf-8-sig auto-strips BOM
            return f.read()
    except UnicodeDecodeError:
        # Fall back: replace undecodable bytes with the replacement char and warn.
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        # Strip BOM manually in case utf-8-sig path wasn't reached
        if content.startswith(UTF8_BOM):
            content = content[1:]
        print(
            f"[PREPROCESSOR] WARNING: '{path}' contains invalid UTF-8 sequences. "
            "Replacement characters (\ufffd) have been substituted. "
            "Save the file as UTF-8 to silence this warning.",
            file=sys.stderr,
        )
        return content

class FXPreprocessor:
    def __init__(self, source_file, compiler_constants=None):
        self.source_file = source_file
        self.processed_files: Set[str] = set()
        self.output_lines = []
        self.constants: Dict[str, str] = {}
        self.lib_dirs: List[str] = []
        # Maps each output line index (0-based) -> (filename, local_line_number 1-based)
        self.line_map: List[tuple] = []
        # Stack of directories for the files currently being processed (for local imports)
        self._dir_stack: List[Path] = []

        if compiler_constants:
            self.constants.update(compiler_constants)
    
    def process(self) -> str:
        """Main processing pipeline"""
        # Step 1: Process all imports and build content in memory
        self._process_file(self.source_file)
        
        # Step 2: Build combined source
        combined_source = '\n'.join(self.output_lines)
        
        # Step 4: Keep replacing constants until no more replacements occur
        replaced = True
        iteration = 0
        while replaced:
            iteration += 1
            print(f"[PREPROCESSOR] constant substitution passes: {iteration}")
            replaced = False
            lines = combined_source.split('\n')
            new_lines = []
            
            for line in lines:
                new_line = self._substitute_constants(line)
                if new_line != line:
                    replaced = True
                new_lines.append(new_line)
            
            combined_source = '\n'.join(new_lines)
        ending = "es." if iteration > 1 else "."
        print(f"[PREPROCESSOR] Completed after {iteration} constant pass{ending}")
        
        # Step 5: Write to build/tmp.fx
        build_dir = Path("build")
        build_dir.mkdir(exist_ok=True)
        output_file = build_dir / "tmp.fx"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(combined_source)
        
        print(f"[PREPROCESSOR] Generated: {output_file}")
        print(f"[PREPROCESSOR] Processed {len(self.processed_files)} file(s)")
        
        return combined_source
    
    def _strip_comments(self, content: str) -> str:
        """Strip all comments from content: // and /// ... ///
        String literals (double or single quoted) are passed through unchanged
        so that URLs like https:// inside strings are not treated as comments."""
        result = []
        i = 0
        n = len(content)

        while i < n:
            c = content[i]

            # ── String literal: skip over it intact ──────────────────────────
            if c == '"' or c == "'":
                quote = c
                result.append(c)
                i += 1
                while i < n:
                    sc = content[i]
                    result.append(sc)
                    if sc == '\\' and i + 1 < n:
                        # Escaped character — consume both chars so an escaped
                        # quote doesn't end the string prematurely.
                        i += 1
                        result.append(content[i])
                    elif sc == quote:
                        break
                    i += 1
                i += 1
                continue

            # ── Block comment: /// ... /// ────────────────────────────────────
            if i + 2 < n and c == '/' and content[i+1] == '/' and content[i+2] == '/':
                i += 3
                while i < n:
                    if i + 2 < n and content[i] == '/' and content[i+1] == '/' and content[i+2] == '/':
                        i += 3
                        break
                    i += 1
                continue

            # ── Line comment: // ──────────────────────────────────────────────
            if i + 1 < n and c == '/' and content[i+1] == '/':
                while i < n and content[i] != '\n':
                    i += 1
                continue

            # ── Regular character ─────────────────────────────────────────────
            result.append(c)
            i += 1

        return ''.join(result)
    
    # ------------------------------------------------------------------
    # Import resolution -- three distinct search domains
    # ------------------------------------------------------------------

    def _resolve_path_local(self, filepath: str) -> Optional[Path]:
        """Resolve a local import (#import "file.fx").
        Search order:
          1. Relative to the importing file's directory (top of _dir_stack).
          2. Relative to the initial source file's directory.
          3. CWD.
          4. Any directories added via #dir.
        """
        locations: List[Path] = []

        # 1. Relative to the currently-processing file's directory
        if self._dir_stack:
            locations.append(self._dir_stack[-1] / filepath)

        # 2. Relative to the root source file's directory
        root_dir = Path(self.source_file).resolve().parent
        locations.append(root_dir / filepath)

        # 3. CWD
        locations.append(Path.cwd() / filepath)

        # 4. Extra dirs from #dir
        for lib_dir in self.lib_dirs:
            locations.append(Path(lib_dir) / filepath)

        for loc in locations:
            if loc.exists():
                return loc.resolve()
        return None

    def _resolve_path_stdlib(self, filepath: str) -> Optional[Path]:
        """Resolve a stdlib import (#import <file.fx>).
        Only the stdlib trees under FLUXC_SRCDIR are searched.
        """
        cwd = FLUXC_SRCDIR
        locations = [
            cwd / "src" / "stdlib" / filepath,
            cwd / "src" / "stdlib" / "runtime" / filepath,
            cwd / "src" / "stdlib" / "functions" / filepath,
            cwd / "src" / "stdlib" / "builtins" / filepath,
            cwd / "src" / "stdlib" / "utility" / filepath,
        ]
        for loc in locations:
            if Path(loc).exists():
                return Path(loc).resolve()
        return None

    def _resolve_path(self, filepath: str) -> Optional[Path]:
        """Legacy resolver used internally (e.g. for the root source file).
        Searches local, stdlib, and package trees in order.
        """
        path = Path(filepath)
        if path.exists():
            return path.resolve()

        result = self._resolve_path_local(filepath)
        if result:
            return result
        result = self._resolve_path_stdlib(filepath)
        if result:
            return result
        return self._resolve_path_stdlib(filepath)
    
    def _process_package(self, package_name: str):
        """Resolve and import a package by name via #package.
        Reads .fpm/packages/<name>/package.json, finds the entrypoint,
        and processes it with the package directory on the dir stack so
        relative imports inside the package resolve correctly.
        """
        fpm_packages_dir = Path.cwd() / ".fpm" / "packages"
        package_dir = fpm_packages_dir / package_name
        if not package_dir.is_dir():
            raise FileNotFoundError(
                f"[PREPROCESSOR] Package '{package_name}' not found in {fpm_packages_dir}"
            )

        manifest_path = package_dir / "package.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"[PREPROCESSOR] Package '{package_name}' is missing package.json"
            )

        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        # Support both {"entrypoint": "..."} at top level and nested under "entries"
        entries = manifest.get("entries", manifest)
        entrypoint = entries.get("entrypoint")
        if not entrypoint:
            raise KeyError(
                f"[PREPROCESSOR] Package '{package_name}' package.json has no 'entrypoint' field"
            )

        entrypoint_path = (package_dir / entrypoint).resolve()
        if not entrypoint_path.exists():
            raise FileNotFoundError(
                f"[PREPROCESSOR] Package '{package_name}' entrypoint '{entrypoint}' not found"
            )

        # Check if already processed (guard by absolute path)
        abs_path = str(entrypoint_path)
        if abs_path in self.processed_files:
            return

        print(f"[PREPROCESSOR] Package import: {package_name} -> {entrypoint}")

        # Push the package directory so the entrypoint's local imports resolve within it
        self._dir_stack.append(package_dir)
        self._process_file(str(entrypoint_path), resolver=self._resolve_path_local)
        self._dir_stack.pop()

    def _process_file(self, filepath: str, resolver=None):
        """Process a file and its imports.
        resolver: callable(filepath)->Optional[Path] to use instead of _resolve_path.
        """
        if resolver is None:
            resolver = self._resolve_path
        # Normalize path separators to the OS convention so that Windows-style
        # backslash paths in source files (e.g. runtime\runtime.fx) resolve
        # correctly on both Windows and Unix.
        filepath = filepath.replace('\\', os.sep).replace('/', os.sep)
        resolved_path = resolver(filepath)
        
        if not resolved_path:
            # Check if it looks like a stdlib file used with "" instead of <>
            if resolver.__name__ == "_resolve_path_local" and self._resolve_path_stdlib(filepath):
                print(f'[PREPROCESSOR] Could not find local import: {filepath}', flush=True)
                print(f'#import "{filepath}";', flush=True)
                print(f'--------^', flush=True)
                print(f'#import <{filepath}>; // try this', flush=True)
            raise FileNotFoundError(f"Could not find import: {filepath}")
        
        # Avoid circular imports
        abs_path = str(resolved_path.resolve())
        if abs_path in self.processed_files:
            return
        
        self.processed_files.add(abs_path)
        print(f"[PREPROCESSOR] Processing: {filepath}")
        
        # Read the file and strip comments immediately
        file_content = _read_source_file(resolved_path)
        
        # Strip all comments before processing
        file_content = self._strip_comments(file_content)
        
        # Enforce semicolons on directives before processing
        for lineno, raw_line in enumerate(file_content.splitlines(), start=1):
            s = raw_line.strip()
            if s.startswith("#import") or s.startswith("#package") or s.startswith("#warn") or s.startswith("#stop") or s.startswith("#def") or s.startswith("#dir"):
                if not s.endswith(';'):
                    directive = s.split()[0]
                    raise SyntaxError(f"[PREPROCESSOR] {directive} directive missing semicolon in {filepath} at line {lineno}")
        
        # Process line by line, tracking current file and local line number
        lines = file_content.splitlines()
        prev_current_file = getattr(self, '_current_file', None)
        prev_current_lines = getattr(self, '_current_lines', None)
        self._current_file = str(resolved_path)
        self._current_lines = lines
        # Push this file's directory onto the stack so nested local imports resolve correctly
        self._dir_stack.append(resolved_path.parent)
        i = 0
        while i < len(lines):
            self._current_local_lineno = i + 1  # 1-based
            i = self._process_line(lines, i)
        self._dir_stack.pop()
        self._current_file = prev_current_file
        self._current_lines = prev_current_lines
    
    def _process_line(self, lines: List[str], i: int) -> int:
        """Process a single line, return next line index"""
        line = lines[i]
        
        # Skip empty lines but preserve them in line_map so line numbers stay accurate
        if not line.strip():
            self.line_map.append((getattr(self, '_current_file', self.source_file), self._current_local_lineno))
            self.output_lines.append('')
            return i + 1
        
        stripped = line.strip()
        
        # Check for #dir
        if stripped.startswith("#dir"):
            if not stripped.rstrip().endswith(';'):
                raise SyntaxError(f"[PREPROCESSOR] #dir directive missing semicolon at line {i + 1}")
            start_idx = line.find('"')
            if start_idx != -1:
                end_idx = line.find('"', start_idx + 1)
                if end_idx != -1:
                    dir_path = line[start_idx + 1:end_idx]
                    # Normalize: replace backslashes with forward slashes
                    dir_path = dir_path.replace('\\', '/')
                    if dir_path not in self.lib_dirs:
                        self.lib_dirs.append(dir_path)
                        print(f"[PREPROCESSOR] Added library directory: {dir_path}")
            self.line_map.append((getattr(self, '_current_file', self.source_file), self._current_local_lineno))
            self.output_lines.append('')
            return i + 1
        
        # Check for #def
        if stripped.startswith("#def"):
            # Find the semicolon
            semicolon_pos = line.find(';')
            if semicolon_pos == -1:
                semicolon_pos = len(line)
            
            # Extract the part up to semicolon
            constant_line = line[:semicolon_pos].strip()
            
            parts = constant_line.split()
            if len(parts) >= 3:
                constant_name = parts[1]
                # Join the rest as the value (skip #def and constant_name)
                constant_value = ' '.join(parts[2:]).strip()
                
                # Remove any trailing semicolon if it's still there
                if constant_value.endswith(';'):
                    constant_value = constant_value.rstrip(';').strip()
                    
                self.constants[constant_name] = constant_value
                print(f"[PREPROCESSOR] Defined constant: {constant_name} = {constant_value}")
            self.line_map.append((getattr(self, '_current_file', self.source_file), self._current_local_lineno))
            self.output_lines.append('')
            return i + 1
        
        # Check for #ifdef
        if stripped.startswith("#ifdef"):
            parts = line.split()
            if len(parts) >= 2:
                constant_name = parts[1]
                return self._process_conditional_block(lines, i, constant_name, False)
        
        # Check for #ifndef
        if stripped.startswith("#ifndef"):
            parts = line.split()
            if len(parts) >= 2:
                constant_name = parts[1]
                return self._process_conditional_block(lines, i, constant_name, True)
        
        # Check for #package
        if stripped.startswith("#package"):
            if not stripped.rstrip().endswith(';'):
                raise SyntaxError(f"[PREPROCESSOR] #package directive missing semicolon at line {i + 1}")
            rest = stripped[len('#package'):].rstrip(';').strip()
            package_names = [p.strip() for p in rest.split(',') if p.strip()]
            for pkg_name in package_names:
                self._process_package(pkg_name)
            self.line_map.append((getattr(self, '_current_file', self.source_file), self._current_local_lineno))
            self.output_lines.append('')
            return i + 1

                # Check for import
        if stripped.startswith("#import"):
            if not stripped.rstrip().endswith(';'):
                raise SyntaxError(f"[PREPROCESSOR] #import directive missing semicolon at line {i + 1}")
            # Determine import kind and extract filenames.
            #
            #   #import "file.fx";          -- local: resolve relative to importing file / root dir / CWD
            #   #import <file.fx>;          -- stdlib: search only stdlib trees
            #   #import @"pkg/file.fx";     -- package: search only .fpm/packages tree
            #
            # Multiple imports on one line are supported for all forms.

            rest = line[line.index('#import') + len('#import'):].rstrip(';').strip()

            # Parse each token from left to right
            j = 0
            while j < len(rest):
                c = rest[j]

                # Stdlib import: <...>
                if c == '<':
                    j += 1
                    end = rest.find('>', j)
                    if end == -1:
                        raise SyntaxError(f"[PREPROCESSOR] Unterminated <> in #import at line {i + 1}")
                    std_path = rest[j:end].strip()
                    print(f"[PREPROCESSOR] Stdlib import: {std_path}")
                    self._process_file(std_path, resolver=self._resolve_path_stdlib)
                    j = end + 1

                # Local import: "..."
                elif c == '"':
                    j += 1
                    end = rest.find('"', j)
                    if end == -1:
                        raise SyntaxError(f"[PREPROCESSOR] Unterminated \"\" in #import at line {i + 1}")
                    local_path = rest[j:end].strip()
                    print(f"[PREPROCESSOR] Local import: {local_path}")
                    self._process_file(local_path, resolver=self._resolve_path_local)
                    j = end + 1

                else:
                    j += 1

            self.line_map.append((getattr(self, '_current_file', self.source_file), self._current_local_lineno))
            self.output_lines.append('')
            return i + 1
        
        # Check for #warn
        if stripped.startswith("#warn"):
            if not stripped.rstrip().endswith(';'):
                raise SyntaxError(f"[PREPROCESSOR] #warn directive missing semicolon at line {i + 1}")
            start_idx = line.find('"')
            if start_idx != -1:
                end_idx = line.find('"', start_idx + 1)
                if end_idx != -1:
                    message = line[start_idx + 1:end_idx]
                    print(f"[PREPROCESSOR] {message}")
            self.line_map.append((getattr(self, '_current_file', self.source_file), self._current_local_lineno))
            self.output_lines.append('')
            return i + 1
        
        # Check for #stop
        if stripped.startswith("#stop"):
            if not stripped.rstrip().endswith(';'):
                raise SyntaxError(f"[PREPROCESSOR] #stop directive missing semicolon at line {i + 1}")
            start_idx = line.find('"')
            if start_idx != -1:
                end_idx = line.find('"', start_idx + 1)
                if end_idx != -1:
                    message = line[start_idx + 1:end_idx]
                    print(f"[PREPROCESSOR] {message}")
            print("Compilation failed, preprocessor stopped by constant.")
            raise SystemExit(1)
        
        # Check for #endif - skip it
        if stripped.startswith("#endif;"):
            self.line_map.append((getattr(self, '_current_file', self.source_file), self._current_local_lineno))
            self.output_lines.append('')
            return i + 1
        
        # Check for #else - skip it (handled in conditional processing)
        if stripped == "#else":
            self.line_map.append((getattr(self, '_current_file', self.source_file), self._current_local_lineno))
            self.output_lines.append('')
            return i + 1
        
        # Regular line - do constant substitution
        processed_line = self._substitute_constants(line)
        self.line_map.append((getattr(self, '_current_file', self.source_file), self._current_local_lineno))
        self.output_lines.append(processed_line)
        return i + 1
    
    def _process_conditional_block(self, lines: List[str], start_i: int, constant_name: str, is_ifndef: bool) -> int:
        """Process an #ifdef/#ifndef block and return next line index after #endif"""
        # Get constant value
        constant_value = self.constants.get(constant_name)
        
        # Evaluate condition
        if is_ifndef:
            condition_true = constant_value is None or constant_value == '0'
        else:
            condition_true = constant_value is not None and constant_value != '0'
        i = start_i + 1
        depth = 1
        in_else = False
        else_seen = False
        
        # Emit a blank entry for the #ifdef/#ifndef line itself so it is accounted for
        self.line_map.append((getattr(self, '_current_file', self.source_file), self._current_local_lineno))
        self.output_lines.append('')
        
        # Store lines that should be included, paired with their original local line numbers
        lines_to_include = []
        origins_to_include = []  # parallel list: local line number (1-based) for each entry
        # excluded_origins tracks lines NOT included (for blank line_map entries)
        excluded_origins = []
        # Base local lineno at block entry; used to compute correct original line for each
        # sub-array line when this function is called recursively with a lines_to_include slice.
        entry_local_lineno = self._current_local_lineno
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Handle nested conditionals
            if stripped.startswith("#ifdef") or stripped.startswith("#ifndef"):
                depth += 1
            
            # Check for #else at our depth level
            if stripped == "#else" and depth == 1:
                if else_seen:
                    raise SyntaxError("Multiple #else directives in same conditional block")
                else_seen = True
                in_else = True
                # Emit blank for the #else line
                self.line_map.append((getattr(self, '_current_file', self.source_file), i + 1))
                self.output_lines.append('')
                i += 1
                continue
            
            # Check for #endif
            if stripped.startswith("#endif;"):
                depth -= 1
                if depth == 0:
                    # Emit blank for excluded lines so line_map stays in sync
                    for orig in excluded_origins:
                        self.line_map.append((getattr(self, '_current_file', self.source_file), orig))
                        self.output_lines.append('')
                    # Emit blank for the #endif line itself
                    self.line_map.append((getattr(self, '_current_file', self.source_file), i + 1))
                    self.output_lines.append('')
                    # End of our block - process collected lines
                    if lines_to_include:
                        j = 0
                        while j < len(lines_to_include):
                            self._current_local_lineno = origins_to_include[j]
                            j = self._process_line(lines_to_include, j)
                    return i + 1
            
            # Collect lines based on condition
            # original_lineno: correct local line in the original file for line i
            original_lineno = entry_local_lineno + (i - start_i)
            if depth > 1:
                # Inside nested block - always include
                if (condition_true and not in_else) or (not condition_true and in_else):
                    lines_to_include.append(line)
                    origins_to_include.append(original_lineno)
                else:
                    excluded_origins.append(original_lineno)
            else:
                # Our depth level
                if (condition_true and not in_else) or (not condition_true and in_else):
                    lines_to_include.append(line)
                    origins_to_include.append(original_lineno)
                else:
                    excluded_origins.append(original_lineno)
            
            i += 1
        
        raise SyntaxError(f"Unclosed conditional block starting at line {start_i + 1}")
    
    def _substitute_constants(self, line: str) -> str:
        """Simple constant substitution - replace constant names with their values"""
        if not line or line.strip() == ';':
            return line
        
        # Split line into tokens, preserving whitespace structure
        result_parts = []
        in_quotes = False
        current_token = ""
        
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
                current_token += char
            elif char.isspace() or char in '.,;:()[]{}+-*/%=!<>|&^~':
                # End of token
                if current_token:
                    # Check if token is a constant
                    if not in_quotes and current_token in self.constants:
                        # Get constant value and strip trailing semicolon if present
                        constant_value = self.constants[current_token]
                        # Remove trailing semicolon if it's at the end
                        if constant_value.endswith(';'):
                            constant_value = constant_value.rstrip(';').strip()
                        result_parts.append(constant_value)
                    else:
                        result_parts.append(current_token)
                    current_token = ""
                result_parts.append(char)
            else:
                current_token += char
        
        # Handle last token
        if current_token:
            if not in_quotes and current_token in self.constants:
                constant_value = self.constants[current_token]
                # Remove trailing semicolon if it's at the end
                if constant_value.endswith(';'):
                    constant_value = constant_value.rstrip(';').strip()
                result_parts.append(constant_value)
            else:
                result_parts.append(current_token)
        
        return ''.join(result_parts)


# Usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python preprocessor.py <source_file.fx>")
        sys.exit(1)
    
    preprocessor = FXPreprocessor(sys.argv[1])
    preprocessor.process()