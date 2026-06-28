// Author: Karac V. Thweatt

// argparse.fx - Command-Line Argument Parsing Library
// Provides structured parsing of argc/argv with support for:
//   flags    (-v, --verbose)
//   options  (-o <value>, --output <value>)
//   positional arguments

#ifndef FLUX_STANDARD
#def FLUX_STANDARD 1;
#endif;

#ifndef FLUX_ARGPARSE
#def FLUX_ARGPARSE 1;

#ifndef FLUX_STANDARD_TYPES
#import <types.fx>;
#endif;

#ifndef FLUX_STANDARD_STRINGS
#import <string_utilities.fx>;
#endif;

namespace argparse
{
    // Argument kinds
    global const int ARG_FLAG       = 0,  // Boolean switch: -v / --verbose
                     ARG_OPTION     = 1,  // Takes a value: -o val / --output val
                     ARG_POSITIONAL = 2;  // Bare positional value

    // Error codes
    global const int ERR_OK            =  0,
                     ERR_UNKNOWN_ARG   = -1,
                     ERR_MISSING_VALUE = -2,
                     ERR_NULL_PARSER   = -3,
                     ERR_TOO_MANY_ARGS = -4,
                     ERR_ALLOC_FAILED  = -5;

    // Capacity limits (fixed, no realloc needed)
    global const int ARGPARSE_MAX_DEFS        = 64,
                     ARGPARSE_MAX_POSITIONALS = 32;

    // String literals hoisted to avoid per-call stack allocation
    global noopstr _S_USAGE        = g"Usage: ",
                   _S_OPTIONS_HDR  = g" [options] [args...]",
                   _S_OPTIONS_SEC  = g"\nOptions:",
                   _S_HELP_LINE    = g"  -h, --help          Show this help message",
                   _S_INDENT       = g"  ",
                   _S_SEP          = g", ",
                   _S_VALUE_TAG    = g" <value>",
                   _S_REQUIRED_TAG = g" (required)",
                   _S_DEFAULT_PRE  = g" [default: ",
                   _S_DEFAULT_POST = g"]",
                   _S_ERR_UNKNOWN  = g"argparse: unknown argument: ",
                   _S_ERR_NOVAL    = g"argparse: option requires a value: ",
                   _S_ERR_MISSING  = g"argparse: required option missing: ",
                   _S_HELP_LONG    = g"--help",
                   _S_HELP_SHORT   = g"-h";

    // A single argument definition registered by the user
    struct ArgDef
    {
        byte* short_name,   // e.g. "-v"  (null if not used)
              long_name,    // e.g. "--verbose" (null if not used)
              help,         // Help text
              value,        // Filled in for ARG_OPTION after parse
              default_val;  // Default value for ARG_OPTION (null if none)
        int   kind;         // ARG_FLAG or ARG_OPTION
        bool  required,     // Whether the option must be present
              found;        // Set to true after parsing
    };

    object ArgParser
    {
        ArgDef[ARGPARSE_MAX_DEFS] defs;
        int   def_count,
              positional_count,
              last_error;

        byte*[ARGPARSE_MAX_POSITIONALS] positionals;
        byte* program_name,   // argv[0]
              description;    // Program description for help text

        def __init() -> this
        {
            return this;
        };

        def __exit() -> void
        {
            return;
        };

        def __expr() -> int
        {
            return this.last_error;
        };

        // ============ REGISTRATION ============

        // Add a flag (boolean switch). short_name or long_name may be null.
        def add_flag(byte* short_name, byte* long_name, byte* help) -> bool
        {
            if (this.def_count >= ARGPARSE_MAX_DEFS)
            {
                this.last_error = ERR_TOO_MANY_ARGS;
                return false;
            };

            this.defs[this.def_count].short_name  = short_name;
            this.defs[this.def_count].long_name   = long_name;
            this.defs[this.def_count].help        = help;
            this.defs[this.def_count].kind        = ARG_FLAG;
            this.defs[this.def_count].required    = false;
            this.defs[this.def_count].found       = false;
            this.defs[this.def_count].value       = (byte*)0;
            this.defs[this.def_count].default_val = (byte*)0;
            this.def_count = this.def_count + 1;
            return true;
        };

        // Add an option that takes a value. Pass null for default_val if none.
        def add_option(byte* short_name, byte* long_name, byte* help, bool required, byte* default_val) -> bool
        {
            if (this.def_count >= ARGPARSE_MAX_DEFS)
            {
                this.last_error = ERR_TOO_MANY_ARGS;
                return false;
            };

            this.defs[this.def_count].short_name  = short_name;
            this.defs[this.def_count].long_name   = long_name;
            this.defs[this.def_count].help        = help;
            this.defs[this.def_count].kind        = ARG_OPTION;
            this.defs[this.def_count].required    = required;
            this.defs[this.def_count].found       = false;
            this.defs[this.def_count].value       = default_val;
            this.defs[this.def_count].default_val = default_val;
            this.def_count = this.def_count + 1;
            return true;
        };

        def set_description(byte* desc) -> void
        {
            this.description = desc;
        };

        // ============ PARSING ============

        // Parse argc/argv. Returns ERR_OK on success or a negative error code.
        def parse(int argc, byte** argv) -> int
        {
            if (argc >= 1)
            {
                this.program_name = argv[0];
            };

            byte* arg;
            int   i = 1,
                  matched;
            bool  short_match,
                  long_match;

            while (i < argc)
            {
                arg = argv[i];

                // Built-in --help / -h
                if (this._str_eq(arg, _S_HELP_LONG) | this._str_eq(arg, _S_HELP_SHORT))
                {
                    this.print_help();
                    return ERR_OK;
                };

                if (arg[0] == '-')
                {
                    matched = -1;

                    for (int d; d < this.def_count; d = d + 1)
                    {
                        short_match = false;
                        long_match  = false;

                        if ((u64)this.defs[d].short_name != 0)
                        {
                            short_match = this._str_eq(arg, this.defs[d].short_name);
                        };

                        if ((u64)this.defs[d].long_name != 0)
                        {
                            long_match = this._str_eq(arg, this.defs[d].long_name);
                        };

                        if (short_match | long_match)
                        {
                            matched = d;
                            break;
                        };
                    };

                    if (matched == -1)
                    {
                        print(_S_ERR_UNKNOWN);
                        println(arg);
                        this.last_error = ERR_UNKNOWN_ARG;
                        return ERR_UNKNOWN_ARG;
                    };

                    this.defs[matched].found = true;

                    if (this.defs[matched].kind == ARG_OPTION)
                    {
                        i = i + 1;
                        if (i >= argc)
                        {
                            print(_S_ERR_NOVAL);
                            println(arg);
                            this.last_error = ERR_MISSING_VALUE;
                            return ERR_MISSING_VALUE;
                        };
                        this.defs[matched].value = argv[i];
                    };
                }
                else
                {
                    if (this.positional_count >= ARGPARSE_MAX_POSITIONALS)
                    {
                        this.last_error = ERR_TOO_MANY_ARGS;
                        return ERR_TOO_MANY_ARGS;
                    };
                    this.positionals[this.positional_count] = arg;
                    this.positional_count = this.positional_count + 1;
                };

                i = i + 1;
            };

            // Check required options
            for (int d; d < this.def_count; d = d + 1)
            {
                if (this.defs[d].required & !this.defs[d].found)
                {
                    print(_S_ERR_MISSING);
                    if ((u64)this.defs[d].long_name != 0)
                    {
                        println(this.defs[d].long_name);
                    }
                    else
                    {
                        println(this.defs[d].short_name);
                    };
                    this.last_error = ERR_MISSING_VALUE;
                    return ERR_MISSING_VALUE;
                };
            };

            this.last_error = ERR_OK;
            return ERR_OK;
        };

        // ============ RESULT ACCESSORS ============

        // Check if a flag/option was present. Pass short or long name.
        def is_set(byte* name) -> bool
        {
            bool short_match,
                 long_match;

            for (int d; d < this.def_count; d = d + 1)
            {
                short_match = false;
                long_match  = false;

                if ((u64)this.defs[d].short_name != 0)
                {
                    short_match = this._str_eq(name, this.defs[d].short_name);
                };
                if ((u64)this.defs[d].long_name != 0)
                {
                    long_match = this._str_eq(name, this.defs[d].long_name);
                };

                if (short_match | long_match)
                {
                    return this.defs[d].found;
                };
            };
            return false;
        };

        // Get the value of an ARG_OPTION. Returns null if not found or not set.
        def get_value(byte* name) -> byte*
        {
            bool short_match,
                 long_match;

            for (int d; d < this.def_count; d = d + 1)
            {
                short_match = false;
                long_match  = false;

                if ((u64)this.defs[d].short_name != 0)
                {
                    short_match = this._str_eq(name, this.defs[d].short_name);
                };
                if ((u64)this.defs[d].long_name != 0)
                {
                    long_match = this._str_eq(name, this.defs[d].long_name);
                };

                if (short_match | long_match)
                {
                    return this.defs[d].value;
                };
            };
            return (byte*)0;
        };

        // Get positional argument by index (0-based). Returns null if out of range.
        def get_positional(int index) -> byte*
        {
            if (index < 0 | index >= this.positional_count)
            {
                return (byte*)0;
            };
            return this.positionals[index];
        };

        def positional_count_get() -> int { return this.positional_count; };
        def error_code()           -> int { return this.last_error; };

        // ============ HELP ============

        def print_help() -> void
        {
            bool has_short,
                 has_long;

            print(_S_USAGE);
            print(this.program_name);
            println(_S_OPTIONS_HDR);

            if ((u64)this.description != 0 & this.description[0] != 0)
            {
                println(this.description);
            };

            println(_S_OPTIONS_SEC);
            println(_S_HELP_LINE);

            for (int d; d < this.def_count; d = d + 1)
            {
                has_short = (u64)this.defs[d].short_name != 0;
                has_long  = (u64)this.defs[d].long_name  != 0;

                print(_S_INDENT);
                if (has_short)         { print(this.defs[d].short_name); };
                if (has_short & has_long) { print(_S_SEP); };
                if (has_long)          { print(this.defs[d].long_name); };

                if (this.defs[d].kind == ARG_OPTION) { print(_S_VALUE_TAG); };
                if (this.defs[d].required)            { print(_S_REQUIRED_TAG); };

                print(_S_INDENT);
                print(this.defs[d].help);

                if (this.defs[d].kind == ARG_OPTION & (u64)this.defs[d].default_val != 0)
                {
                    print(_S_DEFAULT_PRE);
                    print(this.defs[d].default_val);
                    print(_S_DEFAULT_POST);
                };

                print();
            };
        };

        // ============ INTERNAL HELPERS ============

        def _str_eq(byte* a, byte* b) -> bool
        {
            int i;
            while (a[i] != 0 & b[i] != 0)
            {
                if (a[i] != b[i]) { return false; };
                i = i + 1;
            };
            return a[i] == b[i];
        };
    };
};

#endif; // FLUX_ARGPARSE
