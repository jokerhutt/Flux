```
root dir  
|  
|-- build\              // all programs build here:  build\program\program.ll IR output  
|-- build\tmp.fx        // Preprocessor temporary output for builds  
|-- config\             // All configuration  
|-- config\flux_configuration.cfg // Compiler configuration  
|-- docs\               // All docoumentation  
|-- examples\           // Battle-tested production ready working examples  
|-- scripts\            // Scripts like run_tests.py, compiles all test files in tests\  
|-- tests\              // All Flux source file (.fx) testing (weird_test.fx, 64bit_test.fx)  
|-- src\                // Compiler & Standard Library source  
|-- src\stdlib\         // Standard Library source  
|-- src\compiler\       // Compiler source
|   `--> fmacros.py     // Compiler macros  
|   `--> ferrors.py     // Compiler errors  
|   `--> fast.py        // AST  
|   `--> fpreprocess.py // Preprocessor  
|   `--> flexer.py      // Lexer  
|   `--> fparser.py     // Parser  
|	`--> ftypesys.py    // Type System  
|   `--> fcodegen.py    // Code Generation
|   `--> fvmcodegen.py  // FVM Code Generation
|	`--> futilities.py  // Utility functions  
|	`--> flogger.py     // Logging  
|	`--> fconfig.py     // Config helper  
|   `--> fc.py          // Compiler front-end  
|  
`--> fxc.py             // Compiler front-end entrypoint, root, calls to fc.py in src\compiler\
`--> fpm.py             // Flux Package Manager
`--> fvm.py             // Flux Virtual Machine for comptime execution of Flux code & REPL
`--> fvm_test.py        // Tests the FVM
`--> frepl.py           // REPL for Flux, uses the FVM
```