#import <standard.fx>;

using standard::io::console;

def main(int argc, byte** argv) -> int
{
    if (argc > 2)
    {
        print("Too many arguments given.
Usage: compile program.fx

Invokes Python on the Flux compiler backend on program.fx");
        return 0;
    }
    else if (argc == 2)
    {
        println(argv[0]);
        println(argv[1]);
        byte* command = f"python fxc.py {argv[0]}";
        println(f"COMMAND: {command}");
        system(command);
        return 0;
    }
    else if (argc == 0)
    {
        print("Mock Flux Compiler, written in Flux. Calls Python on the Flux compiler.
Usage: compile program.fx");
        return 0;
    };

    return 0;
};