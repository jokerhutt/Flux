#import <standard.fx>;
 
using standard::io::console;

comptime
{
    def foo() -> void
    {
        compiler.io.console.print("comptime foo!!!\n");
    };
};

comptime
{
    foo();

    int x;
    int* px = @x;

    compiler.io.console.print(f"Hello from comptime! {ulong(px)}\n");
};

def main() -> int
{
    return 0;
};
