#import <standard.fx>;

using standard::io::console;

comptime
{
    compiler.io.console.print("Hello, stage 1!\n");
	emitflux
    {
        comptime
        {
            compiler.io.console.print("Hello, stage 2!\n");
            emitflux
            {
                def test() -> void
                {
                    println("Hello World!");
                };
            };
        };
    };
};

def main() -> int
{
    test();
	return 0;
};