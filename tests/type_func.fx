#import <standard.fx>;
 
using standard::io::console;

comptime
{
    compiler.io.console.print("Hello from comptime!\n");

    emitflux
    {
        def comp() -> void { println("Comptime generated func."); };
    };
};
 
byte.my_byte_func<T>(T x) -> byte
{
    return _ - x;
};

"".my_str_func() -> ""
{
    return _ + ", World!";
};

struct TestStru
{
    int x;
};

TestStru.my_tstru_func<T>(T y) -> TestStru
{
    return {_.x - y};
};

def main() -> int
{
    byte a = 100;
    TestStru t = {100};
    println("Hello".my_str_func());
    a = a.my_byte_func(3);
    t = t.my_tstru_func(2);
    println([a, "\0"]);
    println([t.x, "\0"]);

    comp();
    return 0;
};