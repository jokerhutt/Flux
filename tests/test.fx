#import <standard.fx>;
 
using standard::io::console;

comptime
{
    constraint MyCS(A)
    {
        A !@ A
    };

    struct MyStructA<T>
    {
        int a, b, c, d;
    };

    union MyUnionA
    {
        int a;
        char b;
    };

    def foo<T: byte*, :{MyCS}>(T x) -> void
    {
        compiler.io.console.print(f"comptime {x}!!!\n");
    };

    char.mybfunc() -> char
    {
        return _ + 1;
    };

    namespace A
    {
        def Afoo() -> void;
        namespace B {};
    };

    signed data{8} as nbyte;

    compiler.io.console.print(sizeof(nbyte));

    bool k = true;
    if (k | k)
    {
        compiler.io.console.print(f"k!\n");
    };

    compiler.io.console.print(f"{sizeof(MyStructA)}\n");

    int a = 10, b = 20;
    long c = [a, b];
    int d, e = [c];
    int[2] f = [a, b];
    if (a in f)
    {
        compiler.io.console.print(f"a in f\n");
    };
    do
    {
        compiler.io.console.print("DO LOOP!\n");
        break;
    } while (true);
//label myLabel:
    while (true)
    {
        compiler.io.console.print("WHILE LOOP!\n");
        break;
    };
    //goto myLabel;
    compiler.io.console.print(f"{c}\n");
    compiler.io.console.print(f"{d}\n");
    compiler.io.console.print(f"{e}\n");
    compiler.io.console.print(f"{f[0]}\n");
};

comptime
{
    enum MyEnum1
    {
        Thing1,
        Thing2,
        Thing3
    };

    union MyU1 { int a; };

    MyEnum1 me1;
    me1._ = MyU1;
};

comptime
{
    def bar() -> void
    {
        compiler.io.console.print("bar!!!\n");
    };

    def{}* pb()->void = @bar;

    pb();

    ulong abar = *pb;
    compiler.io.console.print(f"Address of bar() = {abar}\n");
};

comptime
{
    foo("Hello World!");

    char b = 'b';

    int* pp = @a;
    int aa = *pp;

    if (aa is a)
    {
        compiler.io.console.print("AA\n");
    };

    MyStructA ns;

    ns.a = 25;

    compiler.io.console.print(b.mybfunc());
    compiler.io.console.print(f"{ns.a}\n");

    if (a != b)
    {
        compiler.io.console.print("!=\n");
    };
};

def main() -> int
{
    return 0;
};
