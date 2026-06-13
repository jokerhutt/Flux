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

    a = a xor b;
    b = b xor a;
    a = a xor b;

    compiler.io.console.print(f"{a} {b}\n");
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
    struct TB
    {
        bool a,b,c,d,e;
    };
    bool tb = true;

    signed data{5} as i5;

    i5 my5 = 0b10110;

    TB mytb from my5;

    compiler.io.console.print(f"mytb.c = {mytb.c}\n");

    if (typeof(tb) == typeof(bool))
    {
        compiler.io.console.print("tb is bool\n");
    };

    compiler.io.console.print("Size of TB: ");
    compiler.io.console.print(sizeof(TB));
    compiler.io.console.print("\n");
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
    compiler.io.console.print(f"Address of bar() at comptime = {abar}\n");

    //jump abar;
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

comptime
{
    int zk;
    if (zk is void)
    {
        compiler.io.console.print("zk is void\n");
    };

    ulong x = 69;

    void* vpx @= x;

    long lx = long(*vpx);

    compiler.io.console.print(f"{lx}\n");
};

comptime
{
    int x = 5;

    fluxvm
    {
        LOCAL_GET x
        PUSH 10
        ADD
        LOCAL_SET x
    };

    compiler.io.console.print(f"FluxVM modified int x = {x}\n");
};

comptime
{
    def zod() -> void
    {
    };

    def zed() <~ void
    {
        compiler.io.console.print("Infinity!\n");
        escape zod();
    };
    compiler.io.console.print("before zed()\n");
    zed();
    compiler.io.console.print("after zod()\n");
    // execution escapes here
    compiler.fvm.dump("C:\\Users\\kvthw\\Flux\\test.fvm");
};



def main() -> int
{
    return 0;
};
