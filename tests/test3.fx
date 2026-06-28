#import <standard.fx>;

using standard::io::console;

def foo() -> byte*
{
    return "Hello!";
};

def main() -> int
{
    auto a = 10ul;
    auto c = foo();

    if (typeof(a) == typeof(unsigned long))
    {
        println("Success 1!");
    };

    if (typeof(c) == typeof(byte*))
    {
        println("Success 2!");
    };

    return 0;
};