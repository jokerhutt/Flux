#import <standard.fx>;

using standard::io::console;

def foo<T: "" | byte*>(T x, T y) -> T
{
    return f"{x}, {y}";
};

def main() -> int
{
    print(foo(5, "World!"));
    return 0;
};