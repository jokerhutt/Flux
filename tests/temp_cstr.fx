#import <standard.fx>;

using standard::io::console;

def foo<T: "" | byte*>(T x) -> T
{
    return f"{x}, World!";
};

def main() -> int
{
    print(foo("Hello"));
    return 0;
};