#import <standard.fx>;

using standard::io::console;

constra MyCS(A,B)
{
    A !`>= B
};

def foo<T: long, U: int, :{MyCS}>(T x, U y) -> U
{
    return x + y;
};

def main() -> int
{
    println(foo(10l, 20));
    return 0;
};