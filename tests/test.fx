#import <standard.fx>;
 
using standard::io::console;

constra MyCS(A)
{
    A !`< A    // This is a unary expression, read as the relation !`< "between A types"
};

def foo<T: int, :{MyCS}>(T x) -> byte
{
    return 5 + x; // lowering would occur here, violating MyCS
};

def main() -> int
{
    foo(10);
    return 0;
};