#import <standard.fx>;
 
using standard::io::console;

~byte*.f<T: "", U: "", T ~ U>(T x, U y) -> ""
{
    return _ + f"{x} {y}";
};

def main() -> int
{
    ~byte* x = "Hello";
    byte* y = ~x.f(",", "World!");
    println(y);
    return 0;
};