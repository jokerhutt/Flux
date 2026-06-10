#import <standard.fx>;
 
using standard::io::console;

byte* z = "f";

constra MyCS(A,B)
{
    A ~= B
};

i"".i"{}":{z}<T: "", U: "", :{MyCS}>(T x, U y) -> ""
{
    return _ + i"{} {}":{x; y;};
};

def main() -> int
{
    println(i"Hello":{}.i"{}":{z}(",", "World!"));
    return 0;
};