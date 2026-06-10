#import <standard.fx>;
 
using standard::io::console;

byte* z = "f";

constra MyCS1(A,B)
{
    A ~ B
};

constra MyCS2(A,B)
{
    A !~ B
};

constra MyCS3(M,N) = MyCS1 + MyCS2;

i"".i"{}":{z}<T: "", U: "", :{MyCS3}>(T x, U y) -> ""
{
    return _ + i"{} {}":{x; y;};
};

def main() -> int
{
    println(i"Hello":{}.i"{}":{z}(",", "World!"));
    return 0;
};