#import <standard.fx>;
 
using standard::io::console;

byte* z = "f";

i"".i"{}":{z}<!+T: "", !+U: "", T ~ U>(T x, U y) -> ""
{
    return _ + i"{} {}":{x; y;};
};

def main() -> int
{
    println(i"Hello":{}.i"{}":{z}(",", "World!"));
    return 0;
};