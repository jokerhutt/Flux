#import <standard.fx>;
 
using standard::io::console;

byte* z = "f";

i"".i"{}":{z}<T>(T x) -> ""
{
    return _ + i", W{}rld!":{x;};
};

def main() -> int
{
    println(i"Hello":{}.i"{}":{z}(0));
    return 0;
};