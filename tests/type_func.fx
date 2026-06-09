#import <standard.fx>;
 
using standard::io::console;

byte* z = "f";

i"".f"{z}"<T>(T x) -> ""
{
    return _ + f", W{x}rld!";
};

def main() -> int
{
    println(i"Hello":{}.f"{z}"(0));
    return 0;
};