#import <standard.fx>;
 
using standard::io::console;

byte* z = "f";

i"".f"{z}"() -> ""
{
    return _ + ", World!";
};

def main() -> int
{
    println(i"Hello":{}.f"{z}"());
    return 0;
};