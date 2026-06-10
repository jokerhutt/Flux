#import <standard.fx>;
 
using standard::io::console;

byte* z = "f",
      t = "T",
      u = "U";

#def NULLSTR "";

NULLSTR.f<~$t: NULLSTR, ~$u: NULLSTR, ~$t ~ ~$u>(~$t x, ~$u y) -> NULLSTR
{
    return _ + i"{} {}":{x; y;};
};

def main() -> int
{
    println("Hello".f(",", "World!"));
    return 0;
};