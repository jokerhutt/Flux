#import <standard.fx>;
 
using standard::io::console;
 
byte.my_byte_func<T>(T x) -> byte
{
    return _ - x;
};

struct TestStru
{
    int x;
};

TestStru.my_tstru_func<T>(T y) -> TestStru
{
    return {_.x - y};
};

def main() -> int
{
    byte a = 100;
    TestStru t = {100};
    a = a.my_byte_func(3);
    t = t.my_tstru_func(2);
    println([a, "\0"]);
    println([t.x, "\0"]);
    return 0;
};