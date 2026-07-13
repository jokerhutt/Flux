#import <standard.fx>;

using standard::io::console;

object test
{
    def __init() -> this
    {
        return this;
    };

    def __expr() -> test*
    {
        return this;
    };

    def __exit() -> void { (void)this; };

    def pv() -> void { print("TEST\n\0"); };
};

def main() -> int
{
    test t();

    t.__exit();

    t.pv();    // ERROR
    return 0;
};