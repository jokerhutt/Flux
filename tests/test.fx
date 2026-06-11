#import <standard.fx>;
 
using standard::io::console;

trait MyTrait1
{
    def foo() -> void,
        bar() -> int;
};

trait MyTrait2
{
    def hehe() -> void,
        haha() -> void;
};

interface MyInter(A: MyTrait1, B: MyTrait2)
{
    A : B
    {
        haha() -> void
    };

    B : A
    {
        foo() -> void,
        bar() -> void
    };
};

object Test3; // Forward ref

MyTrait2 object Test2
{
    def __init() -> this { return this; };
    def __expr() -> this { return this; };
    def __exit() -> void {};

    def hehe() -> void
    {};

    def haha() -> void
    {};

    private : Test3
    {
        def hoho() -> void { println("hoho"); };
    };
};

MyTrait1 object Test1
{
    def __init() -> this { return this; };
    def __expr() -> this { return this; };
    def __exit() -> void {};

    def foo() -> void
    {
    };

    def bar() -> int
    {
        return 0;
    };
} : MyInter(this, Test2);

object Test3 : Test2
{
    def __init() -> this { return this; };
    def __expr() -> this { return this; };
    def __exit() -> void {};
};
 
def main() -> int
{
    Test3 t3();

    t3.hoho();

    return 0;
};
