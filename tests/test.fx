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

MyTrait2 object Test2
{
    def __init() -> this { return this; };
    def __expr() -> this { return this; };
    def __exit() -> void {};

    def hehe() -> void
    {};

    def haha() -> void
    {};
};

MyTrait1 object Test1
{
    def __init() -> this { return this; };
    def __expr() -> this { return this; };
    def __exit() -> void {};

    def foo() -> void
    {
        Test2 x();
        x.hehe();
    };

    def bar() -> int
    {
        return 0;
    };
} : MyInter(this, Test2);
 
def main() -> int
{
    Test1 t1();
    Test2 t2();

    t1().foo();

    return 0;
};
