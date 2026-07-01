#import <standard.fx>;

using standard::io::console;

struct A<T>
{
	T x;
};

struct B<U> : A<T>
{
	U y;
};

def main() -> int
{
	B<int,long> b = {6,9};

	println(f"{b.x}{b.y}");

	return 0;
};