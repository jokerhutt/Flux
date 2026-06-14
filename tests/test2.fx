comptime
{
	compiler.fvm.loadlib("kernel32","dll");
	//compiler.fvm.trace.begin();
	compiler.import.stdlib("standard.fx");
	//compiler.fvm.trace.end();

	using standard::io::console;

	def main() -> int
	{
		compiler.io.console.print("COMPTIME!\n");
		print("Hello at comptime using regular print!\n");
		return 0;
	};

	FRTStartup();
	compiler.fvm.dump("C:\\Users\\kvthw\\Flux\\test2.fvm");
};

def !!FRTStartup() -> int { return 0; };