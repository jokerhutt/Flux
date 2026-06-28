// argparse_demo.fx
// Demonstrates argparse.fx: flags, options, positionals, defaults, required args.
//
// Try:
//   argparse_demo -v -o out.txt input.txt
//   argparse_demo --output result.bin -n 8 file1.txt file2.txt
//   argparse_demo --help
//   argparse_demo              (missing required --output -> error)

#import <standard.fx>, <argparse.fx>;

using standard::io::console,
      argparse;

def main(int argc, byte** argv) -> int
{
    ArgParser parser();
    parser.set_description(g"A demo tool that processes files.");

    parser.add_flag  (g"-v",  g"--verbose", g"Enable verbose output");
    parser.add_flag  (g"-d",  g"--dry-run", g"Simulate without writing anything");
    parser.add_option(g"-o",  g"--output",  g"Output file",        true,  (byte*)0);
    parser.add_option(g"-n",  g"--count",   g"Number of passes",   false, g"1");
    parser.add_option((byte*)0, g"--format", g"Output format",     false, g"binary");

    if (parser.parse(argc, argv) != ERR_OK)
    {
        return 1;
    };

    // Verbose flag
    if (parser.is_set(g"-v"))
    {
        println(g"[verbose] Verbose mode enabled.");
    };

    // Dry-run flag
    if (parser.is_set(g"--dry-run"))
    {
        println(g"[dry-run] No files will be written.");
    };

    // Required option --output
    byte* outfile = parser.get_value(g"--output");
    print(g"Output file : ");
    if ((u64)outfile != 0) { println(outfile); }
    else                   { println(g"(none)"); };

    // Option with default --count
    byte* count = parser.get_value(g"--count");
    print(g"Pass count  : ");
    if ((u64)count != 0) { println(count); }
    else                 { println(g"(none)"); };

    // Option with default --format
    byte* fmt = parser.get_value(g"--format");
    print(g"Format      : ");
    if ((u64)fmt != 0) { println(fmt); }
    else               { println(g"(none)"); };

    // Positional arguments
    int npos = parser.positional_count_get();
    if (npos == 0)
    {
        println(g"No input files provided.");
    }
    else
    {
        print(g"Input files : ");
        println(npos);
        for (int i; i < npos; i = i + 1)
        {
            print(g"  [");
            print(i);
            print(g"] ");
            println(parser.get_positional(i));
        };
    };

    return 0;
};
