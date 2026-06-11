#import <standard.fx>;
 
using standard::io::console;
 
def main() -> int
{
    noopstr[][] maze = [
        ["S", "O", "X", "O", "O", "O"],
        ["X", "O", "X", "O", "X", "O"],
        ["O", "O", "O", "O", "X", "O"],
        ["O", "X", "X", "O", "X", "O"],
        ["O", "O", "O", "O", "O", "O"],
        ["X", "X", "X", "X", "X", "E"]
    ];

    print(maze[0][0]);
    return 0;
};
