#import <standard.fx>, <windows.fx>;

using standard::system::windows;

def main() -> int
{
    Window win("Flux Window", 800, 600, CW_USEDEFAULT, CW_USEDEFAULT);
    SetForegroundWindow(win.handle);
    BringWindowToTop(win.handle);

    while (win.process_messages())
    {
        //print("In main loop!\n");
    };

    system("pause\0");

    return 0;
};
