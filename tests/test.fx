#import <standard.fx>;

using standard::io::console;

struct Vec3
{
    float x, y, z;
};

def dot(Vec3 a, Vec3 b) -> float
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
};

def main() -> int
{
    Vec3 u = { x = 1.0f, y = 2.0f, z = 3.0f };
    Vec3 v = { x = 4.0f, y = 5.0f, z = 6.0f };
    float d = dot(u, v);
    println(f"dot(u, v) = {d}");
    return 0;
};
