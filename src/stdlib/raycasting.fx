// raycasting.fx - Flux 3D Raycasting & Rasterization Engine
//
// A full 3D raycasting engine with retained 2.5D capability.
//
// ARCHITECTURE:
//   Two rendering modes share a single u64* pixel buffer and a double*
//   per-pixel depth buffer (screen_w * screen_h doubles, Z in view space).
//
//   2.5D MODE  (raycaster::rc_*)
//     Tile-based DDA wall raycaster in the style of the original library.
//     All original structs, functions and constants are preserved verbatim
//     except the pixel buffer is now u64* and all math is double-precision.
//     Backward-compatible: existing 2.5D scenes compile with a cast on the
//     buffer pointer and removal of the old float depth_buf argument.
//
//   3D MODE  (raycaster::r3d_*)
//     True 3D engine layered on the same depth buffer so both modes can be
//     composited in a single frame.
//
//     Transform pipeline (all double-precision):
//       World -> View (camera-space) via DMat4 view matrix
//       View  -> Clip via DMat4 perspective projection
//       Clip  -> NDC via perspective divide (w-divide)
//       NDC   -> Screen via viewport transform
//
//     Mesh rendering:
//       R3DMesh        - heap vertex + index + UV arrays
//       R3DMeshInst    - instance: mesh ref + world TRS transform
//       Per triangle:
//         - World -> view transform
//         - Backface cull (view-space cross product)
//         - Frustum clip (Sutherland-Hodgman, all 6 clip-space planes)
//           emitting 1 or 2 output triangles per input triangle
//         - Perspective-correct UV interpolation via stored 1/w at each vertex
//         - Scanline rasterizer with per-pixel depth test against depth buffer
//
//     Billboard sprites:
//       R3DSprite      - world position + optional vertical offset + texture
//       Rendered as axis-aligned quads in view space after mesh pass,
//       respecting the shared depth buffer.
//
//     Lighting:
//       R3DLight       - point or directional light with double-precision
//                        color and attenuation
//       Per-triangle flat shading: N dot L computed in world space.
//       Gouraud shading: per-vertex N dot L, barycentric-interpolated.
//       Ambient term always added to avoid pure black.
//
//     Sky:
//       Cylindrical panorama gradient extended to follow camera pitch.
//
// COORDINATE SYSTEM:
//   Right-handed.  +X = East, +Y = Up, +Z = North (into screen).
//   Camera looks toward -Z in view space (standard OpenGL convention).
//   Yaw rotates around Y, pitch around X.
//
// PIXEL FORMAT:
//   u64 pixels are 0xAAAARRRRGGGGBBBB (16 bits per channel, fully opaque
//   when A = 0xFFFF).  Helper functions color64_pack / color64_unpack
//   convert between 16-bit-per-channel u64 and normalised doubles.
//
// DEPTH BUFFER:
//   double* zbuf, size screen_w * screen_h.
//   Stores positive view-space Z (distance from camera along view axis).
//   Initialised to R3D_INF before each frame.  Closer geometry writes
//   smaller Z values and wins the depth test.
//
// USAGE (3D):
//   #import "raycasting.fx";
//   using raycaster;
//
//   R3DCamera  cam;   r3d_camera_init(@cam, 90.0, 1920, 1080, 0.1, 500.0);
//   R3DPlayer  player; r3d_player_init(@player, 0.0, 0.0, 0.0);
//   R3DMesh    mesh;  r3d_mesh_load_raw(@mesh, verts, vert_count, idxs, idx_count, uvs);
//   R3DMeshInst inst; r3d_inst_init(@inst, @mesh);
//   // populate scene, call r3d_render()
//
// USAGE (2.5D, unchanged from original):
//   RCScene scene; rc_scene_init(...);
//   u64* buf = ...; double* zbuf = ...;
//   rc_render(@scene, buf, zbuf);
//

#ifndef FLUX_STANDARD_TYPES
#import "types.fx";
#endif;

#ifndef FLUX_STANDARD_MATH
#import "math.fx";
#endif;

#ifndef FLUX_STANDARD_VECTORS
#import "vectors.fx";
#endif;

#ifndef FLUX_STANDARD_MATRICES
#import "matrices.fx";
#endif;

#ifndef FLUX_STANDARD_MEMORY
#import "memory.fx";
#endif;

#ifndef FLUX_STANDARD_ALLOCATORS
#import "allocators.fx";
#endif;

#ifndef FLUX_RAYCASTING
#def FLUX_RAYCASTING 1;

using standard::vectors;
using standard::math;
using standard::memory::allocators::stdheap;

// ============================================================================
// SHARED CONSTANTS
// ============================================================================

#def RC_EPSILON          0.000001;
#def RC_INF              1.0e18;
#def RC_PI               3.14159265358979323846;
#def RC_TWO_PI           6.28318530717958647692;
#def RC_HALF_PI          1.57079632679489661923;
#def RC_DEG_TO_RAD       0.01745329251994329576;

// Tile type flags
#def RC_TILE_EMPTY       0;
#def RC_TILE_SOLID       1;
#def RC_TILE_DOOR        2;
#def RC_TILE_TRANS       4;

// Wall face identifiers
#def RC_FACE_NONE        0;
#def RC_FACE_X_POS       1;
#def RC_FACE_X_NEG       2;
#def RC_FACE_Y_POS       3;
#def RC_FACE_Y_NEG       4;

// Render pass flags
#def RC_PASS_SKY         1;
#def RC_PASS_WALLS       2;
#def RC_PASS_FLOOR       4;
#def RC_PASS_SPRITES     8;
#def RC_PASS_ALL         15;

#def RC_MAX_SPRITES      256;

// 3D pass flags (ORed into RCScene.passes for composite frames)
#def R3D_PASS_MESHES     16;
#def R3D_PASS_SPRITES    32;
#def R3D_PASS_LIGHTS     64;
#def R3D_PASS_ALL        112;

// Lighting model
#def R3D_LIGHT_DIR       0;   // Directional (infinite distance)
#def R3D_LIGHT_POINT     1;   // Point light with attenuation

// Shading model per mesh instance
#def R3D_SHADE_FLAT      0;   // One N.L per triangle
#def R3D_SHADE_GOURAUD   1;   // Interpolated per-vertex N.L

#def R3D_MAX_LIGHTS      8;
#def R3D_MAX_CLIP_VERTS  9;   // Max verts after full 6-plane frustum clip of one triangle (3+6)

// ============================================================================
// PIXEL FORMAT  (u64 = 0xAAAARRRRGGGGBBBB, 16 bits per channel)
// ============================================================================

// ============================================================================
// DOUBLE-PRECISION VECTOR & MATRIX TYPES
// (stdlib vectors.fx uses float; we need double for 3D precision)
// ============================================================================

struct DVec2 { double x, y; };
struct DVec3 { double x, y, z; };
struct DVec4 { double x, y, z, w; };

// Row-major 4x4 double matrix
struct DMat4
{
    double m00, m01, m02, m03,
           m10, m11, m12, m13,
           m20, m21, m22, m23,
           m30, m31, m32, m33;
};

// ============================================================================
// 2.5D STRUCTS  (unchanged from original, widened to double)
// ============================================================================

struct RCTile
{
    i32  flags,
         tex_wall,
         tex_floor,
         tex_ceil;
    u64  tint;    // 0xAAAARRRRGGGGBBBB
};

struct RCMap
{
    RCTile* cells;
    i32     width, height;
    u64     floor_color,
            ceil_color;
};

// 64-bit ARGB texture surface (16 bits per channel)
struct RCTexture
{
    u64* pixels;
    i32  width, height;
};

struct RCTexturePalette
{
    RCTexture* slots;
    i32        count, cap;
};

struct RCPlayer
{
    double pos_x, pos_y,
           angle,
           move_speed,
           turn_speed;
};

struct RCCamera
{
    double fov_h;
    i32    screen_w, screen_h;
    double view_dist,
           proj_dist,
           half_h,
           plane_x, plane_y,
           dir_x, dir_y;
};

struct RCWallHit
{
    double dist, wall_u;
    i32    tile_x, tile_y,
           face,
           tex_idx,
           draw_top, draw_bot;
    u64    tint;
};

struct RCSprite
{
    double world_x, world_y;
    i32    tex_idx;
    double scale;
    u64    tint;
    double dist_sq;
};

struct RCSky
{
    u64 color_top,
        color_horizon;
};

struct RCScene
{
    RCMap*            map;
    RCPlayer*         player;
    RCCamera*         cam;
    RCTexturePalette* palette;
    RCSprite*         sprites;
    i32               sprite_count;
    RCSky*            sky;
    i32               passes;
};

// ============================================================================
// 3D STRUCTS
// ============================================================================

// A single mesh vertex in object space (position + UV + normal)
struct R3DVertex
{
    double x, y, z,    // Object-space position
           nx, ny, nz, // Object-space normal (unit)
           u, v;       // Texture coordinates
};

// A triangle index triple (references into R3DMesh.verts)
struct R3DTriangle
{
    i32 a, b, c;
};

// Heap-allocated mesh geometry (caller owns lifetime via r3d_mesh_free)
struct R3DMesh
{
    R3DVertex*   verts;
    i32          vert_count;
    R3DTriangle* tris;
    i32          tri_count,
                 tex_idx;    // Texture palette index (0 = untextured)
};

// World-space instance of a mesh
struct R3DMeshInst
{
    R3DMesh*  mesh;
    double    pos_x, pos_y, pos_z,    // World translation
              rot_x, rot_y, rot_z,    // Euler angles (radians), applied YXZ
              scale_x, scale_y, scale_z;
    i32       shade_model;            // R3D_SHADE_FLAT or R3D_SHADE_GOURAUD
    u64       tint;                   // 0 = no tint
};

// 3D light source
struct R3DLight
{
    i32    kind;             // R3D_LIGHT_DIR or R3D_LIGHT_POINT
    double dir_x, dir_y, dir_z,   // Normalised direction (LIGHT_DIR) or
           pos_x, pos_y, pos_z,   // World position (LIGHT_POINT)
           color_r, color_g, color_b,
           intensity,
           atten_const,            // Point light attenuation: 1/(c + l*d + q*d^2)
           atten_linear,
           atten_quad;
};

// 3D player / camera state
struct R3DPlayer
{
    double pos_x, pos_y, pos_z,  // Eye position in world space
           yaw,                  // Horizontal rotation (radians)
           pitch,                // Vertical rotation (radians, clamped +/-89 deg)
           move_speed,
           turn_speed,
           pitch_speed;
};

// 3D projection camera parameters (derived from R3DPlayer each frame)
struct R3DCamera
{
    double fov_h,           // Horizontal FOV (radians)
           fov_v;           // Vertical FOV (radians, derived from fov_h and aspect)
    i32    screen_w, screen_h;
    double near_z, far_z,
           aspect,          // screen_w / screen_h
           proj_dist;       // (screen_w/2) / tan(fov_h/2)

    // View matrix (world -> camera space)
    DMat4  view,
    // Projection matrix (camera -> clip space)
           proj,
    // Combined view-projection
           vp;

    // Pre-extracted camera axes (world space) for billboard orientation
    double right_x, right_y, right_z,
           up_x, up_y, up_z,
           fwd_x, fwd_y, fwd_z,
           eye_x, eye_y, eye_z;
};

// 3D billboard sprite (always faces the camera, respects depth buffer)
struct R3DSprite
{
    double world_x, world_y, world_z,
           vert_offset,   // Vertical displacement from anchor in world units
           width, height; // World-space dimensions
    i32    tex_idx;
    u64    tint;
    double dist_sq;       // Filled by sort pass
};

// Full 3D scene descriptor
struct R3DScene
{
    R3DCamera*    cam;
    R3DPlayer*    player;
    R3DMeshInst** insts;        // Array of pointers to instances
    i32           inst_count;
    R3DSprite*    sprites;
    i32           sprite_count;
    R3DLight*     lights;
    i32           light_count;
    double        ambient_r, ambient_g, ambient_b;
    RCTexturePalette* palette;  // Shared with 2.5D scene (may be null)
    RCSky*        sky;
    i32           passes;       // R3D_PASS_* flags
};

// Internal: a clipped vertex with perspective-correct interpolation data
struct R3DClipVert
{
    double x, y, z, w,   // Clip space
           nx, ny, nz,   // Interpolated normal
           u, v,         // Texture coords
           inv_w;        // 1/w, carried for p-correct interp
};

namespace raycaster
{
    // =========================================================================
    // 64-BIT COLOR HELPERS
    // 0xAAAARRRRGGGGBBBB layout, 16 bits per channel.
    // =========================================================================

    def color64_pack(double r, double g, double b) -> u64
    {
        u64 ri, gi, bi;
        ri = (u64)(clamp(r, 0.0d, 1.0d) * 65535.0d);
        gi = (u64)(clamp(g, 0.0d, 1.0d) * 65535.0d);
        bi = (u64)(clamp(b, 0.0d, 1.0d) * 65535.0d);
        return (u64)0xFFFF000000000000 | (ri << 32) | (gi << 16) | bi;
    };

    def color64_unpack(u64 argb, double* r, double* g, double* b) -> void
    {
        *r = (double)((argb >> 32) & (u64)0xFFFF) / 65535.0;
        *g = (double)((argb >> 16) & (u64)0xFFFF) / 65535.0;
        *b = (double)( argb        & (u64)0xFFFF) / 65535.0;
    };

    def color64_scale(u64 argb, double factor) -> u64
    {
        double r, g, b;
        color64_unpack(argb, @r, @g, @b);
        return color64_pack(r * factor, g * factor, b * factor);
    };

    def color64_lerp(u64 a, u64 b, double t) -> u64
    {
        double ar, ag, ab, br, bg, bb;
        color64_unpack(a, @ar, @ag, @ab);
        color64_unpack(b, @br, @bg, @bb);
        return color64_pack(ar + (br - ar) * t,
                            ag + (bg - ag) * t,
                            ab + (bb - ab) * t);
    };

    def color64_tint(u64 base, u64 tint) -> u64
    {
        double alpha, br, bg, bb, tr, tg, tb;
        alpha = (double)((tint >> 48) & (u64)0xFFFF) / 65535.0;
        if (alpha < RC_EPSILON) { return base; };
        color64_unpack(base, @br, @bg, @bb);
        color64_unpack(tint, @tr, @tg, @tb);
        return color64_pack(br + (tr - br) * alpha,
                            bg + (tg - bg) * alpha,
                            bb + (tb - bb) * alpha);
    };

    def color64_mul(u64 a, u64 b) -> u64
    {
        double ar, ag, ab, br, bg, bb;
        color64_unpack(a, @ar, @ag, @ab);
        color64_unpack(b, @br, @bg, @bb);
        return color64_pack(ar * br, ag * bg, ab * bb);
    };

    // Modulate base color by an RGB triple (e.g. light contribution)
    def color64_light(u64 base, double lr, double lg, double lb) -> u64
    {
        double r, g, b;
        color64_unpack(base, @r, @g, @b);
        return color64_pack(r * lr, g * lg, b * lb);
    };

    // Pre-multiply a u64 color by fixed light RGB into a ready-to-write u64.
    // Used to bake flat lighting once per triangle instead of per pixel.
    def color64_light_bake(u64 base, double lr, double lg, double lb) -> u64
    {
        double r, g, b;
        u64 ri, gi, bi;
        color64_unpack(base, @r, @g, @b);
        r *= lr; g *= lg; b *= lb;
        if (r > 1.0d) { r = 1.0d; }; if (r < 0.0d) { r = 0.0d; };
        if (g > 1.0d) { g = 1.0d; }; if (g < 0.0d) { g = 0.0d; };
        if (b > 1.0d) { b = 1.0d; }; if (b < 0.0d) { b = 0.0d; };
        ri = (u64)(r * 65535.0d);
        gi = (u64)(g * 65535.0d);
        bi = (u64)(b * 65535.0d);
        return (u64)0xFFFF000000000000 | (ri << 32) | (gi << 16) | bi;
    };

    def fog_factor(double dist, double view_dist) -> double
    {
        double f;
        f = 1.0 - (dist / view_dist);
        if (f < 0.0) { f = 0.0; };
        if (f > 1.0) { f = 1.0; };
        return f;
    };

    // =========================================================================
    // DOUBLE-PRECISION VECTOR / MATRIX HELPERS (internal use)
    // =========================================================================

    def dvec3_dot(DVec3 a, DVec3 b) -> double
    {
        return a.x*b.x + a.y*b.y + a.z*b.z;
    };

    def dvec3_cross(DVec3 a, DVec3 b) -> DVec3
    {
        DVec3 r;
        r.x = a.y*b.z - a.z*b.y;
        r.y = a.z*b.x - a.x*b.z;
        r.z = a.x*b.y - a.y*b.x;
        return r;
    };

    def dvec3_length(DVec3 v) -> double
    {
        return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    };

    def dvec3_normalize(DVec3 v) -> DVec3
    {
        double len;
        DVec3  r;
        len = dvec3_length(v);
        if (len < RC_EPSILON) { r.x = 0.0; r.y = 0.0; r.z = 0.0; return r; };
        r.x = v.x / len;
        r.y = v.y / len;
        r.z = v.z / len;
        return r;
    };

    def dvec3_sub(DVec3 a, DVec3 b) -> DVec3
    {
        DVec3 r;
        r.x = a.x - b.x;
        r.y = a.y - b.y;
        r.z = a.z - b.z;
        return r;
    };

    def dvec3_add(DVec3 a, DVec3 b) -> DVec3
    {
        DVec3 r;
        r.x = a.x + b.x;
        r.y = a.y + b.y;
        r.z = a.z + b.z;
        return r;
    };

    def dvec3_scale(DVec3 v, double s) -> DVec3
    {
        DVec3 r;
        r.x = v.x * s;
        r.y = v.y * s;
        r.z = v.z * s;
        return r;
    };

    // DMat4: row-major. Multiply col-vector on right: r = M * v
    def dmat4_identity() -> DMat4
    {
        DMat4 m;
        m.m00=1.0; m.m01=0.0; m.m02=0.0; m.m03=0.0;
        m.m10=0.0; m.m11=1.0; m.m12=0.0; m.m13=0.0;
        m.m20=0.0; m.m21=0.0; m.m22=1.0; m.m23=0.0;
        m.m30=0.0; m.m31=0.0; m.m32=0.0; m.m33=1.0;
        return m;
    };

    def dmat4_mul(DMat4 a, DMat4 b) -> DMat4
    {
        DMat4 r;
        r.m00 = a.m00*b.m00 + a.m01*b.m10 + a.m02*b.m20 + a.m03*b.m30;
        r.m01 = a.m00*b.m01 + a.m01*b.m11 + a.m02*b.m21 + a.m03*b.m31;
        r.m02 = a.m00*b.m02 + a.m01*b.m12 + a.m02*b.m22 + a.m03*b.m32;
        r.m03 = a.m00*b.m03 + a.m01*b.m13 + a.m02*b.m23 + a.m03*b.m33;

        r.m10 = a.m10*b.m00 + a.m11*b.m10 + a.m12*b.m20 + a.m13*b.m30;
        r.m11 = a.m10*b.m01 + a.m11*b.m11 + a.m12*b.m21 + a.m13*b.m31;
        r.m12 = a.m10*b.m02 + a.m11*b.m12 + a.m12*b.m22 + a.m13*b.m32;
        r.m13 = a.m10*b.m03 + a.m11*b.m13 + a.m12*b.m23 + a.m13*b.m33;

        r.m20 = a.m20*b.m00 + a.m21*b.m10 + a.m22*b.m20 + a.m23*b.m30;
        r.m21 = a.m20*b.m01 + a.m21*b.m11 + a.m22*b.m21 + a.m23*b.m31;
        r.m22 = a.m20*b.m02 + a.m21*b.m12 + a.m22*b.m22 + a.m23*b.m32;
        r.m23 = a.m20*b.m03 + a.m21*b.m13 + a.m22*b.m23 + a.m23*b.m33;

        r.m30 = a.m30*b.m00 + a.m31*b.m10 + a.m32*b.m20 + a.m33*b.m30;
        r.m31 = a.m30*b.m01 + a.m31*b.m11 + a.m32*b.m21 + a.m33*b.m31;
        r.m32 = a.m30*b.m02 + a.m31*b.m12 + a.m32*b.m22 + a.m33*b.m32;
        r.m33 = a.m30*b.m03 + a.m31*b.m13 + a.m32*b.m23 + a.m33*b.m33;
        return r;
    };

    def dmat4_mul_vec4(DMat4 m, DVec4 v) -> DVec4
    {
        DVec4 r;
        r.x = m.m00*v.x + m.m01*v.y + m.m02*v.z + m.m03*v.w;
        r.y = m.m10*v.x + m.m11*v.y + m.m12*v.z + m.m13*v.w;
        r.z = m.m20*v.x + m.m21*v.y + m.m22*v.z + m.m23*v.w;
        r.w = m.m30*v.x + m.m31*v.y + m.m32*v.z + m.m33*v.w;
        return r;
    };

    // Look-at view matrix: camera at eye, looking at center, up hint = up
    def dmat4_lookat(DVec3 eye, DVec3 center, DVec3 up) -> DMat4
    {
        DVec3  f, r, u;
        DMat4  m;

        // Forward: eye -> center, then negate (camera looks toward -Z)
        f.x = center.x - eye.x;
        f.y = center.y - eye.y;
        f.z = center.z - eye.z;
        f = dvec3_normalize(f);

        r = dvec3_cross(f, up);
        r = dvec3_normalize(r);

        u = dvec3_cross(r, f);

        // Build row-major view matrix (right, up, -fwd rows, then translation)
        m.m00 =  r.x; m.m01 =  r.y; m.m02 =  r.z;
        m.m03 = -(r.x*eye.x + r.y*eye.y + r.z*eye.z);

        m.m10 =  u.x; m.m11 =  u.y; m.m12 =  u.z;
        m.m13 = -(u.x*eye.x + u.y*eye.y + u.z*eye.z);

        m.m20 = -f.x; m.m21 = -f.y; m.m22 = -f.z;
        m.m23 =  (f.x*eye.x + f.y*eye.y + f.z*eye.z);

        m.m30 = 0.0; m.m31 = 0.0; m.m32 = 0.0; m.m33 = 1.0;
        return m;
    };

    // Symmetric perspective projection (GL convention: maps to [-1,1] NDC)
    def dmat4_perspective(double fov_y_rad, double aspect,
                          double near_z, double far_z) -> DMat4
    {
        double f, nf;
        DMat4  m;
        f  = 1.0 / tan(fov_y_rad * 0.5);
        nf = 1.0 / (near_z - far_z);

        m.m00 = f / aspect; m.m01 = 0.0; m.m02 = 0.0;                        m.m03 = 0.0;
        m.m10 = 0.0;        m.m11 = f;   m.m12 = 0.0;                        m.m13 = 0.0;
        m.m20 = 0.0;        m.m21 = 0.0; m.m22 = (far_z + near_z) * nf;     m.m23 = (2.0 * far_z * near_z) * nf;
        m.m30 = 0.0;        m.m31 = 0.0; m.m32 = -1.0;                       m.m33 = 0.0;
        return m;
    };

    // TRS (translate * rotateY * rotateX * rotateZ * scale) model matrix
    def dmat4_trs(double tx, double ty, double tz,
                  double rx, double ry, double rz,
                  double sx, double sy, double sz) -> DMat4
    {
        double cx, sx2, cy, sy2, cz, sz2;
        DMat4  s, r, t, tmp;

        cx  = cos(rx); sx2 = sin(rx);
        cy  = cos(ry); sy2 = sin(ry);
        cz  = cos(rz); sz2 = sin(rz);

        // Scale
        s = dmat4_identity();
        s.m00 = sx; s.m11 = sy; s.m22 = sz;

        // Rotation: R = Ry * Rx * Rz
        DMat4 rx_mat, ry_mat, rz_mat;

        rx_mat = dmat4_identity();
        rx_mat.m11 =  cx; rx_mat.m12 = -sx2;
        rx_mat.m21 =  sx2; rx_mat.m22 =  cx;

        ry_mat = dmat4_identity();
        ry_mat.m00 =  cy; ry_mat.m02 =  sy2;
        ry_mat.m20 = -sy2; ry_mat.m22 =  cy;

        rz_mat = dmat4_identity();
        rz_mat.m00 =  cz; rz_mat.m01 = -sz2;
        rz_mat.m10 =  sz2; rz_mat.m11 =  cz;

        tmp = dmat4_mul(ry_mat, rx_mat);
        r   = dmat4_mul(tmp, rz_mat);

        // Translation
        t = dmat4_identity();
        t.m03 = tx; t.m13 = ty; t.m23 = tz;

        // Result: T * R * S
        tmp = dmat4_mul(r, s);
        return dmat4_mul(t, tmp);
    };

    // Normal matrix: transpose of inverse of upper-left 3x3 of model matrix
    // For uniform scaling this equals the rotation sub-matrix; we use cofactor
    // method to avoid full inversion for the common case.
    def dmat4_normal_mat(DMat4 m, DMat4* out) -> void
    {
        // Cofactors of upper-left 3x3
        double a00, a01, a02, a10, a11, a12, a20, a21, a22,
               det, inv_det;

        a00 = m.m00; a01 = m.m01; a02 = m.m02;
        a10 = m.m10; a11 = m.m11; a12 = m.m12;
        a20 = m.m20; a21 = m.m21; a22 = m.m22;

        det = a00*(a11*a22 - a12*a21)
            - a01*(a10*a22 - a12*a20)
            + a02*(a10*a21 - a11*a20);

        if (abs(det) < RC_EPSILON)
        {
            *out = dmat4_identity();
            return;
        };

        inv_det = 1.0 / det;

        // Transposed cofactors == adjugate (= inverse * det), then divide by det
        out.m00 =  (a11*a22 - a12*a21) * inv_det;
        out.m01 = -(a10*a22 - a12*a20) * inv_det;
        out.m02 =  (a10*a21 - a11*a20) * inv_det;
        out.m10 = -(a01*a22 - a02*a21) * inv_det;
        out.m11 =  (a00*a22 - a02*a20) * inv_det;
        out.m12 = -(a00*a21 - a01*a20) * inv_det;
        out.m20 =  (a01*a12 - a02*a11) * inv_det;
        out.m21 = -(a00*a12 - a02*a10) * inv_det;
        out.m22 =  (a00*a11 - a01*a10) * inv_det;

        out.m03 = 0.0; out.m13 = 0.0; out.m23 = 0.0;
        out.m30 = 0.0; out.m31 = 0.0; out.m32 = 0.0; out.m33 = 1.0;
    };

    // =========================================================================
    // TEXTURE SAMPLING (64-bit)
    // =========================================================================

    def rc_tex_sample64(RCTexture* tex, double u, double v) -> u64
    {
        i32    tx, ty;
        double fu, fv;

        fu = u - (double)(i32)u;
        fv = v - (double)(i32)v;
        if (fu < 0.0) { fu += 1.0; };
        if (fv < 0.0) { fv += 1.0; };

        tx = (i32)(fu * (double)tex.width);
        ty = (i32)(fv * (double)tex.height);

        if (tx >= tex.width)  { tx = tex.width  - 1; };
        if (ty >= tex.height) { ty = tex.height - 1; };

        return tex.pixels[ty * tex.width + tx];
    };

    // Bilinear texture sample (higher quality for mesh surfaces)
    def rc_tex_sample_bilinear(RCTexture* tex, double u, double v) -> u64
    {
        double fu, fv, fx, fy, tx_d, ty_d;
        i32    x0, y0, x1, y1;
        double r00, g00, b00, r10, g10, b10, r01, g01, b01, r11, g11, b11;
        double r, g, b;
        u64    p00, p10, p01, p11;

        fu = u - (double)(i32)u;
        fv = v - (double)(i32)v;
        if (fu < 0.0) { fu += 1.0; };
        if (fv < 0.0) { fv += 1.0; };

        tx_d = fu * (double)(tex.width  - 1);
        ty_d = fv * (double)(tex.height - 1);

        x0 = (i32)tx_d;
        y0 = (i32)ty_d;
        x1 = x0 + 1;
        y1 = y0 + 1;

        if (x1 >= tex.width)  { x1 = tex.width  - 1; };
        if (y1 >= tex.height) { y1 = tex.height - 1; };

        fx = tx_d - (double)x0;
        fy = ty_d - (double)y0;

        p00 = tex.pixels[y0 * tex.width + x0];
        p10 = tex.pixels[y0 * tex.width + x1];
        p01 = tex.pixels[y1 * tex.width + x0];
        p11 = tex.pixels[y1 * tex.width + x1];

        color64_unpack(p00, @r00, @g00, @b00);
        color64_unpack(p10, @r10, @g10, @b10);
        color64_unpack(p01, @r01, @g01, @b01);
        color64_unpack(p11, @r11, @g11, @b11);

        r = r00*(1.0-fx)*(1.0-fy) + r10*fx*(1.0-fy) + r01*(1.0-fx)*fy + r11*fx*fy;
        g = g00*(1.0-fx)*(1.0-fy) + g10*fx*(1.0-fy) + g01*(1.0-fx)*fy + g11*fx*fy;
        b = b00*(1.0-fx)*(1.0-fy) + b10*fx*(1.0-fy) + b01*(1.0-fx)*fy + b11*fx*fy;

        return color64_pack(r, g, b);
    };

    // =========================================================================
    // MAP API  (2.5D, unchanged in semantics)
    // =========================================================================

    def rc_map_init(RCMap* m, i32 width, i32 height) -> void
    {
        size_t bytes;
        i32    i, total;

        total = width * height;
        bytes = (size_t)(total * (i32)(sizeof(RCTile) / 8));

        m.cells       = (RCTile*)fmalloc(bytes);
        m.width       = width;
        m.height      = height;
        m.floor_color = (u64)0xFFFF404040404040;
        m.ceil_color  = (u64)0xFFFF808080808080;

        i = 0;
        while (i < total)
        {
            m.cells[i].flags     = RC_TILE_EMPTY;
            m.cells[i].tex_wall  = 0;
            m.cells[i].tex_floor = 0;
            m.cells[i].tex_ceil  = 0;
            m.cells[i].tint      = 0;
            i++;
        };
    };

    def rc_map_free(RCMap* m) -> void
    {
        ffree((u64)m.cells);
        m.cells  = (RCTile*)0;
        m.width  = 0;
        m.height = 0;
    };

    def rc_map_get(RCMap* m, i32 x, i32 y) -> RCTile
    {
        RCTile empty;
        empty.flags     = RC_TILE_EMPTY;
        empty.tex_wall  = 0;
        empty.tex_floor = 0;
        empty.tex_ceil  = 0;
        empty.tint      = 0;
        if (x < 0 | x >= m.width | y < 0 | y >= m.height) { return empty; };
        return *m.cells[y * m.width + x];
    };

    def rc_map_set(RCMap* m, i32 x, i32 y, RCTile tile) -> void
    {
        if (x < 0 | x >= m.width | y < 0 | y >= m.height) { return; };
        m.cells[y * m.width + x] = tile;
    };

    def rc_map_set_solid(RCMap* m, i32 x, i32 y, i32 tex, u64 tint) -> void
    {
        RCTile t;
        t.flags     = RC_TILE_SOLID;
        t.tex_wall  = tex;
        t.tex_floor = 0;
        t.tex_ceil  = 0;
        t.tint      = tint;
        rc_map_set(m, x, y, t);
    };

    // =========================================================================
    // 2.5D PLAYER & CAMERA API
    // =========================================================================

    def rc_player_init(RCPlayer* p, double x, double y, double angle) -> void
    {
        p.pos_x      = x;
        p.pos_y      = y;
        p.angle      = angle;
        p.move_speed = 0.004;
        p.turn_speed = 0.001;
    };

    def rc_camera_init(RCCamera* cam, double fov_deg,
                       i32 sw, i32 sh, double view_dist) -> void
    {
        double fov_rad, half_fov;
        fov_rad  = fov_deg * RC_DEG_TO_RAD;
        half_fov = fov_rad * 0.5;

        cam.fov_h     = fov_rad;
        cam.screen_w  = sw;
        cam.screen_h  = sh;
        cam.view_dist = view_dist;
        cam.proj_dist = ((double)sw * 0.5) / tan(half_fov);
        cam.half_h    = (double)sh * 0.5;
        cam.dir_x     = 1.0;
        cam.dir_y     = 0.0;
        cam.plane_x   = 0.0;
        cam.plane_y   = tan(half_fov);
    };

    def rc_camera_sync(RCCamera* cam, RCPlayer* p) -> void
    {
        double a, half_fov;
        a        = p.angle;
        half_fov = cam.fov_h * 0.5;
        cam.dir_x   = cos(a);
        cam.dir_y   = sin(a);
        cam.plane_x = -sin(a) * tan(half_fov);
        cam.plane_y =  cos(a) * tan(half_fov);
    };

    def rc_player_move(RCPlayer* p, RCMap* m, double forward, double strafe) -> void
    {
        double nx, ny, dx, dy, cos_a, sin_a;
        RCTile t;

        cos_a = cos(p.angle);
        sin_a = sin(p.angle);

        dx = cos_a * forward - sin_a * strafe;
        dy = sin_a * forward + cos_a * strafe;

        nx = p.pos_x + dx;
        ny = p.pos_y + dy;

        t = rc_map_get(m, (i32)nx, (i32)p.pos_y);
        if ((t.flags & RC_TILE_SOLID) == 0) { p.pos_x = nx; };

        t = rc_map_get(m, (i32)p.pos_x, (i32)ny);
        if ((t.flags & RC_TILE_SOLID) == 0) { p.pos_y = ny; };
    };

    def rc_player_turn(RCPlayer* p, double delta_rad) -> void
    {
        p.angle += delta_rad;
        while (p.angle >= RC_TWO_PI) { p.angle -= RC_TWO_PI; };
        while (p.angle <  0.0)       { p.angle += RC_TWO_PI; };
    };

    // =========================================================================
    // TEXTURE PALETTE API
    // =========================================================================

    def rc_palette_init(RCTexturePalette* pal, i32 initial_cap) -> void
    {
        size_t bytes;
        bytes       = (size_t)(initial_cap * (i32)(sizeof(RCTexture) / 8));
        pal.slots   = (RCTexture*)fmalloc(bytes);
        pal.count   = 0;
        pal.cap     = initial_cap;
    };

    def rc_palette_free(RCTexturePalette* pal) -> void
    {
        i32 i;

        while (i < pal.count)
        {
            if ((u64)pal.slots[i].pixels != (u64)0)
            {
                ffree((u64)pal.slots[i].pixels);
            };
            i++;
        };
        ffree((u64)pal.slots);
        pal.slots = (RCTexture*)0;
        pal.count = 0;
    };

    def rc_palette_add(RCTexturePalette* pal, u64* pixels, i32 w, i32 h) -> i32
    {
        RCTexture* new_buf;
        size_t     new_bytes;
        i32        idx;

        if (pal.count >= pal.cap)
        {
            pal.cap   = pal.cap * 2;
            new_bytes = (size_t)(pal.cap * (i32)(sizeof(RCTexture) / 8));
            new_buf   = (RCTexture*)fmalloc(new_bytes);
            memcpy((void*)new_buf, (void*)pal.slots,
                   (size_t)(pal.count * (i32)(sizeof(RCTexture) / 8)));
            ffree((u64)pal.slots);
            pal.slots = new_buf;
        };

        idx = pal.count;
        pal.slots[idx].pixels = pixels;
        pal.slots[idx].width  = w;
        pal.slots[idx].height = h;
        pal.count++;
        return idx;
    };

    // =========================================================================
    // 2.5D DDA WALL RAYCASTER
    // =========================================================================

    def rc_cast_walls(RCMap*     m,
                      RCCamera*  cam,
                      RCPlayer*  p,
                      RCWallHit* hits,
                      double*    depth_buf) -> void
    {
        i32    col, map_x, map_y, step_x, step_y, hit_face;
        double ray_dir_x, ray_dir_y;
        double delta_dist_x, delta_dist_y;
        double side_dist_x, side_dist_y;
        double perp_dist, wall_u, cam_x;
        bool   hit_solid;
        i32    draw_start, draw_end, line_height,
               fy;
        RCTile tile;

        while (col < cam.screen_w)
        {
            cam_x = 2.0 * ((double)col / (double)cam.screen_w) - 1.0;

            ray_dir_x = cam.dir_x + cam.plane_x * cam_x;
            ray_dir_y = cam.dir_y + cam.plane_y * cam_x;

            map_x = (i32)p.pos_x;
            map_y = (i32)p.pos_y;

            if (abs(ray_dir_x) < RC_EPSILON)
            {
                delta_dist_x = RC_INF;
            }
            else
            {
                delta_dist_x = abs(1.0 / ray_dir_x);
            };

            if (abs(ray_dir_y) < RC_EPSILON)
            {
                delta_dist_y = RC_INF;
            }
            else
            {
                delta_dist_y = abs(1.0 / ray_dir_y);
            };

            if (ray_dir_x < 0.0)
            {
                step_x      = -1;
                side_dist_x = (p.pos_x - (double)map_x) * delta_dist_x;
            }
            else
            {
                step_x      = 1;
                side_dist_x = ((double)(map_x + 1) - p.pos_x) * delta_dist_x;
            };

            if (ray_dir_y < 0.0)
            {
                step_y      = -1;
                side_dist_y = (p.pos_y - (double)map_y) * delta_dist_y;
            }
            else
            {
                step_y      = 1;
                side_dist_y = ((double)(map_y + 1) - p.pos_y) * delta_dist_y;
            };

            hit_solid = false;
            hit_face  = RC_FACE_NONE;
            perp_dist = RC_INF;

            while (!hit_solid)
            {
                if (side_dist_x < side_dist_y)
                {
                    side_dist_x += delta_dist_x;
                    map_x       += step_x;
                    hit_face     = (step_x > 0) ? RC_FACE_X_NEG : RC_FACE_X_POS;
                }
                else
                {
                    side_dist_y += delta_dist_y;
                    map_y       += step_y;
                    hit_face     = (step_y > 0) ? RC_FACE_Y_NEG : RC_FACE_Y_POS;
                };

                tile = rc_map_get(m, map_x, map_y);

                if (tile.flags & RC_TILE_SOLID) { hit_solid = true; };

                if (hit_face == RC_FACE_X_POS | hit_face == RC_FACE_X_NEG)
                {
                    perp_dist = side_dist_x - delta_dist_x;
                }
                else
                {
                    perp_dist = side_dist_y - delta_dist_y;
                };

                if (perp_dist > cam.view_dist) { hit_solid = true; };
            };

            if (hit_face == RC_FACE_X_POS | hit_face == RC_FACE_X_NEG)
            {
                perp_dist = side_dist_x - delta_dist_x;
            }
            else
            {
                perp_dist = side_dist_y - delta_dist_y;
            };

            // Write the column's view-space depth to the full depth buffer
            // (covers screen_h pixels in this column, same Z for all of them
            //  in 2.5D mode — the 3D mesh pass will overwrite with finer Z)
            {
                fy = 0;
                while (fy < cam.screen_h)
                {
                    depth_buf[fy * cam.screen_w + col] = perp_dist;
                    fy++;
                };
            };

            if (hit_face == RC_FACE_X_POS | hit_face == RC_FACE_X_NEG)
            {
                wall_u = p.pos_y + perp_dist * ray_dir_y;
            }
            else
            {
                wall_u = p.pos_x + perp_dist * ray_dir_x;
            };
            wall_u -= (double)(i64)wall_u;

            if (perp_dist < RC_EPSILON) { perp_dist = RC_EPSILON; };
            line_height = (i32)(cam.proj_dist / perp_dist);

            draw_start = (i32)(cam.half_h - (double)line_height * 0.5);
            draw_end   = (i32)(cam.half_h + (double)line_height * 0.5);

            if (draw_start < 0)            { draw_start = 0; };
            if (draw_end   >= cam.screen_h) { draw_end = cam.screen_h - 1; };

            hits[col].dist     = perp_dist;
            hits[col].wall_u   = wall_u;
            hits[col].tile_x   = map_x;
            hits[col].tile_y   = map_y;
            hits[col].face     = hit_face;
            hits[col].tex_idx  = tile.tex_wall;
            hits[col].tint     = tile.tint;
            hits[col].draw_top = draw_start;
            hits[col].draw_bot = draw_end;

            col++;
        };
    };

    // =========================================================================
    // 2.5D FLOOR / CEILING RAYCASTER
    // =========================================================================

    def rc_cast_floor(RCMap*            m,
                      RCCamera*         cam,
                      RCPlayer*         p,
                      RCWallHit*        hits,
                      RCTexturePalette* palette,
                      double*           depth_buf,
                      u64*              buf) -> void
    {
        i32    row, col, cell_x, cell_y;
        double row_dist, floor_x, floor_y, step_x, step_y;
        double tex_u, tex_v, shade;
        u64    floor_col, ceil_col;
        RCTile tile;
        double left_ray_x, left_ray_y, right_ray_x, right_ray_y;
        double pos_z;

        pos_z = cam.half_h;

        while (row < cam.screen_h)
        {
            left_ray_x  = cam.dir_x - cam.plane_x;
            left_ray_y  = cam.dir_y - cam.plane_y;
            right_ray_x = cam.dir_x + cam.plane_x;
            right_ray_y = cam.dir_y + cam.plane_y;

            if (row < (i32)pos_z)
            {
                // Ceiling
                if ((i32)pos_z - row == 0) { row++; continue; };
                row_dist = pos_z / (pos_z - (double)row);

                step_x  = row_dist * (right_ray_x - left_ray_x) / (double)cam.screen_w;
                step_y  = row_dist * (right_ray_y - left_ray_y) / (double)cam.screen_w;
                floor_x = p.pos_x + row_dist * left_ray_x;
                floor_y = p.pos_y + row_dist * left_ray_y;
                shade   = fog_factor(row_dist, cam.view_dist);

                col = 0;
                while (col < cam.screen_w)
                {
                    cell_x = (i32)floor_x;
                    cell_y = (i32)floor_y;
                    tile   = rc_map_get(m, cell_x, cell_y);

                    if (palette != (RCTexturePalette*)0 & tile.tex_ceil > 0 &
                        tile.tex_ceil <= palette.count)
                    {
                        tex_u    = floor_x - (double)cell_x;
                        tex_v    = floor_y - (double)cell_y;
                        ceil_col = rc_tex_sample64(@palette.slots[tile.tex_ceil - 1], tex_u, tex_v);
                    }
                    else
                    {
                        ceil_col = m.ceil_color;
                    };

                    if (tile.tint != 0) { ceil_col = color64_tint(ceil_col, tile.tint); };
                    ceil_col = color64_scale(ceil_col, shade);

                    // Depth test: only write if no 3D geometry is closer
                    if (row_dist < depth_buf[row * cam.screen_w + col])
                    {
                        buf[row * cam.screen_w + col] = ceil_col;
                    };

                    floor_x += step_x;
                    floor_y += step_y;
                    col++;
                };
            }
            else
            {
                // Floor
                if (row - (i32)pos_z == 0) { row++; continue; };
                row_dist = pos_z / ((double)row - pos_z);

                step_x  = row_dist * (right_ray_x - left_ray_x) / (double)cam.screen_w;
                step_y  = row_dist * (right_ray_y - left_ray_y) / (double)cam.screen_w;
                floor_x = p.pos_x + row_dist * left_ray_x;
                floor_y = p.pos_y + row_dist * left_ray_y;
                shade   = fog_factor(row_dist, cam.view_dist);

                col = 0;
                while (col < cam.screen_w)
                {
                    cell_x    = (i32)floor_x;
                    cell_y    = (i32)floor_y;
                    tile      = rc_map_get(m, cell_x, cell_y);

                    if (palette != (RCTexturePalette*)0 & tile.tex_floor > 0 &
                        tile.tex_floor <= palette.count)
                    {
                        tex_u     = floor_x - (double)cell_x;
                        tex_v     = floor_y - (double)cell_y;
                        floor_col = rc_tex_sample64(@palette.slots[tile.tex_floor - 1], tex_u, tex_v);
                    }
                    else
                    {
                        floor_col = m.floor_color;
                    };

                    if (tile.tint != 0) { floor_col = color64_tint(floor_col, tile.tint); };
                    floor_col = color64_scale(floor_col, shade);

                    if (row_dist < depth_buf[row * cam.screen_w + col])
                    {
                        buf[row * cam.screen_w + col] = floor_col;
                    };

                    floor_x += step_x;
                    floor_y += step_y;
                    col++;
                };
            };

            row++;
        };
    };

    // =========================================================================
    // 2.5D SKY PASS
    // =========================================================================

    def rc_draw_sky(RCSky* sky, RCCamera* cam, u64* buf) -> void
    {
        i32    row, col, half;
        double t;
        u64    sky_col;

        half = (i32)cam.half_h;
        while (row < half)
        {
            t       = (double)row / (double)half;
            sky_col = color64_lerp(sky.color_top, sky.color_horizon, t);
            col     = 0;
            while (col < cam.screen_w)
            {
                buf[row * cam.screen_w + col] = sky_col;
                col++;
            };
            row++;
        };
    };

    // =========================================================================
    // 2.5D WALL STRIP DRAW
    // =========================================================================

    def rc_draw_walls(RCCamera*         cam,
                      RCWallHit*        hits,
                      RCTexturePalette* palette,
                      double*           depth_buf,
                      u64*              buf) -> void
    {
        i32    col, y, tex_h;
        double shade, v_step, v_pos, tex_v, tex_u;
        u64    wall_col;
        RCTexture* tex;

        while (col < cam.screen_w)
        {
            shade = fog_factor(hits[col].dist, cam.view_dist);

            tex = (RCTexture*)0;
            if (palette != (RCTexturePalette*)0 & hits[col].tex_idx > 0 &
                hits[col].tex_idx <= palette.count)
            {
                tex   = @palette.slots[hits[col].tex_idx - 1];
                tex_h = tex.height;
            }
            else
            {
                tex_h = 64;
            };

            v_step = (double)tex_h / (double)(hits[col].draw_bot - hits[col].draw_top + 1);
            v_pos  = 0.0;
            tex_u  = hits[col].wall_u;

            // Darken X-axis faces for a cheap lighting cue
            if (hits[col].face == RC_FACE_X_POS | hits[col].face == RC_FACE_X_NEG)
            {
                shade *= 0.7;
            };

            y = hits[col].draw_top;
            while (y <= hits[col].draw_bot)
            {
                // Depth test against 3D geometry already in the buffer
                if (hits[col].dist >= depth_buf[y * cam.screen_w + col]) { v_pos += v_step; y++; continue; };

                if (tex != (RCTexture*)0)
                {
                    tex_v    = v_pos / (double)tex_h;
                    wall_col = rc_tex_sample64(tex, tex_u, tex_v);
                }
                else
                {
                    wall_col = hits[col].tint != 0 ? hits[col].tint : (u64)0xFFFFAAAAAAAAAAAA;
                };

                if (hits[col].tint != 0 & tex != (RCTexture*)0)
                {
                    wall_col = color64_tint(wall_col, hits[col].tint);
                };

                wall_col = color64_scale(wall_col, shade);
                buf[y * cam.screen_w + col] = wall_col;
                depth_buf[y * cam.screen_w + col] = hits[col].dist;

                v_pos += v_step;
                y++;
            };

            col++;
        };
    };

    // =========================================================================
    // 2.5D SPRITE PASS
    // =========================================================================

    def rc_sprite_distances(RCSprite* sprites, i32 count, RCPlayer* p) -> void
    {
        i32    i;
        double dx, dy;
        while (i < count)
        {
            dx = sprites[i].world_x - p.pos_x;
            dy = sprites[i].world_y - p.pos_y;
            sprites[i].dist_sq = dx*dx + dy*dy;
            i++;
        };
    };

    def rc_sprite_sort(RCSprite* sprites, i32 count) -> void
    {
        i32      i, j;
        RCSprite tmp;
        i = 1;
        while (i < count)
        {
            tmp = sprites[i];
            j   = i - 1;
            while (j >= 0 & sprites[j].dist_sq < tmp.dist_sq)
            {
                sprites[j + 1] = sprites[j];
                j--;
            };
            sprites[j + 1] = tmp;
            i++;
        };
    };

    def rc_draw_sprites(RCCamera*         cam,
                        RCPlayer*         p,
                        RCSprite*         sprites,
                        i32               sprite_count,
                        double*           depth_buf,
                        RCTexturePalette* palette,
                        u64*              buf) -> void
    {
        i32    s, x, y,
               sprite_screen_x,
               sprite_height, sprite_width,
               draw_start_y, draw_end_y,
               draw_start_x, draw_end_x,
               tex_x, tex_y;
        double sprite_x, sprite_y,
               inv_det, transform_x, transform_y,
               tex_u, tex_v, shade, det;
        u64    sprite_col;
        RCTexture* tex;

        det     = cam.plane_x * cam.dir_y - cam.dir_x * cam.plane_y;
        inv_det = 1.0 / (det + RC_EPSILON);

        rc_sprite_distances(sprites, sprite_count, p);
        rc_sprite_sort(sprites, sprite_count);

        s = 0;
        while (s < sprite_count)
        {
            sprite_x = sprites[s].world_x - p.pos_x;
            sprite_y = sprites[s].world_y - p.pos_y;

            transform_x = inv_det * (cam.dir_y * sprite_x - cam.dir_x * sprite_y);
            transform_y = inv_det * (-cam.plane_y * sprite_x + cam.plane_x * sprite_y);

            if (transform_y <= 0.0) { s++; continue; };

            sprite_screen_x = (i32)(((double)cam.screen_w * 0.5) *
                              (1.0 + transform_x / transform_y));

            sprite_height = (i32)(abs(cam.proj_dist / transform_y) * sprites[s].scale);
            sprite_width  = sprite_height;

            draw_start_y = (i32)(cam.half_h - (double)sprite_height * 0.5);
            draw_end_y   = (i32)(cam.half_h + (double)sprite_height * 0.5);
            if (draw_start_y < 0)             { draw_start_y = 0; };
            if (draw_end_y   >= cam.screen_h)  { draw_end_y = cam.screen_h - 1; };

            draw_start_x = sprite_screen_x - sprite_width  / 2;
            draw_end_x   = sprite_screen_x + sprite_width  / 2;
            if (draw_start_x < 0)             { draw_start_x = 0; };
            if (draw_end_x   >= cam.screen_w)  { draw_end_x = cam.screen_w - 1; };

            tex = (RCTexture*)0;
            if (palette != (RCTexturePalette*)0 & sprites[s].tex_idx > 0 &
                sprites[s].tex_idx <= palette.count)
            {
                tex = @palette.slots[sprites[s].tex_idx - 1];
            };

            shade = fog_factor(sqrt(sprites[s].dist_sq), cam.view_dist);

            x = draw_start_x;
            while (x <= draw_end_x)
            {
                tex_u = (double)(x - (sprite_screen_x - sprite_width / 2)) /
                        (double)(sprite_width + 1);

                y = draw_start_y;
                while (y <= draw_end_y)
                {
                    if (transform_y >= depth_buf[y * cam.screen_w + x]) { y++; continue; };

                    tex_v = (double)(y - draw_start_y) / (double)(sprite_height + 1);

                    if (tex != (RCTexture*)0)
                    {
                        sprite_col = rc_tex_sample64(tex, tex_u, tex_v);
                        // Magenta keyed transparency (top 16-bit channel: 0xFFFF)
                        if ((sprite_col & (u64)0x0000FFFFFFFFFFFF) == (u64)0x0000FFFF0000FFFF)
                        {
                            y++;
                            continue;
                        };
                    }
                    else
                    {
                        sprite_col = sprites[s].tint != 0 ? sprites[s].tint : (u64)0xFFFFFFFFFFFFFFFF;
                    };

                    if (sprites[s].tint != 0 & tex != (RCTexture*)0)
                    {
                        sprite_col = color64_tint(sprite_col, sprites[s].tint);
                    };

                    sprite_col = color64_scale(sprite_col, shade);
                    buf[y * cam.screen_w + x]            = sprite_col;
                    depth_buf[y * cam.screen_w + x]      = transform_y;

                    y++;
                };
                x++;
            };

            s++;
        };
    };

    // =========================================================================
    // 2.5D RENDER ENTRY
    //
    // zbuf must be caller-allocated: screen_w * screen_h * sizeof(double) bytes,
    // pre-filled to RC_INF before calling.  Shared with 3D pass.
    // =========================================================================

    def rc_render(RCScene* scene, u64* buf, double* zbuf) -> void
    {
        RCWallHit* hits;
        size_t     hit_bytes;
        i32        sw;

        sw        = scene.cam.screen_w;
        hit_bytes = (size_t)(sw * (i32)(sizeof(RCWallHit) / 8));
        hits      = (RCWallHit*)fmalloc(hit_bytes);

        if ((scene.passes & RC_PASS_SKY) & scene.sky != (RCSky*)0)
        {
            rc_draw_sky(scene.sky, scene.cam, buf);
        };

        if (scene.passes & RC_PASS_FLOOR)
        {
            rc_cast_floor(scene.map, scene.cam, scene.player,
                          hits, scene.palette, zbuf, buf);
        };

        if (scene.passes & RC_PASS_WALLS)
        {
            rc_cast_walls(scene.map, scene.cam, scene.player, hits, zbuf);
            rc_draw_walls(scene.cam, hits, scene.palette, zbuf, buf);
        };

        if ((scene.passes & RC_PASS_SPRITES) &
            scene.sprites != (RCSprite*)0 &
            scene.sprite_count > 0)
        {
            rc_draw_sprites(scene.cam, scene.player,
                            scene.sprites, scene.sprite_count,
                            zbuf, scene.palette, buf);
        };

        ffree((u64)hits);
    };

    def rc_scene_init(RCScene*          scene,
                      RCMap*            map,
                      RCPlayer*         player,
                      RCCamera*         cam,
                      RCTexturePalette* palette,
                      RCSky*            sky) -> void
    {
        scene.map          = map;
        scene.player       = player;
        scene.cam          = cam;
        scene.palette      = palette;
        scene.sprites      = (RCSprite*)0;
        scene.sprite_count = 0;
        scene.sky          = sky;
        scene.passes       = RC_PASS_ALL;
    };

    def rc_scene_set_sprites(RCScene* scene, RCSprite* sprites, i32 count) -> void
    {
        scene.sprites      = sprites;
        scene.sprite_count = count;
    };

    // =========================================================================
    // 3D PLAYER & CAMERA API
    // =========================================================================

    def r3d_player_init(R3DPlayer* p,
                        double x, double y, double z) -> void
    {
        p.pos_x      = x;
        p.pos_y      = y;
        p.pos_z      = z;
        p.yaw        = 0.0;
        p.pitch      = 0.0;
        p.move_speed = 0.1;
        p.turn_speed = 0.002;
        p.pitch_speed= 0.002;
    };

    def r3d_player_move(R3DPlayer* p, double forward, double strafe, double up) -> void
    {
        double cy, sy;
        cy = cos(p.yaw);
        sy = sin(p.yaw);
        p.pos_x += (cy * forward + sy * strafe);
        p.pos_y += up;
        p.pos_z += (sy * forward - cy * strafe);
    };

    def r3d_player_turn(R3DPlayer* p, double dyaw, double dpitch) -> void
    {
        double max_pitch;
        p.yaw += dyaw;
        while (p.yaw >= RC_TWO_PI) { p.yaw -= RC_TWO_PI; };
        while (p.yaw < 0.0)        { p.yaw += RC_TWO_PI; };

        p.pitch += dpitch;
        max_pitch = RC_HALF_PI * 0.99;
        if (p.pitch >  max_pitch) { p.pitch =  max_pitch; };
        if (p.pitch < -max_pitch) { p.pitch = -max_pitch; };
    };

    def r3d_camera_init(R3DCamera* cam,
                        double fov_h_deg,
                        i32 sw, i32 sh,
                        double near_z, double far_z) -> void
    {
        double fov_h_rad, half_fov;

        cam.screen_w  = sw;
        cam.screen_h  = sh;
        cam.near_z    = near_z;
        cam.far_z     = far_z;
        cam.aspect    = (double)sw / (double)sh;
        cam.fov_h     = fov_h_deg * RC_DEG_TO_RAD;

        half_fov      = cam.fov_h * 0.5;
        cam.fov_v     = 2.0 * atan(tan(half_fov) / cam.aspect);
        cam.proj_dist = ((double)sw * 0.5) / tan(half_fov);

        cam.view = dmat4_identity();
        cam.proj = dmat4_perspective(cam.fov_v, cam.aspect, near_z, far_z);
        cam.vp   = cam.proj;

        cam.eye_x = 0.0; cam.eye_y = 0.0; cam.eye_z = 0.0;
        cam.fwd_x = 0.0; cam.fwd_y = 0.0; cam.fwd_z = -1.0;
        cam.right_x= 1.0; cam.right_y= 0.0; cam.right_z= 0.0;
        cam.up_x  = 0.0; cam.up_y  = 1.0; cam.up_z  = 0.0;
    };

    // Build view matrix and extract camera axes from R3DPlayer state.
    // Call once per frame before r3d_render().
    def r3d_camera_sync(R3DCamera* cam, R3DPlayer* p) -> void
    {
        DVec3  eye, center, up_hint;
        double cy, sy, cp, sp;
        double fwd_x, fwd_y, fwd_z;

        cy = cos(p.yaw);   sy = sin(p.yaw);
        cp = cos(p.pitch); sp = sin(p.pitch);

        // Forward vector from yaw + pitch
        fwd_x = sy * cp;
        fwd_y = sp;
        fwd_z = -cy * cp;

        eye.x = p.pos_x; eye.y = p.pos_y; eye.z = p.pos_z;
        center.x = eye.x + fwd_x;
        center.y = eye.y + fwd_y;
        center.z = eye.z + fwd_z;
        up_hint.x = 0.0; up_hint.y = 1.0; up_hint.z = 0.0;

        cam.view = dmat4_lookat(eye, center, up_hint);
        cam.vp   = dmat4_mul(cam.proj, cam.view);

        cam.eye_x = p.pos_x;
        cam.eye_y = p.pos_y;
        cam.eye_z = p.pos_z;

        cam.fwd_x = fwd_x;
        cam.fwd_y = fwd_y;
        cam.fwd_z = fwd_z;

        // Right = cross(fwd, world_up) normalised
        cam.right_x = cy * cp;
        cam.right_y = 0.0;
        cam.right_z = sy * cp;
        {
            double rl;
            rl = sqrt(cam.right_x*cam.right_x + cam.right_z*cam.right_z);
            if (rl > RC_EPSILON)
            {
                cam.right_x /= rl;
                cam.right_z /= rl;
            };
        };

        // Up = cross(right, fwd)
        cam.up_x = cam.right_y * fwd_z - cam.right_z * fwd_y;
        cam.up_y = cam.right_z * fwd_x - cam.right_x * fwd_z;
        cam.up_z = cam.right_x * fwd_y - cam.right_y * fwd_x;
        {
            double ul;
            ul = sqrt(cam.up_x*cam.up_x + cam.up_y*cam.up_y + cam.up_z*cam.up_z);
            if (ul > RC_EPSILON)
            {
                cam.up_x /= ul;
                cam.up_y /= ul;
                cam.up_z /= ul;
            };
        };
    };

    // =========================================================================
    // 3D MESH API
    // =========================================================================

    def r3d_mesh_init(R3DMesh* mesh,
                      i32 vert_count, i32 tri_count, i32 tex_idx) -> void
    {
        size_t vbytes, tbytes;
        vbytes        = (size_t)(vert_count * (i32)(sizeof(R3DVertex)   / 8));
        tbytes        = (size_t)(tri_count  * (i32)(sizeof(R3DTriangle) / 8));
        mesh.verts     = (R3DVertex*)fmalloc(vbytes);
        mesh.tris      = (R3DTriangle*)fmalloc(tbytes);
        mesh.vert_count= vert_count;
        mesh.tri_count = tri_count;
        mesh.tex_idx   = tex_idx;
    };

    def r3d_mesh_free(R3DMesh* mesh) -> void
    {
        ffree((u64)mesh.verts);
        ffree((u64)mesh.tris);
        mesh.verts      = (R3DVertex*)0;
        mesh.tris       = (R3DTriangle*)0;
        mesh.vert_count = 0;
        mesh.tri_count  = 0;
    };

    // Compute smooth per-vertex normals by averaging face normals.
    // Call after filling mesh.verts and mesh.tris.
    def r3d_mesh_compute_normals(R3DMesh* mesh) -> void
    {
        i32    i;
        double ax, ay, az, bx, by, bz, cx, cy, cz;
        double ex, ey, ez, fx, fy, fz, nx, ny, nz, len;
        R3DVertex* va;
        R3DVertex* vb;
        R3DVertex* vc;

        // Zero normals
        i = 0;
        while (i < mesh.vert_count)
        {
            mesh.verts[i].nx = 0.0;
            mesh.verts[i].ny = 0.0;
            mesh.verts[i].nz = 0.0;
            i++;
        };

        // Accumulate face normals
        i = 0;
        while (i < mesh.tri_count)
        {
            va = @mesh.verts[mesh.tris[i].a];
            vb = @mesh.verts[mesh.tris[i].b];
            vc = @mesh.verts[mesh.tris[i].c];

            ex = vb.x - va.x; ey = vb.y - va.y; ez = vb.z - va.z;
            fx = vc.x - va.x; fy = vc.y - va.y; fz = vc.z - va.z;

            nx = ey*fz - ez*fy;
            ny = ez*fx - ex*fz;
            nz = ex*fy - ey*fx;

            va.nx += nx; va.ny += ny; va.nz += nz;
            vb.nx += nx; vb.ny += ny; vb.nz += nz;
            vc.nx += nx; vc.ny += ny; vc.nz += nz;

            i++;
        };

        // Normalise
        i = 0;
        while (i < mesh.vert_count)
        {
            len = sqrt(mesh.verts[i].nx * mesh.verts[i].nx +
                       mesh.verts[i].ny * mesh.verts[i].ny +
                       mesh.verts[i].nz * mesh.verts[i].nz);
            if (len > RC_EPSILON)
            {
                mesh.verts[i].nx /= len;
                mesh.verts[i].ny /= len;
                mesh.verts[i].nz /= len;
            };
            i++;
        };
    };

    def r3d_inst_init(R3DMeshInst* inst, R3DMesh* mesh) -> void
    {
        inst.mesh        = mesh;
        inst.pos_x       = 0.0;
        inst.pos_y       = 0.0;
        inst.pos_z       = 0.0;
        inst.rot_x       = 0.0;
        inst.rot_y       = 0.0;
        inst.rot_z       = 0.0;
        inst.scale_x     = 1.0;
        inst.scale_y     = 1.0;
        inst.scale_z     = 1.0;
        inst.shade_model = R3D_SHADE_GOURAUD;
        inst.tint        = 0;
    };

    // =========================================================================
    // 3D LIGHTING
    // =========================================================================

    def r3d_light_directional(R3DLight* l,
                               double dx, double dy, double dz,
                               double r, double g, double b,
                               double intensity) -> void
    {
        double len;
        l.kind      = R3D_LIGHT_DIR;
        len = sqrt(dx*dx + dy*dy + dz*dz);
        if (len > RC_EPSILON) { dx /= len; dy /= len; dz /= len; };
        l.dir_x     = dx;
        l.dir_y     = dy;
        l.dir_z     = dz;
        l.color_r   = r;
        l.color_g   = g;
        l.color_b   = b;
        l.intensity = intensity;
        l.atten_const  = 1.0;
        l.atten_linear = 0.0;
        l.atten_quad   = 0.0;
    };

    def r3d_light_point(R3DLight* l,
                        double px, double py, double pz,
                        double r, double g, double b,
                        double intensity,
                        double kc, double kl, double kq) -> void
    {
        l.kind         = R3D_LIGHT_POINT;
        l.pos_x        = px;
        l.pos_y        = py;
        l.pos_z        = pz;
        l.color_r      = r;
        l.color_g      = g;
        l.color_b      = b;
        l.intensity    = intensity;
        l.atten_const  = kc;
        l.atten_linear = kl;
        l.atten_quad   = kq;
    };

    // Compute combined RGB light contribution for a surface point + normal.
    // Returns (lr, lg, lb) in [0, inf); caller multiplies into pixel color.
    def r3d_eval_lighting(R3DLight*  lights,
                          i32        light_count,
                          double     amb_r, double amb_g, double amb_b,
                          double     wx, double wy, double wz,   // surface world pos
                          double     nx, double ny, double nz,   // unit surface normal
                          double*    out_r, double*  out_g, double* out_b) -> void
    {
        i32    i;
        double lr, lg, lb, ndotl, d, atten, ldx, ldy, ldz, llen;

        *out_r = amb_r;
        *out_g = amb_g;
        *out_b = amb_b;

        i = 0;
        while (i < light_count)
        {
            switch (lights[i].kind)
            {
                case (R3D_LIGHT_DIR)
                {
                    // Negate direction: light_dir points TOWARD the surface,
                    // we want the direction from surface to light.
                    ndotl = -(lights[i].dir_x * nx +
                               lights[i].dir_y * ny +
                               lights[i].dir_z * nz);
                    if (ndotl < 0.0) { ndotl = 0.0; };

                    *out_r += lights[i].color_r * lights[i].intensity * ndotl;
                    *out_g += lights[i].color_g * lights[i].intensity * ndotl;
                    *out_b += lights[i].color_b * lights[i].intensity * ndotl;
                }
                case (R3D_LIGHT_POINT)
                {
                    ldx  = lights[i].pos_x - wx;
                    ldy  = lights[i].pos_y - wy;
                    ldz  = lights[i].pos_z - wz;
                    llen = sqrt(ldx*ldx + ldy*ldy + ldz*ldz);

                    if (llen < RC_EPSILON) { i++; continue; };

                    ldx /= llen; ldy /= llen; ldz /= llen;

                    ndotl = ldx*nx + ldy*ny + ldz*nz;
                    if (ndotl < 0.0) { ndotl = 0.0; };

                    atten = 1.0 / (lights[i].atten_const +
                                   lights[i].atten_linear * llen +
                                   lights[i].atten_quad   * llen * llen);

                    *out_r += lights[i].color_r * lights[i].intensity * ndotl * atten;
                    *out_g += lights[i].color_g * lights[i].intensity * ndotl * atten;
                    *out_b += lights[i].color_b * lights[i].intensity * ndotl * atten;
                }
                default {};
            };

            i++;
        };

        if (*out_r > 1.0) { *out_r = 1.0; };
        if (*out_g > 1.0) { *out_g = 1.0; };
        if (*out_b > 1.0) { *out_b = 1.0; };
    };

    // =========================================================================
    // FULL FRUSTUM CLIPPING  (Sutherland-Hodgman, all 6 clip-space planes)
    //
    // Clip-space homogeneous planes (standard perspective, right-handed):
    //   Near:   w >= near_z          (was the only plane clipped before)
    //   Far:    w >= -z  i.e. z >= -w (equivalently, -w <= z)
    //   Left:   x >= -w
    //   Right:  x <=  w
    //   Bottom: y >= -w
    //   Top:    y <=  w
    //
    // Uses ping-pong between two local buffers so no heap allocation needed.
    // Input: 3 vertices.  Output: up to R3D_MAX_CLIP_VERTS vertices.
    // =========================================================================

    // Linearly interpolate between two clip vertices at parameter t in [0,1]
    def r3d_clip_lerp(R3DClipVert* a, R3DClipVert* b, double t,
                      R3DClipVert* out) -> void
    {
        out.x    = a.x   + (b.x   - a.x)   * t;
        out.y    = a.y   + (b.y   - a.y)   * t;
        out.z    = a.z   + (b.z   - a.z)   * t;
        out.w    = a.w   + (b.w   - a.w)   * t;
        out.nx   = a.nx  + (b.nx  - a.nx)  * t;
        out.ny   = a.ny  + (b.ny  - a.ny)  * t;
        out.nz   = a.nz  + (b.nz  - a.nz)  * t;
        out.u    = a.u   + (b.u   - a.u)   * t;
        out.v    = a.v   + (b.v   - a.v)   * t;
        out.inv_w = (out.w > RC_EPSILON) ? (1.0d / out.w) : 0.0d;
    };

    // Clip against all 6 frustum planes using Sutherland-Hodgman.
    // in_verts: 3 input vertices.  out_verts: output polygon (up to R3D_MAX_CLIP_VERTS).
    // Returns output vertex count (0 if fully outside).
    def r3d_clip_frustum(R3DClipVert* in_verts, i32 in_count,
                         double near_z,
                         R3DClipVert* out_verts) -> i32
    {
        // Ping-pong buffers: clip result of each plane feeds the next
        R3DClipVert[R3D_MAX_CLIP_VERTS] buf_a;
        R3DClipVert[R3D_MAX_CLIP_VERTS] buf_b;
        R3DClipVert* src;
        R3DClipVert* dst;
        R3DClipVert* cur;
        R3DClipVert* nxt;
        i32  src_count, dst_count, i, next_i, plane;
        double cur_d, nxt_d, t;
        bool   cur_in, nxt_in;
        R3DClipVert tmp;

        // Copy input into buf_a
        src = @buf_a[0];
        i = 0;
        while (i < in_count) { buf_a[i] = *(@in_verts[i]); i++; };
        src_count = in_count;

        // Plane 0: Near   — w >= near_z         →  d = w - near_z  (custom near distance)
        // Plane 1: Far    — z <= w               →  d = w - z
        // Plane 2: Left   — x >= -w              →  d = x + w
        // Plane 3: Right  — x <= w               →  d = w - x
        // Plane 4: Bottom — y >= -w              →  d = y + w
        // Plane 5: Top    — y <= w               →  d = w - y

        plane = 0;
        while (plane < 6)
        {
            if (src_count < 3) { return 0; };

            dst = (plane & 1) ? @buf_a[0] : @buf_b[0];
            src = (plane & 1) ? @buf_b[0] : @buf_a[0];

            dst_count = 0;
            i = 0;
            while (i < src_count)
            {
                next_i = (i + 1) % src_count;

                cur = @src[i];
                nxt = @src[next_i];

                switch (plane)
                {
                    case (0) { cur_d = cur.w - near_z; nxt_d = nxt.w - near_z; }
                    case (1) { cur_d = cur.w - cur.z;  nxt_d = nxt.w - nxt.z;  }
                    case (2) { cur_d = cur.x + cur.w;  nxt_d = nxt.x + nxt.w;  }
                    case (3) { cur_d = cur.w - cur.x;  nxt_d = nxt.w - nxt.x;  }
                    case (4) { cur_d = cur.y + cur.w;  nxt_d = nxt.y + nxt.w;  }
                    default  { cur_d = cur.w - cur.y;  nxt_d = nxt.w - nxt.y;  };
                };

                cur_in = cur_d >= 0.0d;
                nxt_in = nxt_d >= 0.0d;

                if (cur_in)
                {
                    dst[dst_count] = *cur;
                    dst_count++;
                };

                if (cur_in != nxt_in)
                {
                    t = cur_d / (cur_d - nxt_d);
                    r3d_clip_lerp(cur, nxt, t, @tmp);
                    dst[dst_count] = tmp;
                    dst_count++;
                };

                i++;
            };

            src_count = dst_count;
            plane++;
        };

        // After 6 planes (even number), result is in buf_a
        src = @buf_a[0];
        i = 0;
        while (i < src_count) { out_verts[i] = *(@src[i]); i++; };
        return src_count;
    };

    // =========================================================================
    // SCANLINE TRIANGLE RASTERIZER
    //
    // Rasterizes a screen-space triangle into buf, writing depth to zbuf.
    // All three vertices carry pre-divided (1/w) and perspective-correct u/v.
    // =========================================================================

    def r3d_draw_triangle(i32 sw, i32 sh,
                          double ax, double ay, double az, double a_inv_w,
                          double au, double av,
                          double anx, double any, double anz,
                          double bx, double by, double bz, double b_inv_w,
                          double bu, double bv,
                          double bnx, double bny, double bnz,
                          double cx, double cy2, double cz, double c_inv_w,
                          double cu, double cv,
                          double cnx, double cny, double cnz,
                          RCTexture*  tex,
                          u64         tint,
                          i32         shade_model,
                          R3DLight*   lights,
                          i32         light_count,
                          double      amb_r, double amb_g, double amb_b,
                          double      wx_a, double wy_a, double wz_a,
                          double      wx_b, double wy_b, double wz_b,
                          double      wx_c, double wy_c, double wz_c,
                          double*     zbuf,
                          u64*        buf) -> void
    {
        // ---- Sort vertices by Y (insertion sort, 3 elements) ----
        double tmp_d;
        i32    y_top, y_mid, y_bot;

        if (ay > by)
        {
            tmp_d = ax; ax = bx; bx = tmp_d;
            tmp_d = ay; ay = by; by = tmp_d;
            tmp_d = az; az = bz; bz = tmp_d;
            tmp_d = a_inv_w; a_inv_w = b_inv_w; b_inv_w = tmp_d;
            tmp_d = au; au = bu; bu = tmp_d;
            tmp_d = av; av = bv; bv = tmp_d;
            tmp_d = anx; anx = bnx; bnx = tmp_d;
            tmp_d = any; any = bny; bny = tmp_d;
            tmp_d = anz; anz = bnz; bnz = tmp_d;
            tmp_d = wx_a; wx_a = wx_b; wx_b = tmp_d;
            tmp_d = wy_a; wy_a = wy_b; wy_b = tmp_d;
            tmp_d = wz_a; wz_a = wz_b; wz_b = tmp_d;
        };

        if (ay > cy2)
        {
            tmp_d = ax; ax = cx; cx = tmp_d;
            tmp_d = ay; ay = cy2; cy2 = tmp_d;
            tmp_d = az; az = cz; cz = tmp_d;
            tmp_d = a_inv_w; a_inv_w = c_inv_w; c_inv_w = tmp_d;
            tmp_d = au; au = cu; cu = tmp_d;
            tmp_d = av; av = cv; cv = tmp_d;
            tmp_d = anx; anx = cnx; cnx = tmp_d;
            tmp_d = any; any = cny; cny = tmp_d;
            tmp_d = anz; anz = cnz; cnz = tmp_d;
            tmp_d = wx_a; wx_a = wx_c; wx_c = tmp_d;
            tmp_d = wy_a; wy_a = wy_c; wy_c = tmp_d;
            tmp_d = wz_a; wz_a = wz_c; wz_c = tmp_d;
        };

        if (by > cy2)
        {
            tmp_d = bx; bx = cx; cx = tmp_d;
            tmp_d = by; by = cy2; cy2 = tmp_d;
            tmp_d = bz; bz = cz; cz = tmp_d;
            tmp_d = b_inv_w; b_inv_w = c_inv_w; c_inv_w = tmp_d;
            tmp_d = bu; bu = cu; cu = tmp_d;
            tmp_d = bv; bv = cv; cv = tmp_d;
            tmp_d = bnx; bnx = cnx; cnx = tmp_d;
            tmp_d = bny; bny = cny; cny = tmp_d;
            tmp_d = bnz; bnz = cnz; cnz = tmp_d;
            tmp_d = wx_b; wx_b = wx_c; wx_c = tmp_d;
            tmp_d = wy_b; wy_b = wy_c; wy_c = tmp_d;
            tmp_d = wz_b; wz_b = wz_c; wz_c = tmp_d;
        };

        y_top = (i32)(ay + 0.5d);
        y_mid = (i32)(by + 0.5d);
        y_bot = (i32)(cy2 - 0.5d);

        if (y_top < 0)   { y_top = 0; };
        if (y_bot >= sh) { y_bot = sh - 1; };

        if (y_top > y_bot) { return; };

        // ---- Pre-compute lighting ----
        // Flat: bake once per triangle. Gouraud: per-vertex values for interpolation.
        double flat_lr, flat_lg, flat_lb;
        flat_lr = 0.0d; flat_lg = 0.0d; flat_lb = 0.0d;

        if (shade_model == R3D_SHADE_FLAT)
        {
            double fwx, fwy, fwz, fnx, fny, fnz;
            fwx = (wx_a + wx_b + wx_c) / 3.0d;
            fwy = (wy_a + wy_b + wy_c) / 3.0d;
            fwz = (wz_a + wz_b + wz_c) / 3.0d;
            fnx = (anx + bnx + cnx);
            fny = (any + bny + cny);
            fnz = (anz + bnz + cnz);
            double flen;
            flen = sqrt(fnx*fnx + fny*fny + fnz*fnz);
            if (flen > RC_EPSILON) { fnx /= flen; fny /= flen; fnz /= flen; };
            r3d_eval_lighting(lights, light_count,
                              amb_r, amb_g, amb_b,
                              fwx, fwy, fwz,
                              fnx, fny, fnz,
                              @flat_lr, @flat_lg, @flat_lb);
        };

        double va_lr, va_lg, va_lb;
        double vb_lr, vb_lg, vb_lb;
        double vc_lr, vc_lg, vc_lb;
        va_lr = 0.0d; va_lg = 0.0d; va_lb = 0.0d;
        vb_lr = 0.0d; vb_lg = 0.0d; vb_lb = 0.0d;
        vc_lr = 0.0d; vc_lg = 0.0d; vc_lb = 0.0d;

        if (shade_model == R3D_SHADE_GOURAUD)
        {
            r3d_eval_lighting(lights, light_count, amb_r, amb_g, amb_b,
                              wx_a, wy_a, wz_a, anx, any, anz,
                              @va_lr, @va_lg, @va_lb);
            r3d_eval_lighting(lights, light_count, amb_r, amb_g, amb_b,
                              wx_b, wy_b, wz_b, bnx, bny, bnz,
                              @vb_lr, @vb_lg, @vb_lb);
            r3d_eval_lighting(lights, light_count, amb_r, amb_g, amb_b,
                              wx_c, wy_c, wz_c, cnx, cny, cnz,
                              @vc_lr, @vc_lg, @vc_lb);
        };

        // ---- Precompute UV in perspective space (u/w, v/w at each vertex) ----
        double au_w, av_w, bu_w, bv_w, cu_w, cv_w;
        au_w = au * a_inv_w; av_w = av * a_inv_w;
        bu_w = bu * b_inv_w; bv_w = bv * b_inv_w;
        cu_w = cu * c_inv_w; cv_w = cv * c_inv_w;

        // ---- Edge deltas ----
        double dy_ac, dy_ab, dy_bc;
        dy_ac = cy2 - ay;
        dy_ab = by  - ay;
        dy_bc = cy2 - by;

        if (abs(dy_ac) < RC_EPSILON) { return; };

        // Reciprocals for incremental stepping
        double inv_dy_ac, inv_dy_ab, inv_dy_bc;
        inv_dy_ac = 1.0d / dy_ac;
        inv_dy_ab = (abs(dy_ab) > 0.5d) ? (1.0d / dy_ab) : 0.0d;
        inv_dy_bc = (abs(dy_bc) > 0.5d) ? (1.0d / dy_bc) : 0.0d;

        // Per-scanline step rates along long edge (a->c)
        double dxac, dinv_wac, du_wac, dv_wac;
        double dnxac, dnyac, dnzac;
        double dlrac, dlgac, dlbac;
        double dwxac, dwyac, dwzac;
        dxac    = (cx   - ax)     * inv_dy_ac;
        dinv_wac= (c_inv_w - a_inv_w) * inv_dy_ac;
        du_wac  = (cu_w  - au_w)  * inv_dy_ac;
        dv_wac  = (cv_w  - av_w)  * inv_dy_ac;
        dnxac   = (cnx   - anx)   * inv_dy_ac;
        dnyac   = (cny   - any)   * inv_dy_ac;
        dnzac   = (cnz   - anz)   * inv_dy_ac;
        dlrac   = (vc_lr - va_lr) * inv_dy_ac;
        dlgac   = (vc_lg - va_lg) * inv_dy_ac;
        dlbac   = (vc_lb - va_lb) * inv_dy_ac;
        dwxac   = (wx_c  - wx_a)  * inv_dy_ac;
        dwyac   = (wy_c  - wy_a)  * inv_dy_ac;
        dwzac   = (wz_c  - wz_a)  * inv_dy_ac;

        // Per-scanline step rates along upper short edge (a->b)
        double dxab, dinv_wab, du_wab, dv_wab;
        double dnxab, dnyab, dnzab;
        double dlrab, dlgab, dlbab;
        double dwxab, dwyab, dwzab;
        dxab    = (bx   - ax)     * inv_dy_ab;
        dinv_wab= (b_inv_w - a_inv_w) * inv_dy_ab;
        du_wab  = (bu_w  - au_w)  * inv_dy_ab;
        dv_wab  = (bv_w  - av_w)  * inv_dy_ab;
        dnxab   = (bnx   - anx)   * inv_dy_ab;
        dnyab   = (bny   - any)   * inv_dy_ab;
        dnzab   = (bnz   - anz)   * inv_dy_ab;
        dlrab   = (vb_lr - va_lr) * inv_dy_ab;
        dlgab   = (vb_lg - va_lg) * inv_dy_ab;
        dlbab   = (vb_lb - va_lb) * inv_dy_ab;
        dwxab   = (wx_b  - wx_a)  * inv_dy_ab;
        dwyab   = (wy_b  - wy_a)  * inv_dy_ab;
        dwzab   = (wz_b  - wz_a)  * inv_dy_ab;

        // Per-scanline step rates along lower short edge (b->c)
        double dxbc, dinv_wbc, du_wbc, dv_wbc;
        double dnxbc, dnybc, dnzbc;
        double dlrbc, dlgbc, dlbbc;
        double dwxbc, dwybc, dwzbc;
        dxbc    = (cx   - bx)     * inv_dy_bc;
        dinv_wbc= (c_inv_w - b_inv_w) * inv_dy_bc;
        du_wbc  = (cu_w  - bu_w)  * inv_dy_bc;
        dv_wbc  = (cv_w  - bv_w)  * inv_dy_bc;
        dnxbc   = (cnx   - bnx)   * inv_dy_bc;
        dnybc   = (cny   - bny)   * inv_dy_bc;
        dnzbc   = (cnz   - bnz)   * inv_dy_bc;
        dlrbc   = (vc_lr - vb_lr) * inv_dy_bc;
        dlgbc   = (vc_lg - vb_lg) * inv_dy_bc;
        dlbbc   = (vc_lb - vb_lb) * inv_dy_bc;
        dwxbc   = (wx_c  - wx_b)  * inv_dy_bc;
        dwybc   = (wy_c  - wy_b)  * inv_dy_bc;
        dwzbc   = (wz_c  - wz_b)  * inv_dy_bc;

        // Determine if long edge is on right or left
        bool long_edge_right;
        if (abs(dy_ab) > 0.5d)
        {
            long_edge_right = (ax + dxac * dy_ab) > bx;
        }
        else
        {
            long_edge_right = ax > bx;
        };

        // ---- Initialize edge walkers at y_top ----
        double t0;
        t0 = ((double)y_top + 0.5d - ay);

        // Long edge starting values
        double lx, linv_w, lu_w, lv_w, lnx, lny, lnz, llr, llg, llb, lwx, lwy, lwz;
        lx    = ax     + dxac    * t0;
        linv_w= a_inv_w+ dinv_wac* t0;
        lu_w  = au_w   + du_wac  * t0;
        lv_w  = av_w   + dv_wac  * t0;
        lnx   = anx    + dnxac   * t0;
        lny   = any    + dnyac   * t0;
        lnz   = anz    + dnzac   * t0;
        llr   = va_lr  + dlrac   * t0;
        llg   = va_lg  + dlgac   * t0;
        llb   = va_lb  + dlbac   * t0;
        lwx   = wx_a   + dwxac   * t0;
        lwy   = wy_a   + dwyac   * t0;
        lwz   = wz_a   + dwzac   * t0;

        // Short edge starting values (upper half a->b)
        double ts;
        ts = t0;
        double sx, sinv_w, su_w, sv_w, snx, sny, snz, slr, slg, slb, swx, swy, swz;
        sx    = ax     + dxab    * ts;
        sinv_w= a_inv_w+ dinv_wab* ts;
        su_w  = au_w   + du_wab  * ts;
        sv_w  = av_w   + dv_wab  * ts;
        snx   = anx    + dnxab   * ts;
        sny   = any    + dnyab   * ts;
        snz   = anz    + dnzab   * ts;
        slr   = va_lr  + dlrab   * ts;
        slg   = va_lg  + dlgab   * ts;
        slb   = va_lb  + dlbab   * ts;
        swx   = wx_a   + dwxab   * ts;
        swy   = wy_a   + dwyab   * ts;
        swz   = wz_a   + dwzab   * ts;

        i32 y, x, px_start, px_end, row_base;
        double span, inv_span;
        double x_left, x_right, inv_w_left, inv_w_right;
        double u_w_left, u_w_right, v_w_left, v_w_right;
        double nx_left, nx_right, ny_left, ny_right, nz_left, nz_right;
        double lr_left, lr_right, lg_left, lg_right, lb_left, lb_right;
        double wx_left, wx_right, wy_left, wy_right, wz_left, wz_right;
        double px_inv_w, px_z, px_u, px_v;
        double px_lr, px_lg, px_lb;
        u64    px_col;

        // Whether the short edge has transitioned to b->c
        bool in_lower;
        in_lower = (y_top >= y_mid);

        // If starting in lower half, reinitialize short edge to b->c
        if (in_lower)
        {
            double tb;
            tb    = ((double)y_top + 0.5d - by);
            sx    = bx     + dxbc    * tb;
            sinv_w= b_inv_w+ dinv_wbc* tb;
            su_w  = bu_w   + du_wbc  * tb;
            sv_w  = bv_w   + dv_wbc  * tb;
            snx   = bnx    + dnxbc   * tb;
            sny   = bny    + dnybc   * tb;
            snz   = bnz    + dnzbc   * tb;
            slr   = vb_lr  + dlrbc   * tb;
            slg   = vb_lg  + dlgbc   * tb;
            slb   = vb_lb  + dlbbc   * tb;
            swx   = wx_b   + dwxbc   * tb;
            swy   = wy_b   + dwybc   * tb;
            swz   = wz_b   + dwzbc   * tb;
        };

        y = y_top;
        while (y <= y_bot)
        {
            // Transition short edge to lower half when needed
            if (!in_lower & y >= y_mid)
            {
                in_lower = true;
                double tb;
                tb    = ((double)y + 0.5d - by);
                sx    = bx     + dxbc    * tb;
                sinv_w= b_inv_w+ dinv_wbc* tb;
                su_w  = bu_w   + du_wbc  * tb;
                sv_w  = bv_w   + dv_wbc  * tb;
                snx   = bnx    + dnxbc   * tb;
                sny   = bny    + dnybc   * tb;
                snz   = bnz    + dnzbc   * tb;
                slr   = vb_lr  + dlrbc   * tb;
                slg   = vb_lg  + dlgbc   * tb;
                slb   = vb_lb  + dlbbc   * tb;
                swx   = wx_b   + dwxbc   * tb;
                swy   = wy_b   + dwybc   * tb;
                swz   = wz_b   + dwzbc   * tb;
            };

            // Assign left/right from long and short edge walkers
            if (long_edge_right)
            {
                x_left    = sx;     x_right    = lx;
                inv_w_left= sinv_w; inv_w_right= linv_w;
                u_w_left  = su_w;   u_w_right  = lu_w;
                v_w_left  = sv_w;   v_w_right  = lv_w;
                nx_left   = snx;    nx_right   = lnx;
                ny_left   = sny;    ny_right   = lny;
                nz_left   = snz;    nz_right   = lnz;
                lr_left   = slr;    lr_right   = llr;
                lg_left   = slg;    lg_right   = llg;
                lb_left   = slb;    lb_right   = llb;
                wx_left   = swx;    wx_right   = lwx;
                wy_left   = swy;    wy_right   = lwy;
                wz_left   = swz;    wz_right   = lwz;
            }
            else
            {
                x_left    = lx;     x_right    = sx;
                inv_w_left= linv_w; inv_w_right= sinv_w;
                u_w_left  = lu_w;   u_w_right  = su_w;
                v_w_left  = lv_w;   v_w_right  = sv_w;
                nx_left   = lnx;    nx_right   = snx;
                ny_left   = lny;    ny_right   = sny;
                nz_left   = lnz;    nz_right   = snz;
                lr_left   = llr;    lr_right   = slr;
                lg_left   = llg;    lg_right   = slg;
                lb_left   = llb;    lb_right   = slb;
                wx_left   = lwx;    wx_right   = swx;
                wy_left   = lwy;    wy_right   = swy;
                wz_left   = lwz;    wz_right   = swz;
            };

            px_start = (i32)(x_left  + 0.5d);
            px_end   = (i32)(x_right - 0.5d);
            if (px_start < 0)   { px_start = 0; };
            if (px_end   >= sw) { px_end   = sw - 1; };

            span = x_right - x_left;
            if (span < 0.5d) { y++; lx+=dxac; linv_w+=dinv_wac; lu_w+=du_wac; lv_w+=dv_wac; lnx+=dnxac; lny+=dnyac; lnz+=dnzac; llr+=dlrac; llg+=dlgac; llb+=dlbac; lwx+=dwxac; lwy+=dwyac; lwz+=dwzac; if (in_lower) { sx+=dxbc; sinv_w+=dinv_wbc; su_w+=du_wbc; sv_w+=dv_wbc; snx+=dnxbc; sny+=dnybc; snz+=dnzbc; slr+=dlrbc; slg+=dlgbc; slb+=dlbbc; swx+=dwxbc; swy+=dwybc; swz+=dwzbc; } else { sx+=dxab; sinv_w+=dinv_wab; su_w+=du_wab; sv_w+=dv_wab; snx+=dnxab; sny+=dnyab; snz+=dnzab; slr+=dlrab; slg+=dlgab; slb+=dlbab; swx+=dwxab; swy+=dwyab; swz+=dwzab; }; continue; };

            inv_span = 1.0d / span;

            // Precompute per-pixel step rates across the scanline
            double d_inv_w, d_u_w, d_v_w, d_nx, d_ny, d_nz;
            double d_lr, d_lg, d_lb, d_wx, d_wy, d_wz;
            d_inv_w = (inv_w_right - inv_w_left) * inv_span;
            d_u_w   = (u_w_right   - u_w_left)   * inv_span;
            d_v_w   = (v_w_right   - v_w_left)   * inv_span;
            d_nx    = (nx_right    - nx_left)     * inv_span;
            d_ny    = (ny_right    - ny_left)     * inv_span;
            d_nz    = (nz_right    - nz_left)     * inv_span;
            d_lr    = (lr_right    - lr_left)     * inv_span;
            d_lg    = (lg_right    - lg_left)     * inv_span;
            d_lb    = (lb_right    - lb_left)     * inv_span;
            d_wx    = (wx_right    - wx_left)     * inv_span;
            d_wy    = (wy_right    - wy_left)     * inv_span;
            d_wz    = (wz_right    - wz_left)     * inv_span;

            // Initialize pixel walker at px_start center
            double t0x;
            t0x = ((double)px_start + 0.5d - x_left) * inv_span;
            if (t0x < 0.0d) { t0x = 0.0d; };
            if (t0x > 1.0d) { t0x = 1.0d; };

            double cur_inv_w, cur_u_w, cur_v_w;
            double cur_nx, cur_ny, cur_nz;
            double cur_lr, cur_lg, cur_lb;
            double cur_wx, cur_wy, cur_wz;
            cur_inv_w = inv_w_left + d_inv_w * ((double)px_start + 0.5d - x_left);
            cur_u_w   = u_w_left   + d_u_w   * ((double)px_start + 0.5d - x_left);
            cur_v_w   = v_w_left   + d_v_w   * ((double)px_start + 0.5d - x_left);
            cur_nx    = nx_left    + d_nx    * ((double)px_start + 0.5d - x_left);
            cur_ny    = ny_left    + d_ny    * ((double)px_start + 0.5d - x_left);
            cur_nz    = nz_left    + d_nz    * ((double)px_start + 0.5d - x_left);
            cur_lr    = lr_left    + d_lr    * ((double)px_start + 0.5d - x_left);
            cur_lg    = lg_left    + d_lg    * ((double)px_start + 0.5d - x_left);
            cur_lb    = lb_left    + d_lb    * ((double)px_start + 0.5d - x_left);
            cur_wx    = wx_left    + d_wx    * ((double)px_start + 0.5d - x_left);
            cur_wy    = wy_left    + d_wy    * ((double)px_start + 0.5d - x_left);
            cur_wz    = wz_left    + d_wz    * ((double)px_start + 0.5d - x_left);

            // #3: Cache row base index to avoid per-pixel multiply
            row_base = y * sw;

            x = px_start;
            while (x <= px_end)
            {
                px_inv_w = cur_inv_w;
                if (px_inv_w < RC_EPSILON) { x++; cur_inv_w+=d_inv_w; cur_u_w+=d_u_w; cur_v_w+=d_v_w; cur_nx+=d_nx; cur_ny+=d_ny; cur_nz+=d_nz; cur_lr+=d_lr; cur_lg+=d_lg; cur_lb+=d_lb; cur_wx+=d_wx; cur_wy+=d_wy; cur_wz+=d_wz; continue; };

                px_z = 1.0d / px_inv_w;

                // Depth test — use cached row_base
                if (px_z >= zbuf[row_base + x]) { x++; cur_inv_w+=d_inv_w; cur_u_w+=d_u_w; cur_v_w+=d_v_w; cur_nx+=d_nx; cur_ny+=d_ny; cur_nz+=d_nz; cur_lr+=d_lr; cur_lg+=d_lg; cur_lb+=d_lb; cur_wx+=d_wx; cur_wy+=d_wy; cur_wz+=d_wz; continue; };

                // Recover perspective-correct UVs
                px_u = cur_u_w * px_z;
                px_v = cur_v_w * px_z;
                if (px_u < 0.0d) { px_u = 0.0d; } elif (px_u > 1.0d) { px_u = 1.0d; };
                if (px_v < 0.0d) { px_v = 0.0d; } elif (px_v > 1.0d) { px_v = 1.0d; };

                // Sample texture
                if (tex != (RCTexture*)0)
                {
                    px_col = rc_tex_sample_bilinear(tex, px_u, px_v);
                    if ((px_col & (u64)0x0000FFFFFFFFFFFF) == (u64)0x0000FFFF0000FFFF)
                    {
                        x++; cur_inv_w+=d_inv_w; cur_u_w+=d_u_w; cur_v_w+=d_v_w; cur_nx+=d_nx; cur_ny+=d_ny; cur_nz+=d_nz; cur_lr+=d_lr; cur_lg+=d_lg; cur_lb+=d_lb; cur_wx+=d_wx; cur_wy+=d_wy; cur_wz+=d_wz;
                        continue;
                    };
                }
                else
                {
                    px_col = (u64)0xFFFFFFFFFFFFFFFF;
                };

                if (tint != 0) { px_col = color64_tint(px_col, tint); };

                // #4: Apply lighting
                if (shade_model == R3D_SHADE_FLAT)
                {
                    // Flat: pre-baked color, no per-pixel unpack/pack
                    px_col = color64_light_bake(px_col, flat_lr, flat_lg, flat_lb);
                }
                elif (shade_model == R3D_SHADE_GOURAUD)
                {
                    // Gouraud: interpolated light RGB, incremental
                    px_lr = cur_lr;
                    px_lg = cur_lg;
                    px_lb = cur_lb;
                    px_col = color64_light(px_col, px_lr, px_lg, px_lb);
                };

                buf[row_base + x]  = px_col;
                zbuf[row_base + x] = px_z;

                x++;
                cur_inv_w+=d_inv_w; cur_u_w+=d_u_w; cur_v_w+=d_v_w;
                cur_nx+=d_nx; cur_ny+=d_ny; cur_nz+=d_nz;
                cur_lr+=d_lr; cur_lg+=d_lg; cur_lb+=d_lb;
                cur_wx+=d_wx; cur_wy+=d_wy; cur_wz+=d_wz;
            };

            // Step edge walkers by one scanline
            lx+=dxac; linv_w+=dinv_wac; lu_w+=du_wac; lv_w+=dv_wac;
            lnx+=dnxac; lny+=dnyac; lnz+=dnzac;
            llr+=dlrac; llg+=dlgac; llb+=dlbac;
            lwx+=dwxac; lwy+=dwyac; lwz+=dwzac;

            if (in_lower)
            {
                sx+=dxbc; sinv_w+=dinv_wbc; su_w+=du_wbc; sv_w+=dv_wbc;
                snx+=dnxbc; sny+=dnybc; snz+=dnzbc;
                slr+=dlrbc; slg+=dlgbc; slb+=dlbbc;
                swx+=dwxbc; swy+=dwybc; swz+=dwzbc;
            }
            else
            {
                sx+=dxab; sinv_w+=dinv_wab; su_w+=du_wab; sv_w+=dv_wab;
                snx+=dnxab; sny+=dnyab; snz+=dnzab;
                slr+=dlrab; slg+=dlgab; slb+=dlbab;
                swx+=dwxab; swy+=dwyab; swz+=dwzab;
            };

            y++;
        };
    };

    // =========================================================================
    // 3D MESH RENDER PASS
    // =========================================================================

    def r3d_draw_mesh_inst(R3DMeshInst*       inst,
                           R3DCamera*         cam,
                           R3DLight*          lights,
                           i32                light_count,
                           double             amb_r, double amb_g, double amb_b,
                           RCTexturePalette*  palette,
                           double*            zbuf,
                           u64*               buf) -> void
    {
        i32       t, v;
        R3DMesh*  mesh;
        DMat4     model, normal_mat;
        RCTexture* tex;

        mesh = inst.mesh;
        if (mesh == (R3DMesh*)0) { return; };

        // Build model matrix
        model = dmat4_trs(inst.pos_x, inst.pos_y, inst.pos_z,
                          inst.rot_x, inst.rot_y, inst.rot_z,
                          inst.scale_x, inst.scale_y, inst.scale_z);

        // Normal matrix: transpose of inverse of model upper-left 3x3
        dmat4_normal_mat(model, @normal_mat);

        // #6: Detect uniform scale — if so, normals are already unit after transform
        // and renormalization can be skipped per-vertex.
        double sx_diff, sy_diff;
        bool uniform_scale;
        sx_diff = inst.scale_x - inst.scale_y;
        sy_diff = inst.scale_y - inst.scale_z;
        if (sx_diff < 0.0d) { sx_diff = -sx_diff; };
        if (sy_diff < 0.0d) { sy_diff = -sy_diff; };
        uniform_scale = (sx_diff < RC_EPSILON & sy_diff < RC_EPSILON);

        // Resolve texture
        tex = (RCTexture*)0;
        if (palette != (RCTexturePalette*)0 & mesh.tex_idx > 0 &
            mesh.tex_idx <= palette.count)
        {
            tex = @palette.slots[mesh.tex_idx - 1];
        };

        // #5: Pre-transform all vertices into world space and clip space once.
        // Allocates on the stack — mesh.vert_count must be bounded.
        // world positions (x,y,z), world normals (nx,ny,nz), clip (x,y,z,w)
        double* wpos_x;  double* wpos_y;  double* wpos_z;
        double* wnrm_x;  double* wnrm_y;  double* wnrm_z;
        double* cpos_x;  double* cpos_y;  double* cpos_z;  double* cpos_w;

        i32 vc;
        vc = mesh.vert_count;
        wpos_x = (double*)fmalloc((u64)(vc * 8));
        wpos_y = (double*)fmalloc((u64)(vc * 8));
        wpos_z = (double*)fmalloc((u64)(vc * 8));
        wnrm_x = (double*)fmalloc((u64)(vc * 8));
        wnrm_y = (double*)fmalloc((u64)(vc * 8));
        wnrm_z = (double*)fmalloc((u64)(vc * 8));
        cpos_x = (double*)fmalloc((u64)(vc * 8));
        cpos_y = (double*)fmalloc((u64)(vc * 8));
        cpos_z = (double*)fmalloc((u64)(vc * 8));
        cpos_w = (double*)fmalloc((u64)(vc * 8));

        v = 0;
        while (v < vc)
        {
            R3DVertex* vt;
            vt = @mesh.verts[v];

            DVec4 p, w, n, wn, c;
            p.x = vt.x; p.y = vt.y; p.z = vt.z; p.w = 1.0d;
            w  = dmat4_mul_vec4(model, p);
            wpos_x[v] = w.x; wpos_y[v] = w.y; wpos_z[v] = w.z;

            n.x = vt.nx; n.y = vt.ny; n.z = vt.nz; n.w = 0.0d;
            wn = dmat4_mul_vec4(normal_mat, n);
            if (!uniform_scale)
            {
                double len;
                len = sqrt(wn.x*wn.x + wn.y*wn.y + wn.z*wn.z);
                if (len > RC_EPSILON) { wn.x /= len; wn.y /= len; wn.z /= len; };
            };
            wnrm_x[v] = wn.x; wnrm_y[v] = wn.y; wnrm_z[v] = wn.z;

            c = dmat4_mul_vec4(cam.vp, w);
            cpos_x[v] = c.x; cpos_y[v] = c.y; cpos_z[v] = c.z; cpos_w[v] = c.w;

            v++;
        };

        // Precompute screen half-dimensions outside triangle loop
        double hw, hh;
        hw = (double)cam.screen_w * 0.5d;
        hh = (double)cam.screen_h * 0.5d;

        t = 0;
        while (t < mesh.tri_count)
        {
            i32 ia, ib, ic;
            ia = mesh.tris[t].a;
            ib = mesh.tris[t].b;
            ic = mesh.tris[t].c;

            R3DVertex* va;
            R3DVertex* vb;
            R3DVertex* vc2;
            va  = @mesh.verts[ia];
            vb  = @mesh.verts[ib];
            vc2 = @mesh.verts[ic];

            // Load pre-transformed data
            DVec4 wa, wb, wc, ca, cb, cc;
            wa.x = wpos_x[ia]; wa.y = wpos_y[ia]; wa.z = wpos_z[ia]; wa.w = 1.0d;
            wb.x = wpos_x[ib]; wb.y = wpos_y[ib]; wb.z = wpos_z[ib]; wb.w = 1.0d;
            wc.x = wpos_x[ic]; wc.y = wpos_y[ic]; wc.z = wpos_z[ic]; wc.w = 1.0d;
            ca.x = cpos_x[ia]; ca.y = cpos_y[ia]; ca.z = cpos_z[ia]; ca.w = cpos_w[ia];
            cb.x = cpos_x[ib]; cb.y = cpos_y[ib]; cb.z = cpos_z[ib]; cb.w = cpos_w[ib];
            cc.x = cpos_x[ic]; cc.y = cpos_y[ic]; cc.z = cpos_z[ic]; cc.w = cpos_w[ic];

            DVec4 wna, wnb, wnc;
            wna.x = wnrm_x[ia]; wna.y = wnrm_y[ia]; wna.z = wnrm_z[ia]; wna.w = 0.0d;
            wnb.x = wnrm_x[ib]; wnb.y = wnrm_y[ib]; wnb.z = wnrm_z[ib]; wnb.w = 0.0d;
            wnc.x = wnrm_x[ic]; wnc.y = wnrm_y[ic]; wnc.z = wnrm_z[ic]; wnc.w = 0.0d;

            // ---- Backface cull in world space ----
            // Face normal (from pre-transformed world normals, averaged).
            // If the normal faces away from the camera, skip the triangle.
            {
                double fnx, fny, fnz, vdx, vdy, vdz, ndotv;
                fnx = wna.x + wnb.x + wnc.x;
                fny = wna.y + wnb.y + wnc.y;
                fnz = wna.z + wnb.z + wnc.z;
                vdx = wa.x - cam.eye_x;
                vdy = wa.y - cam.eye_y;
                vdz = wa.z - cam.eye_z;
                ndotv = fnx*vdx + fny*vdy + fnz*vdz;
                if (ndotv >= 0.0d) { t++; continue; };
            };

            // ---- Full frustum clip (all 6 planes) ----
            R3DClipVert[3] clip_in;
            R3DClipVert[R3D_MAX_CLIP_VERTS] clip_out;
            i32 clip_count;

            clip_in[0].x = ca.x; clip_in[0].y = ca.y;
            clip_in[0].z = ca.z; clip_in[0].w = ca.w;
            clip_in[0].nx = wna.x; clip_in[0].ny = wna.y; clip_in[0].nz = wna.z;
            clip_in[0].u = va.u; clip_in[0].v = va.v;
            clip_in[0].inv_w = (ca.w > cam.near_z) ? (1.0d / ca.w) : (1.0d / cam.near_z);

            clip_in[1].x = cb.x; clip_in[1].y = cb.y;
            clip_in[1].z = cb.z; clip_in[1].w = cb.w;
            clip_in[1].nx = wnb.x; clip_in[1].ny = wnb.y; clip_in[1].nz = wnb.z;
            clip_in[1].u = vb.u; clip_in[1].v = vb.v;
            clip_in[1].inv_w = (cb.w > cam.near_z) ? (1.0d / cb.w) : (1.0d / cam.near_z);

            clip_in[2].x = cc.x; clip_in[2].y = cc.y;
            clip_in[2].z = cc.z; clip_in[2].w = cc.w;
            clip_in[2].nx = wnc.x; clip_in[2].ny = wnc.y; clip_in[2].nz = wnc.z;
            clip_in[2].u = vc2.u; clip_in[2].v = vc2.v;
            clip_in[2].inv_w = (cc.w > cam.near_z) ? (1.0d / cc.w) : (1.0d / cam.near_z);

            clip_count = r3d_clip_frustum(@clip_in[0], 3, cam.near_z, @clip_out[0]);

            if (clip_count < 3) { t++; continue; };

            // Fan-triangulate and rasterize
            i32 fan;
            fan = 1;
            while (fan < clip_count - 1)
            {
                R3DClipVert* v0;
                R3DClipVert* v1;
                R3DClipVert* v2;
                v0 = @clip_out[0];
                v1 = @clip_out[fan];
                v2 = @clip_out[fan + 1];

                double inv_w0, inv_w1, inv_w2;
                inv_w0 = (v0.w > cam.near_z) ? (1.0d / v0.w) : (1.0d / cam.near_z);
                inv_w1 = (v1.w > cam.near_z) ? (1.0d / v1.w) : (1.0d / cam.near_z);
                inv_w2 = (v2.w > cam.near_z) ? (1.0d / v2.w) : (1.0d / cam.near_z);

                double sx0, sy0, sz0, sx1, sy1, sz1, sx2, sy2, sz2;
                sx0 = ( v0.x * inv_w0 + 1.0d) * hw;
                sy0 = (-v0.y * inv_w0 + 1.0d) * hh;
                sz0 = v0.z * inv_w0;
                sx1 = ( v1.x * inv_w1 + 1.0d) * hw;
                sy1 = (-v1.y * inv_w1 + 1.0d) * hh;
                sz1 = v1.z * inv_w1;
                sx2 = ( v2.x * inv_w2 + 1.0d) * hw;
                sy2 = (-v2.y * inv_w2 + 1.0d) * hh;
                sz2 = v2.z * inv_w2;

                r3d_draw_triangle(
                    cam.screen_w, cam.screen_h,
                    sx0, sy0, sz0, inv_w0, v0.u, v0.v,
                    v0.nx, v0.ny, v0.nz,
                    sx1, sy1, sz1, inv_w1, v1.u, v1.v,
                    v1.nx, v1.ny, v1.nz,
                    sx2, sy2, sz2, inv_w2, v2.u, v2.v,
                    v2.nx, v2.ny, v2.nz,
                    tex,
                    inst.tint,
                    inst.shade_model,
                    lights,
                    light_count,
                    amb_r, amb_g, amb_b,
                    wa.x, wa.y, wa.z,
                    wb.x, wb.y, wb.z,
                    wc.x, wc.y, wc.z,
                    zbuf,
                    buf
                );

                fan++;
            };

            t++;
        };

        // Free pre-transformed vertex arrays
        ffree((u64)wpos_x); ffree((u64)wpos_y); ffree((u64)wpos_z);
        ffree((u64)wnrm_x); ffree((u64)wnrm_y); ffree((u64)wnrm_z);
        ffree((u64)cpos_x); ffree((u64)cpos_y); ffree((u64)cpos_z); ffree((u64)cpos_w);
    };

    // =========================================================================
    // 3D BILLBOARD SPRITE PASS
    // =========================================================================

    def r3d_sprite_sort(R3DSprite* sprites, i32 count, R3DCamera* cam) -> void
    {
        i32       i, j;
        R3DSprite tmp;
        double    dx, dy, dz;

        // Compute squared distances
        i = 0;
        while (i < count)
        {
            dx = sprites[i].world_x - cam.eye_x;
            dy = sprites[i].world_y - cam.eye_y;
            dz = sprites[i].world_z - cam.eye_z;
            sprites[i].dist_sq = dx*dx + dy*dy + dz*dz;
            i++;
        };

        // Insertion sort descending
        i = 1;
        while (i < count)
        {
            tmp = sprites[i];
            j   = i - 1;
            while (j >= 0 & sprites[j].dist_sq < tmp.dist_sq)
            {
                sprites[j + 1] = sprites[j];
                j--;
            };
            sprites[j + 1] = tmp;
            i++;
        };
    };

    def r3d_draw_sprites(R3DCamera*         cam,
                         R3DSprite*         sprites,
                         i32                sprite_count,
                         RCTexturePalette*  palette,
                         double*            zbuf,
                         u64*               buf) -> void
    {
        i32    s;
        double hw, hh;

        hw = (double)cam.screen_w * 0.5;
        hh = (double)cam.screen_h * 0.5;

        r3d_sprite_sort(sprites, sprite_count, cam);

        s = 0;
        while (s < sprite_count)
        {
            R3DSprite* sp;
            sp = @sprites[s];

            // Sprite anchor in world space (vertical offset applied on Y)
            double cx2, cy2, cz2;
            cx2 = sp.world_x;
            cy2 = sp.world_y + sp.vert_offset;
            cz2 = sp.world_z;

            // Transform center to clip space
            DVec4 wpos, cpos;
            wpos.x = cx2; wpos.y = cy2; wpos.z = cz2; wpos.w = 1.0;
            cpos = dmat4_mul_vec4(cam.vp, wpos);

            if (cpos.w < cam.near_z) { s++; continue; };

            double inv_w_c, scx, scy;
            inv_w_c = 1.0 / cpos.w;
            scx = ( cpos.x * inv_w_c + 1.0) * hw;
            scy = (-cpos.y * inv_w_c + 1.0) * hh;

            // Project half-extents: use right axis for width, up axis for height
            double half_w, half_h_ext;
            half_w     = sp.width  * 0.5;
            half_h_ext = sp.height * 0.5;

            // Project corners via VP (use camera right/up in view space to build quad)
            DVec4 p_tl, p_tr, p_bl, p_br;
            double rw, rh;

            // Right and up contribution to NDC span at this depth
            rw = half_w * cam.proj_dist * inv_w_c;
            rh = half_h_ext * cam.proj_dist * inv_w_c;

            i32 scx_left, scx_right, scy_top, scy_bot;
            scx_left  = (i32)(scx - rw);
            scx_right = (i32)(scx + rw);
            scy_top   = (i32)(scy - rh);
            scy_bot   = (i32)(scy + rh);

            if (scx_right <= 0 | scx_left >= cam.screen_w)  { s++; continue; };
            if (scy_bot   <= 0 | scy_top  >= cam.screen_h)  { s++; continue; };

            if (scx_left  < 0)             { scx_left  = 0; };
            if (scx_right >= cam.screen_w) { scx_right = cam.screen_w - 1; };
            if (scy_top   < 0)             { scy_top   = 0; };
            if (scy_bot   >= cam.screen_h) { scy_bot   = cam.screen_h - 1; };

            double view_z;
            view_z = 1.0 / inv_w_c;

            RCTexture* tex;
            tex = (RCTexture*)0;
            if (palette != (RCTexturePalette*)0 & sp.tex_idx > 0 &
                sp.tex_idx <= palette.count)
            {
                tex = @palette.slots[sp.tex_idx - 1];
            };

            i32    px, py;
            double tex_u, tex_v;
            u64    px_col;

            py = scy_top;
            while (py <= scy_bot)
            {
                px = scx_left;
                while (px <= scx_right)
                {
                    if (view_z >= zbuf[py * cam.screen_w + px]) { px++; continue; };

                    tex_u = (double)(px - scx_left) / (double)(scx_right - scx_left + 1);
                    tex_v = (double)(py - scy_top)  / (double)(scy_bot   - scy_top  + 1);

                    if (tex != (RCTexture*)0)
                    {
                        px_col = rc_tex_sample_bilinear(tex, tex_u, tex_v);
                        if ((px_col & (u64)0x0000FFFFFFFFFFFF) == (u64)0x0000FFFF0000FFFF)
                        {
                            px++;
                            continue;
                        };
                    }
                    else
                    {
                        px_col = sp.tint != 0 ? sp.tint : (u64)0xFFFFFFFFFFFFFFFF;
                    };

                    if (sp.tint != 0 & tex != (RCTexture*)0)
                    {
                        px_col = color64_tint(px_col, sp.tint);
                    };

                    buf[py * cam.screen_w + px]  = px_col;
                    zbuf[py * cam.screen_w + px] = view_z;

                    px++;
                };
                py++;
            };

            s++;
        };
    };

    // =========================================================================
    // 3D SKY PASS (pitch-aware cylindrical gradient)
    // =========================================================================

    def r3d_draw_sky(RCSky* sky, R3DCamera* cam, R3DPlayer* p, u64* buf) -> void
    {
        i32    row, col;
        double t, horizon_y, pitch_offset, norm_y;
        u64    sky_col;

        // Horizon is shifted up/down by pitch
        // pitch = 0 => horizon at screen center
        // positive pitch (look up) => horizon shifts down
        pitch_offset = p.pitch / (cam.fov_v * 0.5);
        horizon_y    = (double)cam.screen_h * (0.5 - pitch_offset * 0.5);

        row = 0;
        while (row < cam.screen_h)
        {
            if ((double)row >= horizon_y) { row++; continue; };

            // Normalise [0, horizon_y) -> [0, 1]
            norm_y  = (double)row / horizon_y;
            sky_col = color64_lerp(sky.color_top, sky.color_horizon, norm_y);

            col = 0;
            while (col < cam.screen_w)
            {
                buf[row * cam.screen_w + col] = sky_col;
                col++;
            };
            row++;
        };
    };

    // =========================================================================
    // 3D RENDER ENTRY POINT
    //
    // zbuf: caller-allocated double[screen_w * screen_h].
    // buf:  caller-allocated u64[screen_w * screen_h].
    // Both buffers are cleared internally at the start of each call.
    // r3d_camera_sync() must be called before this.
    // =========================================================================

    def r3d_render(R3DScene* scene,
                   u64*      buf,
                   double*   zbuf) -> void
    {
        i32 i, total_px;

        // Clear both buffers at the start of every frame.
        // buf is zeroed via mem_fill (byte 0 = u64 0).
        // zbuf requires a double fill loop since RC_INF is not zero.
        total_px = scene.cam.screen_w * scene.cam.screen_h;
        mem_fill((void*)buf, (byte)0, (size_t)((u64)total_px * 8));
        i = 0;
        while (i < total_px)
        {
            zbuf[i] = RC_INF;
            i++;
        };

        if (scene.sky != (RCSky*)0)
        {
            r3d_draw_sky(scene.sky, scene.cam, scene.player, buf);
        };

        if (scene.passes & R3D_PASS_MESHES)
        {
            i = 0;
            while (i < scene.inst_count)
            {
                if (scene.insts[i] != (R3DMeshInst*)0)
                {
                    r3d_draw_mesh_inst(
                        scene.insts[i],
                        scene.cam,
                        scene.lights,
                        scene.light_count,
                        scene.ambient_r,
                        scene.ambient_g,
                        scene.ambient_b,
                        scene.palette,
                        zbuf,
                        buf
                    );
                };
                i++;
            };
        };

        if ((scene.passes & R3D_PASS_SPRITES) &
            scene.sprites != (R3DSprite*)0 &
            scene.sprite_count > 0)
        {
            r3d_draw_sprites(scene.cam,
                             scene.sprites, scene.sprite_count,
                             scene.palette, zbuf, buf);
        };
    };

    // =========================================================================
    // CONVENIENCE SCENE BUILDERS
    // =========================================================================

    def r3d_scene_init(R3DScene*         scene,
                       R3DCamera*        cam,
                       R3DPlayer*        player,
                       RCTexturePalette* palette,
                       RCSky*            sky) -> void
    {
        scene.cam           = cam;
        scene.player        = player;
        scene.insts         = (R3DMeshInst**)0;
        scene.inst_count    = 0;
        scene.sprites       = (R3DSprite*)0;
        scene.sprite_count  = 0;
        scene.lights        = (R3DLight*)0;
        scene.light_count   = 0;
        scene.ambient_r     = 0.1;
        scene.ambient_g     = 0.1;
        scene.ambient_b     = 0.1;
        scene.palette       = palette;
        scene.sky           = sky;
        scene.passes        = R3D_PASS_ALL;
    };

    def r3d_scene_set_insts(R3DScene* scene,
                            R3DMeshInst** insts, i32 count) -> void
    {
        scene.insts      = insts;
        scene.inst_count = count;
    };

    def r3d_scene_set_sprites(R3DScene* scene,
                               R3DSprite* sprites, i32 count) -> void
    {
        scene.sprites      = sprites;
        scene.sprite_count = count;
    };

    def r3d_scene_set_lights(R3DScene* scene,
                              R3DLight* lights, i32 count) -> void
    {
        scene.lights      = lights;
        scene.light_count = count;
    };

    def r3d_scene_set_ambient(R3DScene* scene,
                               double r, double g, double b) -> void
    {
        scene.ambient_r = r;
        scene.ambient_g = g;
        scene.ambient_b = b;
    };

    // =========================================================================
    // COMPOSITE FRAME HELPER
    //
    // Renders a 2.5D scene and a 3D scene into the same buffer in one call.
    // zbuf is allocated and freed internally.
    //
    // Render order:
    //   1. Sky (3D pitch-aware if r3d_scene is non-null, else 2.5D flat)
    //   2. 2.5D floor/ceiling
    //   3. 2.5D walls
    //   4. 2.5D sprites
    //   5. 3D meshes
    //   6. 3D billboard sprites
    //
    // Either scene pointer may be null to skip that pass.
    // =========================================================================

    def rc_render_composite(RCScene*  rc_scene,
                             R3DScene* r3d_scene,
                             u64*      buf) -> void
    {
        i32    sw, sh, total_px;
        double* zbuf;
        size_t  zbytes;
        i32     i;

        if (rc_scene != (RCScene*)0)
        {
            sw = rc_scene.cam.screen_w;
            sh = rc_scene.cam.screen_h;
        }
        elif (r3d_scene != (R3DScene*)0)
        {
            sw = r3d_scene.cam.screen_w;
            sh = r3d_scene.cam.screen_h;
        }
        else
        {
            return;
        };

        total_px = sw * sh;
        zbytes   = (size_t)(total_px * (i32)(sizeof(double) / 8));
        zbuf     = (double*)fmalloc(zbytes);

        // Fill zbuf with effective infinity
        i = 0;
        while (i < total_px)
        {
            zbuf[i] = RC_INF;
            i++;
        };

        if (rc_scene != (RCScene*)0)
        {
            rc_render(rc_scene, buf, zbuf);
        };

        if (r3d_scene != (R3DScene*)0)
        {
            r3d_render(r3d_scene, buf, zbuf);
        };

        ffree((u64)zbuf);
    };

};  // namespace raycaster

#endif;
