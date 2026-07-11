#!/usr/bin/env python3
"""
Flux Type Geometry Constraint Visualizer (pygame renderer)
Antialiased lines, circles, and text via pygame + pygame.gfxdraw.

Controls:
  LMB drag    rotate
  Scroll      zoom
  RMB         select node
  R           reset camera
  Enter       visualize expression
  Escape      clear input
"""

import pygame
import pygame.gfxdraw
import math
import colorsys
import sys
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

OP_INFO = {
    "~=":    ("compatible",          (76,  175, 80),  False),
    "!~=":   ("incompatible",        (244, 67,  54),  False),
    "!@":    ("no address-of",       (255, 152, 0),   True),
    "!`<":   ("no narrowing",        (156, 39,  176), True),
    "!`<=":  ("no narrowing (pair)", (123, 31,  162), False),
    "!`>":   ("no widening",         (33,  150, 243), True),
    "!`>=":  ("no widening (pair)",  (21,  101, 192), False),
    "!-=":   ("no signed ops",       (96,  125, 139), False),
}

@dataclass
class Relation:
    lhs: list
    op: str
    rhs: str
    inferred: bool = False
    bracket_group: bool = False

@dataclass
class TypeNode:
    name: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    sx: float = 0.0
    sy: float = 0.0
    sz: float = 0.0

# ---------------------------------------------------------------------------
# Parser (unchanged from tkinter version)
# ---------------------------------------------------------------------------

OPERATORS = ["!`<=", "!`>=", "!~=", "!-=", "!`<", "!`>", "~=", "!@"]

def tokenize(expr: str):
    expr = expr.strip()
    tokens = []
    i = 0
    while i < len(expr):
        if expr[i].isspace():
            i += 1
            continue
        if expr[i] == '[':
            tokens.append(('[', 'LBRACKET'))
            i += 1
            continue
        if expr[i] == ']':
            tokens.append((']', 'RBRACKET'))
            i += 1
            continue
        if expr[i] == '&':
            tokens.append(('&', 'AMP'))
            i += 1
            continue
        matched = False
        for op in OPERATORS:
            if expr[i:i+len(op)] == op:
                tokens.append((op, 'OP'))
                i += len(op)
                matched = True
                break
        if matched:
            continue
        if expr[i].isalpha() or expr[i] == '_':
            j = i
            while j < len(expr) and (expr[j].isalnum() or expr[j] == '_'):
                j += 1
            tokens.append((expr[i:j], 'IDENT'))
            i = j
            continue
        i += 1
    return tokens

def parse_constraint_expr(expr: str):
    tokens = tokenize(expr)
    relations = []
    type_names = set()
    idx = 0

    def peek(offset=0):
        i = idx + offset
        return tokens[i] if i < len(tokens) else None

    def is_op():
        t = peek()
        return t is not None and t[1] == 'OP'

    def consume_op():
        nonlocal idx
        op = tokens[idx][0]
        idx += 1
        return op

    def parse_id_list():
        nonlocal idx
        names = []
        t = peek()
        if t is None or t[1] != 'IDENT':
            return names
        names.append(t[0])
        type_names.add(t[0])
        idx += 1
        while peek() and peek()[1] == 'AMP':
            nxt = tokens[idx + 1] if idx + 1 < len(tokens) else None
            if nxt and nxt[1] == 'IDENT':
                idx += 1
                names.append(nxt[0])
                type_names.add(nxt[0])
                idx += 1
            else:
                break
        return names

    def parse_bracket_group():
        nonlocal idx
        idx += 1
        sub_lhs = parse_id_list()
        sub_bracket_names = list(sub_lhs)
        while is_op():
            op = consume_op()
            in_bracket_inner = peek() and peek()[1] == 'LBRACKET'
            if in_bracket_inner:
                sub_rhs = parse_bracket_group()
            else:
                sub_rhs = parse_id_list()
            sub_bracket_names += [n for n in sub_rhs if n not in sub_bracket_names]
            ind = OP_INFO.get(op, ("", (255,255,255), False))[2]
            if ind:
                for n in sub_lhs:
                    relations.append(Relation([n], op, n, bracket_group=True))
                for n in sub_rhs:
                    relations.append(Relation([n], op, n, bracket_group=True))
            else:
                for ln in sub_lhs:
                    for rn in sub_rhs:
                        relations.append(Relation([ln], op, rn, bracket_group=True))
            sub_lhs = sub_rhs
        if peek() and peek()[1] == 'RBRACKET':
            idx += 1
        return sub_bracket_names

    def parse_segment():
        nonlocal idx
        if peek() and peek()[1] == 'LBRACKET':
            names = parse_bracket_group()
            is_brk = True
        else:
            names = parse_id_list()
            is_brk = False
        # keep consuming & IDENT or & [...] entries until neither matches
        while peek() and peek()[1] == 'AMP':
            nxt = tokens[idx + 1] if idx + 1 < len(tokens) else None
            if nxt and nxt[1] == 'LBRACKET':
                idx += 1  # consume &
                brk_names = parse_bracket_group()
                names += [n for n in brk_names if n not in names]
                is_brk = True
            elif nxt and nxt[1] == 'IDENT':
                idx += 1  # consume &
                names += parse_id_list()
            else:
                break
        return names, is_brk

    lhs, lhs_bracket = parse_segment()
    if not lhs:
        return relations, type_names

    while is_op():
        op = consume_op()
        rhs, rhs_bracket = parse_segment()
        if not rhs:
            break
        ind = OP_INFO.get(op, ("", (255,255,255), False))[2]
        if ind:
            for n in lhs:
                relations.append(Relation([n], op, n, bracket_group=lhs_bracket))
            for n in rhs:
                relations.append(Relation([n], op, n, bracket_group=rhs_bracket))
        else:
            for ln in lhs:
                for rn in rhs:
                    relations.append(Relation([ln], op, rn,
                                              bracket_group=lhs_bracket or rhs_bracket))
        lhs = rhs
        lhs_bracket = rhs_bracket

    return relations, type_names

def infer_relations(relations, type_names):
    incompat = set()
    compat = set()
    # collect all explicit (non-inferred) op pairs keyed by canonical pair
    explicit = {}   # canonical pair -> set of ops
    for r in relations:
        if r.inferred:
            continue
        key = tuple(sorted([r.lhs[0], r.rhs]))
        explicit.setdefault(key, set()).add(r.op)
        if r.op == "!~=":
            incompat.add(key)
        elif r.op == "~=":
            compat.add(key)

    inferred = []
    names = list(type_names)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            for c in names:
                if c == a or c == b:
                    continue
                ab = tuple(sorted([a, b]))
                bc = tuple(sorted([b, c]))
                ac = tuple(sorted([a, c]))
                if ab in incompat and bc in incompat and ac not in compat:
                    # only emit if the explicit ops for this pair don't contradict
                    explicit_ops = explicit.get(ac, set())
                    if "!~=" not in explicit_ops:
                        inferred.append(Relation([a], "~=", c, inferred=True))
                        compat.add(ac)
    return inferred

# ---------------------------------------------------------------------------
# 3D math
# ---------------------------------------------------------------------------

def sphere_layout(n):
    pts = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (i / max(n - 1, 1)) * 2.0
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        pts.append((x * 2.0, y * 2.0, z * 2.0))
    return pts

def rotate_x(x, y, z, a):
    c, s = math.cos(a), math.sin(a)
    return x, y * c - z * s, y * s + z * c

def rotate_y(x, y, z, a):
    c, s = math.cos(a), math.sin(a)
    return x * c + z * s, y, -x * s + z * c

def project(x, y, z, fov, cx, cy):
    dz = z + 6.0
    if dz < 0.01:
        dz = 0.01
    return x * fov / dz + cx, -y * fov / dz + cy, dz

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def depth_fade_rgb(rgb, dz, min_dz=3.0, max_dz=12.0):
    t = max(0.0, min(1.0, (dz - min_dz) / (max_dz - min_dz)))
    alpha = 1.0 - t * 0.65
    bg = (30, 30, 30)
    r = int(rgb[0] * alpha + bg[0] * (1 - alpha))
    g = int(rgb[1] * alpha + bg[1] * (1 - alpha))
    b = int(rgb[2] * alpha + bg[2] * (1 - alpha))
    return (r, g, b)

def with_alpha(rgb, a):
    return (rgb[0], rgb[1], rgb[2], a)

def depth_scaled_size(base_size, dz, min_dz=3.0, max_dz=12.0):
    """Scale a font size inversely with depth -- closer = bigger."""
    t = max(0.0, min(1.0, (dz - min_dz) / (max_dz - min_dz)))
    scale = 1.6 - t * 0.8   # ranges from 1.6 (close) down to 0.8 (far)
    return max(7, int(round(base_size * scale)))

def node_color_for(name):
    h = (hash(name) % 360) / 360.0
    rv, gv, bv = colorsys.hsv_to_rgb(h, 0.65, 0.90)
    return (int(rv * 255), int(gv * 255), int(bv * 255))

# ---------------------------------------------------------------------------
# AA draw helpers
# ---------------------------------------------------------------------------

def draw_aa_line(surf, color, x1, y1, x2, y2, width=1, alpha=255):
    """Draw an antialiased line with given width and alpha."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if width <= 1:
        pygame.gfxdraw.aacircle(surf, x1, y1, 1, (*color, alpha))
        pygame.gfxdraw.line(surf, x1, y1, x2, y2, (*color, alpha))
        return
    # for thicker lines draw multiple AA lines offset perpendicularly
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy) or 1
    px = -dy / length
    py = dx / length
    for w in range(-(width // 2), width // 2 + 1):
        ox = int(px * w)
        oy = int(py * w)
        pygame.gfxdraw.line(surf, x1 + ox, y1 + oy, x2 + ox, y2 + oy, (*color, alpha))

def draw_aa_circle(surf, color, cx, cy, r, alpha=255, fill=True, outline=True):
    cx, cy, r = int(cx), int(cy), int(r)
    if r < 1:
        return
    if fill:
        pygame.gfxdraw.filled_circle(surf, cx, cy, r, (*color, alpha))
    if outline:
        pygame.gfxdraw.aacircle(surf, cx, cy, r, (*color, min(255, alpha + 40)))

def _draw_arrowhead(surf, color, tip_x, tip_y, ux, uy, alpha, arrow_len=12, arrow_w=5):
    """Draw a single AA arrowhead triangle pointing in direction (ux, uy) at (tip_x, tip_y)."""
    px = -uy * arrow_w
    py =  ux * arrow_w
    base_x = tip_x - ux * arrow_len
    base_y = tip_y - uy * arrow_len
    pts = [
        (int(tip_x),        int(tip_y)),
        (int(base_x + px),  int(base_y + py)),
        (int(base_x - px),  int(base_y - py)),
    ]
    pygame.gfxdraw.filled_trigon(surf, pts[0][0], pts[0][1],
                                  pts[1][0], pts[1][1],
                                  pts[2][0], pts[2][1], (*color, alpha))
    pygame.gfxdraw.aatrigon(surf, pts[0][0], pts[0][1],
                              pts[1][0], pts[1][1],
                              pts[2][0], pts[2][1], (*color, alpha))

def draw_arrow(surf, color, x1, y1, x2, y2, width=2, alpha=255, dash=False):
    """Draw AA double-headed arrow between (x1,y1) and (x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy) or 1
    ux = dx / length
    uy = dy / length

    arrow_len = 12

    # shorten both ends to make room for arrowheads
    sx = x1 + ux * arrow_len
    sy = y1 + uy * arrow_len
    ex = x2 - ux * arrow_len
    ey = y2 - uy * arrow_len

    if dash:
        seg = 10
        gap = 6
        ddx = ex - sx
        ddy = ey - sy
        seg_len = math.hypot(ddx, ddy) or 1
        t = 0.0
        while t < 1.0:
            t0 = t
            t1 = min(t + seg / seg_len, 1.0)
            draw_aa_line(surf, color,
                         sx + ddx * t0, sy + ddy * t0,
                         sx + ddx * t1, sy + ddy * t1,
                         width, alpha)
            t += (seg + gap) / seg_len
    else:
        draw_aa_line(surf, color, sx, sy, ex, ey, width, alpha)

    # arrowhead at x2 end (pointing forward)
    _draw_arrowhead(surf, color, x2, y2,  ux,  uy, alpha, arrow_len)
    # arrowhead at x1 end (pointing backward)
    _draw_arrowhead(surf, color, x1, y1, -ux, -uy, alpha, arrow_len)

def draw_aa_dashed_circle(surf, color, cx, cy, r, alpha=200, dash_deg=18):
    """Draw a dashed AA circle (for halos)."""
    cx, cy, r = int(cx), int(cy), int(r)
    steps = 360 // dash_deg
    for i in range(steps):
        if i % 2 == 0:
            continue
        a0 = math.radians(i * dash_deg)
        a1 = math.radians((i + 1) * dash_deg)
        for t in range(8):
            tt = t / 7.0
            a = a0 + (a1 - a0) * tt
            x = int(cx + math.cos(a) * r)
            y = int(cy + math.sin(a) * r)
            if 0 <= x < surf.get_width() and 0 <= y < surf.get_height():
                pygame.gfxdraw.pixel(surf, x, y, (*color, alpha))

# ---------------------------------------------------------------------------
# Text rendering with pygame.font
# ---------------------------------------------------------------------------

_font_cache = {}

def get_font(size, bold=False):
    key = (size, bold)
    if key not in _font_cache:
        try:
            _font_cache[key] = pygame.font.SysFont("consolas,monospace", size, bold=bold)
        except:
            _font_cache[key] = pygame.font.Font(None, size)
    return _font_cache[key]

def draw_text(surf, text, x, y, size=13, color=(220,220,220), bold=False,
              anchor="center", alpha=255):
    font = get_font(size, bold)
    rendered = font.render(text, True, color)
    if alpha < 255:
        rendered.set_alpha(alpha)
    rect = rendered.get_rect()
    if anchor == "center":
        rect.center = (int(x), int(y))
    elif anchor == "topleft":
        rect.topleft = (int(x), int(y))
    elif anchor == "midleft":
        rect.midleft = (int(x), int(y))
    surf.blit(rendered, rect)
    return rect

# ---------------------------------------------------------------------------
# Panel surfaces (legend / info)
# ---------------------------------------------------------------------------

PANEL_W = 230
SIDEBAR_BG = (22, 22, 22)
BG_COLOR = (18, 18, 20)

CANVAS_W = 1100
CANVAS_H = 720
TOP_H = 46

def build_legend_surf(fonts_ready=True):
    w, h = PANEL_W, CANVAS_H - TOP_H
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    surf.fill((22, 22, 22, 245))

    y = 12
    draw_text(surf, "Operators", 10, y, size=13, bold=True,
              color=(210,210,210), anchor="topleft")
    y += 22

    for op, (meaning, color, ind) in OP_INFO.items():
        # color swatch
        pygame.draw.rect(surf, color, (10, y - 1, 6, 13))
        scope = " [ind]" if ind else ""
        draw_text(surf, op, 22, y, size=11, bold=True,
                  color=color, anchor="topleft")
        draw_text(surf, meaning + scope, 72, y, size=10,
                  color=(130, 130, 130), anchor="topleft")
        y += 16

    y += 4
    pygame.draw.line(surf, (55, 55, 55), (8, y), (w - 8, y))
    y += 8

    draw_text(surf, "-- inferred relation", 10, y, size=10,
              color=(80, 80, 80), anchor="topleft")
    y += 15
    draw_text(surf, "[ ] bracket group", 10, y, size=10,
              color=(100, 100, 100), anchor="topleft")
    y += 20

    pygame.draw.line(surf, (55, 55, 55), (8, y), (w - 8, y))
    y += 8

    draw_text(surf, "Controls", 10, y, size=12, bold=True,
              color=(180, 180, 180), anchor="topleft")
    y += 18
    for txt in ["LMB drag: rotate", "Scroll: zoom",
                "RMB: select node", "R: reset camera",
                "Enter: visualize"]:
        draw_text(surf, txt, 10, y, size=10, color=(90, 90, 90), anchor="topleft")
        y += 14

    return surf

def build_info_surf(node, relations, w=PANEL_W, h=200):
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    surf.fill((22, 22, 22, 0))
    if node is None:
        draw_text(surf, "RMB a node to inspect", 8, 8, size=10,
                  color=(80, 80, 80), anchor="topleft")
        return surf

    y = 8
    draw_text(surf, f"Type: {node.name}", 8, y, size=12, bold=True,
              color=(180, 210, 255), anchor="topleft")
    y += 20

    for r in relations:
        if node.name not in r.lhs and node.name != r.rhs:
            continue
        _, color, _ = OP_INFO.get(r.op, ("?", (200,200,200), False))
        tag = " (inf)" if r.inferred else ""
        tag2 = " [brk]" if r.bracket_group else ""
        if r.lhs[0] == r.rhs:
            line = f"{r.lhs[0]} {r.op}{tag}{tag2}"
        else:
            line = f"{r.lhs[0]} {r.op} {r.rhs}{tag}{tag2}"
        draw_text(surf, line, 8, y, size=10, color=color, anchor="topleft")
        y += 13
        _, meaning, _ = OP_INFO.get(r.op, ("?", (200,200,200), False))
        draw_text(surf, f"  -> {meaning}", 8, y, size=9,
                  color=(90, 90, 90), anchor="topleft")
        y += 13
        if y > h - 16:
            break
    return surf

# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------

NODE_RADIUS = 20
LANE_SPACING = 15

class FluxTypeGeomViz:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Flux Type Geometry Visualizer")

        self.W = CANVAS_W
        self.H = CANVAS_H
        self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
        pygame.scrap.init()
        self.clock = pygame.time.Clock()

        self.nodes: list[TypeNode] = []
        self.relations: list[Relation] = []
        self.selected = None

        self.rot_x = 0.2
        self.rot_y = 0.4
        self.zoom = 1.0
        self.auto_rotate = True
        self.dragging = False
        self.drag_last = (0, 0)
        self.idle_timer = 0       # ms since last user interaction
        self.ease_t = 1.0         # 0.0=stopped 1.0=full speed (eased in)
        self.IDLE_DELAY = 5000    # ms before auto-rotate resumes

        # text input
        self.input_text = "D !~= B & [A !@ A] !~= C !`< D !-= A"
        self.input_active = False
        self.cursor_visible = True
        self.cursor_timer = 0
        self.cursor_pos = len(self.input_text)  # insertion point
        self.sel_start = 0                       # selection anchor
        self.sel_end = len(self.input_text)      # selection end (== cursor_pos when no sel)
        self._input_rect = None                  # set during draw, used for hit-testing
        self._txt_x_offset = 0                   # horizontal scroll offset in pixels
        self._sel_dragging = False               # dragging selection in text box

        # status
        self.status = "Press Enter to visualize"

        self._parse_and_build()

        # pre-render static legend
        self.legend_surf = build_legend_surf()
        self.info_surf = build_info_surf(None, [])

    # -----------------------------------------------------------------------
    # Parse
    # -----------------------------------------------------------------------

    def _parse_and_build(self):
        expr = self.input_text.strip()
        if not expr:
            return
        try:
            rels, names = parse_constraint_expr(expr)
        except Exception as ex:
            self.status = f"Parse error: {ex}"
            return

        inferred = infer_relations(rels, names)
        self.relations = rels + inferred

        name_list = sorted(names)
        pts = sphere_layout(len(name_list))
        self.nodes = [TypeNode(nm, *pts[i]) for i, nm in enumerate(name_list)]
        self.selected = None
        self.info_surf = build_info_surf(None, [])

        n_inf = len(inferred)
        self.status = (f"{len(name_list)} types  |  {len(rels)} explicit  |  "
                       f"{n_inf} inferred  |  drag=rotate  scroll=zoom  R=reset")

    def _try_live_update(self):
        """
        Called on every keystroke. Parses silently; only rebuilds the graph
        if the set of type names has changed. This means the graph stays stable
        while the user is typing operators between names, and snaps in as soon
        as a new (or removed) name is resolved.
        """
        expr = self.input_text.strip()
        if not expr:
            return
        try:
            rels, names = parse_constraint_expr(expr)
        except Exception:
            return  # mid-type parse errors are silently ignored

        if not names:
            return  # nothing to show yet

        current_names = {n.name for n in self.nodes}
        if names == current_names:
            # same nodes -- just refresh relations in place (new edges may have appeared)
            inferred = infer_relations(rels, names)
            self.relations = rels + inferred
            n_inf = len(inferred)
            self.status = (f"{len(names)} types  |  {len(rels)} explicit  |  "
                           f"{n_inf} inferred  |  drag=rotate  scroll=zoom  R=reset")
            return

        # name set changed -- full rebuild (new nodes get sphere positions)
        inferred = infer_relations(rels, names)
        self.relations = rels + inferred

        name_list = sorted(names)
        pts = sphere_layout(len(name_list))
        self.nodes = [TypeNode(nm, *pts[i]) for i, nm in enumerate(name_list)]
        self.selected = None
        self.info_surf = build_info_surf(None, [])

        n_inf = len(inferred)
        self.status = (f"{len(name_list)} types  |  {len(rels)} explicit  |  "
                       f"{n_inf} inferred  |  drag=rotate  scroll=zoom  R=reset")

    # -----------------------------------------------------------------------
    # Camera
    # -----------------------------------------------------------------------

    def _reset_camera(self):
        self.rot_x = 0.2
        self.rot_y = 0.4
        self.zoom = 1.0

    def _project_nodes(self):
        cw = self.W - PANEL_W
        ch = self.H - TOP_H
        cx = cw / 2
        cy = TOP_H + ch / 2
        fov = 480 * self.zoom

        for node in self.nodes:
            x, y, z = node.x, node.y, node.z
            x, y, z = rotate_x(x, y, z, self.rot_x)
            x, y, z = rotate_y(x, y, z, self.rot_y)
            sx, sy, dz = project(x, y, z, fov, cx, cy)
            node.sx = sx
            node.sy = sy
            node.sz = dz

    def _node_by_name(self, name):
        for n in self.nodes:
            if n.name == name:
                return n
        return None

    # -----------------------------------------------------------------------
    # Draw
    # -----------------------------------------------------------------------

    def _draw_frame(self):
        self.screen.fill(BG_COLOR)

        # decorative ring
        cw = self.W - PANEL_W
        ch = self.H - TOP_H
        cx = cw // 2
        cy = TOP_H + ch // 2
        ring_r = int(min(cw, ch) * 0.42)
        pygame.gfxdraw.aacircle(self.screen, cx, cy, ring_r, (45, 45, 45, 120))

        if self.nodes:
            self._project_nodes()
            self._draw_edges()
            sorted_nodes = sorted(self.nodes, key=lambda n: n.sz, reverse=True)
            for node in sorted_nodes:
                self._draw_node(node)

        self._draw_top_bar()
        self._draw_panel()
        self._draw_status()

    def _draw_edges(self):
        drawn_self = set()

        # group pairwise
        pair_groups = {}
        self_rels = []
        for r in self.relations:
            if r.lhs[0] == r.rhs:
                self_rels.append(r)
                continue
            lnode = self._node_by_name(r.lhs[0])
            rnode = self._node_by_name(r.rhs)
            if lnode is None or rnode is None:
                continue
            key = tuple(sorted([r.lhs[0], r.rhs]))
            forward = (r.lhs[0] == key[0])
            pair_groups.setdefault(key, []).append((r, forward))

        # self-referential halos -- group by node first so we can space radially
        node_self_ops = {}   # node_name -> [(op, color)]
        for r in self_rels:
            key2 = (r.lhs[0], r.op)
            if key2 in drawn_self:
                continue
            drawn_self.add(key2)
            _, color, _ = OP_INFO.get(r.op, ("?", (150,150,150), False))
            node_self_ops.setdefault(r.lhs[0], []).append((r.op, color))

        for node_name, ops in node_self_ops.items():
            lnode = self._node_by_name(node_name)
            if lnode is None:
                continue
            dz = lnode.sz
            alpha = max(60, int(255 * (1.0 - max(0, (dz - 3.0) / 9.0) * 0.65)))
            nr = int(NODE_RADIUS * max(0.5, min(1.3, 6.0 / dz)))
            cx, cy = int(lnode.sx), int(lnode.sy)

            n_ops = len(ops)
            # spread labels radially; start at top-right (315 deg) and space evenly
            # use a base radius slightly outside the node for the label anchor
            label_r = nr + 22

            for i, (op, color) in enumerate(ops):
                fade = depth_fade_rgb(color, dz)

                # draw one shared dashed halo per node (first op draws it)
                if i == 0:
                    r_halo = nr + 12
                    draw_aa_dashed_circle(self.screen, fade, cx, cy, r_halo, alpha=alpha)

                # radial angle: spread evenly starting from -45 deg (top-right)
                angle_deg = -45 + i * (360 / n_ops)
                angle_rad = math.radians(angle_deg)
                lx = cx + math.cos(angle_rad) * label_r
                ly = cy + math.sin(angle_rad) * label_r

                # small connecting tick from halo edge to label
                tick_r = nr + 13
                tx = cx + math.cos(angle_rad) * tick_r
                ty = cy + math.sin(angle_rad) * tick_r
                draw_aa_line(self.screen, fade,
                             tx, ty, lx, ly, width=1, alpha=max(40, alpha - 60))

                draw_text(self.screen, op, lx, ly,
                          size=depth_scaled_size(10, dz),
                          color=fade, anchor="center")

        # pairwise edges
        for (nameA, nameB), entries in pair_groups.items():
            nodeA = self._node_by_name(nameA)
            nodeB = self._node_by_name(nameB)
            if nodeA is None or nodeB is None:
                continue

            x1, y1 = nodeA.sx, nodeA.sy
            x2, y2 = nodeB.sx, nodeB.sy
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy) or 1
            px = -dy / length
            py = dx / length

            n = len(entries)
            label_fracs = [(i + 1) / (n + 1) for i in range(n)]

            avg_dz = (nodeA.sz + nodeB.sz) / 2
            alpha = max(40, int(255 * (1.0 - max(0, (avg_dz - 3.0) / 9.0) * 0.65)))

            for i, (r, _forward) in enumerate(entries):
                _, color, _ = OP_INFO.get(r.op, ("?", (150,150,150), False))
                fade = depth_fade_rgb(color, avg_dz)
                line_alpha = max(30, alpha - (60 if r.inferred else 0))
                width = 1 if r.inferred else 2

                offset = (i - (n - 1) / 2.0) * LANE_SPACING
                ox = px * offset
                oy = py * offset

                ax = x1 + ox
                ay = y1 + oy
                bx = x2 + ox
                by = y2 + oy

                # shorten both ends to node edge
                scale_start = max(0.0, min(1.0, NODE_RADIUS / (length or 1)))
                scale_end   = max(0.0, min(1.0, (length - NODE_RADIUS) / (length or 1)))
                ssx = ax + (bx - ax) * scale_start
                ssy = ay + (by - ay) * scale_start
                eex = ax + (bx - ax) * scale_end
                eey = ay + (by - ay) * scale_end

                draw_arrow(self.screen, fade, ssx, ssy, eex, eey,
                           width=width, alpha=line_alpha, dash=r.inferred)

                # label at fractional position
                t = label_fracs[i]
                lx = ax + (bx - ax) * t
                ly = ay + (by - ay) * t
                nudge = 13
                lx += px * nudge
                ly += py * nudge

                label_color = fade if not r.inferred else (70, 70, 70)
                label_alpha = max(40, line_alpha)
                lbl_size = depth_scaled_size(10, avg_dz)
                draw_text(self.screen, r.op, lx, ly, size=lbl_size,
                          color=label_color, anchor="center")
                if r.bracket_group:
                    draw_text(self.screen, "[grp]", lx, ly + lbl_size + 3, size=max(7, lbl_size - 2),
                              color=(60, 60, 60), anchor="center")

    def _draw_node(self, node):
        dz = node.sz
        scale = max(0.5, min(1.3, 6.0 / dz))
        nr = int(NODE_RADIUS * scale)
        cx, cy = int(node.sx), int(node.sy)
        alpha = max(80, int(255 * (1.0 - max(0, (dz - 3.0) / 9.0) * 0.55)))

        base_color = node_color_for(node.name)
        fade = depth_fade_rgb(base_color, dz)

        is_sel = self.selected is not None and self.selected.name == node.name

        if is_sel:
            # glow ring
            for rr in range(nr + 8, nr + 3, -1):
                a = max(0, int(60 * (1 - (rr - nr) / 8.0)))
                pygame.gfxdraw.aacircle(self.screen, cx, cy, rr, (255, 255, 255, a))

        # filled circle
        draw_aa_circle(self.screen, fade, cx, cy, nr, alpha=alpha, fill=True, outline=True)

        # outline
        outline_col = (255, 255, 255) if is_sel else (80, 80, 80)
        pygame.gfxdraw.aacircle(self.screen, cx, cy, nr, (*outline_col, alpha))

        # label
        draw_text(self.screen, node.name, cx, cy, size=14, bold=True,
                  color=(255, 255, 255), anchor="center")

    # -----------------------------------------------------------------------
    # Text box helpers
    # -----------------------------------------------------------------------

    def _char_x(self, pos, font):
        """Pixel x offset of character index pos within the text, relative to text origin."""
        return font.size(self.input_text[:pos])[0]

    def _pos_from_x(self, px, font):
        """Return the character index closest to pixel offset px from text origin."""
        text = self.input_text
        best = len(text)
        for i in range(len(text) + 1):
            cx = font.size(text[:i])[0]
            if cx >= px:
                # pick whichever side of this char is closer
                if i > 0:
                    prev_cx = font.size(text[:i-1])[0]
                    if abs(prev_cx - px) < abs(cx - px):
                        return i - 1
                return i
        return best

    def _scroll_to_cursor(self, clip_w, font):
        """Adjust _txt_x_offset so cursor is visible inside clip_w."""
        cx = self._char_x(self.cursor_pos, font)
        # cursor pixel position on screen relative to clip left
        screen_cx = cx + self._txt_x_offset
        margin = 6
        if screen_cx < margin:
            self._txt_x_offset = -cx + margin
        elif screen_cx > clip_w - margin:
            self._txt_x_offset = clip_w - cx - margin

    def _sel_range(self):
        """Return (lo, hi) selection indices, always lo <= hi."""
        a, b = self.sel_start, self.sel_end
        return (min(a, b), max(a, b))

    def _has_sel(self):
        return self.sel_start != self.sel_end

    def _delete_selection(self):
        lo, hi = self._sel_range()
        self.input_text = self.input_text[:lo] + self.input_text[hi:]
        self.cursor_pos = lo
        self.sel_start = lo
        self.sel_end = lo

    def _move_cursor(self, new_pos, extend_sel):
        self.cursor_pos = max(0, min(len(self.input_text), new_pos))
        if extend_sel:
            self.sel_end = self.cursor_pos
        else:
            self.sel_start = self.cursor_pos
            self.sel_end = self.cursor_pos

    def _word_left(self):
        p = self.cursor_pos - 1
        while p > 0 and not self.input_text[p-1].isalnum():
            p -= 1
        while p > 0 and self.input_text[p-1].isalnum():
            p -= 1
        return p

    def _word_right(self):
        p = self.cursor_pos
        while p < len(self.input_text) and not self.input_text[p].isalnum():
            p += 1
        while p < len(self.input_text) and self.input_text[p].isalnum():
            p += 1
        return p

    # -----------------------------------------------------------------------
    # Draw top bar
    # -----------------------------------------------------------------------

    def _draw_top_bar(self):
        bar_w = self.W - PANEL_W
        pygame.draw.rect(self.screen, (28, 28, 32), (0, 0, bar_w, TOP_H))
        pygame.draw.line(self.screen, (50, 50, 55), (0, TOP_H - 1), (bar_w, TOP_H - 1))

        draw_text(self.screen, "Constraint:", 10, TOP_H // 2, size=12,
                  color=(150, 150, 150), anchor="midleft")

        input_x = 100
        input_w = bar_w - input_x - 120
        input_rect = pygame.Rect(input_x, 7, input_w, TOP_H - 14)
        self._input_rect = input_rect
        clip_rect = input_rect.inflate(-8, -4)

        box_color = (45, 50, 60) if self.input_active else (35, 35, 40)
        border_color = (80, 130, 200) if self.input_active else (55, 55, 60)
        pygame.draw.rect(self.screen, box_color, input_rect, border_radius=4)
        pygame.draw.rect(self.screen, border_color, input_rect, 1, border_radius=4)

        font = get_font(12)
        ty = input_rect.y + (input_rect.h - font.get_height()) // 2

        self._scroll_to_cursor(clip_rect.width, font)
        tx = clip_rect.x + self._txt_x_offset

        self.screen.set_clip(clip_rect)

        # selection highlight
        if self.input_active and self._has_sel():
            lo, hi = self._sel_range()
            sel_x0 = tx + self._char_x(lo, font)
            sel_x1 = tx + self._char_x(hi, font)
            sel_rect = pygame.Rect(sel_x0, ty, sel_x1 - sel_x0, font.get_height())
            sel_rect = sel_rect.clip(clip_rect)
            pygame.draw.rect(self.screen, (60, 100, 180), sel_rect)

        # text
        txt_surf = font.render(self.input_text, True, (210, 220, 235))
        self.screen.blit(txt_surf, (tx, ty))

        # cursor
        if self.input_active and self.cursor_visible and not self._has_sel():
            cx = tx + self._char_x(self.cursor_pos, font)
            pygame.draw.line(self.screen, (200, 220, 255),
                             (int(cx), ty + 1), (int(cx), ty + font.get_height() - 1), 1)

        self.screen.set_clip(None)

        # Visualize button
        btn_x = bar_w - 115
        btn_rect = pygame.Rect(btn_x, 8, 105, TOP_H - 16)
        pygame.draw.rect(self.screen, (45, 80, 130), btn_rect, border_radius=4)
        pygame.draw.rect(self.screen, (70, 110, 180), btn_rect, 1, border_radius=4)
        draw_text(self.screen, "Visualize", btn_rect.centerx, btn_rect.centery,
                  size=12, bold=True, color=(200, 220, 255), anchor="center")
        self._btn_rect = btn_rect

        ar_x = self.W - PANEL_W + 8
        ar_y = TOP_H // 2
        col = (80, 200, 120) if self.auto_rotate else (120, 120, 120)
        draw_text(self.screen, "Auto-rotate", ar_x, ar_y, size=11,
                  color=col, anchor="midleft")
        self._ar_label_rect = pygame.Rect(ar_x, ar_y - 10, 100, 20)

    def _draw_panel(self):
        px = self.W - PANEL_W
        # legend
        self.screen.blit(self.legend_surf, (px, TOP_H))
        # info panel at bottom of legend
        legend_h = self.legend_surf.get_height()
        info_y = TOP_H + legend_h - 220
        pygame.draw.line(self.screen, (50, 50, 55),
                         (px + 8, info_y), (self.W - 8, info_y))
        draw_text(self.screen, "Node Info", px + 10, info_y + 8, size=11,
                  bold=True, color=(160, 160, 160), anchor="topleft")
        self.screen.blit(self.info_surf, (px, info_y + 24))

    def _draw_status(self):
        sy = self.H - 20
        draw_text(self.screen, self.status, 10, sy, size=10,
                  color=(70, 70, 70), anchor="topleft")

    # -----------------------------------------------------------------------
    # Events
    # -----------------------------------------------------------------------

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.VIDEORESIZE:
                self.W, self.H = event.w, event.h
                self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
                self.legend_surf = build_legend_surf()

            elif event.type == pygame.KEYDOWN:
                ctrl = event.mod & pygame.KMOD_CTRL
                shift = event.mod & pygame.KMOD_SHIFT

                if event.key == pygame.K_r and not self.input_active:
                    self._reset_camera()
                elif event.key == pygame.K_RETURN:
                    self._parse_and_build()
                    self.input_active = False
                elif event.key == pygame.K_ESCAPE:
                    self.input_active = False
                elif self.input_active:
                    if ctrl and event.key == pygame.K_a:
                        # select all
                        self.sel_start = 0
                        self.sel_end = len(self.input_text)
                        self.cursor_pos = len(self.input_text)
                    elif ctrl and event.key == pygame.K_c:
                        # copy selection
                        lo, hi = self._sel_range()
                        if lo < hi:
                            pygame.scrap.put(pygame.SCRAP_TEXT,
                                             self.input_text[lo:hi].encode())
                    elif ctrl and event.key == pygame.K_x:
                        # cut selection
                        lo, hi = self._sel_range()
                        if lo < hi:
                            pygame.scrap.put(pygame.SCRAP_TEXT,
                                             self.input_text[lo:hi].encode())
                            self._delete_selection()
                            self._try_live_update()
                    elif ctrl and event.key == pygame.K_v:
                        # paste
                        data = pygame.scrap.get(pygame.SCRAP_TEXT)
                        if data:
                            text = data.decode('utf-8', errors='ignore').rstrip(chr(0))
                            if self._has_sel():
                                self._delete_selection()
                            p = self.cursor_pos
                            self.input_text = self.input_text[:p] + text + self.input_text[p:]
                            self._move_cursor(p + len(text), False)
                            self._try_live_update()
                    elif event.key == pygame.K_BACKSPACE:
                        if self._has_sel():
                            self._delete_selection()
                        elif ctrl:
                            # delete word left
                            new_p = self._word_left()
                            self.input_text = self.input_text[:new_p] + self.input_text[self.cursor_pos:]
                            self._move_cursor(new_p, False)
                        else:
                            if self.cursor_pos > 0:
                                p = self.cursor_pos
                                self.input_text = self.input_text[:p-1] + self.input_text[p:]
                                self._move_cursor(p - 1, False)
                        self._try_live_update()
                    elif event.key == pygame.K_DELETE:
                        if self._has_sel():
                            self._delete_selection()
                        elif ctrl:
                            # delete word right
                            new_p = self._word_right()
                            self.input_text = self.input_text[:self.cursor_pos] + self.input_text[new_p:]
                        else:
                            p = self.cursor_pos
                            self.input_text = self.input_text[:p] + self.input_text[p+1:]
                        self._try_live_update()
                    elif event.key == pygame.K_LEFT:
                        if not shift and self._has_sel():
                            lo, _ = self._sel_range()
                            self._move_cursor(lo, False)
                        elif ctrl:
                            self._move_cursor(self._word_left(), shift)
                        else:
                            self._move_cursor(self.cursor_pos - 1, shift)
                    elif event.key == pygame.K_RIGHT:
                        if not shift and self._has_sel():
                            _, hi = self._sel_range()
                            self._move_cursor(hi, False)
                        elif ctrl:
                            self._move_cursor(self._word_right(), shift)
                        else:
                            self._move_cursor(self.cursor_pos + 1, shift)
                    elif event.key == pygame.K_HOME:
                        self._move_cursor(0, shift)
                    elif event.key == pygame.K_END:
                        self._move_cursor(len(self.input_text), shift)
                    else:
                        ch = event.unicode
                        if ch and ch.isprintable():
                            if self._has_sel():
                                self._delete_selection()
                            p = self.cursor_pos
                            self.input_text = self.input_text[:p] + ch + self.input_text[p:]
                            self._move_cursor(p + 1, False)
                            self._try_live_update()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                shift = pygame.key.get_mods() & pygame.KMOD_SHIFT

                if event.button == 1:
                    if my < TOP_H:
                        if hasattr(self, '_btn_rect') and self._btn_rect.collidepoint(mx, my):
                            self._parse_and_build()
                        elif hasattr(self, '_ar_label_rect') and self._ar_label_rect.collidepoint(mx, my):
                            self.auto_rotate = not self.auto_rotate
                            self.idle_timer = 0
                            self.ease_t = 0.0 if not self.auto_rotate else 1.0
                        elif self._input_rect and self._input_rect.collidepoint(mx, my):
                            self.input_active = True
                            self._sel_dragging = True
                            font = get_font(12)
                            clip_x = self._input_rect.x + 4
                            px = mx - clip_x - self._txt_x_offset
                            pos = self._pos_from_x(px, font)
                            if shift:
                                self._move_cursor(pos, True)
                            else:
                                self._move_cursor(pos, False)
                                self.sel_start = pos
                        else:
                            self.input_active = False
                            self._sel_dragging = False
                    else:
                        self.input_active = False
                        self._sel_dragging = False
                        self.dragging = True
                        self.drag_last = (mx, my)
                        self.auto_rotate = False
                        self.idle_timer = 0
                        self.ease_t = 0.0

                elif event.button == 3:
                    if my >= TOP_H:
                        best = None
                        best_d = 35.0
                        for node in self.nodes:
                            d = math.hypot(node.sx - mx, node.sy - my)
                            if d < best_d:
                                best_d = d
                                best = node
                        self.selected = best
                        self.info_surf = build_info_surf(best, self.relations,
                                                         w=PANEL_W, h=220)

                elif event.button == 4:
                    if my >= TOP_H:
                        self.zoom = min(5.0, self.zoom * 1.1)
                elif event.button == 5:
                    if my >= TOP_H:
                        self.zoom = max(0.15, self.zoom * 0.9)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self._sel_dragging = False
                    if self.dragging:
                        self.dragging = False
                        self.idle_timer = 0
                        self.ease_t = 0.0

            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                if self._sel_dragging and self._input_rect and self.input_active:
                    font = get_font(12)
                    clip_x = self._input_rect.x + 4
                    px = mx - clip_x - self._txt_x_offset
                    pos = self._pos_from_x(px, font)
                    self.cursor_pos = pos
                    self.sel_end = pos
                elif self.dragging:
                    dx = mx - self.drag_last[0]
                    dy = my - self.drag_last[1]
                    self.rot_y += dx * 0.007
                    self.rot_x += dy * 0.007
                    self.drag_last = (mx, my)

        return True

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60)

            running = self._handle_events()

            # auto-rotate with 5s idle delay and ease-in
            if not self.auto_rotate:
                self.idle_timer += dt
                if self.idle_timer >= self.IDLE_DELAY:
                    self.auto_rotate = True
                    # ease_t stays at 0, will ramp up below

            if self.auto_rotate:
                # ease_t ramps from 0 to 1 over ~1.5 seconds
                if self.ease_t < 1.0:
                    self.ease_t = min(1.0, self.ease_t + dt / 1500.0)
                # smoothstep
                t = self.ease_t * self.ease_t * (3.0 - 2.0 * self.ease_t)
                self.rot_y += 0.006 * t

            # cursor blink
            self.cursor_timer += dt
            if self.cursor_timer >= 530:
                self.cursor_timer = 0
                self.cursor_visible = not self.cursor_visible

            self._draw_frame()
            pygame.display.flip()

        pygame.quit()


def main():
    app = FluxTypeGeomViz()
    app.run()

if __name__ == "__main__":
    main()