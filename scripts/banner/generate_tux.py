#!/usr/bin/env python3
"""Generate the animated GitHub profile banners with Tux (Linux penguin)."""

from __future__ import annotations

import html
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
LOGOS = Path(__file__).resolve().parent / "logos"

W, H = 1180, 610
INTRO_SECONDS = 3.2
TRANSITION_SECONDS = 1.3
LOGO_HOLD_SECONDS = 4.0
TRAVELLER_COUNT = 900
HOLD_PARTICLE_COUNT = 2_400
SEED = 314159
LOGO_SUFFIXES = {".png", ".webp"}
LOGO_ALIASES = {"kube": "kubernetes"}
PREFERRED_LOGO_ORDER = ("linux",)

YAML_ROWS = [
    (0, "profile", ""),
    (1, "subject", "Usuario444"),
    (1, "role", "Redes &amp; Seguridad Informatica"),
    (1, "study", "Cisco CCNA (Switching &amp; Routing)"),
    (1, "focus", "Redes · Infraestructura · Seguridad"),
    (1, "status", "[ITN ✓]  [SRWE ✓]  [ENSA ⋯]"),
    (1, "tools", "Wireshark · Packet Tracer · GNS3"),
    (0, "stack", ""),
    (1, "layer_2", "802.1Q · LACP · STP/RSTP · ARP"),
    (1, "layer_3", "OSPF · IPv4/IPv6 · HSRP · NAT"),
    (1, "systems", "Linux · Cisco IOS"),
    (1, "lab_env", "PNETLab · GNS3 · Wireshark"),
    (1, "dev_db", "Java · PostgreSQL · Postman"),
    (0, "info", ""),
    (1, "state", "con sueño ☕"),
    (1, "github", "Usuario444"),
    (1, "protocol", "ping exitoso"),
]

THEMES = {
    "dark": {
        "bg":      "#080C14",
        "panel":   "#0D131F",
        "panel2":  "#101726",
        "line":    "#1F293D",
        "muted":   "#8291A8",
        "text":    "#F0E6F0",
        "portrait":"#F78CA0",   # city pop pink
        "tux":     "#000000",   # black Tux animation
        "chrome":  "#C9B1D9",   # city pop lavender
        "accent":  "#F78CA0",
        "shadow":  "#02050B",
    },
    "light": {
        "bg":      "#FDF0F3",
        "panel":   "#FFFFFF",
        "panel2":  "#FDE8EE",
        "line":    "#F0C0CE",
        "muted":   "#9B7B8A",
        "text":    "#2D1A24",
        "portrait":"#E05F80",
        "tux":     "#000000",   # black Tux animation
        "chrome":  "#7B5EA7",
        "accent":  "#E05F80",
        "shadow":  "#D4A0B0",
    },
}


def edge_connected(mask: np.ndarray) -> np.ndarray:
    connected = np.zeros(mask.shape, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    height, width = mask.shape
    for x in range(width):
        queue.extend(((0, x), (height - 1, x)))
    for y in range(height):
        queue.extend(((y, 0), (y, width - 1)))
    while queue:
        y, x = queue.popleft()
        if connected[y, x] or not mask[y, x]:
            continue
        connected[y, x] = True
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < height and 0 <= nx < width:
                queue.append((ny, nx))
    return connected


def normalize_logo(image: Image.Image) -> Image.Image:
    icon = image.convert("RGBA")
    alpha = icon.getchannel("A")
    if alpha.getextrema() == (255, 255):
        gray = np.asarray(ImageOps.grayscale(icon))
        background = edge_connected(gray >= 220)
        visible = ~background
        visible = np.asarray(
            Image.fromarray((visible * 255).astype("uint8"))
            .filter(ImageFilter.MaxFilter(3))
        ) > 0
        exterior = edge_connected(~visible)
        icon.putalpha(Image.fromarray(np.where(exterior, 0, 255).astype("uint8")))
    icon.thumbnail((320, 320), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    offset = ((400 - icon.width) // 2, (400 - icon.height) // 2)
    canvas.alpha_composite(icon, offset)
    return canvas


def make_logos() -> dict[str, Image.Image]:
    linux_path = LOGOS / "linux.png"
    if not linux_path.exists():
        raise SystemExit(f"Missing {linux_path}")
    with Image.open(linux_path) as img:
        return {"linux": normalize_logo(img)}


def logo_silhouette_points(image: Image.Image) -> np.ndarray:
    alpha = np.asarray(image.getchannel("A"))
    ys, xs = np.where(alpha > 127)
    return np.column_stack((89 + xs * 0.675, 188 + ys * 0.675)).astype(np.float32)


def sample_logo_points(
    image: Image.Image, rng: np.random.Generator, count: int
) -> np.ndarray:
    points = logo_silhouette_points(image)
    chosen = rng.choice(len(points), count, replace=len(points) < count)
    return points[chosen]


def portrait_points(theme: str, rng: np.random.Generator) -> np.ndarray:
    """Tux is our primary portrait, rendered densely in Floyd-Steinberg dither lattice."""
    linux_path = LOGOS / "linux.png"
    with Image.open(linux_path) as img:
        norm = normalize_logo(img)
    pts = logo_silhouette_points(norm)
    if len(pts) > 14000:
        pts = pts[rng.choice(len(pts), 14000, replace=False)]
    return pts


def transport(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    rows, cols = linear_sum_assignment(cdist(source, target, metric="sqeuclidean"))
    ordered = np.empty_like(target)
    ordered[rows] = target[cols]
    return ordered


def num(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def time_num(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def point_path(points: np.ndarray) -> str:
    if not len(points):
        return ""
    integer = np.rint(points).astype(int)
    unique = sorted({(int(x), int(y)) for x, y in integer}, key=lambda p: (p[1], p[0]))
    chunks: list[str] = []
    i = 0
    while i < len(unique):
        x0, y = unique[i]
        x1 = x0
        i += 1
        while i < len(unique) and unique[i][1] == y and unique[i][0] <= x1 + 1:
            x1 = unique[i][0]
            i += 1
        chunks.append(f"M{x0} {y}h{x1 - x0 + 1}")
    return "".join(chunks)


def particle_path(points: np.ndarray) -> str:
    integer = np.rint(points).astype(int)
    unique = sorted({(int(x), int(y)) for x, y in integer}, key=lambda p: (p[1], p[0]))
    return "".join(f"M{x} {y}h1" for x, y in unique)


def animate_values(points: list[np.ndarray], index: int) -> str:
    return ";".join(f"{num(p[index, 0])} {num(p[index, 1])}" for p in points)


def render_svg(
    theme_name: str,
    portrait: np.ndarray,
    logo_points: dict[str, np.ndarray],
    logo_hold_particles: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> str:
    t = THEMES[theme_name]
    n = min(TRAVELLER_COUNT, len(portrait))
    source = portrait[rng.choice(len(portrait), n, replace=False)]
    targets: list[np.ndarray] = []
    current = source
    for name, points in logo_points.items():
        current = transport(current, points[:n])
        targets.append(current)

    times = [0.0, 3.0]
    frames = [source, source]
    for target in targets:
        times.extend((times[-1] + TRANSITION_SECONDS,
                      times[-1] + TRANSITION_SECONDS + LOGO_HOLD_SECONDS))
        frames.extend((target, target))

    # Add gentle drift back to source for seamless loop
    times.append(times[-1] + TRANSITION_SECONDS)
    frames.append(source)

    loop_duration = times[-1]
    key_times = ";".join(time_num(time / loop_duration) for time in times)

    opacities: list[float] = [0.0]
    for _ in targets:
        opacities.extend((0.0, 0.94, 0.94, 0.0))
    opacities.append(0.0)
    opacity_values = ";".join(num(v) for v in opacities)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="610" '
        'viewBox="0 0 1180 610" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">Usuario444's live system profile</title>",
        '<desc id="desc">Animated terminal profile for Usuario444 with Linux Tux particle system.</desc>',
        "<defs>",
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">',
        f'<feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="{t["shadow"]}" flood-opacity=".28"/>'
        "</filter>",
        '<filter id="glow" x="-100%" y="-100%" width="300%" height="300%">',
        f'<feGaussianBlur stdDeviation="3" result="b"/><feFlood flood-color="{t["chrome"]}" '
        'flood-opacity=".35"/><feComposite in2="b" operator="in"/>'
        '<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '<clipPath id="visualClip"><rect x="49" y="124" width="390" height="414" rx="3"/></clipPath>',
        (
            '<radialGradient id="tuxBacklight" cx="50%" cy="50%" r="50%">'
            '<stop offset="0%" stop-color="#E2E8F0" stop-opacity="0.92"/>'
            '<stop offset="52%" stop-color="#94A3B8" stop-opacity="0.45"/>'
            '<stop offset="85%" stop-color="#1E293B" stop-opacity="0.15"/>'
            '<stop offset="100%" stop-color="#101726" stop-opacity="0"/>'
            '</radialGradient>'
            if theme_name == "dark"
            else '<radialGradient id="tuxBacklight" cx="50%" cy="50%" r="50%">'
            '<stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.9"/>'
            '<stop offset="100%" stop-color="#FDE8EE" stop-opacity="0"/>'
            '</radialGradient>'
        ),
        "</defs>",
        f'<rect width="{W}" height="{H}" rx="18" fill="{t["bg"]}"/>',
        f'<rect x="13" y="13" width="1154" height="584" rx="13" fill="{t["panel"]}" '
        f'stroke="{t["line"]}" filter="url(#shadow)"/>',
        f'<path d="M13 62H1167" stroke="{t["line"]}"/>',
        '<circle cx="38" cy="38" r="6" fill="#FF5F57"/>'
        '<circle cx="59" cy="38" r="6" fill="#FEBC2E"/>'
        '<circle cx="80" cy="38" r="6" fill="#28C840"/>',
        f'<text x="590" y="43" text-anchor="middle" fill="{t["muted"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
        'letter-spacing=".4">vim profile.yml</text>',
        f'<rect x="35" y="88" width="418" height="472" rx="6" fill="{t["panel2"]}" '
        f'stroke="{t["line"]}"/>',
        f'<path d="M35 124H453" stroke="{t["line"]}"/>',
        f'<text x="49" y="111" fill="{t["chrome"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
        'font-weight="700" letter-spacing="1.2">VISUAL.MAP</text>',
        f'<text x="438" y="111" text-anchor="end" fill="{t["muted"]}" '
        'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">TUX / 1-BIT</text>',
        f'<path d="M49 141h12M49 141v12M439 141h-12M439 141v12M49 539h12M49 539v-12'
        f'M439 539h-12M439 539v-12" fill="none" stroke="{t["chrome"]}" opacity=".55"/>',
        '<rect x="49" y="124" width="390" height="414" rx="3" fill="url(#tuxBacklight)"/>',
        '<g clip-path="url(#visualClip)" shape-rendering="crispEdges">',
        '<g opacity="1">',
    ]

    first_centroid = targets[0].mean(axis=0)
    band_ids = rng.integers(0, 94, size=len(portrait))
    noise = rng.normal(0, 4, size=(94, 2))
    for band in range(94):
        pts = portrait[band_ids == band]
        if not len(pts):
            continue
        centroid = pts.mean(axis=0)
        delta = (first_centroid - centroid) * 0.18 + noise[band]
        drift_positions = ["0 0", "0 0", f"{num(delta[0])} {num(delta[1])}",
                           f"{num(delta[0])} {num(delta[1])}"]
        drift_positions.extend(["0 0"] * (len(frames) - len(drift_positions)))
        drift_opacity = [".94", ".94", "0", "0"]
        drift_opacity.extend(["0"] * (len(frames) - len(drift_opacity) - 1))
        drift_opacity.append(".94")
        drift_position_values = ";".join(drift_positions)
        drift_opacity_values = ";".join(drift_opacity)
        d = point_path(pts)
        parts.append(
            f'<path d="{d}" fill="none" stroke="{t["tux"]}" stroke-width="1" '
            'opacity=".94">'
            f'<animateTransform attributeName="transform" type="translate" begin="{INTRO_SECONDS}s" '
            f'dur="{loop_duration}s" repeatCount="indefinite" calcMode="linear" '
            f'keyTimes="{key_times}" values="{drift_position_values}"/>'
            f'<animate attributeName="opacity" begin="{INTRO_SECONDS}s" dur="{loop_duration}s" '
            f'repeatCount="indefinite" keyTimes="{key_times}" '
            f'values="{drift_opacity_values}"/></path>'
        )

    for i in range(n):
        positions = animate_values(frames, i)
        parts.append(
            f'<path d="M-.65-.65h1.3v1.3h-1.3z" fill="{t["tux"]}">'
            f'<animateTransform attributeName="transform" type="translate" begin="{INTRO_SECONDS}s" '
            f'dur="{loop_duration}s" repeatCount="indefinite" calcMode="linear" '
            f'keyTimes="{key_times}" values="{positions}"/>'
            f'<animate attributeName="opacity" begin="{INTRO_SECONDS}s" dur="{loop_duration}s" '
            f'repeatCount="indefinite" calcMode="linear" keyTimes="{key_times}" '
            f'values="{opacity_values}"/></path>'
        )

    for index, (name, particles) in enumerate(logo_hold_particles.items()):
        visible = ["0"] * len(frames)
        visible[index * 2 + 2] = ".82"
        visible[index * 2 + 3] = ".82"
        parts.append(
            f'<path d="{particle_path(particles)}" fill="none" stroke="{t["tux"]}" '
            'stroke-width="1" opacity="0">'
            f'<animate attributeName="opacity" begin="{INTRO_SECONDS}s" dur="{loop_duration}s" '
            f'repeatCount="indefinite" calcMode="linear" keyTimes="{key_times}" '
            f'values="{";".join(visible)}"/></path>'
        )
    parts.append("</g>")

    intro_ids = rng.integers(0, 60, size=len(portrait))
    order = rng.permutation(60)
    starts = np.empty(60)
    starts[order] = np.linspace(0.05, 1.2, 60)
    for group in range(60):
        pts = portrait[intro_ids == group]
        if not len(pts):
            continue
        parts.append(
            f'<path d="{point_path(pts)}" fill="none" stroke="{t["tux"]}" '
            'stroke-width="1" opacity="0">'
            f'<animate attributeName="opacity" begin="{num(starts[group])}s" dur=".8s" '
            'values="0;1" fill="freeze"/>'
            '<animate attributeName="opacity" begin="3.08s" dur=".12s" values="1;0" fill="freeze"/>'
            "</path>"
        )
    parts.extend(
        [
            "</g>",
            f'<text x="58" y="551" fill="{t["muted"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="10">'
            f'PTS {len(portrait):05d} · LINUX TUX SILHOUETTE</text>',
            f'<rect x="474" y="88" width="672" height="472" rx="6" fill="{t["panel2"]}" '
            f'stroke="{t["line"]}"/>',
            f'<path d="M474 124H1146" stroke="{t["line"]}"/>',
            f'<text x="490" y="111" fill="{t["chrome"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
            'font-weight="700" letter-spacing=".5">profile.yml</text>',
            f'<text x="580" y="111" fill="{t["muted"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">[YAML]</text>',
            f'<rect x="996" y="94" width="132" height="24" rx="12" fill="{t["chrome"]}" opacity=".16" '
            f'stroke="{t["chrome"]}"/>',
            f'<text x="1062" y="111" text-anchor="middle" fill="{t["chrome"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" '
            'font-weight="700">@Usuario444</text>',
        ]
    )

    row_y = 148.0
    for idx, (indent, key, value) in enumerate(YAML_ROWS, 1):
        line_num = f"{idx:2d}"
        if indent == 0:
            content = f'<tspan fill="{t["chrome"]}" font-weight="700">{html.escape(key)}:</tspan>'
            text_x = 525.0
        else:
            content = (
                f'<tspan fill="{t["portrait"]}">{html.escape(key)}: </tspan>'
                f'<tspan fill="{t["text"]}">{value}</tspan>'
            )
            text_x = 542.0

        parts.extend(
            [
                f'<text x="506" y="{num(row_y)}" text-anchor="end" fill="{t["muted"]}" opacity=".45" '
                'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13">'
                f"{line_num}</text>",
                f'<text x="{num(text_x)}" y="{num(row_y)}" '
                'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13">'
                f"{content}</text>",
            ]
        )
        row_y += 21.5

    parts.extend(
        [
            f'<path d="M474 526H1146" stroke="{t["line"]}"/>',
            f'<rect x="475" y="527" width="670" height="32" fill="{t["panel"]}" rx="0 0 5 5"/>',
            f'<rect x="485" y="533" width="72" height="20" rx="3" fill="{t["portrait"]}"/>',
            f'<text x="521" y="547" text-anchor="middle" fill="{t["bg"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11" font-weight="700">NORMAL</text>',
            f'<text x="569" y="547" fill="{t["text"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12" font-weight="600">profile.yml</text>',
            f'<text x="740" y="547" fill="{t["muted"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">[utf-8]</text>',
            f'<text x="1134" y="547" text-anchor="end" fill="{t["muted"]}" '
            'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">17L, 482B  100%  17:1</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    logos = make_logos()

    for index, theme in enumerate(THEMES):
        rng = np.random.default_rng(SEED + index)
        portrait = portrait_points(theme, rng)
        sampled = {
            name: sample_logo_points(image, rng, TRAVELLER_COUNT)
            for name, image in logos.items()
        }
        hold_particles = {
            name: sample_logo_points(image, rng, HOLD_PARTICLE_COUNT)
            for name, image in logos.items()
        }
        svg = render_svg(theme, portrait, sampled, hold_particles, rng)
        output = ASSETS / f"banner-{theme}.v12.svg"
        output.write_text(svg, encoding="utf-8")
        byte_size = output.stat().st_size
        print(
            f"Wrote {output.name}: {byte_size:,} bytes "
            f"({byte_size / 1024:.1f} KiB), {len(portrait):,} points"
        )


if __name__ == "__main__":
    main()
