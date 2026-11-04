#!/usr/bin/env python3
"""
Generate the PF AI Lab 5.4.0 application icon (pf_algo.ico).
Uses Pillow to create an icon matching PleneufTrading.com branding.

Palette extracted from https://pleneuftrading.com/:
  - Navy:  #004C8C  (primary text, data points)
  - Teal:  #00B4D8  (chart gradient start/end)
  - Cyan:  #00C8C8  (chart gradient mid, "TRADING" text)
  - Green: #00D084  (bullish data points)
  - White: #FFFFFF  (text on dark backgrounds)

The icon reproduces the site's upward chart-line motif with "PF" text.
"""

import sys
import os


def generate_icon(output_path='pf_algo.ico'):
    """Generate a multi-size .ico file matching PleneufTrading branding."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("[WARN] Pillow not installed. Skipping icon generation.")
        print("       Install with: pip install Pillow")
        return False

    # ── PleneufTrading brand colors ──
    NAVY    = (0, 76, 140, 255)      # #004C8C
    TEAL    = (0, 180, 216, 255)     # #00B4D8
    CYAN    = (0, 200, 200, 255)     # #00C8C8
    GREEN   = (0, 208, 132, 255)    # #00D084
    WHITE   = (255, 255, 255, 255)
    BG_DARK = (0, 38, 70, 255)       # dark navy background

    sizes = [16, 32, 48, 64, 128, 256]
    images = []

    for size in sizes:
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        padding = max(1, size // 16)
        corner_radius = size // 6

        # ── Background: rounded navy rectangle ──
        draw.rounded_rectangle(
            [padding, padding, size - padding, size - padding],
            radius=corner_radius,
            fill=BG_DARK
        )

        # ── Teal accent bar at top ──
        bar_h = max(2, size // 12)
        draw.rounded_rectangle(
            [padding, padding, size - padding, padding + bar_h],
            radius=corner_radius,
            fill=TEAL
        )

        # ── Chart line (upward trend) — the PleneufTrading signature ──
        if size >= 32:
            # Define chart points as fractions of icon size
            # Mimics the SVG: M20,50 L50,35 L80,45 L100,15 L120,45 L150,35 L180,50
            chart_points_norm = [
                (0.15, 0.55),
                (0.30, 0.42),
                (0.45, 0.50),
                (0.55, 0.22),  # peak
                (0.65, 0.50),
                (0.78, 0.42),
                (0.88, 0.55),
            ]
            chart_points = [
                (int(px * size), int(py * size))
                for px, py in chart_points_norm
            ]

            # Chart line (teal gradient approximation → use cyan)
            line_w = max(1, size // 18)
            draw.line(chart_points, fill=CYAN, width=line_w, joint='curve')

            # Data point dots
            dot_colors = [TEAL, NAVY, GREEN, NAVY, GREEN, NAVY, TEAL]
            dot_r = max(1, size // 28)
            for (cx, cy), color in zip(chart_points, dot_colors):
                # Use lighter versions for visibility on dark background
                visible_color = color
                if color == NAVY:
                    visible_color = TEAL  # navy dots invisible on dark bg
                draw.ellipse(
                    [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
                    fill=visible_color
                )

            # Peak highlight (larger dot at the highest point)
            peak = chart_points[3]
            peak_r = max(2, size // 20)
            draw.ellipse(
                [peak[0] - peak_r, peak[1] - peak_r,
                 peak[0] + peak_r, peak[1] + peak_r],
                fill=GREEN
            )

        # ── "PF" text (bottom area, white on navy) ──
        if size >= 32:
            try:
                font_size = max(8, size // 4)
                font_names = [
                    'arialbd.ttf', 'Arial Bold.ttf', 'arial.ttf',
                    'DejaVuSans-Bold.ttf', 'DejaVuSans.ttf',
                    'LiberationSans-Bold.ttf', 'FreeSans.ttf',
                ]
                font = None
                for fn in font_names:
                    try:
                        font = ImageFont.truetype(fn, font_size)
                        break
                    except (OSError, IOError):
                        continue
                if font is None:
                    font = ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()

            text = "PF"
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = (size - tw) // 2
            ty = int(size * 0.65)

            # Only draw if it fits above the bottom padding
            if ty + th < size - padding:
                draw.text((tx, ty), text, fill=WHITE, font=font)

        # ── Thin green bottom accent line ──
        if size >= 32:
            bottom_y = size - padding - max(1, size // 20)
            draw.line(
                [(padding + size // 6, bottom_y),
                 (size - padding - size // 6, bottom_y)],
                fill=GREEN, width=max(1, size // 32)
            )

        images.append(img)

    # Save as .ico with multiple sizes
    # Pillow ICO requires the largest image first for proper multi-size output
    images_reversed = list(reversed(images))
    images_reversed[0].save(
        output_path,
        format='ICO',
        append_images=images_reversed[1:]
    )
    print(f"[OK] Icon generated: {output_path}")
    print(f"     Branding: PleneufTrading.com palette")
    print(f"     Navy #004C8C | Teal #00B4D8 | Cyan #00C8C8 | Green #00D084")
    return True


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    output = os.path.join(script_dir, 'pf_algo.ico')
    if generate_icon(output):
        print(f"     Size: {os.path.getsize(output)} bytes")
    else:
        sys.exit(1)
