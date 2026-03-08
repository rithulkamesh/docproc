"""
Scene 2: IntroScene
Documents (PDF, DOCX, PPTX, XLSX) flow into docproc and produce clean.md.
Gruvbox dark theme.
"""

from manim import *

# ── Gruvbox Dark Palette ─────────────────────────────────────────────────────
BG_COLOR = "#282828"   # gruvbox bg
FG       = "#ebdbb2"   # gruvbox fg
ACCENT   = "#83a598"   # gruvbox bright-blue  (docproc / engine)
GRAY     = "#928374"   # gruvbox gray
GREEN    = "#b8bb26"   # gruvbox bright-green (output)
ORANGE   = "#fe8019"   # gruvbox bright-orange
YELLOW   = "#fabd2f"   # gruvbox bright-yellow
RED_G    = "#fb4934"   # gruvbox bright-red
BLUE_G   = "#83a598"   # gruvbox bright-blue
DARK_BG  = "#1d2021"   # gruvbox bg-hard


# Per-format chip colors (all Gruvbox hues)
DOC_COLORS = {
    "PDF":  "#fb4934",  # bright-red
    "DOCX": "#83a598",  # bright-blue
    "PPTX": "#fe8019",  # bright-orange
    "XLSX": "#b8bb26",  # bright-green
}


class IntroScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── Document chips (left column) ─────────────────────────────────────
        doc_nodes = VGroup(
            *[self._doc_chip(lbl, col) for lbl, col in DOC_COLORS.items()]
        ).arrange(DOWN, buff=0.35)
        doc_nodes.to_edge(LEFT, buff=1.5)

        # ── docproc engine box (centre) ──────────────────────────────────────
        engine_rect = RoundedRectangle(
            corner_radius=0.15, width=2.6, height=1.1,
            fill_color=DARK_BG, fill_opacity=1,
            stroke_color=ACCENT, stroke_width=2.5,
        )
        engine_label = Text("docproc", font="Courier New", font_size=26,
                            color=ACCENT, weight=BOLD)
        engine_label.move_to(engine_rect.get_center())
        engine = VGroup(engine_rect, engine_label).move_to(ORIGIN)

        # ── clean.md output box (right) ──────────────────────────────────────
        out_rect = RoundedRectangle(
            corner_radius=0.15, width=2.6, height=1.1,
            fill_color=DARK_BG, fill_opacity=1,
            stroke_color=GREEN, stroke_width=2.5,
        )
        out_label = Text("clean.md", font="Courier New", font_size=24,
                         color=GREEN, weight=BOLD)
        out_label.move_to(out_rect.get_center())
        output = VGroup(out_rect, out_label).to_edge(RIGHT, buff=1.5)

        # ── Subtitle ─────────────────────────────────────────────────────────
        subtitle = Text(
            "Document  →  Markdown  →  AI",
            font="Helvetica", font_size=26, color=GRAY
        ).to_edge(DOWN, buff=0.55)

        # Subtle glow behind engine
        pulse = Circle(radius=1.4, stroke_width=0)
        pulse.set_fill(ACCENT, opacity=0.07)
        pulse.move_to(engine.get_center())

        # ── Animations ──────────────────────────────────────────────────────
        self.play(FadeIn(pulse, scale=0.6), FadeIn(engine, scale=0.85), run_time=0.7)

        self.play(
            LaggedStart(
                *[FadeIn(d, shift=RIGHT * 0.25) for d in doc_nodes],
                lag_ratio=0.2,
            ),
            run_time=0.8,
        )

        arrows_in = [
            Arrow(d.get_right(), engine_rect.get_left(), buff=0.12,
                  color=d[1].color, stroke_width=2,
                  max_tip_length_to_length_ratio=0.15)
            for d in doc_nodes
        ]
        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows_in], lag_ratio=0.15),
            run_time=1.0,
        )

        # Pulse engine
        self.play(
            pulse.animate.scale(1.15).set_fill(opacity=0.14),
            engine_label.animate.set_color(FG),
            run_time=0.4,
        )
        self.play(
            pulse.animate.scale(1 / 1.15).set_fill(opacity=0.07),
            engine_label.animate.set_color(ACCENT),
            run_time=0.4,
        )

        arrow_out = Arrow(
            engine_rect.get_right(), out_rect.get_left(),
            buff=0.12, color=GREEN, stroke_width=2.5,
            max_tip_length_to_length_ratio=0.15
        )
        self.play(GrowArrow(arrow_out), FadeIn(output, shift=RIGHT * 0.3), run_time=0.8)
        self.play(Write(subtitle), run_time=0.7)
        self.wait(1.2)

    @staticmethod
    def _doc_chip(label: str, color: str) -> VGroup:
        rect = RoundedRectangle(
            corner_radius=0.1, width=1.5, height=0.52,
            fill_color=DARK_BG, fill_opacity=1,
            stroke_color=color, stroke_width=1.8,
        )
        text = Text(label, font="Courier New", font_size=20, color=color, weight=BOLD)
        text.move_to(rect.get_center())
        return VGroup(rect, text)
