"""
Scene 1: ProblemScene
Shows a typical broken document-parsing pipeline with red error highlights.
Gruvbox dark theme.
"""

from manim import *

# ── Gruvbox Dark Palette ─────────────────────────────────────────────────────
BG_COLOR = "#282828"   # gruvbox bg
FG       = "#ebdbb2"   # gruvbox fg (warm cream)
ACCENT   = "#83a598"   # gruvbox bright-blue
ERROR    = "#fb4934"   # gruvbox bright-red
GRAY     = "#928374"   # gruvbox gray
DARK_BG  = "#1d2021"   # gruvbox bg-hard (inner fills)


class ProblemScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── Pipeline nodes ──────────────────────────────────────────────────
        pdf_box    = self._node("PDF",             color=ACCENT)
        parser_box = self._node("Typical\nParser", color=GRAY)
        output_box = self._node("Broken\nMarkdown",color=ERROR)

        pdf_box.to_edge(LEFT, buff=1.2)
        parser_box.move_to(ORIGIN)
        output_box.to_edge(RIGHT, buff=1.2)

        arrow1 = Arrow(
            pdf_box.get_right(), parser_box.get_left(),
            buff=0.15, color=GRAY, stroke_width=2.5
        )
        arrow2 = Arrow(
            parser_box.get_right(), output_box.get_left(),
            buff=0.15, color=ERROR, stroke_width=2.5
        )

        pipeline = VGroup(pdf_box, parser_box, output_box, arrow1, arrow2)
        pipeline.shift(UP * 0.8)

        # ── Broken output lines ──────────────────────────────────────────────
        broken_lines = VGroup(
            Text("Equation:  ???",          font="Courier New", font_size=22, color=ERROR),
            Text("Figure:    image_42.png", font="Courier New", font_size=22, color=ERROR),
            Text("Table:     [missing]",    font="Courier New", font_size=22, color=ERROR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        broken_lines.to_edge(DOWN, buff=1.8)

        # Dashed border around broken block (DashedVMobject is the CE 0.20 API)
        border_solid = SurroundingRectangle(
            broken_lines, color=ERROR, stroke_width=1.5,
            corner_radius=0.1, buff=0.2
        )
        border = DashedVMobject(border_solid, num_dashes=28, dashed_ratio=0.55)

        # ── Caption ─────────────────────────────────────────────────────────
        caption = Text(
            "Structure is lost", font="Helvetica", font_size=28,
            color=FG, weight=BOLD
        ).to_edge(DOWN, buff=0.5)

        # ── Cross icon on output box ─────────────────────────────────────────
        cross = Cross(output_box, stroke_color=ERROR, stroke_width=3, scale_factor=0.35)
        cross.move_to(output_box.get_center())

        # ── Animations ──────────────────────────────────────────────────────
        self.play(FadeIn(pdf_box, shift=RIGHT * 0.3), run_time=0.5)
        self.play(GrowArrow(arrow1), FadeIn(parser_box, shift=RIGHT * 0.3), run_time=0.6)
        self.play(GrowArrow(arrow2), FadeIn(output_box, shift=RIGHT * 0.3), run_time=0.6)
        self.play(Create(cross), run_time=0.4)
        self.wait(0.3)

        self.play(
            LaggedStart(
                *[FadeIn(line, shift=LEFT * 0.2) for line in broken_lines],
                lag_ratio=0.25,
            ),
            run_time=1.0,
        )
        self.play(Create(border), run_time=0.5)
        self.play(Write(caption), run_time=0.8)
        self.wait(1.2)

    # ── helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _node(label: str, color: str = FG) -> VGroup:
        rect = RoundedRectangle(
            corner_radius=0.12, width=2.2, height=0.9,
            fill_color=DARK_BG, fill_opacity=1,
            stroke_color=color, stroke_width=2,
        )
        text = Text(label, font="Helvetica", font_size=22, color=color, weight=BOLD)
        text.move_to(rect.get_center())
        return VGroup(rect, text)
