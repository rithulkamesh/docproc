"""
Scene 3: VisionScene
diagram.png → Vision Model → caption text, plus equation image → text.
Gruvbox dark theme.
"""

from manim import *

# ── Gruvbox Dark Palette ─────────────────────────────────────────────────────
BG_COLOR = "#282828"   # gruvbox bg
FG       = "#ebdbb2"   # gruvbox fg
ACCENT   = "#83a598"   # gruvbox bright-blue
GRAY     = "#928374"   # gruvbox gray
AQUA     = "#8ec07c"   # gruvbox bright-aqua  (output caption)
YELLOW   = "#fabd2f"   # gruvbox bright-yellow (eq label)
DARK_BG  = "#1d2021"   # gruvbox bg-hard


class VisionScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── Image placeholder ────────────────────────────────────────────────
        img_rect = Rectangle(
            width=2.2, height=1.6,
            fill_color=DARK_BG, fill_opacity=1,
            stroke_color=GRAY, stroke_width=2,
        )
        img_icon  = Text("🖼", font_size=36).move_to(img_rect.get_center())
        img_label = Text("diagram.png", font="Courier New", font_size=18, color=GRAY)
        img_label.next_to(img_rect, DOWN, buff=0.18)
        image_group = VGroup(img_rect, img_icon, img_label)

        # ── Vision model box ─────────────────────────────────────────────────
        vision_rect = RoundedRectangle(
            corner_radius=0.14, width=2.6, height=1.0,
            fill_color=DARK_BG, fill_opacity=1,
            stroke_color=ACCENT, stroke_width=2.2,
        )
        vision_label = Text("Vision Model", font="Helvetica", font_size=22,
                            color=ACCENT, weight=BOLD)
        vision_label.move_to(vision_rect.get_center())
        vision_box = VGroup(vision_rect, vision_label)

        # ── Markdown caption output ──────────────────────────────────────────
        md_out = Text(
            '> Figure: Electric field\n  around a point charge',
            font="Courier New", font_size=19, color=AQUA,
        )
        md_bg = BackgroundRectangle(md_out, fill_color=DARK_BG, buff=0.2, fill_opacity=1)
        md_group = VGroup(md_bg, md_out)

        # ── Equation image placeholder ───────────────────────────────────────
        eq_rect = Rectangle(
            width=2.6, height=0.8,
            fill_color=DARK_BG, fill_opacity=1,
            stroke_color=GRAY, stroke_width=1.5,
        )
        eq_img_label = Text("equation.png", font="Courier New", font_size=18, color=GRAY)
        eq_img_label.move_to(eq_rect.get_center())
        eq_img_group = VGroup(eq_rect, eq_img_label)

        # Equation text target (no LaTeX compiler needed — Text uses Pango/Cairo)
        latex_out = Text(
            "E = q / (4π ε₀ r²)",
            font="Courier New", font_size=28, color=FG
        )

        # ── Layout ──────────────────────────────────────────────────────────
        # Top row arrangement
        top_row = VGroup(image_group, vision_box, md_group)
        top_row.arrange(RIGHT, buff=1.0)
        top_row.shift(UP * 1.2)

        arrow_iv = Arrow(
            img_rect.get_right(), vision_rect.get_left(),
            buff=0.15, color=GRAY, stroke_width=2,
        )
        arrow_vm = Arrow(
            vision_rect.get_right(), md_group.get_left(),
            buff=0.15, color=AQUA, stroke_width=2,
        )

        # Bottom row arrangement
        bottom_row = VGroup(eq_img_group, latex_out)
        bottom_row.arrange(RIGHT, buff=3.0)
        bottom_row.shift(DOWN * 1.5)

        arrow_eq = Arrow(
            eq_rect.get_right(), latex_out.get_left(),
            buff=0.2, color=YELLOW, stroke_width=2,
        )
        # Extra space guarantees it doesn't render as 'LaTeXextraction'
        eq_label = Text("LaTeX  extraction", font="Helvetica", font_size=18, color=GRAY)
        eq_label.next_to(arrow_eq, UP, buff=0.15)

        # ── Divider ──────────────────────────────────────────────────────────
        divider = Line(LEFT * 6, RIGHT * 6, stroke_color=GRAY,
                       stroke_width=0.8, stroke_opacity=0.35)
        divider.shift(DOWN * 0.55)

        # ── Animations ──────────────────────────────────────────────────────
        self.play(FadeIn(image_group, shift=RIGHT * 0.2), run_time=0.6)
        self.play(GrowArrow(arrow_iv), FadeIn(vision_box, scale=0.9), run_time=0.7)
        self.play(GrowArrow(arrow_vm), FadeIn(md_group, shift=RIGHT * 0.2), run_time=0.7)
        self.wait(0.3)

        self.play(Create(divider), run_time=0.3)
        self.play(FadeIn(eq_img_group, shift=RIGHT * 0.2), run_time=0.5)
        self.play(GrowArrow(arrow_eq), Write(eq_label), run_time=0.6)
        self.play(
            ReplacementTransform(eq_img_group.copy(), latex_out),
            run_time=1.0,
        )
        self.wait(1.2)
