"""
Scene 5: ArchitectureScene
System architecture: User Upload → S3 → Queue → docproc Worker →
Markdown → Embeddings → PgVector → Study Tools
Gruvbox dark theme.
"""

from manim import *

# ── Gruvbox Dark Palette ─────────────────────────────────────────────────────
BG_COLOR = "#282828"   # gruvbox bg
FG       = "#ebdbb2"   # gruvbox fg
ACCENT   = "#83a598"   # gruvbox bright-blue
GRAY     = "#928374"   # gruvbox gray
DARK_BG  = "#1d2021"   # gruvbox bg-hard

# Each arch step gets its own Gruvbox hue (cycling through the bright palette)
ARCH_STEPS = [
    ("User Upload",     "#ebdbb2"),  # fg
    ("S3 Storage",      "#fe8019"),  # bright-orange
    ("Queue",           "#fabd2f"),  # bright-yellow
    ("docproc Worker",  "#83a598"),  # bright-blue  (accent)
    ("Markdown",        "#b8bb26"),  # bright-green
    ("Embeddings",      "#8ec07c"),  # bright-aqua
    ("PgVector",        "#d3869b"),  # bright-purple
    ("Study Tools",     "#fb4934"),  # bright-red
]


class ArchitectureScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        mid = len(ARCH_STEPS) // 2       # 4
        left_steps  = ARCH_STEPS[:mid]
        right_steps = ARCH_STEPS[mid:]

        left_nodes  = [self._arch_box(*s) for s in left_steps]
        right_nodes = [self._arch_box(*s) for s in right_steps]

        left_col  = VGroup(*left_nodes).arrange(DOWN, buff=0.48)
        right_col = VGroup(*right_nodes).arrange(DOWN, buff=0.48)

        left_col.to_edge(LEFT, buff=2.0)
        right_col.to_edge(RIGHT, buff=2.0)
        right_col.align_to(left_col, UP)

        left_arrows  = self._chain_arrows(left_nodes,  GRAY)
        right_arrows = self._chain_arrows(right_nodes, GRAY)

        bridge = CurvedArrow(
            left_nodes[-1].get_right(),
            right_nodes[0].get_left(),
            color=ACCENT,
            stroke_width=2.2,
            angle=-TAU / 6,
        )

        # ── Title ─────────────────────────────────────────────────────────────
        title = Text("System Architecture", font="Helvetica",
                     font_size=30, color=FG, weight=BOLD)
        title.to_edge(UP, buff=0.45)
        underline = Line(
            title.get_left(), title.get_right(),
            stroke_color=ACCENT, stroke_width=2,
        ).next_to(title, DOWN, buff=0.08)

        # ── Animations ──────────────────────────────────────────────────────
        self.play(Write(title), Create(underline), run_time=0.7)

        self.play(FadeIn(left_nodes[0], shift=RIGHT * 0.15), run_time=0.35)
        for i, arrow in enumerate(left_arrows):
            self.play(
                GrowArrow(arrow),
                FadeIn(left_nodes[i + 1], shift=RIGHT * 0.15),
                run_time=0.32,
            )

        self.play(Create(bridge), run_time=0.5)

        self.play(FadeIn(right_nodes[0], shift=LEFT * 0.15), run_time=0.35)
        for i, arrow in enumerate(right_arrows):
            self.play(
                GrowArrow(arrow),
                FadeIn(right_nodes[i + 1], shift=LEFT * 0.15),
                run_time=0.32,
            )

        self.wait(1.2)

    @staticmethod
    def _arch_box(label: str, color: str) -> VGroup:
        rect = RoundedRectangle(
            corner_radius=0.12, width=2.5, height=0.62,
            fill_color=DARK_BG, fill_opacity=1,
            stroke_color=color, stroke_width=2,
        )
        text = Text(label, font="Helvetica", font_size=20, color=color, weight=BOLD)
        text.move_to(rect.get_center())
        return VGroup(rect, text)

    @staticmethod
    def _chain_arrows(nodes: list, color: str) -> list:
        return [
            Arrow(
                nodes[i].get_bottom(), nodes[i + 1].get_top(),
                buff=0.08, color=color, stroke_width=1.8,
                max_tip_length_to_length_ratio=0.2,
            )
            for i in range(len(nodes) - 1)
        ]
