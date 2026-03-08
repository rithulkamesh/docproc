"""
Scene 4: PipelineScene
Animated AI pipeline: Document → docproc → Markdown → Embeddings → Vector DB → LLM
Then output cards: Chat, Notes, Flashcards, Assessments.
Gruvbox dark theme.
"""

from manim import *

# ── Gruvbox Dark Palette ─────────────────────────────────────────────────────
BG_COLOR = "#282828"   # gruvbox bg
FG       = "#ebdbb2"   # gruvbox fg
ACCENT   = "#83a598"   # gruvbox bright-blue
GRAY     = "#928374"   # gruvbox gray
GREEN    = "#b8bb26"   # gruvbox bright-green
AQUA     = "#8ec07c"   # gruvbox bright-aqua
YELLOW   = "#fabd2f"   # gruvbox bright-yellow
PURPLE   = "#d3869b"   # gruvbox bright-purple
ORANGE   = "#fe8019"   # gruvbox bright-orange
DARK_BG  = "#1d2021"   # gruvbox bg-hard


# Pipeline step (label, gruvbox color)
PIPELINE_STEPS = [
    ("Document",   FG),
    ("docproc",    ACCENT),
    ("Markdown",   GREEN),
    ("Embeddings", AQUA),
    ("Vector DB",  PURPLE),
    ("LLM",        YELLOW),
]

OUTPUT_COLOR = ORANGE


class PipelineScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── Build pipeline column ────────────────────────────────────────────
        nodes  = [self._pipeline_node(label, color) for label, color in PIPELINE_STEPS]
        pipe_group = VGroup(*nodes).arrange(DOWN, buff=0.55)
        pipe_group.scale(0.95).to_edge(LEFT, buff=2.2)

        arrows = [
            Arrow(
                nodes[i].get_bottom(), nodes[i + 1].get_top(),
                buff=0.08, color=GRAY, stroke_width=2,
                max_tip_length_to_length_ratio=0.18,
            )
            for i in range(len(nodes) - 1)
        ]

        # ── Output cards (right side) ────────────────────────────────────────
        cards = VGroup(
            *[self._output_card(lbl) for lbl in ["Chat", "Notes", "Flashcards", "Assessments"]]
        ).arrange(DOWN, buff=0.32)
        cards.to_edge(RIGHT, buff=1.8)

        bridge_arrow = Arrow(
            nodes[-1].get_right(), cards.get_left(),
            buff=0.25, color=OUTPUT_COLOR, stroke_width=2.5,
            max_tip_length_to_length_ratio=0.15,
        )

        outputs_label = Text("Outputs", font="Helvetica", font_size=22,
                             color=GRAY, weight=BOLD)
        outputs_label.next_to(cards, UP, buff=0.3)

        # ── Animate pipeline ─────────────────────────────────────────────────
        self.play(FadeIn(nodes[0], scale=0.85), run_time=0.4)
        for i, arrow in enumerate(arrows):
            self.play(
                GrowArrow(arrow),
                FadeIn(nodes[i + 1], scale=0.85),
                run_time=0.38,
            )

        self.wait(0.2)
        self.play(GrowArrow(bridge_arrow), Write(outputs_label), run_time=0.6)
        self.play(
            LaggedStart(
                *[FadeIn(card, shift=RIGHT * 0.2) for card in cards],
                lag_ratio=0.2,
            ),
            run_time=1.0,
        )
        self.wait(1.2)

    @staticmethod
    def _pipeline_node(label: str, color: str) -> VGroup:
        rect = RoundedRectangle(
            corner_radius=0.1, width=2.1, height=0.6,
            fill_color=DARK_BG, fill_opacity=1,
            stroke_color=color, stroke_width=2,
        )
        text = Text(label, font="Helvetica", font_size=21, color=color, weight=BOLD)
        text.move_to(rect.get_center())
        return VGroup(rect, text)

    @staticmethod
    def _output_card(label: str) -> VGroup:
        rect = RoundedRectangle(
            corner_radius=0.12, width=2.4, height=0.58,
            fill_color=DARK_BG, fill_opacity=1,
            stroke_color=OUTPUT_COLOR, stroke_width=1.8,
        )
        text = Text(label, font="Helvetica", font_size=22, color=FG, weight=BOLD)
        text.move_to(rect.get_center())
        return VGroup(rect, text)
