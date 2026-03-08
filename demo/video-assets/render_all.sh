#!/usr/bin/env bash
# render_all.sh  –  Render all five docproc demo scenes with Manim CE
# Usage:  bash render_all.sh [quality_flag]
# Quality flags: -ql (low), -qm (medium), -qh (high, default), -qk (4K)

set -e
Q=${1:--qh}
DIR="$(cd "$(dirname "$0")" && pwd)"

SCENES=(
    "scene1_problem.py:ProblemScene"
    "scene2_intro.py:IntroScene"
    "scene3_vision.py:VisionScene"
    "scene4_pipeline.py:PipelineScene"
    "scene5_architecture.py:ArchitectureScene"
)

echo "▶ Rendering docproc demo scenes (quality: $Q) …"
for entry in "${SCENES[@]}"; do
    FILE="${entry%%:*}"
    CLASS="${entry##*:}"
    echo "  → $CLASS"
    manim "$Q" "$DIR/$FILE" "$CLASS"
done

echo ""
echo "✅  All scenes rendered. Output in: $DIR/media/"
