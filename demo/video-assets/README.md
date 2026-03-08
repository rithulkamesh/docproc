# docproc Demo – Manim Animation Scenes

Animated scenes built with **Manim Community Edition** for the `docproc` developer tool demo.

## Scenes

| File | Class | Description |
|------|-------|-------------|
| `scene1_problem.py` | `ProblemScene` | Broken parser pipeline with red error highlights |
| `scene2_intro.py`   | `IntroScene`   | PDF/DOCX/PPTX/XLSX → docproc → clean.md |
| `scene3_vision.py`  | `VisionScene`  | Image & equation → Vision Model → LaTeX/caption |
| `scene4_pipeline.py`| `PipelineScene`| Full AI pipeline with Chat/Notes/Flashcards output |
| `scene5_architecture.py` | `ArchitectureScene` | Two-column system architecture diagram |

## Requirements

```bash
pip install manim
```

## Render a single scene

```bash
# High quality (1080p, 60 fps)
manim -qh scene1_problem.py ProblemScene

# Low quality preview
manim -ql scene1_problem.py ProblemScene
```

## Render all scenes at once

```bash
bash render_all.sh        # high quality (default)
bash render_all.sh -ql    # low quality (fast preview)
bash render_all.sh -qk    # 4K
```

Output MP4 files are written to `./media/videos/`.

## Visual Style

- **Background**: `#0b0b0b`
- **Accent**: `#6366F1` (indigo)
- **Error**: `#ef4444` (red)
- **Text**: white
- **Fonts**: Helvetica (labels), Courier New (code/filenames)
