# Project-specific user requirements

## Human-selected first-frame SAM prompts

For any new first-frame SAM / SAM 2 / SAM2D segmentation in this project:

- Present the actual first RGB frame through an interactive tool and let Nelson personally choose the initial prompt. Do not click on the user's behalf, automatically select a foreground point, or derive the prompt from a GT mask.
- Do not run prompt-dependent segmentation before the user's submitted click is available. An already saved user prompt may be reused for reproducing that exact sequence and frame without asking the user to click again.
- Persist the user-selected pixel coordinates and labels, source sequence and frame, original image size and SHA256, paired depth and camera metadata, and the selected SAM model/configuration. Record any additional user correction prompts as well.
- The interactive tool must accumulate multiple positive and negative points, support undo, and update the predicted mask after each edit. Do not replace this workflow with a one-point submission followed by another chat turn. Persist the final point set and the user-confirmed mask candidate for exact reproduction.
- Repeated experiments and comparisons must reuse the saved user prompt unchanged. A different frame, image hash, object, or changed prompt needs a new user selection or explicit user instruction.
- Keep automatically generated legacy clicks separate and label their provenance honestly; do not relabel them as user clicks.
- For RGB-D point-cloud previews, use the resulting predicted mask, paired registered depth, and supplied camera intrinsics. Exclude invalid depth and state coordinate units; do not silently substitute GT masks, CAD geometry, or inferred back surfaces.
- Every generated "目标彩色三维点云 · 仅首帧实测可见表面" result must default to an in-page interactive 3D viewer: drag to rotate, wheel to zoom, right-drag to pan, and reset view. Keep RGB colors and a uniform spatial scale. Do not show only a static PNG or require a separate viewer page; static output is an export/fallback, not the primary result.

Established by user request on 2026-10-18. First interactive run: `experiments/2026-10-18-drill-user-prompt/`.
