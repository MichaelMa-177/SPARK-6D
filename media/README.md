# SPARK-6D comparison videos

Both videos are wall-clock aligned: a method that produces poses slower than
30 FPS holds its latest result, while a faster method advances to the newest
pose available at each output timestamp.

| File | Comparison | Resolution | FPS | Duration | SHA-256 |
|---|---|---:|---:|---:|---|
| `spark6d_official_engineering_comparison.mp4` | Official / engineered official / SPARK-6D | 1920×480 | 30 | 53.90 s | `cac5c22165524224e7c464f03fa8633a2e8e2f68f8aaee69f5744dba8eff33c6` |
| `spark6d_foundationpose_realtime_comparison.mp4` | Official / SPARK-6D / FoundationPose++ | 1920×480 | 30 | 74.27 s | `d871129b5832990a2897e6fe80691c635783b991d7d3efa9f40856c19b42beb7` |

Detailed protocols and machine-readable outputs remain in the local
`results/official_engineering_comparison_2026-08-20/` and
`results/demo_comparison_2026-08-20/` directories; those large generated
directories are not included in the Git repository.
