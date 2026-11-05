# Complete-state hash transfer microbenchmark

The original refiner checkpoint contains 134 tensors (68,175,248 bytes). On the RTX 4070 SUPER, 30 measured calls per method in three alternating blocks produced mean 61.557659 ms for the original observer and 46.118990 ms for the batched observer (25.0800% lower latency). All 78 calls including warmup reproduced every original per-tensor digest exactly. Every call rehashed all bytes; no digest was cached and no check was omitted.

Both observers hash dtype, shape and the same native tensor bytes. The batched observer concatenates native bytes by device within a 64 MiB transfer bound and preserves the original tensor order and fallback for oversized values. Timing includes CUDA synchronization around each complete observer call; checkpoint loading is excluded. The three repeats share one process. These are descriptive timings on checkpoint tensors, not a live tracker, end-to-end speedup or accuracy result. Actual runtime integration and pose/update parity remain pending.

The package contains only numerical summaries, protocol, source hashes and this description. It contains no weights, source code, images, point trajectories or raw poses.
