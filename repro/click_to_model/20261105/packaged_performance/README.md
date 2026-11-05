# Current native CLI performance experiments

Both actual fixed30 runs completed all inputs with 45 callbacks and 29 committed updates. Each passed 621 numeric and 1,146 structural comparisons against the original CLI. Every full-model state check remains: 221 total, 220 in tracking, with 134 tensors each. The optional adapter uses the previously tested batched algorithm at the current packaged CLI boundaries; it is not a new tracking algorithm.

Recorded 29-frame tracking elapsed: control 19.811279 s, batched 18.166096 s. The control overlapped the CPU storage experiment; therefore these are confounded descriptive measurements, not a causal speedup estimate. Both arms have identical outer-observer topology. No GT or accuracy evaluation was added.

The independent CPU storage experiment covers all 787 saved NPZ archives and 9,435 array members, three rotated-order rounds and 7,083 successful write/readback trials. Every decompressed NPY member and array descriptor/value byte matches. All original per-method outcomes are retained in the summary. Default compression, DEFLATE level 1 and stored encoding were compared; timings include writes and ZIP close but no fsync. The host was not isolated.

Native evidence alone averages about 4.167 s with default compression and 0.192 s stored, with only 8,146,384 extra bytes. Motion evidence grows much more when stored. Thus the next proposed runtime experiment changes only the native evidence codec. This derived tradeoff is not yet an online tracking result.

This package contains numeric summaries, limitations and content hashes only. It includes no source, images, weights, raw poses, trajectories or GT. New code remains local and default runtime behavior remains unchanged.
