# Native evidence storage: six actual 30-input trials

Three control/stored pairs use the predeclared order: control_0, stored_0, stored_1, control_1, control_2, stored_2. Every run completed 30 inputs, 29 tracking frames, 45 callbacks and 29 committed updates. Each independent audit reported 621 exact numeric and 1,146 exact structural comparisons against the reviewed ordinary CLI reference.

Only native evidence NPZ uses ZIP_STORED in the candidate arm; motion and other product archives keep NumPy default DEFLATE. This export independently reduces each saved 29-element latency array, rehashes every NPZ against its audit pin, and counts ZIP members, actual codecs and archive sizes without decoding member arrays. The JSON gives all six measurements, each paired difference, and three-repeat mean/min/max. Negative stored-minus-control timing means less elapsed time.

These are descriptive measurements on one development prefix. There is no significance, generalization or deployment-FPS claim. No heavy compression benchmark ran concurrently with these six GPU trials, but desktop/system activity was not isolated; the first pair followed a 76-check CPU contract, and small preparation/parity checks occurred around trials. Read the distinct tracking, child and parent timing boundaries in summary.json. Writes exclude fsync. Archive size excludes non-NPZ outputs. Empty archives do not expose a member codec.

The package contains aggregates and source/artifact hashes only. It contains no raw trajectories, pose arrays, images, model weights or source code. The export performs no GT read, model evaluation, GPU work or upload. Independent neural/gradient recomputation is outside the parity scope.
