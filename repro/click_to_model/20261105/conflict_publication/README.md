# Lossless aggregate shards, publication format v2

This package contains exactly the same numerical results as the original aggregate package. It is a storage-format change only: **41 newly published candidates, including 2 additional errors at MSSD < 0.05D and 29 at MSPD < 5 pixels**, remain unchanged. No pose, threshold, protocol, count, rate, association or trust statistic changed. The rule remains an output-only development experiment and is not the production default. Read `original_README.md` for the full result and limitations, including the absent real secondary bridge in the historical synthetic controls.

Every file is strictly smaller than 70,000 bytes. The 3.52 MB original package is represented with repeated JSON object field names stored once as ordered column schemas, and CSVs split by case/world while preserving original row order. No compression utility, binary file or base64 payload is required. No source code is included.

## Summary representation

`summary.schemas.json` lists ordered object-key columns. Each `summary.caseNN.json` encodes one original case/world: an object is represented by `s` (its column-schema index) and `v` (its ordered values). Values are recursively encoded by the same rule; ordinary arrays and JSON scalars keep their original types. Pair the schema's keys with the encoded values recursively to recover the object. `summary.metadata.json` contains every remaining original summary field. `package_index.json` records case order and the original root key order, allowing exact reconstruction of `summary.json`.

All original statistics are retained: 21 thresholds, all case/world/scope combinations, three variants, six-cell error/publication counts, complete paired counts, risk bounds, coverage, synthetic associations and per-scope trust counts. Undefined synthetic BOP scopes remain explicitly undefined with null rates. Hidden stay/exchange still refer to the same observation stream and are not independent videos.

## CSV representation

`threshold_counts.caseNN.partNN.csv` contains the original threshold rows for that case/world; all parts retain the original header. The corresponding `paired_transitions.caseNN.partNN.csv` files retain the complete paired rows. To reconstruct either original CSV, use its first header once and append the data rows of each listed part in `package_index.json` order. Preserve the original CRLF line endings. There are exactly **2,016 threshold variant rows** and **672 paired threshold/scope rows**.

## Provenance and equivalence

`package_index.json` binds all six original files by exact byte size and SHA-256. `original_manifest.json`, `original_README.md` and `sources.json` are byte-identical copies from the original package. The original summary's exact formatting is two-space JSON indentation with one final newline. Independent reconstruction has reproduced **all six original file SHA-256 hashes exactly**, including every CSV byte and the 3,016,945-byte original summary. The new `manifest.json` binds every current file.

This package contains only existing aggregate numerics, schema/column metadata, paired counts, provenance hashes and explanations. It contains no per-frame trajectory, pose matrix, raw GT, image, weight, model, log or source code. The associated local round-trip audit was performed solely on the two public packages and did not reopen experiment labels or raw GT.
