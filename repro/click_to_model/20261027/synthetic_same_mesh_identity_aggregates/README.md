# Same-mesh synthetic identity: actual two-stream result

Two actual neural runs each completed all 180 inputs (initialization 0 plus 179 tracking frames). These are synthetic development results. The partially occluded continuous stream retained target association on **177/179** tracking frames and matched neither instance on 2.

The ambiguous video was inferred **once**. Its identical RGB-D observations admit two persistent-ID worlds. Under `stay`, its output matched target on **132/179** and neither on 47. Under `exchange`, it matched target on **55/179**, distractor on **77/179**, and neither on 47. **All 70 post-opening tracking frames match the distractor in the exchange world.** These two worlds are alternate explanations of the same observations and output, not two independent videos or additional samples.

The reported adjacent identity-switch count is zero because the 47 neither-associated frames break the consecutive target/distractor chain. **Zero in this restricted switch counter does not mean identity was preserved.** The post-occlusion identity failure remains explicit.

Association uses full-original-vertex ADD <= 0.1 model diameter against each instance. `target` and `distractor` mean only that instance passes; `both` means both pass; `neither` means both finite errors fail; `unknown` means no qualified finite association. Initialization, all 180 inputs, all 179 tracking frames, and every fixed phase are shown separately. Hidden frames are never filtered out.

All seven actual trust flags are false. The product does not report an explicit abstention field; its missing value is retained as unknown reporting state. Unverified outputs and incorrect working poses are not successful abstentions. Verified-subset risk is null because that subset is empty.

`association_phase_counts.csv` contains 22 aggregate rows. `timing.csv` contains two actual runs, not three worlds. `source_hashes.csv` pins the evaluated receipts, stored-error rows, reports, and recomputation source. `experiment_scope.json` records denominator and scope details. No raw poses, GT matrices, point trajectories, images, or local paths are published.

Timings are instrumented measurements. Parent, child, initialization, registration, admission, and tracking are overlapping scopes and cannot be summed. Tracking FPS excludes final output serialization; parent-inclusive input throughput includes startup and orchestration. No deployment speed or solved-identity claim is made.

The aggregate check independently recomputed labels, clocks, phase summaries and all stored BOP/ADD threshold counts from saved per-frame errors. It did not reread GT or recompute geometry. It was implemented by the posthoc author with separate formulas; it is not external experimental replication.
