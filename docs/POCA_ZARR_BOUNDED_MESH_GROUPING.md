# Bounded shared-mesh grouping investigation

2026-11-03. D279 source inspection; design boundary only, not implemented or measured.

ObjectListMultiObjectDisplayCommand::rebuild currently owns one combined set of position/normal/feature/index buffers, retained local vectors, model-index vectors, model texture and ListDrawRange/ObjectFeatureRange tables for the whole collection. It assembles by list index across children. A successful display marks the entire ObjectLists/ObjectList family handled, suppressing all child rendering. Features, transformed buffers, ellipsoids, picking, sorting and cleanup depend on those global ranges.

The 256 MiB preflight charges six copies of the estimated complete payload before flattening. This bounds additional CPU scratch/combined/retained/GPU exposure conservatively while individual child resources coexist. Raising it would not demonstrate available GPU memory. Splitting the draw call alone would still construct the large combined vectors and retain all buffer sets, so it does not fix the peak.

Existing per-child resources and the frame-local CollectionRenderPlan provide useful building blocks, but this renderer does not currently own independent bounded groups or claim per-list/sub-element ownership. Creating multiple borrowed MyMultipleObject wrappers would conflict with child ownership and command dispatch; that is not a safe shortcut.

A coherent follow-up would extend this renderer with an owned batch-resource structure grouping buffers, retained vectors, source identities, per-child transforms and local ranges. Metadata-only planning must bound each group before generation, build one group at a time without a whole-scene temporary, and enforce a TOTAL admitted budget as well as a per-group peak. N groups of 256 MiB are not a bounded solution.

It must also cover partial admission: claim exactly the lists/sub-elements actually drawn, leave oversized/unadmitted children to their existing individual commands, preserve cross-group transparent order and per-child settings, and propagate feature/selection/LUT/normal/ellipsoid/transform changes to the correct resources. Context loss, picking, cancellation and disposal need the same ownership guarantees as current commands. Meshes larger than one admissible group still require their individual path or a separately designed chunk strategy.

Selected-only packing alone is insufficient: later selection changes need invalidation/replanning and may enlarge the admitted payload; current geometry size/offsets span hidden lists too. Reuse of existing buffers could save duplication, but shared draw submission cannot assume arbitrary child buffers are laid out compatibly.

This exceeds a small local rebuild correction and has therefore been kept separate as requested. Before implementation, obtain per-family build/upload/resident bytes and representative selected meshes, establish a total working-set target against actual hardware, then test transforms/settings/selection, transparency, partial fallback and context cleanup. The current source change neither raises the ceiling nor changes shared-mesh ownership.