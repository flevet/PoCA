# OME-Zarr positioning preflight: progress and metadata profiling

2026-11-03. Source delivery; runtime acceptance and performance comparison pending.

## Original symptom and behavior

The 0 / 0 ready label came from start() sending a reconstruction-progress signal before metadata was known. PocaZarrPositioning::read() then scanned all children without intermediate notifications. The background work was concurrent, but the status bar had no metadata events to consume.

ProgressiveObjectLoad now has a dedicated preflightProgressChanged(completed, total, stage) signal. The root manifest parser reports the declared child count immediately after JSON parsing, before full root validation. Only the count is observed at this point; the complete existing validation must succeed before children can be used.

The bar starts indeterminate while reading the root. Once the count is available, it shows 0/N scanned, advances only when a complete child's manifest and spatial bounds have been processed, and ends at N/N scanned. Compact stages identify root validation, child manifests, image metadata/bounds, quantitative bounds, overlaps, and positioning-dialog preparation. N/N means the child scan is done; overlap checking/dialog setup may still be running. No byte-I/O percentages or reconstructed-child readiness are inferred.

Worker updates enter the existing GUI owner through queued invocations protected by the pipeline's receiver mutex. Intermediate updates use one pending mailbox event. Initial determinate and final child-count boundaries are explicit. Monotone sequence numbers discard stale boundary snapshots so a newer overlap/dialog stage cannot be overwritten. Both metadata and reconstruction signals use queued status-panel connections, preserving phase order even when worker completion is fast. Labels have a fixed width, elision and a tooltip; dataset names are omitted. After positioning approval, the same bar restarts at 0/N ready.

## Cancellation, ownership, and scope

Cancellation checkpoints precede each child, each image/quantitative metadata operation, and overlap comparisons. An additional checkpoint precedes publishing a completed child. A filesystem read/canonicalization or a manifest-validation invocation already in progress is not interrupted; cancellation takes effect at the next checkpoint. No widget is accessed by the worker, and cancel never joins filesystem work on the GUI thread.

Callbacks are synchronous read() arguments and are not retained by the positioning plan. They cannot create a plan/pipeline reference cycle. Pending invocations are bound to the receiver's QObject lifetime, and panel connections are bound to the panel. The existing receiver revocation, CPU-state ownership, producer queue, exit-only join, ordered reconstruction, Engine registration, attachment, macro parameters, and loading/cancel lifecycle remain in place.

Original public read(), detect(), manifest read(path), and two-argument omeZarrArrayPath entrypoints remain as forwarding overloads. No new production source files, libraries or source registrations are needed.

## Measurements now available

PocaZarrLoadReport is reused, with profiling of low-level metadata operations enabled only inside a Preflight scan scope.

| Output | Measured scope |
| --- | --- |
| Preflight scan | Worker root read/validation through child metadata and overlap detection |
| Preflight root manifest | Root JSON group read and complete existing manifest validation |
| Preflight child manifest | Individual child path resolution, JSON read, full validation, identity and transform checks |
| Preflight image metadata | OME metadata parsing and existing level-path checks, level-0 array shape read, bounds conversion |
| Preflight quantitative bounds | Component-path validation, group JSON read and bounding-box metadata validation |
| Path validation | Existing PoCA path validation, including nested canonicalization and link/reparse checks |
| Path canonicalization | Relative-path checks, two explicit weakly_canonical calls, containment comparison |
| Metadata JSON reads | File open and JSON parse attempts |
| Preflight spatial overlaps | Existing sweep and pair comparisons, with cancellation checks |
| dialog_preparation_s | Actual dialog constructor/setup including the initial preview, excluding exec()/user dwell |
| preview_preparation_s | Sum of preview computation and setPreview calls; excludes message-box/user dwell |

Inclusive rows overlap and must not be summed. Self time subtracts direct timed children. Explicit call counters cover JSON attempts, repeated JSON paths within the scan, weakly_canonical calls, PoCA path validations, and existence/link/reparse checks inside that validator; they do not count the standard library's internal filesystem calls.

Normal output is a bounded summary: stage totals, counters, and the top three child totals/manifest/image/component operations per category. Identities use numeric child/image/component indices, not long filesystem paths. The report retains each child's total/self elapsed time; existing zarrLoadPerformance verbosity additionally prints every child timing.

The worker scan excludes positioning-dialog dwell and reconstruction. Dialog setup and preview preparation are logged separately; the initial preview is nested inside dialog setup and must not be added twice. The existing broader positioning_dialog_begin/end profile includes user interaction and remains distinct.

One existing boundary is intentionally unchanged: LoaderZarr::createProgressiveLoad reads and validates the root synchronously for format/routing before start() creates the displayed worker preflight. The worker then reads a fresh root. Thus there are two root parses across the whole open operation, but one within the measured worker scan. The duplicate-path counter applies to the worker scan only. Routing latency should be recorded separately when comparing the complete File/Open experience. No cached routing manifest or shortened routing validation was introduced.

## Confirmed redundant work and safe optimization

Originally each pocaZarrPath call made four explicit weakly_canonical calls: canonical root and target inside omeZarrArrayPath, then the same root again for the component walk and again for self-reference rejection. The new overload returns the canonical root already computed by that invocation, reducing four calls to two. This is a source-confirmed reduction in filesystem operations; its wall-time benefit is not measured yet.

The target is still canonicalized and compared against the canonical root. All relative-path, traversal, safe-segment, self-reference, exists, symlink and Windows reparse-point checks remain. Every quantitative component still performs its own original path checks. There is no persistent cache, no cross-operation validation cache, no skipped malformed-data validation, and no parallel metadata parsing.

Within a child scan, each manifest/image group/level-0 shape/quantitative group is read through the existing metadata routines. Progress performs no additional filesystem reads. The synthetic RAW+LABEL fixture asserts exactly 1+5*N JSON reads and zero repeated JSON paths for N children. Measurements on real data are needed before deciding whether further path or JSON reuse is justified.

Pixels, mesh vertices, faces and scientific feature arrays are not read by preflight. Camera/projection, visibility/pyramids, GPU initialization, mesh drawing/batching, DataAssembler algorithms, scientific features, file format, child assembly and attachment are unchanged.

## Evidence and before/after results

| Observation | Seconds | Cache classification |
| --- | ---: | --- |
| User-supplied previous 83-child preflight | 45.26 | Unknown |
| User-supplied latest 83-child preflight | 6.76 | Unknown |
| Instrumented after-change scan | UNMEASURED | UNMEASURED |
| Before/after stage breakdown on the same store | UNMEASURED | UNMEASURED |

The supplied timings have no stage/call/cache receipts, so they cannot establish the source of variability or identify a measured dominant bottleneck. Filesystem cache effects, storage latency and external contention remain hypotheses, not findings. No speedup, cold-cache classification or runtime-correctness claim is made.

## Authored verification and pending acceptance

Existing Tests-menu fixtures were extended, without executing them:

- Positioning: synthetic 1/20/83/305-child manifests, monotone 0/N through N/N callbacks, stage transitions, per-child timing records, exact JSON-read counts, zero duplicate JSON reads, bounds/overlap/model equivalence, and compact path-free summary output.
- Cancellation: deterministic stop after three complete children with no next-child reads; reread after cancellation; cancellation during overlap detection.
- Malformation: invalid child transform must identify child[1], its dataset name and the transform problem.
- Progressive loading: queued GUI-thread metadata phase, initial 0/N and final N/N, no return to unknown counts, 0/N ready transition, existing reconstruction/attachment/transform checks retained.
- Mid-scan GUI cancellation: a test-only worker gate holds metadata scanning after child 1; three GUI timer ticks must occur before cancel/release; no placement, reconstruction or attachment may follow.
- Schema: two canonicalization calls for a path resolution, identical resolved path, and existing traversal/reserved-name/case-collision checks retained.
- Existing metadata-only fixtures still verify no numeric payload reads; existing repositioning fixtures exercise saved transformations and packing algorithms.

Static review checks declarations, definitions, call sites, callback lifetime, queued event ordering, unchanged producer/reconstruction bodies, existing AUTOMOC/test registration, UTF-8/BOM/CRLF consistency, quoting artifacts and git diff whitespace. Static checks do not establish compilation or runtime behavior.

For runtime acceptance, use a matching user-built executable with the existing optional source-test menu enabled; no build/configuration was performed here. Run the positioning/progressive/schema actions, then open the same real 83-child store in before/after executables with identical configuration. Capture normal Preflight summaries and optionally zarrLoadPerformance child details. Record routing latency separately, cancel at root/early/middle/late scan and overlaps, and confirm a kept-original and repositioned complete load.

Record first-access and immediate repeat runs separately. Label a run cold-cache only when the cache state was controlled and documented; restarting PoCA alone does not prove cold cache. Do not flush system caches or alter storage state implicitly. Compare repeated scan medians and stage/call totals, excluding dialog dwell. For security acceptance, verify traversal plus symlink/Windows junction/reparse fixtures in an environment authorized to create those links; no links or filesystem test data were created here.

No configure, compile, test, application, backend or benchmark execution occurred. Runtime cancellation, GUI responsiveness, correctness and before/after timing remain UNCONFIRMED.
