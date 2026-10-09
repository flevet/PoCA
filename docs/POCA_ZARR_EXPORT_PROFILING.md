# Complete OME-Zarr dataset export profiling

2026-10-09. Diagnostic-only, SOURCE-ONLY implementation. No export-performance optimization was intentionally implemented. Runtime behavior was not verified. No CMake/configuration/build, PoCA/backend application, tests, benchmarks, Python or helper executable was run.

## Current call flow

The unchanged GUI dispatches CommandInfo to OmeZarrExportCommand::execute. saveDatasetOmeZarr and saveSelectedDatasetOmeZarr resolve the owner/selected child through PocaZarrDatasetExporter::target and call save. The original save entrypoint now owns one PocaZarrExportReport and delegates to an overload accepting a report for inspection/source fixtures.

1. preflight checks options, owner/child/component identity, manifests, image capabilities/calibration and persisted feature/display state before creating staging.
2. OmeZarrExportStore resolves the destination and validates replacement safety, then creates its unique sibling staging directory.
3. exportObject captures its dataset manifest, exports components, saves command characteristics, validates/writes manifests and group metadata. The same method serves single objects, selected children and every multiple child.
4. exportMultipleObject exports root components, serializes every child in existing order through exportObject, appends child entries, and serializes root characteristics/hierarchy/manifests. OrganoGraph stays with its existing root owner; nothing is duplicated into children.
5. exportComponents calls writePocaZarrExtension first. Its existing enumeration writes DetectionSets and direct/ObjectLists meshes via writePocaZarrPoints/writePocaZarrMeshes. It then writes image lists or standalone RAW/LABEL components with the existing association/path rules.
6. publish writes final root metadata, repeats destination safety validation, renames an old destination to a sibling backup when applicable, renames staging into place, and removes the backup. On failure, the same rollback/destructor cleanup runs before the partial report is printed.

The image-only saveOmeZarr public path keeps its existing behavior and does not independently create a dataset report. Shared serializers observe a report only during complete dataset export or an explicitly supplied diagnostic scope.

## Source/image pyramid work

RAW exportTypedImage in OmeZarrExport.cpp obtains the existing omeZarrExportLevels plan, creates each level through ZarrArrayWriter::create and invokes writeOmeZarrExportLevel in OmeZarrExportPixels.hpp. Level zero copies resident pixels or calls the existing full-resolution region reader. Positive native levels use regional readers or the existing native full-level view. Generated levels reread the already-written parent through ZarrArrayWriter::read and call Image<T>::downsampleRegion, then copy the result into the existing write buffer. The parent read is timed separately from downsampling and value writing; parent allocation/interval preparation remains level self time.

LABEL writeLabel in OmeZarrLabelsExport.cpp uses its existing matching native level or sampleOmeZarrLabelChunk in OmeZarrLabelSampling.hpp. Sampling includes nearest-neighbor coordinate mapping/selection; its tile source access is a nested timer. Its self time therefore excludes source access. No nearest-neighbor policy, RAW Average policy, XY stopping threshold, dropped odd sample, level count, calibration, chunk shape or codec was changed.

The report aggregates arrays, voxels, input bytes and value-write seconds by the actual output level number. It records resident/nonresident sources and requested scientific read bytes separately from observed Zarr backend read bytes. ZarrArrayAccess observes existing physical opens and successful reads, including deferred feature opens. ZarrImageStorage suppresses existing per-region debug/verbose logging only while an export report is active; dataset/image/level progress replaces that noise.

## One report, bounded observation and ownership

PocaZarrExportReport.hpp/.cpp is the only new production source pair. It owns counts, inclusive/self timing totals, progress, failure context and four fixed top-five groups: Dataset, Images, ObjectListMesh, Arrays (setup + value writes). Each record holds only fixed-category count/time deltas. It does not retain every feature/array/sample timing or size. Existing component classes and serializers keep their responsibilities; no generic profiler framework was added.

A nested Stage provides RAII synchronous scope observation via a thread-local pointer, matching the existing load-report pattern. No worker, locking/synchronization subsystem, callback lifetime extension or retained source reporter pointer is introduced. Array writers retain only category, level, successful byte count and setup/write durations; they consult the current scope, close their backend handle before ranking, and never own the report. Original writer/mesh entrypoints remain available as delegating overloads.

The existing CommandStateStorage gains an optional persistencePhase marker. Its default does not collect diagnostics in nonprofiling adapters. The Zarr command adapter owns at most one live observation stage, closes it before switching phases and destroys it before the enclosing command stage. This is a storage-neutral persistence observation hook, not an error/data fallback. OrganoGraph uses it without a dependency on the loader plugin. Neither phase names nor report data enter saved JSON or numeric arrays.

## Timing labels

All times are steady-clock wall seconds. Every row prints inclusive | direct children | self/unclassified. Nested totals overlap and must not be summed as siblings. Only directly nested spans debit parent self; descendant detail summaries are informational. Per-child progress prints inclusive Images/DetectionSets/ObjectListMesh/Features/Metadata totals and dataset total/self. Fixed top-five lists retain identities and useful count deltas (RAW/LABEL, level-zero voxels, mesh components/objects/vertices/faces, feature records).

| Area | Exact labels |
| --- | --- |
| Whole export | TOTAL EXPORT; Preflight; Staging setup; Actual data export; Export planning/manifest preparation; Multiple child export; Multiple root assembly; Dataset; Root metadata/manifests; Command characteristics; OrganoGraph; OrganoGraph without persisted state; Final transaction commit |
| Transactions/metadata | Destination safety checks; Staging directory creation; Final manifest completion; Rename old destination to backup; Rename staging to destination; Rollback rename; Old destination cleanup; Temporary cleanup; Metadata; Existing dataset manifest validation; Existing quantitative manifest validation; Quantitative component planning |
| Images | Images; Image source preparation/acquisition; Label source preparation/acquisition; Image level N; Level 0 preparation; Positive level source preparation; Image source access/copy; Image native source access/copy; Image display bounds reduction; Generated parent reads; Pyramid generation/downsampling; Label source access/copy; Label pyramid sampling; Label sampling source access/copy; Image metadata assembly; RAW image group metadata write; Label metadata assembly/write; Level N value writes |
| Common array operations | Array setup; Value writes; CATEGORY array setup; CATEGORY value writes; Zarr source array opening; Zarr source value reads |
| Features | Features; Feature discovery/enumeration; Feature serialization; Feature source access/materialization; Feature metadata/statistics preparation; Feature metadata/statistics assembly; Feature metadata/statistics group writing |
| Points | DetectionSets; Detection component preparation; Coordinate source access/packing; Detection metadata assembly/write |
| Meshes | ObjectListMesh; Mesh component/manifest preparation; Mesh geometry extraction/access; Mesh existing conversion validation/storage checks; Mesh existing indexed validation; Mesh geometry packing/serialization; Normal acquisition/copy/regeneration; Vertex normal serialization; Face normal serialization; Validation provenance serialization; Mesh rendering axes; Mesh remaining metadata assembly/write |
| Graph | OrganoGraph hierarchy/mapping/source indices; OrganoGraph metadata/configuration; OrganoGraph scalar result preparation; OrganoGraph embedding/PCA/UMAP preparation; OrganoGraph spatial curve preparation; OrganoGraph numeric scalar arrays; OrganoGraph embedding/PCA/UMAP arrays; OrganoGraph spatial curve arrays; OrganoGraph plot/embedding display state |

CATEGORY is exactly Image, Coordinate, Mesh geometry, Vertex normal, Face normal, Mesh auxiliary, Feature, OrganoGraph, Command characteristic (Other is available for a manually scoped generic writer). Per-image RAW ranking stops before associated LABEL serialization so each image is ranked once. Its final RAW group document is written after associated labels by the existing algorithm and is reported separately; the ranking excludes that final document write. LABEL ranking includes its own final metadata write. Image metadata scopes contain persisted-feature children, so use self when examining metadata-only work.

Feature display/descriptor assembly is separate from source access and writes. Array setup times the writer/backend initialization; caller-side array JSON construction/dump remains serializer self time. The final display/feature manifests and mesh validation provenance are fields in their existing component/image documents: their final JSON dump/disk write cannot be attributed independently to each field without changing serialization. The report times provenance construction and the enclosing document write. Geometry traversal/packing remains one existing streaming workflow; offsets, vertices and faces are classified together as Mesh geometry. Existing conversion validation/storage checks are measured inside fromMeshes and charged as a child of geometry extraction, without rerunning validation.

## Counters and byte meanings

Component/image/feature counts describe encountered export work; completed Dataset scopes increment Datasets exported. Partial summaries can include encountered components/images/features that failed later. Arrays created, arrays written, write calls, elements and bytes count successful operations. Array setup attempts and Write attempts expose failed backend work separately. One array written is counted on its first successful write; subsequent regions do not count it again.

- Dataset/component counts: Planned images, Datasets exported, RAW images, LABEL images, Pyramid level arrays, DetectionSets, Detections, ObjectListMesh components, Mesh objects, Vertices, Faces.
- Arrays: Array setup attempts, Arrays created, Arrays written, Write attempts, Write calls, Elements submitted, Bytes submitted/uncompressed; CATEGORY arrays, CATEGORY arrays written and CATEGORY bytes.
- Disjoint submitted-byte categories: Image (RAW/LABEL pixels at every level), Coordinate (DetectionSet positions), Mesh geometry (float64 vertices, uint64 faces and both offsets), Vertex normal, Face normal, Mesh auxiliary (render axes), Feature (image/point/mesh quantitative arrays), OrganoGraph (root numeric state), Command characteristic (other numeric state), Other (manually scoped generic arrays). Total normal bytes is Vertex normal + Face normal. No source/read volume is added to submitted-byte totals.
- Per output level N: Level N arrays, Level N voxels, Level N bytes input. Actual level value-write time is accumulated separately, including failed call time in partial reports.
- Features: Feature records (including intentional reserved/derived skips), Feature arrays skipped (reserved/derived), Feature values, Storage-backed feature sources, Resident feature sources, Unloaded feature sources, Feature sources read from Zarr, Features materialized during export. Physical arrays/bytes use the common writer and reader counters.
- Source access: Resident image sources, Nonresident image sources, Image source region calls, Image source bytes requested, Images with Zarr pixel source reads, Image pixels materialized during export, Native full-level views acquired (may already be cached), Zarr source read calls, Zarr image source read calls, Zarr image source bytes read, Zarr feature/coordinate source bytes read, Zarr source arrays opened during export and Zarr CATEGORY source arrays opened during export. Generated parent read calls/Generated parent bytes read describe destination rereads, separately from source-store reads.
- Mesh state: Indexed-authority mesh sources (resident arrays), CGAL-authority mesh sources, Mesh CGAL-to-indexed conversions, Mesh conversion topology validations, Mesh conversion storage-only checks, Mesh explicit indexed validations, Mesh normal collections regenerated/reused, CGAL mesh collections materialized during export. KD-trees materialized during export is a before/after hasSpatialIndex comparison. Normal regeneration is reported by the existing owning method at its actual branch; no probe regenerates normals.
- Characteristics: OrganoGraph states persisted, Other command states persisted. Graph recognition uses existing characteristicName() == OrganoGraph; command ownership remains unchanged. Null-state commands are timed as OrganoGraph without persisted state rather than added to actual graph persistence time.
- Approximate filesystem operations: Array metadata documents, Group metadata documents written, Group metadata write calls and their summed Metadata documents written. These are operation counts, not unique groups/files/chunks. Existing parent-group rewrites count each successful document write. Backend internal chunk/file counts are unavailable.
- Small arrays: declared uncompressed array bytes, zero-length arrays, disjoint zero/(0,1 KiB]/(1,4 KiB]/(4,64 KiB]/(64 KiB,1 MiB]/>1 MiB bins and integer mean declared payload. Arrays with unknown declared payload are explicitly counted and excluded from mean/bin volume. Every production caller supplies its existing shape/count-based payload; original two-argument callers can report unknown. Overflow in a diagnostic payload returns unknown, never changes scientific export validation. No exact median or per-array size list is retained.

## Progress and failed exports

Console prefix is [PoCA][Zarr][Export]. Dataset progress includes child index/total and original name; image progress includes a global image ordinal/planned total and current dataset/name; every actual level is named. Image, mesh and dataset completion report elapsed/self seconds. No per-feature or per-chunk progress line is printed. Root-only images are included in the global planned total.

The first unwinding failure freezes its dataset/component/feature/image/level/stage context. Transaction cleanup cannot replace that context. After unwinding the existing transaction, save prints a partial summary and rethrows the original exception; existing higher-level error context is preserved. There is no new cancellation API; an exception-based cancellation follows this same path. Cleanup/rollback errors keep existing semantics, including retained old backup and published-but-cleanup-failed states. Publication uses sibling renames, never recursive copy.

## Observed source work, not optimized

The following are source-level observations, not measured bottlenecks:

1. RAW generated pyramids reread/decode the just-written parent before downsampling and rewriting every level. Generated-parent reads, pyramid self time and level/write throughput can distinguish these costs.
2. Every published feature creates its own array and metadata/filesystem entries. Enumeration, array setup, writes, payload bins and declared mean can expose small-array proliferation.
3. Preflight and save paths inspect feature/display state and dataset manifests repeatedly. Quantitative manifest validation/readback and complete manifest validation remain in place, including exporter readback of the extension manifest. Metadata/preflight/validation scopes expose their cost.
4. indexedGeometry converts CGAL authority to a fresh indexed snapshot when indexed backing is absent; conversion checks validate topology or storage as before. Untouched indexed geometry is reused, but normalsForGeometry still copies cached normal vectors into its output. Mutable CGAL authority or missing complete cached properties can regenerate normals. Acquisition/regeneration counters expose this.
5. LABEL generated levels may request repeated source tiles/rows according to the existing sampling workflow. Sampling self and source request/backend bytes expose amplification.
6. Existing overwrite safety recursively inspects the old store at setup and publication. Backup deletion can be expensive. Those existing walks/deletions are separately timed; no new recursive scan is introduced.

Future optimization investigation order, ranked by what these diagnostics can discriminate rather than by unmeasured runtime speed:

1. Compare image source/parent reads, downsampling/sampling self, array setup and value writes by level. Decide whether source access, CPU computation or the opaque backend write dominates.
2. Compare feature setup/metadata cost with small-payload bins and submitted bytes. Assess array proliferation before considering combined/packed formats.
3. Compare mesh access/conversion/validation, normal acquisition/regeneration and feature cost. Check untouched indexed re-export counters stay at zero materialization/regeneration.
4. Compare destination safety, rename and old/temporary cleanup to actual data export. Establish whether filesystem transaction overhead dominates.
5. Compare bounded per-child/per-image distributions to assess future concurrency benefit. Current children are serialized sequentially with existing ownership; shared source handles, commands, source/destination overlap and backend/thread safety still require an independent audit. No concurrency readiness is claimed.
6. Zarr source read/open counters and unmaterialized-state checks expose decode/read/rewrite work for future Zarr-to-Zarr reuse analysis. No copy/reuse path was implemented.

## Limits

The unchanged C ABI returns synchronous create/read/write status, not compression, disk-only timing, actual compressed byte counts or backend chunk traffic. Value-write timers include encoding/compression and storage wait together. The current image array metadata still contains its existing bytes codec; no compressor settings were added/changed. No scientific read, hash, filesystem walk, second source enumeration or CGAL/KD-tree construction is performed to fill counters.

Observed Zarr reads identify backing actually used during export. The core image/feature region-reader APIs do not identify arbitrary callback provenance. Resident pixels may originally have come from Zarr but are correctly counted as resident when the exporter copies them. Histogram.storageBacked distinguishes storage-backed histograms even when already resident, but an arbitrary backing callback is not assumed to be Zarr. Loaded mesh geometry/normals are resident indexed arrays, not lazy Zarr arrays, and their historical origin is not available through existing state; the report says indexed authority instead of inventing a Zarr origin.

Before/after state counters report successful observed transitions, not hidden native-cache internals or failed reconstruction transitions. Physical open counters are operations, not a retained set of unique array identities. Normal acquisition combines vertex/face copying or regeneration as the existing owner does, while the actual vertex/face writes are separate. Array top-five ranks successful setup + completed value-write durations, not array lifetime, close latency or failed-call latency; partial aggregate timers still include failed calls. No profiler-overhead benchmark has been run. One report instance is intended for one synchronous export.

## Source-only fixture coverage (not executed)

| Required case | Existing source fixture/assertion |
| --- | --- |
| 1 single object; 2 multiple aggregation | DatasetContainerFixture::checkExportProfile: one dataset; seven children/top five; independent timing totals |
| 3 RAW; 4 LABEL; 5 levels | Same fixture: 2 RAW, 4 LABEL, six level-zero arrays; generated 257x2 RAW emits two actual levels |
| 6 computation vs write | Generated-parent bytes and separate Pyramid generation/downsampling, Level 1 value writes, Array setup labels |
| 7 image bytes; 8 points bytes | 144 pixel bytes, 60 coordinate bytes, 5 detections |
| 9 geometry; 10 vertex normals; 11 face normals | 480 geometry bytes (including offsets), 144 vertex-normal and 48 face-normal bytes, two arrays each |
| 12 feature arrays/bytes | 23 feature arrays, 55 values, 220 bytes, disjoint total submitted bytes |
| 13 root graph once | LoadCharacteristicFixture::exportRootProfile synthetic characteristic: three children with null graph commands, one root numeric graph state/array, 32 graph bytes, separate no-state timing; actual OrganoPersistenceFixture asserts all phase markers and unchanged five-array layout |
| 14 nested accounting | ZarrArrayWriterMockTests: inclusive >= children, exact inclusive-self-parent-child relation; scope restoration |
| 15 lazy feature/image state | Container lazy checks before/after export; Zarr copy observes 23 feature demand opens/reads and six image sources without materialization; pyramid full-materialization callback throws |
| 16 no CGAL | Zero CGAL/KD-tree transitions, unchanged indexed mesh lazy checks, zero normal regeneration |
| 17 bounded top N | 10,000 synthetic array cost records retain exactly descending top five; seven real children retain five |
| 18 failure/cleanup | Existing late child rollback now inspects partial report, Dataset 4/4, level 0, cleanup timing and observer restoration, plus existing unchanged destination/staging assertions |
| 19 old values/layout | Existing mock metadata/region/source-byte equality and existing round-trip suites retained; repeated complete dataset manifest equality |
| 20 no profiler metadata | Manifest equality and unchanged characteristic count/layout; markers only go to observer |
| Additional bookkeeping | Mock failed write excluded from successful counters; zero/threshold bins; unknown-overflow payload; separate generated-parent reads and actual per-level counters |

No new source test file or test registration action was added. The existing manual PoCA export source suite remains default OFF. Existing independent reader/writer mock CMake source lists add only the report dependency; no configuration/build/execution occurred.

## Modified files

New production pair:
- poca_extra/src/poca_loaderZarrFile/PocaZarrExportReport.hpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrExportReport.cpp

Core owning APIs:
- poca/src/poca_core/General/Command.hpp
- poca/src/poca_core/General/Histogram.hpp
- poca/src/poca_geometry/Geometry/ObjectListMesh.hpp
- poca/src/poca_geometry/Geometry/ObjectListMeshIndexedGeometry.cpp
- poca/src/poca_geometry/Geometry/ObjectListMeshPersistence.cpp

Existing exporter/storage sources:
- poca_extra/src/poca_loaderZarrFile/PocaZarrDataset.hpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrDatasetExporter.cpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrDatasetManifest.cpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrExtensionExport.cpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrFeatures.cpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrPoints.cpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrMeshExport.cpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrMeshDisplay.cpp
- poca_extra/src/poca_loaderZarrFile/OmeZarrExport.cpp
- poca_extra/src/poca_loaderZarrFile/OmeZarrExportPixels.hpp
- poca_extra/src/poca_loaderZarrFile/OmeZarrLabelsExport.cpp
- poca_extra/src/poca_loaderZarrFile/OmeZarrLabelSampling.hpp
- poca_extra/src/poca_loaderZarrFile/OmeZarrExportStore.cpp
- poca_extra/src/poca_loaderZarrFile/ZarrArrayWriter.hpp
- poca_extra/src/poca_loaderZarrFile/ZarrArrayWriter.cpp
- poca_extra/src/poca_loaderZarrFile/ZarrArrayAccess.cpp
- poca_extra/src/poca_loaderZarrFile/ZarrImageStorage.cpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrCommandState.hpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrCommandState.cpp
- poca_extra/src/poca_organographplugin/OrganoGraphPersistence.cpp
- poca_extra/src/poca_organographplugin/OrganoGraphCommand.cpp

Registration/source fixtures:
- poca_extra/src/poca_loaderZarrFile/CMakeLists.txt
- poca_extra/src/poca_loaderZarrFile/backend_tests/CMakeLists.txt
- poca_extra/src/poca_loaderZarrFile/backend_tests/ZarrArrayWriterMockTests.cpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrDatasetContainerTests.cpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrCharacteristicTests.cpp
- poca_extra/src/poca_organographplugin/OrganoGraphPersistenceTests.cpp

Documentation:
- poca/docs/POCA_ZARR_EXPORT_PROFILING.md
- CONTINUITY.md

## Static audit

Values/metadata fields, array shape/dtype/dimensions, source access calls, loop/order/association, existing RAW/LABEL algorithms, codecs/chunks, transaction rename/rollback/delete sequence and command dispatch remain unchanged. Changes consist of observation scopes/counters, optional diagnostic outputs, delegating overloads, progress and source fixtures. No backend ABI implementation, schema, scientific computation, rendering, GUI layout or generated/external build directory was edited. No parallelism or new data materialization path was introduced.

Static verification covers declarations/definitions/call sites, independent mock dependencies and existing source registration, added include resolution, lexical delimiters, CRLF/UTF-8/BOM preservation, git diff --check and targeted quoting-artifact checks. This is source evidence only and cannot establish runtime correctness or overhead. See CONTINUITY.md for final audit receipts.
