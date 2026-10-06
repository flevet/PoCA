# PoCA quantitative data alongside OME-NGFF

## Phase 6.1 — complete dataset containers (2026-10-06)

This section supersedes the image-only scope statements in the historical Phase-6 sections below. Source implementation and static source audit only; compilation, Qt/backend/CGAL behavior and runtime round trips are UNCONFIRMED.

### Actions and recordable commands

| Active entity | Export image as OME-Zarr... | Export dataset as OME-Zarr... | Export selected dataset as OME-Zarr... |
| --- | --- | --- | --- |
| MyObject | Existing action; enabled for the current RAW | Complete active MyObject | Hidden |
| MyMultipleObject | Existing action; enabled for the current child's current RAW | Complete outer object and ALL children | Visible; currentObject() as a standalone MyObject |
| No selected child | Disabled image action | Full export still traverses every available child | Disabled and safely rejected |

The image action retains `saveOmeZarr`, one current RAW, explicitly associated labels, and the existing optional Phase-6 quantitative extension. Dataset actions record `saveDatasetOmeZarr` or `saveSelectedDatasetOmeZarr` through LoaderZarr's `actionNeeded` and normal CommandInfo dispatch. MDI activation, child selection and File-menu opening update action state through PluginInterface::updateActions; no polling timer is used. The active camera's entity is the target, never all open windows.

Both dataset commands accept outputPath, overwrite=false, multiscale=true, componentChunkRows=65536 and chunkX/Y/Z=256/256/4. The command specification includes executeOnObjectOnly=true. MyMultipleObject also recognizes both commands as owner operations for direct/replayed CommandInfo without that parameter, so child command forwarding cannot produce repeated partial exports. Complete export cannot turn off labels or quantitative data; direct options requesting partial export are rejected.

Default GUI filenames use the existing ZarrSafeNames and suffix normalization: the active object name, multiple-object name, or selected child's name followed by .ome.zarr. No internal ID is added. Empty names use the sanitizer's explicit dataset name policy. Errors include exactly "No current dataset to export.", "No datasets are available in the current multiple dataset.", and "No dataset is currently selected." where applicable. Dataset actions do not use the image-only intensity-selection error.

### Class responsibilities and shared serialization

| Class | Responsibility |
| --- | --- |
| PocaZarrDatasetManifest (added) | Captures and restores names/transforms/hierarchy; owns ordered component/image/child descriptors, quantitative references, validation, JSON and persistence |
| PocaZarrDatasetExporter (added) | Resolves full/selected targets; preflights all scientific content; writes one MyObject, all multiple children and every ImagesList entry; delegates payload writers; owns one staging transaction |
| PocaZarrDatasetLoader (added) | Loads declared entries and explicit associations, reuses scalar/point/mesh reconstruction, assembles ordered owned children/aggregate components, restores state, registers only the completed root |
| OmeZarrExportCommand (extended) | Existing image command plus dataset command specs/dispatch and owner-aware copies |
| LoaderZarr (extended) | Three actions, dynamic visibility/enabling, filename/dialog/error collection, command creation and version dispatch |
| Engine (extended) | Unregistered owning assembly for MyObject and MyMultipleObject; one final root registration; components/plugins installed once |
| ImagesList (extended) | Reads explicit source indices and accepts owned unassociated images without selection-based inference |
| MyMultipleObject (extended) | Persistence constructor can bypass grid recomputation; dataset commands execute once at the owner |
| Command / CommandableObject (extended) | copyFor(owner) lets export commands bind the new owner during base construction; the default preserves other commands' existing copy behavior |
| PluginInterface (extended in both distributions) | Default updateActions hook for event-driven menu state |

One shared header declares the three dataset classes; three substantial implementation files separate metadata, export and reconstruction. No class wraps one trivial operation. The existing PocaZarrManifest.cpp now also supplies the manifest class's shared quantitative validator, preserving the v0.1 reader entry point. Existing scalar/pyramid/label/template writers remain in their current files; the new group-writing and label-validation entry points are adapters needed to reuse those implementations. Tiny local matrix conversions and existing safe-name/arithmetic helpers remain stateless.

Exactly one `exportObject` implementation serializes a standalone object, a selected child and every full multiple-object child. Full export never chooses currentObject as its target; only selected export does. Selected export creates kind=object and no one-child wrapper.

### Scientific hierarchy and schema 0.2

The outer store is a **PoCA extension stored in Zarr**, containing standards-compliant NGFF 0.5 image groups where applicable. It is not an official OME-NGFF multiple-dataset standard. Generic OME viewers may need to open an individual contained image group; they need not understand the outer container.

Every root and child has a Zarr v3 group. Root attributes.poca_dataset identifies version=0.2 and manifest=poca. The authoritative manifest is attributes.poca in poca/zarr.json:

- version=0.2, kind=object or multiple_object, exact name and transform.
- components is ordered and has contiguous component_index, exact name and kind: images_list, image, points, mesh_collection or object_lists.
- An images_list descriptor has images with contiguous index, exact entry name, type=raw/label, relative path and source=null or an explicit same-list RAW index.
- An object_lists descriptor also has entry_count. quantitative retains the unchanged Phase-6 component payload references, including container ID/name, ordered entry index/name, plugin and normalized CommandInfo.
- A multiple_object additionally has ordered objects: index, exact name, unique relative path and kind=object; hierarchy nodes with label, level_name, parent, children, object indices and string metadata; grid_boxes; and an optional current_object preference.
- Parent hierarchy nodes precede their children, matching the current addHierarchyNode API. Duplicate visible object/image names are valid; unique indices and deterministic safe paths identify entries.

Conceptual layout (actual safe names depend on original names and collisions):

```text
dataset.ome.zarr/
  zarr.json                         PoCA container; no dummy root image
  poca/zarr.json                    0.2 manifest
  images/ImagesList/
    actin/                         independent NGFF RAW group
      0/ ...
      labels/mask/                 explicitly associated NGFF LABEL
      labels/mask_2/               second label for the same RAW
    actin_2/                       another independent RAW
    orphan_manual/                 scalar LABEL; no image-label.source
  poca/points/...                  existing Phase-6 positions/features
  poca/meshes/...                  existing indexed geometry/object features

multiple.ome.zarr/
  zarr.json
  poca/zarr.json                    kind=multiple_object, ordered child descriptors
  objects/same_name/               complete kind=object dataset
  objects/same_name_2/             another child with the same exact visible name
  objects/...                     all remaining children
```

All ImagesList entries are considered independently of current selection: every RAW, all explicitly associated labels, and every orphan. Associated labels use the existing Phase-4 writer beneath their declared RAW. Orphans use that same scalar label/pyramid implementation at an independent group and remove the source member; type=label and source=null restore LABEL semantics and remain unassociated. Owned insertion avoids addImage's existing current-entry inference; associations are restored after all entries exist, including labels that precede their RAW.

Points-only, mesh-only, LABEL-only and mixed datasets need no RAW or dummy image. DetectionSets and direct top-level ObjectListMesh components need no image association. ObjectLists retains its container, exact entry order/names/plugins/commands; it is not flattened. Top-level component order is independent of path sorting. Every supported component is written once, including supported aggregate components owned directly by MyMultipleObject.

The existing positions, indexed double vertices/u64 faces/offsets, coordinate metadata, nbSlices, bounding boxes, float32 feature arrays, lazy feature sampling and lazy KD-tree paths are reused. No new source-image associations or scientific units are guessed.

### Transforms, metadata and ownership

Source inspection found MyMultipleObject's ordered raw-pointer child vector, unbounded nbColors count, unchecked currentObject access, an owning destructor, hierarchy metadata, selected-object indices and batch-rendering/grid state. Its constructor normally recomputes grid placement. Its interface accepts arbitrary MyObjectInterface pointers, so nested multiple containers are technically possible but are explicitly unsupported by this schema and rejected with child context.

For each object, model and rotation are persisted as 16 row-major float32 values (JSON numeric scalars), and translation as three float32 values. Access/reconstruction converts GLM's column-indexed matrix to this documented representation. The authoritative model matrix, including any scale, is restored directly; rotation and PoCA's stored translation are also restored so subsequent gizmo edits retain the saved decomposition. Nothing is baked into pixels, vertices or localizations. Saved grid boxes and hierarchy node metadata are preserved. The multiple constructor bypasses placement recalculation during reconstruction.

The current-child index is an optional UI preference. Valid indices are restored; malformed/out-of-range preferences are ignored without discarding scientific children. Multi-selection, display visibility/category/colors, batch-rendering switches, camera and command history are not captured as generic dataset state. Label color metadata and already supported mesh feature/display metadata retain their existing payload behavior.

Loader reconstruction uses local unique_ptr owners throughout. Children are assembled without registration, moved into MyMultipleObject once, and deleted by its existing destructor. Aggregate/child component commands and root commands are installed once, after the relevant components exist. Only the completely restored root is registered in Engine. Failure releases all partially reconstructed owners without registering children or an incomplete root. The new command-copy binding avoids exporting through an original owner pointer after MyObject/ImagesList/Image copies.

### Preflight, transaction and compatibility

Complete export preflights the outer object and every child, all top-level components, all ImagesList entries, ObjectLists entry types, quantitative feature types/readability/display metadata, and ROIs stored outside components. Unsupported state is aggregated with object name, child index/name, component or entry name and reason. Skeleton, TrackSet, Voronoi/Delaunay, plugin-private unsupported components, ObjectListPolygon and ROI/annotation state must not be silently omitted. Empty ImagesList/ObjectLists cannot currently be reconstructed and are explicitly rejected. Null/repeated children or aliased owned entries are rejected.

Unsupported-content preflight finishes before creating staging. Pixel/feature reads, geometry validation and metadata/filesystem failures during serialization still fail the entire transaction. Exactly one OmeZarrExportStore stages the full hierarchy; subgroup/child writers do not publish. Only after all children and manifests succeed does the existing rename/backup publication run. A late child failure cleans staging and preserves the old destination. Existing overwrite/link checks and rollback/backup retention remain. The operation remains a synchronous local-filesystem transaction, without a new cross-process or crash-durability guarantee; existing same-store source-handle limitations remain.

LoaderZarr recognizes 0.2 before image-centric loading, restores object or multiple_object without requiring a root image, and rejects malformed supported manifests. Version 0.1 remains readable through its shared quantitative validation and old root-image path. Stores without a PoCA manifest continue through the external normal OME-Zarr image/labels loader. The image export still writes the existing 0.1 extension and retains its established unsupported-component skip policy; the new complete export has the stricter policy described above.

No backend ABI/TensorStore wrapper changes, dependency installations, source changes in poca_zarr_backend or C:/tsb/C:/tsbd, camera changes, image scaling/calibration changes, navigation/detail streaming changes, GPU-budget changes or unrelated refactors were made. The PoCA plugin interface gains a default action hook; the eventual user-controlled build must include the updated interface consistently across core and plugins.

### Source fixtures and requested coverage

PocaZarrDatasetContainerTests.cpp contains one cohesive DatasetContainerFixture, two necessary synthetic unsupported/null-selection fixtures, and one TestRegistry adapter. It is registered as an optional manual Tests/Images action under the existing POCA_ZARR_EXPORT_SOURCE_TESTS flag (OFF by default). These sources were written and statically reviewed only, never executed.

| Requested cases | Source evidence for later execution |
| --- | --- |
| 1–8 GUI | checkGui: three actual QActions, both switch directions, invalid/null child, image action retained; MainWindow activation/menu/selection hooks audited |
| 9–17 every image entry | checkObject: duplicate RAW names, two labels on one RAW, second RAW's label, orphan source=null, exact order/name/type/source; checkImageOnlyVariants adds LABEL-only and LABEL-before-RAW |
| 18–22 quantitative components | checkObject/checkImageOnlyVariants: points with/without images, ordered ObjectLists metadata, direct mesh, direct scalar component, lazy open/re-export; existing point/mesh fixtures retain scientific payload coverage |
| 23–33 multiple datasets | checkMultiple: 1, 4 and 133 children, all children vs last current child, exact duplicate names, paths/order, model/rotation/translation, grid boxes, hierarchy metadata and current preference; empty case in checkPreflight |
| 34–37 selected | checkSelected: command-selected/direct manifest equality, standalone kind=object with no wrapper, changing selection, full direct command owner scope |
| 38–42 mixed children | checkMultiple cycles RAW+LABEL+points+ObjectLists, points-only, mesh-only, orphan LABEL+mesh; aggregate outer points also round-trip |
| 43–46 transaction | checkRollback: final child pixel-reader failure after earlier children wrote staging, old manifest retained, no published objects, staging/backup absent |
| 47–51 unsupported | checkPreflight: unsupported normal component, nested multiple rejection, two different child issues, name/index/reason aggregation, external ROI marker, no destination created; loader/manifest strict supported-kind checks audited |
| 52–56 compatibility | Existing ordinary and v0.1 whole-object/image fixtures remain registered; checkObject/checkMultiple/checkRegistration exercise new 0.2 dispatch; old image command/dialog/labels fixtures retained |
| Additional guards | checkManifestRejection: path alias/traversal, hierarchy cycles, optional invalid current preference, fabricated orphan source; checkCopiedOwner: original deleted before copied object's command runs |

Static source audit covers the 34 requested structural checks: retained image action; correct normal/multiple visibility; all-child traversal; selected-only currentObject; one shared object serializer; every image/orphan/RAW; independent point/direct-mesh components; explicit ObjectLists/multiple hierarchy and ordering; safe duplicate names; persisted transforms; no inferred associations; full unsupported-content preflight; one transaction and rollback; v0.1/external dispatch; unchanged backend/rendering/calibration boundaries; and three substantive OO responsibilities rather than tiny utility files. Final static checks pass 39 UTF-8/consistent-CRLF files, 34 balanced lexical source streams, 78 existing plugin source-list paths, 26 new-source local includes, new-source whitespace and git diff --check. The documented 5-added/34-modified inventory exactly matches the working tree after excluding externally edited AGENTS.md. These are source findings, not runtime assertions.

### Known limitations and later manual validation

Nested MyMultipleObject, empty component lists and unsupported scientific/ROI components are explicit complete-export errors. Source callbacks, objects and destination trees must remain stable for synchronous export. The existing lazy sampling, in-memory mesh reconstruction, scalar dtype restrictions, same-store source handles and local publication limitations remain. Build/link/API/Qt/backend/CGAL behavior remains UNCONFIRMED.

Manual checklist for a separately supplied build and user-controlled validation:

1. Switch normal → multiple → normal MDI windows; confirm both dataset actions update and the old image action remains present.
2. Select different children and RAW/LABEL entries; confirm image applicability and selected-dataset enablement.
3. Validate empty entity/empty multiple/null current-child messages without unchecked vector access.
4. Export a normal object with two RAWs, multiple associated labels and an orphan; open each RAW group in an NGFF-aware viewer.
5. Reopen in PoCA; compare every entry's exact name, type, order and explicit/orphan association, including LABEL-before-RAW.
6. Round-trip LABEL-only, points-only, mesh-only and LABEL+mesh datasets without a dummy RAW.
7. Compare ObjectLists entry order/names/plugin/normalized commands and direct mesh presence.
8. Inspect lazy feature/index state after load and after unchanged re-export; use existing scientific point/mesh fixtures to compare values/topology.
9. Full-export one, several and 133 children with duplicate names and a selected child near the end; count all ordered child datasets.
10. Compare saved/reopened child and outer matrices, rotation/translation, hierarchy labels/membership/metadata and grid boxes; then make a gizmo edit.
11. Export selected child twice after changing selection; compare each standalone store against direct export of that child.
12. Replay both recordable dataset commands; confirm owner execution once and copied objects export their own content.
13. Introduce unsupported components and ROI state in separate children; check all diagnostics and no staging/destination write.
14. Inject a late child read failure over an existing destination; verify previous data, no partial children and staging cleanup.
15. Load ordinary external OME-Zarr, old 0.1 stores and both 0.2 container kinds through Engine; check one root registration and no duplicate commands. Run the optional source fixtures only when separately authorized.

### Phase 6.1 complete file inventory

The externally edited AGENTS.md is preserved and excluded from the implementation inventory below.


Added (5 files):

- `poca_extra/src/poca_loaderZarrFile/PocaZarrDataset.hpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrDatasetContainerTests.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrDatasetExporter.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrDatasetLoader.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrDatasetManifest.cpp`

Modified (34 files):

- `CONTINUITY.md`
- `poca_extra/include/PluginInterface.hpp`
- `poca_extra/src/poca_loaderZarrFile/CMakeLists.txt`
- `poca_extra/src/poca_loaderZarrFile/EXPORT_README.md`
- `poca_extra/src/poca_loaderZarrFile/LoaderZarr.cpp`
- `poca_extra/src/poca_loaderZarrFile/LoaderZarr.hpp`
- `poca_extra/src/poca_loaderZarrFile/LoaderZarrGui.cpp`
- `poca_extra/src/poca_loaderZarrFile/LoaderZarrObject.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExport.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExport.hpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExportCommand.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExportCommand.hpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExportGui.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExportGui.hpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExportMetadataTests.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExportTests.hpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrLabelsExport.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrLabelsExport.hpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrLabelsLoad.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrLabelsLoad.hpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrExtensionExport.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrManifest.cpp`
- `poca_extra/src/poca_loaderZarrFile/README.md`
- `poca/docs/POCA_ZARR_EXTENSION.md`
- `poca/include/PluginInterface.hpp`
- `poca/src/poca_core/General/Command.hpp`
- `poca/src/poca_core/General/CommandableObject.cpp`
- `poca/src/poca_core/General/Engine.hpp`
- `poca/src/poca_core/General/EngineObjectAssembly.cpp`
- `poca/src/poca_core/General/ImagesList.cpp`
- `poca/src/poca_core/General/ImagesList.hpp`
- `poca/src/poca_core/Objects/MyMultipleObject.cpp`
- `poca/src/poca_core/Objects/MyMultipleObject.hpp`
- `poca/src/poca/Widgets/MainWindow.cpp`

Suggested commit: `feat(zarr): export complete MyObject and MyMultipleObject datasets`.

No CMake configure/generate, compilation, linking, installation, application executable, Python script, benchmark, or test was run.

## Historical Phase-6 image-oriented foundation (version 0.1)

2026-10-05 — Phase 6 source implementation. This is a **PoCA extension stored alongside OME-NGFF**, not an OME-NGFF point-cloud or mesh standard. Compilation and runtime behavior are UNCONFIRMED. The source fixtures below are registered for later manual execution and have not been run.

## Scope and entry points

The existing `saveOmeZarr` command exports one selected scalar RAW image, its explicitly associated labels, and supported quantitative components of the owning PoCA object into one local `.ome.zarr` store. DetectionSets, ObjectLists entries with concrete ObjectListMesh geometry, and direct ObjectListMesh components are supported. Images are not duplicated under `poca/`.

`includeLabels=true` and `includePocaData=true` are independent defaults. `componentChunkRows=65536` controls quantitative row chunks independently of image `chunkX/Y/Z`; it must be a positive uint32. The existing File -> Export -> Export image as OME-Zarr... action records these defaults through the existing command dispatch. No additional export action or advanced dialog was introduced.

The direct API is `saveOmeZarr(image, path, options, labels = {}, object = nullptr)`. An explicit owner is required to include quantitative components through this API. The command resolves ownership with Engine and passes that owner to orchestration. Omitting the owner keeps the existing standalone-image API valid.

## Version and hierarchy

`poca/zarr.json` is a normal Zarr v3 group:

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "poca": {
      "version": "0.1",
      "components": [
        {"kind": "points", "path": "points/DetectionSet", "name": "DetectionSet", "component_index": 1}
      ]
    }
  }
}
```

```text
dataset.ome.zarr/
  zarr.json                    # unchanged OME-NGFF 0.5 image group
  0/, 1/, ...                  # scientific image and pyramid
  labels/<safe-label>/...      # standard associated OME labels
  poca/
    zarr.json                  # version and component manifest
    points/<safe-component>/
      zarr.json
      positions/
      features/<safe-feature>/
    meshes/<safe-component>/
      zarr.json
      vertices/, faces/
      vertex_offsets/, face_offsets/
      object_features/<safe-feature>/
      render_axes/             # optional existing rendering cache
```

Every extension container is an ordinary v3 group. Generic array metadata includes shape, data_type, dimension_names, zero fill, regular chunks, default slash chunk-key encoding, attributes and a little-endian bytes codec. Image metadata and its XYZ mapping retain the original image-specific path. Generic `ZarrArrayAccess::readRaw` accepts positive regions of arbitrary nonzero rank and returns packed C-order data through the existing C ABI.

The manifest uses semantic `points` and `mesh_collection` kinds, exact original component names and safe relative paths. Supported top-level component indices are unique; gaps from skipped components are accepted. Unsupported extension versions fail explicitly and cannot trigger image-only fallback. Within version 0.1, unknown kinds with valid basic names/paths are skipped with a diagnostic; unknown auxiliary fields are ignored. Malformed supported components fail the whole load. No older-version conversion is implied.

## Quantitative arrays

| Array | Shape | Dtype | Dimension names | Row chunks |
| --- | --- | --- | --- | --- |
| Point positions | `[N,D]`, D=2 or 3 | float32 | point, coordinate | `[min(N,rows),D]` |
| Point feature | `[N]` | float32 | point | `[min(N,rows)]` |
| Mesh vertices | `[V,3]` | float64 | vertex, coordinate | `[min(V,rows),3]` |
| Mesh faces | `[F,3]` | uint64 | face, corner | `[min(F,rows),3]` |
| Vertex/face offsets | `[O+1]` | uint64 | object_boundary | `[min(O+1,rows)]` |
| Mesh object feature | `[O]` | float32 | object | `[min(O,rows)]` |
| Optional render axes | `[O,3,3]` | float32 | object, axis, coordinate | `[min(O,rows),3,3]` |

Zero-face arrays are permitted and use chunk extents of at least one; zero-sized regions are never read. Components must have positive point/object counts; each mesh object must contain vertices. Count, offset, chunk element and byte/address arithmetic is checked. Current resident PoCA indices constrain point/object/vertex counts to uint32 and mesh triangle corners to uint32; uint64 storage does not remove those application limits.

Both geometry kinds declare `space="poca-world"`. Numeric values are preserved without inventing units, image transforms or parent/source associations. An explicit trustworthy future unit/source association can be auxiliary metadata; this implementation emits none for generic geometry. The existing image calibration and label associations remain separate.

## Point metadata and loading

The component group contains `attributes.poca` with kind, name, space, count, dimension, nbSlices, bounding_box, positions, coordinate_names, coordinate_display, current_feature and features. Bounding boxes use `[xmin,ymin,zmin,xmax,ymax,zmax]`. A valid existing 2D Z range is retained without inventing a third coordinate.

Coordinate names are exactly x/y or x/y/z. They are stored only in the positions matrix and recreated as existing PoCA MyData feature names on load. Coordinate reads select `[start,coordinateIndex]` with shape `[count,1]`; every returned coordinate is checked for finiteness and membership in the persisted bounding box. Export validates coordinates and feature lengths while streaming. Arbitrary additional float features are discovered from BasicComponent::getData rather than a fixed registry.

Opening validates metadata and obtains bounded coordinate/feature samples. It uses persisted count, dimension, bounding box and nbSlices; no full coordinate scan or spatial index is required. Selection starts all true. Display may subsequently require coordinate arrays, independently of spatial indexing.

Storage-backed DetectionSet::getKdTree and getKdTreeCloud call ensureSpatialIndex. The first request materializes only x/y/z, creates the cloud and builds one owned KD-tree under its mutex; later requests reuse it. Existing eager constructors keep their eager behavior. Storage-backed copies clone mutable feature/histogram state and callbacks, retain backing handles, and start without a tree. The destructor deletes its owned tree.

## Feature storage, sampling and display

Supported scientific feature data is explicitly Histogram<float>; other scalar types fail with component/feature/type context instead of lossy conversion. Each feature manifest has exact name, safe relative path and display metadata. The array also stores the exact original name in attributes. Load validates rank 1, exact component length, float32, dimension name, name agreement, nonempty/unique exact names, reserved coordinates, case-insensitive path uniqueness and safe containment.

Display state stores finite min/max, currentMin/currentMax, interaction, scaleLUT and requested log preference. The selected current_feature is preserved and validated. Complete histogram bins are not persisted as scientific values.

At most eight distributed contiguous windows and 32768 values per feature are sampled. Finite samples drive approximate bins/statistics; full scientific float32 values, including signed zero and nonfinite values in extra features, are not replaced by samples. Every lazy feature owns its read-only Zarr handle through value-captured shared ownership. No callback points at the loader or temporary metadata.

Histogram<T> has a generic typed region-reader seam in core, with no Zarr dependency. Resident values take priority over callbacks so resident edits export correctly. Untouched lazy features and coordinate columns stream to another store in bounded chunks without full materialization. Full reads allocate one requested feature, read bounded regions into a temporary vector, and publish it only after success.

An unloaded feature whose filter covers the full known range, or has interaction disabled, contributes no restriction and does not materialize during default selection. Restrictive filtering, explicit data access, CSV/algorithm requests and log transforms may materialize the requested feature. Existing resident selection semantics remain.

Log capability is advertised without eagerly creating log arrays. Saved log preference survives untouched re-export, but the reopened histogram initially displays linearly; an explicit log toggle materializes and computes the log histogram. Active log bounds are serialized in the original value domain. This deliberate UI limitation avoids accidental full-array loading at open. Existing log-domain behavior for nonpositive values remains; invalid/nonfinite display state is rejected.

## Indexed meshes and authoritative measurements

Export enumerates actual Surface_mesh_3_double vertices in iterator order, keeps a deterministic per-object descriptor-to-global-index map, and writes bounded vertex/face buffers. It does not export only expanded rendering triangles. One first pass computes offsets and verifies triangular topology. No second permanent flattened O(V+F) topology buffer is required.

Offsets start at zero, are nondecreasing, have O+1 entries and end exactly at V/F. Faces have three distinct global vertex indices; every index must be below V and inside that object's vertex interval. Load validates all stored face indices before constructing CGAL topology, then remaps to local descriptors. add_face failure is an error. The saved first corner and cyclic orientation are preserved for re-export.

The persistence constructor adopts indexed meshes without repair, stitching, orientation correction, triangulation, PCA measurement calculation or remeshing. Double coordinates remain authoritative in CGAL; float rendering arrays, normals, bounding boxes and centroids are rebuilt for existing APIs. The use_vertex_normals display choice is preserved. Rendering coordinates must be finite and representable by float32. Scientific feature arrays are adopted lazily, with safe ownership; no constructor-generated measurements replace persisted values.

`render_axes` stores existing finite per-object axis directions only as an auxiliary rendering cache. It preserves ellipsoid overlays without recomputing measurements. It is null/absent as an array when the source has no cache; ellipsoid measurements require a matching cache. It is not a general per-vertex/per-face feature registry.

Existing ObjectListMesh `z` may be triangle-corner display data rather than O object values. Export accepts this special case only when noninteractive, length exactly 3F and every value agrees with rendered geometry. It records `derived_triangle_z` display metadata and regenerates that display array on reopen; it never mislabels it as an object feature. A genuine `[O]` z feature uses normal feature storage. Other mismatched feature lengths fail.

Mesh geometry, offsets, rendering arrays and caches are resident after reopen. Per-object scientific features remain lazy until consumers request them. Rendering an ellipsoid can request major/minor/minor2; rendering the current coloring feature can request that feature. This is not out-of-core mesh rendering.

## ObjectLists and whole-object integration

Each container-derived manifest entry adds `container: {id,name,entry_index,entry_name,plugin,command}`. The id groups supported entries; component_index preserves supported top-level order and entry_index preserves supported entry order. Exact user-visible strings, including empty entry/plugin strings, are retained. CommandInfo::toNormalizedJson writes the existing name/params/recordable representation; load validates and restores it with fromJson. Full command history is not stored.

LoaderZarr::loadObject returns nullptr when poca/zarr.json is absent, preserving the ordinary loadData route. When present, it validates the version/manifest, reuses one shared image-and-label helper, builds points/meshes and restores ObjectLists. Supported load failures propagate explicitly rather than returning an incomplete image-only object.

The new owning Engine::createObject overload accepts all components, uses existing addComponentToObject for each container and its children, installs object commands once, and registers exactly one dataset only after assembly succeeds. Components are added in reverse because MyObject inserts at the front. Unique names and unique_ptr ownership prevent ambiguous merges and failure leaks. Existing raw-pointer APIs remain available.

## Transaction, security and memory

Image arrays, labels and poca arrays are all written inside the same existing sibling staging directory. Quantitative destination handles close before extension metadata; poca manifest is validated before existing root publication. Supported component errors abort export and invoke existing staging cleanup/backup rollback. Unsupported components emit one diagnostic and are skipped. No partial supported component is published.

The shared Phase-4 sanitizer retains label behavior: separators/invalid/control characters are sanitized, UTF-8 bytes escaped, Windows device names protected, bases bounded and case-insensitive collisions resolved with deterministic suffixes. Exact display names remain metadata. Extension paths are safe ASCII relative children, cannot be absolute/drive/backslash/traversal/self references, must remain canonically contained, and refuse symlink/reparse surprises.

BasicComponent reports resident histogram/sample/selection storage rather than claiming unloaded feature lengths as RAM. Storage-backed DetectionSet adds resident coordinates/cloud/tree; ObjectListMesh reports measured resident rendering arrays/caches and features. Reports saturate the existing unsigned-int API. CGAL property maps/allocator overhead, map/string/callback overhead and backend internal scratch are not fully measured; these are lower bounds, not a memory-budget guarantee. Source/destination and callbacks must remain stable during synchronous export and copy operations.

## Unsupported scope and known limitations

Skeleton, TrackSet, ROIs, annotations, Delaunay, Voronoi, arbitrary ObjectListPolygon, graph topology, Organograph hierarchy, complete command history/provenance, generic per-vertex/per-face feature registries, remote/cloud edits and out-of-core mesh rendering are not persisted. Unknown component kinds are skipped; unsupported feature types inside supported components abort export.

Image scope stays one selected scalar RAW image plus associated labels, not every RAW image in an ImagesList. Existing restrictions on multichannel/time/plate stores, nonfinite image display metadata, same-store replacement handles and local synchronous transaction crash durability remain. Opening histograms consumes O(features * bounded sample) memory; selection costs O(N) bits; a first KD request costs O(N) coordinates/tree. Explicit large componentChunkRows can increase export scratch memory. All Qt/backend/CGAL/filesystem runtime behavior remains UNCONFIRMED.

## Source fixtures and requested coverage

`POCA_ZARR_EXPORT_SOURCE_TESTS` remains OFF by default. Existing manual Tests/Images registration adds a combined schema/array/lazy/geometry/dataset action and an Engine whole-object action. No source fixture was executed. The following matrix maps every requested case; structural properties also require the stated source audit, not an assertion of runtime proof.

| Requested cases | Source fixtures / static evidence |
| --- | --- |
| 1–9 schema/version/manifest/names/paths | PocaZarrSchemaTests; shared sanitizer + manifest audit |
| 10–15 generic rank/bytes/bounds/dtypes | PocaZarrArrayTests; readRaw interval/arithmetic review; rank-3 axes fixture |
| 16–22 point export | PocaZarrPointsTests: 2D/3D, no duplicate coordinates, arbitrary/constant features, bbox/slices, large chunked reader |
| 23–25 metadata discovery/one-feature access | PocaZarrLazyTests + PointsTests + DatasetTests |
| 26–28 lazy x/y/z columns | PocaZarrPointsTests selected-column reads for both dimensions |
| 29–34 lazy KD/default/restrictive/log | PocaZarrLazyTests; storage constructor and getter audit |
| 35–40 point round trip | PocaZarrPointsTests float bit comparisons, names/count/bbox/slices |
| 41–47 mesh export | PocaZarrMeshTests: two indexed meshes, float64/u64, offsets, globals, nontriangle rejection, object features |
| 48 bounded topology output | Mesh export source audit: two bounded buffers + per-object descriptor map + O(O) offsets |
| 49–57 mesh load | PocaZarrMeshTests: real malformed offsets/faces, cross-object/out-of-range rejection, exact coordinates/connectivity, lazy authoritative features; constructor audit |
| 58–62 ObjectLists | PocaZarrDatasetTests: two entries, order/exact names/plugins/normalized CommandInfo |
| 63–67 transaction | PocaZarrDatasetTests image+label+poca, unsupported skip, failure/old destination/cleanup, OME root and independent options |
| 68–72 whole-object path | PocaZarrDatasetTests ordinary loadObject null + old loadData, actual Engine dispatch with all supported components |
| 73–74 commands/registration | Dataset whole-object fixture command type uniqueness/owner/dataset count; assembly source audit proves one installation path per component/container child |
| 75–76 lazy re-export | PocaZarrPointsTests + MeshTests + DatasetTests: chunk copy and unloaded state; feature-export source audit |

Additional PocaZarrFeatureTests cover rank/length/dtype/domain/original-name rejection, empty names, case collisions, reversed display metadata and scientific nonfinite/signed-zero float32 bits. Mesh fixtures also cover preserved rendering axes and the derived triangle-z recipe. Existing Phase-3/4 image/label fixtures remain registered.

## Static audit

Reviewed all 28 requested audit areas at source level: ordinary image/label route, separate generic image/raw reads, backend dependency boundary, one staging transaction, lazy histogram/KD/copy/selection paths, indexed double/u64 topology without repair, authoritative features/entry metadata, unsupported skips/no guessed provenance, scope, checked arithmetic, ownership and Windows safe names. Final audit passes 66 UTF-8/no-BOM CRLF files, 59 balanced lexical source streams and 230 existing source-list paths, local quoted includes, new-file whitespace, quoting and git diff --check. The inventory matches 32 added and 34 modified files; results are also recorded in CONTINUITY.md. This cannot establish compilation or runtime correctness.

## Manual validation after the user compiles

1. Open an image-only OME-Zarr; verify the old image-only route and image display.
2. Open image + labels; verify label IDs, colors, source association and existing rendering.
3. Export a selected RAW image with a DetectionSet, includePocaData enabled; inspect image and poca manifests.
4. Close/reopen and display detections; compare point count, 2D/3D coordinates, bbox and nbSlices.
5. Compare feature names/current feature/histograms and display bounds; expect approximate bins initially.
6. Access one extra feature; verify that unrelated features stay unloaded. Check a restrictive filter and explicit log toggle.
7. Run a nearest-neighbor/KD operation; verify coordinate loading, one index build and untouched extra features.
8. Export image + one ObjectListMesh; inspect float64 vertices, uint64 faces/offsets and object feature manifests.
9. Reopen and visually compare mesh geometry, normals and ellipsoid overlays; compare exact coordinates/connectivity numerically.
10. Compare all supported object feature values, including added measurements and any derived triangle-z coloring.
11. Round-trip multiple mesh entries in ObjectLists; verify order, entry names, plugins and normalized commands.
12. Re-export an untouched reopened extended store; verify identical supported values and that feature vectors remain unloaded.
13. Use a very large point/feature set; inspect bounded samples, selection RAM, one-feature reads and first KD memory separately.
14. Inject a supported-component/read failure during overwrite; verify the old destination remains and staging/backup cleanup or exact recovery diagnostics.
15. Open the result in an external OME reader; verify image and labels remain understandable while poca/ is ignored.

## Complete Phase-6 changed-file inventory

This inventory separates new files from modified files. CMake files were edited only to register sources, never executed.

### Added

- `poca/docs/POCA_ZARR_EXTENSION.md`
- `poca/src/poca_core/General/EngineObjectAssembly.cpp`
- `poca/src/poca_geometry/Geometry/DetectionSetStorage.cpp`
- `poca/src/poca_geometry/Geometry/ObjectListMeshPersistence.cpp`
- `poca_extra/src/poca_loaderZarrFile/LoaderZarrObject.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrObjectLoad.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrObjectLoad.hpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrArrayTests.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrDatasetTests.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrExtension.hpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrExtensionExport.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrExtensionLoad.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrFeatureDisplay.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrFeatureTests.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrFeatures.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrFeatures.hpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrLazyTests.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrManifest.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrMeshDisplay.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrMeshExport.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrMeshLoad.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrMeshTests.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrMeshes.hpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrPoints.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrPoints.hpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrPointsTests.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrSchema.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrSchema.hpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrSchemaTests.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrTestSupport.hpp`
- `poca_extra/src/poca_loaderZarrFile/ZarrSafeNames.cpp`
- `poca_extra/src/poca_loaderZarrFile/ZarrSafeNames.hpp`

### Modified

- `CONTINUITY.md`
- `poca/src/poca_core/CMakeLists.txt`
- `poca/src/poca_core/General/BasicComponent.cpp`
- `poca/src/poca_core/General/BasicComponent.hpp`
- `poca/src/poca_core/General/BasicComponentList.hpp`
- `poca/src/poca_core/General/Engine.hpp`
- `poca/src/poca_core/General/Histogram.hpp`
- `poca/src/poca_core/General/ImagesList.cpp`
- `poca/src/poca_core/General/ImagesList.hpp`
- `poca/src/poca_core/General/MyData.cpp`
- `poca/src/poca_core/General/MyData.hpp`
- `poca/src/poca_core/Interfaces/HistogramInterface.hpp`
- `poca/src/poca_geometry/CMakeLists.txt`
- `poca/src/poca_geometry/Geometry/DetectionSet.cpp`
- `poca/src/poca_geometry/Geometry/DetectionSet.hpp`
- `poca/src/poca_geometry/Geometry/ObjectListMesh.hpp`
- `poca/src/poca_geometry/Geometry/ObjectLists.cpp`
- `poca/src/poca_geometry/Geometry/ObjectLists.hpp`
- `poca_extra/src/poca_loaderZarrFile/CMakeLists.txt`
- `poca_extra/src/poca_loaderZarrFile/EXPORT_README.md`
- `poca_extra/src/poca_loaderZarrFile/LoaderZarr.cpp`
- `poca_extra/src/poca_loaderZarrFile/LoaderZarr.hpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExport.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExport.hpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExportCommand.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExportGui.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExportGuiTests.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExportMetadataTests.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExportTests.hpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrLabelsLoad.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrLabelsMetadata.cpp`
- `poca_extra/src/poca_loaderZarrFile/README.md`
- `poca_extra/src/poca_loaderZarrFile/ZarrArrayAccess.cpp`
- `poca_extra/src/poca_loaderZarrFile/ZarrArrayAccess.hpp`

Suggested commit: `feat(zarr): persist PoCA points meshes and lazy features`.

No CMake configure/generate, compilation, linking, installation, application executable, Python script, benchmark, or test was run.
