"""Compare two PoCA renderPerformance CSV captures. Standard library only."""
import argparse
import csv
import math
import statistics
from pathlib import Path


def capture(path, camera, warmup, samples, trace):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    cameras = {row["camera_id"] for row in rows if row["kind"] == "paint"}
    if camera is None:
        if len(cameras) != 1:
            raise ValueError(f"{path}: choose --camera from {sorted(cameras)}")
        camera = next(iter(cameras))
    rows = [
        row for row in rows
        if row["kind"] == "paint" and row["camera_id"] == camera
        and (trace or row["interactive"] == "0")
    ]
    identities = [row["frame_id"] for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError(f"{path}: repeated frame IDs; use a fresh file per process/capture")
    rows.sort(key=lambda row: int(row["frame_id"]))
    rows = rows[warmup:]
    if samples:
        if len(rows) < samples:
            raise ValueError(f"{path}: need {samples} post-warmup paints, found {len(rows)}")
        rows = rows[:samples]
    if not rows:
        raise ValueError(f"{path}: no matching post-warmup paints")
    return rows


def summarize(name, rows):
    times = sorted(float(row["cpu_ms"]) for row in rows)
    p95 = times[math.ceil(0.95 * len(times)) - 1]
    print(f"{name}: paints={len(times)} median_ms={statistics.median(times):.4f} p95_ms={p95:.4f}")
    fields = (
        "scene_preparation_us", "object_display_us", "raw_volume_gl_draws_single",
        "raw_volume_gl_draws_multi", "nonraw_volume_gl_draws", "grid_face_draws",
        "grid_line_draws", "bbox_draws", "child_frame_recomputes",
        "frame_geometry_glBufferData_calls", "frame_geometry_glBufferData_bytes",
        "overlap_pair_tests", "overlap_cache_hits", "visibility_calls", "image_share_calls",
        "residency_map_entries_examined", "lod_request_attempts", "lod_requests_enqueued",
        "ready_uploads_processed", "texture3d_allocation_calls", "texture3d_upload_chunks",
        "texture3d_uploaded_bytes", "picking_fbo_creates", "picking_fbo_clears", "depth_clears",
    )
    for field in fields:
        total = sum(int(row[field]) for row in rows)
        print(f"  {field}: total={total} per_paint={total / len(rows):.4f}")
    print("  GPU execution time: NOT MEASURED (gpu_timing_available=0)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--camera", help="Use this camera ID for both files")
    parser.add_argument("--before-camera", help="Camera ID in the before file")
    parser.add_argument("--after-camera", help="Camera ID in the after file")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=120, help="0 keeps the full post-warmup trace")
    parser.add_argument("--trace", action="store_true", help="Include interactive paints for navigation/resize traces")
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 0:
        parser.error("warmup and samples must be nonnegative")
    for name, path, camera in (("before", args.before, args.before_camera), ("after", args.after, args.after_camera)):
        summarize(name, capture(path, camera or args.camera, args.warmup, args.samples, args.trace))


if __name__ == "__main__":
    main()
