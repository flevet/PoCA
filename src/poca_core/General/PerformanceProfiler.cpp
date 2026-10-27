/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      PerformanceProfiler.cpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*/

#include "PerformanceProfiler.hpp"

#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <locale>
#include <QtCore/QFile>
#include <QtCore/QFileInfo>
#include <QtCore/QCoreApplication>
#include <QtCore/QVariant>

namespace poca::core {

	PerformanceProfiler::ScopedTimer::ScopedTimer(const std::string& _category, const std::string& _name)
		: m_category(_category), m_name(_name), m_start(std::chrono::high_resolution_clock::now())
	{
	}

	PerformanceProfiler::ScopedTimer::~ScopedTimer()
	{
		auto end = std::chrono::high_resolution_clock::now();
		double ms = std::chrono::duration<double, std::milli>(end - m_start).count();
		PerformanceProfiler::instance().record(m_category, m_name, ms);
	}

	PerformanceProfiler::ImageTimer::ImageTimer(ImageMetric metric, int preparationRole) : m_metric(metric), m_role(preparationRole)
	{
		auto& profiler = PerformanceProfiler::instance();
		if (!profiler.imageMetricsEnabled()) return;
		m_profiler = &profiler; m_start = std::chrono::steady_clock::now();
		if (m_role >= 0) {
			std::lock_guard<std::mutex> lock(profiler.m_mutex);
			profiler.m_imagePreparationRoles[std::this_thread::get_id()] = m_role;
		}
	}

	PerformanceProfiler::ImageTimer::~ImageTimer()
	{
		if (!m_profiler) return;
		m_profiler->addImageMetric(m_metric, uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - m_start).count()));
		if (m_role >= 0) {
			std::lock_guard<std::mutex> lock(m_profiler->m_mutex);
			m_profiler->m_imagePreparationRoles.erase(std::this_thread::get_id());
		}
	}

	void PerformanceProfiler::setImageMetricsEnabled(bool enabled)
	{
		if (m_imageMetricsEnabled.exchange(enabled, std::memory_order_relaxed) == enabled) return;
		for (auto& value : m_imageMetrics) value.store(0, std::memory_order_relaxed);
		m_imageReportStart = std::chrono::steady_clock::now();
	}

	void PerformanceProfiler::addImageMetric(ImageMetric metric, uint64_t value)
	{
		if (imageMetricsEnabled()) m_imageMetrics[size_t(metric)].fetch_add(value, std::memory_order_relaxed);
	}

	void PerformanceProfiler::recordImageTiffRead(uint64_t planes, uint64_t stackEquivalents, bool firstRead)
	{
		if (!imageMetricsEnabled()) return;
		addImageMetric(ImageMetric::TiffPlanes, planes); addImageMetric(ImageMetric::TiffStackEquivalents, stackEquivalents);
		if (!firstRead) return;
		std::lock_guard<std::mutex> lock(m_mutex);
		const auto role = m_imagePreparationRoles.find(std::this_thread::get_id());
		if (role == m_imagePreparationRoles.end()) return; // Scientific/opening reads have no display role.
		if (role->second == 0) addImageMetric(ImageMetric::TiffPreviewSessions);
		else if (role->second == 1) addImageMetric(ImageMetric::TiffNavigationSessions);
		else if (role->second == 2) addImageMetric(ImageMetric::TiffDetailSessions);
	}

	void PerformanceProfiler::reportImageMetrics(size_t queued, size_t ready, size_t preparing)
	{
		if (!imageMetricsEnabled()) return;
		const auto now = std::chrono::steady_clock::now();
		const double seconds = std::chrono::duration<double>(now - m_imageReportStart).count();
		if (seconds < 1.) return;
		m_imageReportStart = now;
		static const std::array<const char*, size_t(ImageMetric::Count)> names{
			"displays", "display_us", "batches", "geometry_allocations", "descriptor_allocations", "descriptor_uploads",
			"visibility_hits", "visibility_misses", "visibility_us", "picking_passes", "picking_us",
			"upload_frames", "uploads", "upload_us", "preview_preparations", "navigation_preparations", "detail_preparations",
			"preparation_us", "completions", "failures", "refinements_avoided", "detail_retirements",
			"redraw_requests", "redraw_scheduled", "streaming_frames",
			"tiff_preview_sessions", "tiff_navigation_sessions", "tiff_detail_sessions", "tiff_planes", "tiff_stack_equivalents"
		};
		std::ostringstream line;
		line << "[PoCA][ImagePerformance] seconds=" << seconds;
		for (size_t i = 0; i < names.size(); ++i) line << ' ' << names[i] << '=' << m_imageMetrics[i].exchange(0, std::memory_order_relaxed);
		line << " queued=" << queued << " ready=" << ready << " preparing=" << preparing;
		std::cout << line.str() << std::endl;
	}

	PerformanceProfiler& PerformanceProfiler::renderInstance()
	{
		static PerformanceProfiler& profiler = instance();
		return profiler;
	}

	const char* PerformanceProfiler::renderMetricName(RenderMetric metric)
	{
		static const std::array<const char*, size_t(RenderMetric::VisibleBegin)> names{
			"paintgl_calls", "draw_elements_onscreen", "draw_elements_offscreen", "recompute_frame_calls", "child_frame_recomputes",
			"grid_face_draws", "grid_line_draws", "bbox_draws", "raw_volume_gl_draws_single", "raw_volume_gl_draws_multi", "nonraw_volume_gl_draws",
			"other_audited_gl_draws", "picking_gl_draws", "shared_descriptor_glBufferData_calls", "shared_descriptor_glBufferData_bytes",
			"shared_descriptor_glBufferSubData_calls", "shared_descriptor_glBufferSubData_bytes", "frame_geometry_glBufferData_calls", "frame_geometry_glBufferData_bytes",
			"overlap_pair_tests", "overlap_cache_hits", "overlap_cache_misses", "batch_count", "batch_max_images", "visible_images", "culled_images",
			"child_display_dispatches", "picking_fbo_creates", "picking_fbo_clears", "visibility_calls", "visibility_cache_hits", "visibility_cache_misses",
			"image_share_calls", "residency_map_entries_examined", "lod_request_attempts", "lod_requests_enqueued", "lod_requests_deduplicated",
			"lod_priority_requeues", "lod_continuations_enqueued", "ready_uploads_processed", "texture3d_allocation_calls", "texture3d_upload_chunks",
			"texture3d_uploaded_bytes", "stream_redraw_requests", "stream_redraw_scheduled", "depth_clears", "frame_cache_hits", "frame_cache_misses",
			"frame_cache_entries", "frame_cache_bytes", "multi_batch_fallback_incompatible_settings", "multi_batch_fallback_overlays", "multi_batch_fallback_limit",
			"draw_elements_us", "grid_preparation_us", "grid_draw_us", "child_dispatch_us", "visibility_us", "lod_planning_us", "images_list_us",
			"multi_display_us", "overlap_us", "texture_upload_us", "picking_us", "scene_preparation_us", "object_display_us", "lod_requests_deferred", "texture3d_submitted_bytes", "render_frame_stack_spills",
			"objectlist_pick_draw_invocations",
			"objectlist_pick_gl_draws",
			"objectlist_pick_cpu_us",
			"objectlist_pick_fbo_allocations",
			"objectlist_pick_fbo_reallocations",
			"objectlist_pick_fbo_live",
			"objectlist_pick_fbo_peak_live",
			"objectlist_pick_fbo_requested_bytes",
			"objectlist_pick_fbo_unknown_payloads",
			"objectlist_pick_fbo_bind_attempts",
			"objectlist_pick_fbo_binds",
			"objectlist_pick_fbo_bind_failures",
			"objectlist_pick_fbo_clears",
			"objectlist_pick_readpixels_calls",
			"objectlist_pick_readpixels_us",
			"objectlist_pick_components_visited",
			"objectlist_pick_components_rendered",
			"objectlist_pick_read_hits",
			"objectlist_pick_read_misses",
			"objectlist_pick_fbo_outside_draw_cpu_us",
			"objectlist_pick_gpu_supported",
			"objectlist_pick_gpu_unavailable",
			"objectlist_pick_gpu_query_count",
			"objectlist_pick_gpu_sample_count",
			"objectlist_pick_gpu_elapsed_ns",
			"objectlist_pick_gpu_skipped",
			"detectionset_pick_draw_invocations",
			"detectionset_pick_gl_draws",
			"detectionset_pick_cpu_us",
			"detectionset_pick_fbo_allocations",
			"detectionset_pick_fbo_reallocations",
			"detectionset_pick_fbo_live",
			"detectionset_pick_fbo_peak_live",
			"detectionset_pick_fbo_requested_bytes",
			"detectionset_pick_fbo_unknown_payloads",
			"detectionset_pick_fbo_bind_attempts",
			"detectionset_pick_fbo_binds",
			"detectionset_pick_fbo_bind_failures",
			"detectionset_pick_fbo_clears",
			"detectionset_pick_readpixels_calls",
			"detectionset_pick_readpixels_us",
			"detectionset_pick_components_visited",
			"detectionset_pick_components_rendered",
			"detectionset_pick_read_hits",
			"detectionset_pick_read_misses",
			"detectionset_pick_fbo_outside_draw_cpu_us",
			"detectionset_pick_gpu_supported",
			"detectionset_pick_gpu_unavailable",
			"detectionset_pick_gpu_query_count",
			"detectionset_pick_gpu_sample_count",
			"detectionset_pick_gpu_elapsed_ns",
			"detectionset_pick_gpu_skipped",
			"pick_hover_dispatches",
			"pick_left_dispatches",
			"pick_right_dispatches",
			"pick_other_dispatches",
			"local_bounds_calls", "local_bounds_us", "collection_bounds_calls", "collection_bounds_us",
			"collection_preparation_us", "collection_grid_calls", "collection_grid_us", "bbox_preparation_calls", "bbox_preparation_us",
			"objectlist_batching_disabled", "detectionset_batching_disabled"
		};
		if (size_t(metric) < names.size()) return names.at(size_t(metric));
		// Construct once on CSV/report access, never in a disabled or per-object hot path.
		static const auto visibleNames = [] {
			const std::array<const char*, size_t(VisibleFamily::None)> families{
				"objectlist_individual_", "objectlist_shared_", "detectionset_individual_", "detectionset_shared_" };
			const std::array<const char*, size_t(VisibleMetric::Count)> fields{
				"display_calls", "display_us", "offscreen_calls", "components_visited", "components_selected", "components_drawn",
				"unselected", "no_display_bounds", "no_frustum_test", "handled", "fallback", "fallback_no_family", "fallback_settings",
				"missing_command", "fallback_build", "fallback_memory", "builds", "build_us", "build_bytes",
				"build_points", "build_triangle_vertices", "build_ellipsoid_instances", "can_batch_calls", "can_batch_us",
				"transform_calls", "transform_us", "ranges", "triangle_visibility_checks", "triangle_unselected_spans",
				"helper_invocations", "helper_gl_draws", "gl_draws", "draw_ranges", "indexed_triangle_indices", "array_triangle_vertices",
				"submitted_triangles", "wire_triangle_elements", "submitted_points", "submitted_line_vertices",
				"submitted_quad_indices", "submitted_quads", "ellipsoid_instances", "gpu_deferred",
				"points_us", "points_draws", "points_submitted", "points_disabled", "points_empty",
				"outline_us", "outline_draws", "outline_submitted", "outline_disabled", "outline_empty",
				"opaque_us", "opaque_draws", "opaque_submitted", "opaque_disabled", "opaque_empty",
				"translucent_us", "translucent_draws", "translucent_submitted", "translucent_disabled", "translucent_empty",
				"ellipsoids_us", "ellipsoids_draws", "ellipsoids_submitted", "ellipsoids_disabled", "ellipsoids_empty",
				"skeleton_us", "skeleton_draws", "skeleton_submitted", "skeleton_disabled", "skeleton_empty",
				"links_us", "links_draws", "links_submitted", "links_disabled", "links_empty",
				"bbox_us", "bbox_draws", "bbox_submitted", "bbox_disabled", "bbox_empty",
				"frustum_tests", "frustum_us", "frustum_culled", "frustum_bypassed", "frustum_deferred_builds",
				"surface_build_us", "surface_allocations", "surface_upload_bytes",
				"point_build_us", "point_allocations", "point_upload_bytes", "point_deferred_builds", "point_deferred_estimated_bytes",
				"ellipsoid_build_us", "ellipsoid_allocations", "ellipsoid_upload_bytes", "ellipsoid_deferred_builds", "ellipsoid_deferred_estimated_bytes",
				"outline_dormant", "outline_build_us", "outline_allocations", "outline_upload_bytes",
				"lod_planning_us", "lod_tests", "lod_subpixel_samples", "lod_one_to_two_samples", "lod_larger_samples", "lod_unknown_samples",
				"lod_subpixel_triangle_samples", "lod_full_draw_fallbacks", "lod_suppressed_objects", "lod_omitted_triangles", "lod_retained_ranges" };
			std::array<std::string, size_t(VisibleFamily::None) * size_t(VisibleMetric::Count)> result;
			for (size_t family = 0; family < families.size(); ++family)
				for (size_t field = 0; field < fields.size(); ++field)
					result[family * fields.size() + field] = std::string(families[family]) + fields[field];
			result[size_t(VisibleFamily::ObjectListShared) * fields.size() + size_t(VisibleMetric::BuildBytes)] = "objectlist_shared_build_admitted_estimated_peak_bytes";
			return result;
		}();
		return visibleNames.at(size_t(metric) - names.size()).c_str();
	}

	PerformanceProfiler::~PerformanceProfiler() { stopRenderCsv(); }

	void PerformanceProfiler::setRenderMetricsEnabled(bool enabled)
	{
		if (renderMetricsEnabled() == enabled) return;
		if (m_activeRender || std::any_of(m_betweenTimers.begin(), m_betweenTimers.end(), [](bool active) { return active; })) return; // Preserve active scopes.
		m_renderEnabled.store(false, std::memory_order_relaxed);
		stopRenderCsv();
		if (m_renderThread == std::thread::id{}) m_renderThread = std::this_thread::get_id();
		else if (m_renderThread != std::this_thread::get_id()) throw std::logic_error("Render profiler configuration requires its GUI thread");
		m_renderRingNext = m_renderRingSize = m_renderWindowFrames = 0;
		m_renderTotals.fill(0); m_renderBetweenFrames.fill(0);
		for (auto& value : m_renderWorkers) value.store(0, std::memory_order_relaxed);
		m_renderCsvDropped.store(0, std::memory_order_relaxed);
		m_renderReportStart = std::chrono::steady_clock::now();
		if (enabled) {
			++m_renderSession;
			const QByteArray path = qgetenv("POCA_RENDER_PROFILE_CSV");
			if (!path.isEmpty()) {
				if (!QFileInfo(QString::fromLocal8Bit(path)).isAbsolute())
					std::cerr << "[PoCA][RenderPerformance] CSV requires an absolute path; export disabled" << std::endl;
				else {
					m_renderCsvStop = false; m_renderCsvActive = true;
					m_renderCsvThread = std::thread([this, path]() { renderCsvLoop(path.toStdString()); });
				}
			}
		}
		m_renderEnabled.store(enabled, std::memory_order_release);
	}

	PerformanceProfiler::RenderFrame::RenderFrame(uint64_t camera, bool paint, bool offscreen, bool multiple, uint64_t children, int width, int height, bool interactive)
		: RenderFrame(camera, paint, offscreen, multiple, children, width, height, interactive, nullptr, true)
	{
	}

	PerformanceProfiler::RenderFrame::RenderFrame(uint64_t camera, bool paint, bool offscreen, bool multiple, uint64_t children, int width, int height, bool interactive, const char* kind, bool active)
	{
		auto& profiler = renderInstance();
		if (!active || !profiler.renderMetricsEnabled()) return;
		if (std::this_thread::get_id() != profiler.m_renderThread) return;
		// Only a same-camera onscreen draw inside paint is part of that paint record.
		if (!kind && !paint && !offscreen && profiler.m_activeRender && profiler.m_activeRender->paint && profiler.m_activeRender->cameraId == camera) {
			++profiler.m_activeRender->values[size_t(RenderMetric::DrawElementsOnscreen)];
			return;
		}
		m_profiler = &profiler; m_parent = profiler.m_activeRender;
		if (profiler.m_renderDepth < profiler.m_renderStack.size()) m_record = &profiler.m_renderStack[profiler.m_renderDepth];
		else {
			renderCount(RenderMetric::FrameStackSpills);
			m_overflowRecord = std::make_unique<RenderRecord>(); m_record = m_overflowRecord.get();
		}
		++profiler.m_renderDepth;
		*m_record = RenderRecord{};
		m_record->frameId = ++profiler.m_renderSequence; m_record->cameraId = camera; m_record->children = children;
		m_record->paint = paint; m_record->offscreen = offscreen; m_record->multiple = multiple; m_record->interactive = interactive;
		m_record->width = width; m_record->height = height; m_record->kind = kind;
		m_record->timestampUs = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
		if (!kind) ++m_record->values[size_t(paint ? RenderMetric::PaintGLCalls : (offscreen ? RenderMetric::DrawElementsOffscreen : RenderMetric::DrawElementsOnscreen))];
		profiler.m_activeRender = m_record;
		m_start = std::chrono::steady_clock::now();
	}

	PerformanceProfiler::RenderFrame::~RenderFrame()
	{
		if (!m_profiler) return;
		m_record->cpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - m_start).count();
		m_profiler->m_activeRender = m_parent; --m_profiler->m_renderDepth;
		m_profiler->finishRenderRecord(*m_record);
	}

	PerformanceProfiler::RenderTimer::RenderTimer(RenderMetric metric, bool active) : m_metric(metric)
	{
		auto& profiler = renderInstance();
		if (!active || !profiler.renderMetricsEnabled()) return;
		if (std::this_thread::get_id() != profiler.m_renderThread) return;
		auto* record = profiler.m_activeRender;
		if (!record) {
			if (profiler.m_betweenTimers[size_t(metric)]) return;
			profiler.m_betweenTimers[size_t(metric)] = true; m_betweenProfiler = &profiler;
		}
		else {
			if (record->activeTimers[size_t(metric)]) return;
			m_record = record; record->activeTimers[size_t(metric)] = true;
		}
		m_start = std::chrono::steady_clock::now();
	}
	PerformanceProfiler::RenderTimer::~RenderTimer() { stop(); }
	void PerformanceProfiler::RenderTimer::stop()
	{
		if (m_betweenProfiler) {
			if (m_betweenProfiler->renderMetricsEnabled()) m_betweenProfiler->m_renderBetweenFrames[size_t(m_metric)] +=
				uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - m_start).count());
			m_betweenProfiler->m_betweenTimers[size_t(m_metric)] = false; m_betweenProfiler = nullptr; return;
		}
		if (!m_record) return;
		m_record->values[size_t(m_metric)] += uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - m_start).count());
		m_record->activeTimers[size_t(m_metric)] = false;
		m_record = nullptr;
	}

	void PerformanceProfiler::renderCount(RenderMetric metric, uint64_t value)
	{
		auto& profiler = renderInstance();
		if (!profiler.renderMetricsEnabled()) return;
		if (std::this_thread::get_id() != profiler.m_renderThread) return;
		auto& values = profiler.m_activeRender ? profiler.m_activeRender->values : profiler.m_renderBetweenFrames;
		values[size_t(metric)] += value;
	}
	void PerformanceProfiler::renderMaximum(RenderMetric metric, uint64_t value)
	{
		auto& profiler = renderInstance();
		if (!profiler.renderMetricsEnabled()) return;
		if (std::this_thread::get_id() != profiler.m_renderThread) return;
		auto& values = profiler.m_activeRender ? profiler.m_activeRender->values : profiler.m_renderBetweenFrames;
		values[size_t(metric)] = std::max(values[size_t(metric)], value);
	}
	PerformanceProfiler::RenderMetric PerformanceProfiler::pickMetric(PickFamily family, PickMetric metric)
	{
		return RenderMetric(size_t(RenderMetric::ObjectPickDrawInvocations) + size_t(family) * size_t(PickMetric::Count) + size_t(metric));
	}
	void PerformanceProfiler::pickCount(PickFamily family, PickMetric metric, uint64_t value)
	{
		if (family != PickFamily::None) renderCount(pickMetric(family, metric), value);
	}
	void PerformanceProfiler::pickingDraw()
	{
		auto& profiler = renderInstance();
		if (!profiler.renderMetricsEnabled() || std::this_thread::get_id() != profiler.m_renderThread) return;
		auto& values = profiler.m_activeRender ? profiler.m_activeRender->values : profiler.m_renderBetweenFrames;
		++values[size_t(RenderMetric::PickingDraws)];
		if (profiler.m_activePickFamily != PickFamily::None) ++values[size_t(pickMetric(profiler.m_activePickFamily, PickMetric::GlDraws))];
	}
	void PerformanceProfiler::pickFboChange(PickFamily family, int delta, uint64_t bytes, bool known)
	{
		if (family == PickFamily::None) return;
		auto& profiler = renderInstance();
		const auto i = size_t(family);
		// Resource census persists while disabled: fixed scalar bookkeeping only, no GL/locks/allocation.
		if (delta > 0) {
			++profiler.m_pickLive[i]; profiler.m_pickBytes[i] += bytes;
			if (!known) ++profiler.m_pickUnknown[i];
			profiler.m_pickPeak[i] = std::max(profiler.m_pickPeak[i], profiler.m_pickLive[i]);
		}
		else {
			--profiler.m_pickLive[i]; profiler.m_pickBytes[i] -= bytes;
			if (!known) --profiler.m_pickUnknown[i];
		}
	}
	bool PerformanceProfiler::hasPickEvent() const { return m_activeRender && m_activeRender->kind && std::strcmp(m_activeRender->kind, "pick_event") == 0; }
	uint64_t PerformanceProfiler::currentRenderId() const { return m_activeRender ? m_activeRender->frameId : 0; }
	void PerformanceProfiler::renderSource(uint64_t id) { auto& profiler = renderInstance(); if (profiler.m_activeRender) profiler.m_activeRender->sourceRecordId = id; }
	PerformanceProfiler::PickScope::PickScope(PickFamily family) : m_family(family)
	{
		auto& profiler = renderInstance();
		if (!profiler.renderMetricsEnabled() || std::this_thread::get_id() != profiler.m_renderThread) return;
		m_profiler = &profiler; m_previous = profiler.m_activePickFamily; profiler.m_activePickFamily = family;
		pickCount(family, PickMetric::DrawInvocations);
		m_draws = profiler.m_activeRender ? profiler.m_activeRender->values[size_t(pickMetric(family, PickMetric::GlDraws))] : 0;
		m_timer.emplace(pickMetric(family, PickMetric::CpuUs));
	}
	PerformanceProfiler::PickScope::~PickScope()
	{
		if (!m_profiler) return;
		if (m_profiler->m_activeRender && m_profiler->m_activeRender->values[size_t(pickMetric(m_family, PickMetric::GlDraws))] > m_draws)
			pickCount(m_family, PickMetric::ComponentsRendered);
		m_profiler->m_activePickFamily = m_previous;
	}

	PerformanceProfiler::RenderMetric PerformanceProfiler::visibleMetric(VisibleFamily family, VisibleMetric metric)
	{
		return RenderMetric(size_t(RenderMetric::VisibleBegin) + size_t(family) * size_t(VisibleMetric::Count) + size_t(metric));
	}
	PerformanceProfiler::VisibleMetric PerformanceProfiler::visiblePassMetric(VisiblePass pass, size_t offset)
	{
		return VisibleMetric(size_t(VisibleMetric::PointsUs) + size_t(pass) * 5 + offset);
	}
	bool PerformanceProfiler::visibleActive()
	{
		auto& profiler = renderInstance();
		return profiler.renderMetricsEnabled() && std::this_thread::get_id() == profiler.m_renderThread &&
			profiler.m_activeVisibleFamily != VisibleFamily::None;
	}
	void PerformanceProfiler::visibleCount(VisibleFamily family, VisibleMetric metric, uint64_t value)
	{
		if (family != VisibleFamily::None) renderCount(visibleMetric(family, metric), value);
	}
	void PerformanceProfiler::visibleCount(VisibleMetric metric, uint64_t value)
	{
		if (visibleActive()) visibleCount(renderInstance().m_activeVisibleFamily, metric, value);
	}
	void PerformanceProfiler::visibleMode(VisiblePass pass, bool enabled, uint64_t elements)
	{
		if (!enabled) visibleCount(visiblePassMetric(pass, 3));
		else if (!elements) visibleCount(visiblePassMetric(pass, 4));
	}
	void PerformanceProfiler::visibleDraw(VisiblePass pass, unsigned mode, uint64_t elements, bool indexed,
		uint64_t instances, bool wire, uint64_t calls, uint64_t ranges)
	{
		if (!visibleActive() || !calls) return;
		const uint64_t submitted = elements * instances * calls;
		visibleCount(VisibleMetric::GlDraws, calls);
		visibleCount(VisibleMetric::DrawRanges, ranges * calls);
		visibleCount(visiblePassMetric(pass, 1), calls);
		visibleCount(visiblePassMetric(pass, 2), submitted);
		// OpenGL primitive constants; count submitted topology, never fragments/visible triangles.
		if (mode == 0x0004) { // GL_TRIANGLES
			visibleCount(indexed ? VisibleMetric::IndexedTriangleIndices : VisibleMetric::ArrayTriangleVertices, submitted);
			visibleCount(VisibleMetric::SubmittedTriangles, (elements / 3) * instances * calls);
			if (wire) visibleCount(VisibleMetric::WireTriangleElements, submitted);
		}
		else if (mode == 0x0000) visibleCount(VisibleMetric::SubmittedPoints, submitted); // GL_POINTS
		else if (mode == 0x0001) visibleCount(VisibleMetric::SubmittedLineVertices, submitted); // GL_LINES
		else if (mode == 0x0007) { // GL_QUADS
			if (indexed) visibleCount(VisibleMetric::SubmittedQuadIndices, submitted);
			visibleCount(VisibleMetric::SubmittedQuads, (elements / 4) * instances * calls);
		}
		if (pass == VisiblePass::Ellipsoids) visibleCount(VisibleMetric::EllipsoidInstances, instances * calls);
	}
	PerformanceProfiler::VisibleTimer::VisibleTimer(VisibleMetric metric, VisibleFamily family)
	{
		auto& profiler = renderInstance();
		if (!profiler.renderMetricsEnabled() || std::this_thread::get_id() != profiler.m_renderThread) return;
		if (family == VisibleFamily::None) family = profiler.m_activeVisibleFamily;
		if (family != VisibleFamily::None) m_timer.emplace(visibleMetric(family, metric));
	}
	PerformanceProfiler::VisibleScope::VisibleScope(VisibleFamily family, bool offscreen) : m_family(family)
	{
		auto& profiler = renderInstance();
		if (!profiler.renderMetricsEnabled() || std::this_thread::get_id() != profiler.m_renderThread) return;
		m_profiler = &profiler; m_previous = profiler.m_activeVisibleFamily; profiler.m_activeVisibleFamily = family;
		visibleCount(VisibleMetric::DisplayCalls);
		if (offscreen) visibleCount(VisibleMetric::OffscreenCalls);
		auto& values = profiler.m_activeRender ? profiler.m_activeRender->values : profiler.m_renderBetweenFrames;
		m_draws = values[size_t(visibleMetric(family, VisibleMetric::GlDraws))];
		m_timer.emplace(visibleMetric(family, VisibleMetric::DisplayUs));
	}
	PerformanceProfiler::VisibleScope::~VisibleScope()
	{
		if (!m_profiler) return;
		auto& values = m_profiler->m_activeRender ? m_profiler->m_activeRender->values : m_profiler->m_renderBetweenFrames;
		if (values[size_t(visibleMetric(m_family, VisibleMetric::GlDraws))] > m_draws) {
			visibleCount(VisibleMetric::GpuDeferred); // GPU execution requires an external context-aware capture.
			if (m_family == VisibleFamily::ObjectListIndividual || m_family == VisibleFamily::DetectionIndividual)
				visibleCount(VisibleMetric::ComponentsDrawn);
		}
		m_profiler->m_activeVisibleFamily = m_previous;
	}
	PerformanceProfiler::VisibleHelperScope::VisibleHelperScope(VisiblePass pass, unsigned mode, uint64_t elements, bool indexed, bool wire)
		: m_pass(pass), m_mode(mode), m_elements(elements), m_indexed(indexed), m_wire(wire)
	{
		if (!visibleActive()) return;
		m_profiler = &renderInstance();
		auto& values = m_profiler->m_activeRender ? m_profiler->m_activeRender->values : m_profiler->m_renderBetweenFrames;
		m_before = values[size_t(RenderMetric::OtherAuditedDraws)];
		visibleCount(VisibleMetric::HelperInvocations);
		m_timer.emplace(visibleMetric(m_profiler->m_activeVisibleFamily, visiblePassMetric(pass)));
	}
	PerformanceProfiler::VisibleHelperScope::~VisibleHelperScope()
	{
		if (!m_profiler) return;
		auto& values = m_profiler->m_activeRender ? m_profiler->m_activeRender->values : m_profiler->m_renderBetweenFrames;
		const auto draws = values[size_t(RenderMetric::OtherAuditedDraws)] - m_before;
		visibleCount(VisibleMetric::HelperGlDraws, draws);
		visibleDraw(m_pass, m_mode, m_elements, m_indexed, 1, m_wire, draws);
	}

	void PerformanceProfiler::renderWorkerCount(RenderWorkerMetric metric, uint64_t value)
	{
		auto& profiler = renderInstance();
		if (profiler.renderMetricsEnabled()) profiler.m_renderWorkers[size_t(metric)].fetch_add(value, std::memory_order_relaxed);
	}
	void PerformanceProfiler::renderQueues(size_t queued, size_t ready, size_t preparing)
	{
		if (!renderMetricsEnabled() || !m_activeRender) return;
		m_activeRender->queued = int64_t(queued); m_activeRender->ready = int64_t(ready); m_activeRender->preparing = int64_t(preparing);
	}
	std::vector<PerformanceProfiler::RenderRecord> PerformanceProfiler::renderRecords() const
	{
		std::vector<RenderRecord> records; records.reserve(m_renderRingSize);
		for (size_t i = 0; i < m_renderRingSize; ++i)
			records.push_back(m_renderRing[(m_renderRingNext + m_renderRing.size() - m_renderRingSize + i) % m_renderRing.size()]);
		return records;
	}
	std::string PerformanceProfiler::renderCsvHeader()
	{
		std::ostringstream out;
		out << "schema,frame_id,camera_id,kind,offscreen,object_kind,children,width,height,interactive,cpu_ms,queued,ready,preparing,gpu_timing_available";
		for (size_t i = 0; i < size_t(RenderMetric::Count); ++i) out << ',' << renderMetricName(RenderMetric(i));
		out << ",timestamp_unix_us,gpu_source_record_id,gpu_pick_timing_status";
		return out.str();
	}
	std::string PerformanceProfiler::renderCsvRow(const RenderRecord& record)
	{
		std::ostringstream out; out.imbue(std::locale::classic());
		out << "4," << record.frameId << ',' << record.cameraId << ',' << (record.kind ? record.kind : (record.paint ? "paint" : "draw")) << ',' << record.offscreen << ','
			<< (record.multiple ? "MyMultipleObject" : "MyObject") << ',' << record.children << ',' << record.width << ',' << record.height << ','
			<< record.interactive << ',' << record.cpuMs << ',' << record.queued << ',' << record.ready << ',' << record.preparing << ",0";
		for (const auto value : record.values) out << ',' << value;
		const auto unavailable = record.values[size_t(RenderMetric::ObjectPickGpuUnavailable)] + record.values[size_t(RenderMetric::DetectionPickGpuUnavailable)];
		const auto sampled = record.values[size_t(RenderMetric::ObjectPickGpuSamples)] + record.values[size_t(RenderMetric::DetectionPickGpuSamples)];
		const auto supported = record.values[size_t(RenderMetric::ObjectPickGpuSupported)] + record.values[size_t(RenderMetric::DetectionPickGpuSupported)];
		out << ',' << record.timestampUs << ',' << record.sourceRecordId << ',' << (sampled ? "SAMPLED" : (unavailable ? "GPU_PICK_TIMING_UNAVAILABLE" : (supported ? "SUPPORTED_NOT_SAMPLED" : "NOT_OBSERVED")));
		return out.str();
	}
	void PerformanceProfiler::finishRenderRecord(const RenderRecord& completed)
	{
		if (!renderMetricsEnabled()) return;
		RenderRecord record = completed;
		for (size_t family = 0; family < 2; ++family) {
			const auto type = PickFamily(family);
			record.values[size_t(pickMetric(type, PickMetric::FboLive))] = m_pickLive[family];
			record.values[size_t(pickMetric(type, PickMetric::FboPeakLive))] = m_pickPeak[family];
			record.values[size_t(pickMetric(type, PickMetric::FboRequestedBytes))] = m_pickBytes[family];
			record.values[size_t(pickMetric(type, PickMetric::FboUnknownPayloads))] = m_pickUnknown[family];
		}
		m_renderRing[m_renderRingNext] = record;
		m_renderRingNext = (m_renderRingNext + 1) % m_renderRing.size();
		m_renderRingSize = std::min(m_renderRingSize + 1, m_renderRing.size()); ++m_renderWindowFrames;
		for (size_t i = 0; i < record.values.size(); ++i) {
			const bool maximum = i == size_t(RenderMetric::BatchMaxImages) || i == size_t(RenderMetric::FrameCacheEntries) || i == size_t(RenderMetric::FrameCacheBytes);
			if (maximum) m_renderTotals[i] = std::max(m_renderTotals[i], record.values[i]);
			else if (i >= size_t(RenderMetric::ObjectPickDrawInvocations) && i < size_t(RenderMetric::PickHoverDispatches) &&
				((i - size_t(RenderMetric::ObjectPickDrawInvocations)) % size_t(PickMetric::Count) >= size_t(PickMetric::FboLive)) &&
				((i - size_t(RenderMetric::ObjectPickDrawInvocations)) % size_t(PickMetric::Count) <= size_t(PickMetric::FboUnknownPayloads)))
				m_renderTotals[i] = std::max(m_renderTotals[i], record.values[i]);
			else m_renderTotals[i] += record.values[i];
		}
		{
			std::lock_guard<std::mutex> lock(m_renderCsvMutex);
			if (m_renderCsvActive) {
				if (m_renderCsvSize == m_renderCsvQueue.size()) ++m_renderCsvDropped;
				else { m_renderCsvQueue[(m_renderCsvHead + m_renderCsvSize++) % m_renderCsvQueue.size()] = record; m_renderCsvCondition.notify_one(); }
			}
		}
		if (!m_activeRender) reportRenderMetrics(); // Keep nested offscreen work out of logging overhead in the parent.
	}
	void PerformanceProfiler::reportRenderMetrics()
	{
		const auto now = std::chrono::steady_clock::now();
		const auto seconds = std::chrono::duration<double>(now - m_renderReportStart).count();
		if (seconds < 1.) return;
		const size_t samples = std::min(m_renderWindowFrames, m_renderRingSize);
		std::vector<double> paintTimes; paintTimes.reserve(samples);
		for (size_t i = 0; i < samples; ++i) {
			const auto& record = m_renderRing[(m_renderRingNext + m_renderRing.size() - samples + i) % m_renderRing.size()];
			if (record.paint) paintTimes.push_back(record.cpuMs);
		}
		std::sort(paintTimes.begin(), paintTimes.end());
		std::ostringstream out; out.imbue(std::locale::classic());
		out << "[PoCA][RenderPerformance] seconds=" << seconds << " records=" << m_renderWindowFrames << " paint_samples=" << paintTimes.size();
		if (!paintTimes.empty()) {
			const auto n = paintTimes.size(); double total = 0.; for (const double time : paintTimes) total += time;
			out << " paint_mean_ms=" << total / n << " paint_median_ms=" << (paintTimes[(n - 1) / 2] + paintTimes[n / 2]) / 2.
				<< " paint_p95_ms=" << paintTimes[size_t(std::ceil(.95 * n)) - 1] << " paint_max_ms=" << paintTimes.back();
		}
		else out << " paint_ms=not_instrumented";
		for (size_t i = 0; i < m_renderTotals.size(); ++i) {
			if (i >= size_t(RenderMetric::VisibleBegin) && !m_renderTotals[i] && !m_renderBetweenFrames[i]) continue;
			out << ' ' << renderMetricName(RenderMetric(i)) << '=' << m_renderTotals[i];
			if (m_renderBetweenFrames[i]) out << " between_" << renderMetricName(RenderMetric(i)) << '=' << m_renderBetweenFrames[i];
		}
		static const std::array<const char*, size_t(RenderWorkerMetric::Count)> workerNames{
			"worker_preparations", "worker_preparation_us", "worker_admission_deferred", "worker_admission_rejected", "manager_redraw_requests" };
		for (size_t i = 0; i < workerNames.size(); ++i) out << ' ' << workerNames[i] << '=' << m_renderWorkers[i].exchange(0, std::memory_order_relaxed);
		const auto& last = m_renderRing[(m_renderRingNext + m_renderRing.size() - 1) % m_renderRing.size()];
		out << " queue_camera=" << last.cameraId << " queued=" << last.queued << " ready=" << last.ready << " preparing=" << last.preparing
			<< " csv_dropped=" << m_renderCsvDropped.exchange(0) << " gpu_timing_unavailable=1";
		std::cout << out.str() << std::endl;
		m_renderTotals.fill(0); m_renderBetweenFrames.fill(0); m_renderWindowFrames = 0; m_renderReportStart = now;
	}
	void PerformanceProfiler::stopRenderCsv()
	{
		{ std::lock_guard<std::mutex> lock(m_renderCsvMutex); m_renderCsvStop = true; m_renderCsvCondition.notify_one(); }
		if (m_renderCsvThread.joinable()) m_renderCsvThread.join();
		m_renderCsvActive = false; m_renderCsvHead = m_renderCsvSize = 0;
	}
	void PerformanceProfiler::renderCsvLoop(const std::string& path)
	{
		QFile file(QString::fromLocal8Bit(path.c_str()));
		if (!file.open(QIODevice::ReadWrite | QIODevice::Text)) {
			std::cerr << "[PoCA][RenderPerformance] CSV open failed: " << file.errorString().toStdString() << std::endl;
			std::lock_guard<std::mutex> lock(m_renderCsvMutex); m_renderCsvActive = false; return;
		}
		const QByteArray header = QByteArray::fromStdString(renderCsvHeader());
		if (file.size() && file.readLine().trimmed() != header) {
			std::cerr << "[PoCA][RenderPerformance] CSV schema mismatch; export disabled" << std::endl;
			std::lock_guard<std::mutex> lock(m_renderCsvMutex); m_renderCsvActive = false; return;
		}
		if (!file.seek(file.size()) || (!file.size() && file.write(header + '\n') != header.size() + 1)) {
			std::cerr << "[PoCA][RenderPerformance] CSV append/header failed: " << file.errorString().toStdString() << std::endl;
			std::lock_guard<std::mutex> lock(m_renderCsvMutex); m_renderCsvActive = false; return;
		}
		for (;;) {
			RenderRecord record;
			{
				std::unique_lock<std::mutex> lock(m_renderCsvMutex);
				m_renderCsvCondition.wait(lock, [this]() { return m_renderCsvStop || m_renderCsvSize; });
				if (!m_renderCsvSize && m_renderCsvStop) break;
				record = m_renderCsvQueue[m_renderCsvHead]; m_renderCsvHead = (m_renderCsvHead + 1) % m_renderCsvQueue.size(); --m_renderCsvSize;
			}
			const QByteArray row = QByteArray::fromStdString(renderCsvRow(record)) + '\n';
			if (file.write(row) != row.size()) {
				std::cerr << "[PoCA][RenderPerformance] CSV write failed: " << file.errorString().toStdString() << std::endl;
				std::lock_guard<std::mutex> lock(m_renderCsvMutex); m_renderCsvActive = false; m_renderCsvSize = 0; return;
			}
		}
		if (!file.flush()) std::cerr << "[PoCA][RenderPerformance] CSV flush failed: " << file.errorString().toStdString() << std::endl;
	}

	PerformanceProfiler& PerformanceProfiler::instance()
	{
		/*
		 * poca_core is currently built as a static library and is linked both by the
		 * main application and by plugins. A function-local static singleton would
		 * therefore be duplicated once per module, so samples recorded from plugins
		 * would not be visible from the PerformanceWidget living in the application.
		 *
		 * Store the profiler object pointer on the unique QCoreApplication instance,
		 * which is shared process-wide through QtCore, and let all modules retrieve
		 * the same profiler. The object intentionally lives until process shutdown.
		 */
		static const char* propertyName = "poca.core.PerformanceProfiler.instance";

		QCoreApplication* app = QCoreApplication::instance();
		if (app != nullptr) {
			QVariant value = app->property(propertyName);
			if (value.isValid()) {
				PerformanceProfiler* profiler = reinterpret_cast<PerformanceProfiler*>(value.value<quintptr>());
				if (profiler != nullptr)
					return *profiler;
			}

			PerformanceProfiler* profiler = new PerformanceProfiler();
			app->setProperty(propertyName, QVariant::fromValue<quintptr>(reinterpret_cast<quintptr>(profiler)));
			QObject::connect(app, &QCoreApplication::aboutToQuit, app, [profiler]() {
				profiler->m_renderEnabled.store(false, std::memory_order_relaxed);
				profiler->stopRenderCsv(); // Drain before Qt/plugin teardown.
			}, Qt::DirectConnection);
			return *profiler;
		}

		static PerformanceProfiler fallbackProfiler;
		return fallbackProfiler;
	}

	void PerformanceProfiler::setEnabled(bool _enabled)
	{
		std::lock_guard<std::mutex> guard(m_mutex);
		m_enabled = _enabled;
	}

	bool PerformanceProfiler::enabled() const
	{
		std::lock_guard<std::mutex> guard(m_mutex);
		return m_enabled;
	}

	void PerformanceProfiler::record(const std::string& _category, const std::string& _name, double _milliseconds)
	{
		std::lock_guard<std::mutex> guard(m_mutex);
		if (!m_enabled)
			return;

		m_samples.push_back(Sample{ ++m_sequence, _category, _name, _milliseconds });
		while (m_samples.size() > m_maxSamples)
			m_samples.pop_front();
	}

	std::vector<PerformanceProfiler::Sample> PerformanceProfiler::samplesSince(uint64_t _sequence) const
	{
		std::lock_guard<std::mutex> guard(m_mutex);
		std::vector<Sample> samples;
		for (const Sample& sample : m_samples)
			if (sample.sequence > _sequence)
				samples.push_back(sample);
		return samples;
	}

	std::map<std::string, double> PerformanceProfiler::latestByCategory() const
	{
		std::lock_guard<std::mutex> guard(m_mutex);
		std::map<std::string, double> values;
		for (const Sample& sample : m_samples)
			values[sample.category] = sample.milliseconds;
		return values;
	}

	void PerformanceProfiler::clear()
	{
		std::lock_guard<std::mutex> guard(m_mutex);
		m_samples.clear();
	}
}
