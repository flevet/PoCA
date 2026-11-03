/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#ifndef CameraGeometry_hpp__
#define CameraGeometry_hpp__
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <glm/glm.hpp>
namespace poca::opengl {
	// distanceOrtho is projection half-extent; cameraDistance is signed physical
	// eye-to-center distance. Only perspective zoom couples the two.
	inline float cameraHalfFovTangent(float _fov) {
		return std::tan(glm::radians(_fov) / 2.f);
	}
	inline float perspectiveEquivalentDistance(float _extent, float _fov, float _previousDistance) {
		const float sign = _previousDistance < 0.f ? -1.f : 1.f;
		return sign * (std::max(0.0001f, _extent) / cameraHalfFovTangent(_fov));
	}
	inline float orthographicExtentForDistance(float _distance, float _fov) {
		return std::max(0.0001f, std::abs(_distance) * cameraHalfFovTangent(_fov));
	}
	inline float safeInitialCameraDistance(float _extent, const glm::vec3& _sceneSize, float _fov) {
		const float radius = glm::length(_sceneSize) / 2.f;
		// Leave room for the positive near plane at every orientation, even for
		// a deep box fitted using only its existing XY projection extent.
		return std::max(perspectiveEquivalentDistance(_extent, _fov, 1.f), radius * 1.01f + 0.001f);
	}

	inline void applyCameraExtent(float _value, bool _perspective, float _fov, float& _extent, float& _distance) {
		_extent = std::max(0.0001f, _value);
		if (_perspective)
			_distance = perspectiveEquivalentDistance(_extent, _fov, _distance);
	}
	inline void applyCameraZoom(float _delta, bool _perspective, float _fov, float& _extent, float& _distance) {
		if (_perspective) {
			_distance += _delta / cameraHalfFovTangent(_fov);
			_extent = orthographicExtentForDistance(_distance, _fov);
		}
		else
			applyCameraExtent(_extent + _delta, false, _fov, _extent, _distance);
	}
	inline glm::vec2 cameraProjectionFactors(float _width, float _height) {
		glm::vec2 factors(1.f);
		if (_width > _height)
			factors.x = _width / _height;
		else
			factors.y = _height / _width;
		return factors;
	}
	struct CameraBoxFit { float extent, distance; };
	inline CameraBoxFit fitCameraBox(const glm::vec3& size, const glm::mat4& view, const glm::vec2& viewport, float fov) {
		if (!std::isfinite(viewport.x) || !std::isfinite(viewport.y) || viewport.x <= 0 || viewport.y <= 0)
			throw std::invalid_argument("Cannot fit camera to an invalid viewport");
		if (!std::isfinite(fov) || fov <= 0.f || fov >= 180.f)
			throw std::invalid_argument("Cannot fit camera with an invalid field of view");
		for (int axis=0;axis<3;++axis)
			if (!std::isfinite(size[axis]) || size[axis] < 0.f)
				throw std::invalid_argument("Cannot fit camera to invalid box dimensions");
		for (int column=0;column<4;++column) for (int row=0;row<4;++row)
			if (!std::isfinite(view[column][row]))
				throw std::invalid_argument("Cannot fit camera with a nonfinite view");
		const float determinant=glm::determinant(glm::mat3(view));
		if (!std::isfinite(determinant) || std::abs(determinant) < 1e-6f)
			throw std::invalid_argument("Cannot fit camera with a degenerate view");
		const float aspect = viewport.x / viewport.y, tangent = cameraHalfFovTangent(fov);
		if (!std::isfinite(aspect) || aspect <= 0.f || !std::isfinite(tangent) || tangent <= 0.f ||
			!std::isfinite(aspect*tangent) || aspect*tangent <= 0.f)
			throw std::invalid_argument("Camera fit projection is not representable");
		const auto factors = cameraProjectionFactors(viewport.x,viewport.y);
		CameraBoxFit fit{0.0001f,0.f};
		for (int corner=0;corner<8;++corner) {
			const glm::vec3 local((corner&1) ? size.x/2 : -size.x/2,
				(corner&2) ? size.y/2 : -size.y/2,(corner&4) ? size.z/2 : -size.z/2);
			const auto p=glm::mat3(view)*local;
			if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
				throw std::overflow_error("Camera fit corner is not representable");
			fit.extent=std::max(fit.extent,std::max(std::abs(p.x)/factors.x,std::abs(p.y)/factors.y));
			fit.distance=std::max(fit.distance,p.z+std::max(std::abs(p.x)/(aspect*tangent),std::abs(p.y)/tangent));
		}
		// Keep the positive near plane outside the bounding sphere at any orientation.
		fit.distance=std::max(fit.distance,glm::length(size)*.505f+.001f);
		if (!std::isfinite(fit.extent) || !std::isfinite(fit.distance) || fit.extent <= 0.f || fit.distance <= 0.f ||
			!std::isfinite(fit.extent*factors.x) || !std::isfinite(fit.extent*factors.y))
			throw std::overflow_error("Camera fit projection is not representable");
		return fit;
	}
	inline void convertCameraProjection(bool _toPerspective, float _fov, float _factorH, float& _extent, float& _distance) {
		// Ortho extent is based on the smaller viewport dimension; perspective FOV
		// is vertical. Account for portrait viewports at the transition only.
		if (_toPerspective) {
			_distance = perspectiveEquivalentDistance(_extent * _factorH, _fov, _distance);
			_extent = orthographicExtentForDistance(_distance, _fov);
		}
		else
			_extent = std::max(0.0001f, orthographicExtentForDistance(_distance, _fov) / _factorH);
	}
	inline void restoreCameraDistances(float _savedExtent, float _savedDistance, float& _extent, float& _distance) {
		// Undo restores exact independent values, without projection conversion.
		_extent = _savedExtent;
		_distance = _savedDistance;
	}
	struct CameraDepthRange {
		float nearPlane, farPlane;
	};
	inline CameraDepthRange projectionDepthRange(float _distance, const glm::vec3& _sceneSize) {
		const float distance = std::abs(_distance);
		const float d = std::max(std::max(_sceneSize.x, _sceneSize.y) / 2.f, _sceneSize.z);
		const float radius = std::max(1.f, d * std::sqrt(3.f));
		return { std::max(0.001f, distance / 1000.f), distance + radius * 4.f };
	}
	inline glm::vec3 physicalCameraPosition(const glm::vec3& _eye, const glm::vec3& _center, float _distance) {
		glm::vec3 direction = _eye - _center;
		// Preserve the existing direction/zero-distance conventions.
		if (glm::dot(direction, direction) < 1e-8f)
			direction = glm::vec3(0.f, 0.f, 1.f);
		const float distance = std::abs(_distance) < 0.0001f ? (_distance < 0.f ? -0.0001f : 0.0001f) : _distance;
		return _center + glm::normalize(direction) * distance;
	}
}
#endif
