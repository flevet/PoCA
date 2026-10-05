/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#ifndef CameraGeometry_hpp__
#define CameraGeometry_hpp__
#include <algorithm>
#include <cmath>
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
