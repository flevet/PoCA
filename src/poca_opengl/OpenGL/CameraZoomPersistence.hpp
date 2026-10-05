/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#ifndef CameraZoomPersistence_hpp__
#define CameraZoomPersistence_hpp__
#include "CameraGeometry.hpp"
#include <General/json.hpp>
namespace poca::opengl {
	inline void saveCameraZoomState(nlohmann::json& _json, float _extent, float _distance) {
		_json["distanceOrtho"] = _extent;
		_json["cameraDistance"] = _distance;
	}
	// Keep partial-load checkboxes meaningful: ortho zoom restores scale only;
	// ortho view restores physical distance; perspective zoom restores distance.
	// Missing cameraDistance is an explicit legacy-format conversion ONCE at load.
	inline void restoreCameraZoomState(const nlohmann::json& _json, bool _perspective, float _fov,
		bool _view, bool _zoom, float& _extent, float& _distance) {
		if (_zoom && _json.contains("distanceOrtho"))
			applyCameraExtent(_json["distanceOrtho"].get<float>(), _perspective, _fov, _extent, _distance);
		if (_perspective ? _zoom : _view) {
			if (_json.contains("cameraDistance"))
				_distance = _json["cameraDistance"].get<float>();
			else if (_json.contains("distanceOrtho"))
				_distance = perspectiveEquivalentDistance(_json["distanceOrtho"].get<float>(), _fov, _perspective ? _distance : 1.f);
		}
	}
}
#endif
