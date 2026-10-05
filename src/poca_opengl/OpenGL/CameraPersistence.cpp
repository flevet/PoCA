/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#include "Camera.hpp"
#include "CameraZoomPersistence.hpp"
namespace poca::opengl {
	void Camera::saveZoomState(nlohmann::json& _json) const {
		saveCameraZoomState(_json, m_distanceOrtho, m_cameraDistance);
	}
	void Camera::restoreZoomState(const nlohmann::json& _json, bool _view, bool _zoom) {
		const float previousDistance = m_cameraDistance;
		restoreCameraZoomState(_json, isPerspectiveProjection(), m_perspectiveFov,
			_view, _zoom, m_distanceOrtho, m_cameraDistance);
		if (previousDistance != m_cameraDistance)
			updateCamera();
		recalcModelView();
	}
}
