/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      CameraInterface.hpp
*
* Copyright: Florian Levet (2020-2022)
*
* License:   LGPL v3
*
* Homepage:  https://github.com/flevet/PoCA
*
* PoCA is a free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 3 of the License, or (at your option) any later version.
*
* The algorithms that underlie PoCA have required considerable
* development. They are described in the original SR-Tesseler paper,
* doi:10.1038/nmeth.3579. If you use PoCA as part of work (visualization, 
* manipulation, quantification) towards a scientific publication, please include 
* a citation to the original paper.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with this program; if not, write to the Free Software Foundation,
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

#ifndef CameraInterface_hpp__
#define CameraInterface_hpp__

#include <array>
#include <glm/glm.hpp>

#include <Interfaces/MyObjectInterface.hpp>

namespace poca::opengl {

	class Shader;

	class CameraInterface {
	public:
		virtual ~CameraInterface() = default;

		virtual const glm::mat4& getProjectionMatrix() const = 0;
		virtual const glm::mat4& getViewMatrix() const = 0;
		virtual const glm::mat4& getModelMatrix() const = 0;
		//virtual const glm::mat4& getTranslationMatrix() const = 0;
		virtual const glm::mat4& getRotationMatrix() const = 0;

		virtual const glm::vec4& getClipPlaneX() const = 0;
		virtual const glm::vec4& getClipPlaneY() const = 0;
		virtual const glm::vec4& getClipPlaneZ() const = 0;
		virtual const glm::vec4& getClipPlaneW() const = 0;
		virtual const glm::vec4& getClipPlaneH() const = 0;
		virtual const glm::vec4& getClipPlaneT() const = 0;

		virtual const glm::vec3& getCenter() = 0;
		virtual const glm::vec3& getEye() = 0;
		virtual const glm::vec3& getUp() = 0;
		virtual const glm::quat getRotationSum() const = 0;
		virtual const float getOriginalDistanceOrtho() const = 0;

		virtual glm::vec3 getWorldCoordinates(const glm::vec2&) = 0;
		virtual glm::vec2 worldToScreenCoordinates(const glm::vec3&) const = 0;

		virtual void resizeWindow(const int, const int, const int, const int) = 0;
		virtual std::array<int, 2> sizeHintInterface() const = 0;
		virtual void update() = 0;

		virtual poca::core::MyObjectInterface* getObject() = 0;

		virtual Shader* getShader(const std::string&) = 0;

		virtual int getWidth() const = 0;
		virtual int getHeight() const = 0;

		virtual void makeCurrent() = 0;

	protected:
		bool m_sizeChanged{ false };
	};
}

#endif

