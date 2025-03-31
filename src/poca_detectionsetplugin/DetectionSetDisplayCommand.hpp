/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DetectionSetDisplayCommand.hpp
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

#ifndef DetectionSetDisplayCommand_h__
#define DetectionSetDisplayCommand_h__

#include <General/Vec3.hpp>
#include <General/Palette.hpp>
#include <Interfaces/HistogramInterface.hpp>
#include <OpenGL/Camera.hpp>
#include <OpenGL/GLBuffer.hpp>
#include <OpenGL/BasicDisplayCommand.hpp>
#include <Geometry/DetectionSet.hpp>

class DetectionSet; 
class QOpenGLFramebufferObject;

class DetectionSetDisplayCommand : public poca::opengl::BasicDisplayCommand {
public:
	DetectionSetDisplayCommand(poca::geometry::DetectionSet *);
	DetectionSetDisplayCommand(const DetectionSetDisplayCommand &);
	~DetectionSetDisplayCommand();
	
	void execute(poca::core::CommandInfo *);
	poca::core::Command * copy();

	void freeGPUMemory();
	void generateFeatureBuffer(poca::core::HistogramInterface* = NULL);

	QString getInfosLocalization(const int) const;

	inline const poca::opengl::PointSingleGLBuffer <poca::core::Vec3mf>& getPointBuffer() const { return m_pointBuffer; }
	inline const poca::opengl::PointSingleGLBuffer <poca::core::Vec3mf>& getUncertaintiesBuffer() const { return m_uncertaintiesBuffer; }
	inline const poca::opengl::FeatureSingleGLBuffer <float>& getFeatureBuffer() const { return m_featureBuffer; }
	inline const GLfloat getMinOriginalFeature() const { return m_minOriginalFeature; }
	inline const GLfloat getMaxOriginalFeature() const { return m_maxOriginalFeature; }
	inline const GLfloat getActualOriginalFeature() const { return m_actualValueFeature; }

protected:
	void display(poca::opengl::Camera*, const bool, const bool = false);
	void drawElements(poca::opengl::Camera*, const bool);
	void drawPicking(poca::opengl::Camera*);

	void createDisplay();
	void sortWrtCameraPosition(const glm::vec3&, const glm::vec3&);

protected:
	poca::geometry::DetectionSet * m_dset;

	GLuint m_textureLutID;
	GLfloat m_minOriginalFeature, m_maxOriginalFeature, m_actualValueFeature, m_alphaValue;

	poca::opengl::PointSingleGLBuffer <poca::core::Vec3mf> m_pointBuffer, m_uncertaintiesBuffer, m_normalBuffer;
	poca::opengl::PointSingleGLBuffer <float> m_idBuffer;
	poca::opengl::FeatureSingleGLBuffer <float> m_featureBuffer;
	poca::opengl::PointSingleGLBuffer <poca::core::Color4D> m_colorBuffer;

#ifdef DEBUG_NORMAL
	poca::opengl::LineSingleGLBuffer <poca::core::Vec3mf> m_normalDebugBuffer;
#endif
};
#endif // DetectionSetDisplayCommand_h__

