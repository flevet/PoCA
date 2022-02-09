/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectListDisplayCommand.hpp
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

#ifndef VoronoiDiagramDisplayCommand_h__
#define VoronoiDiagramDisplayCommand_h__

#include <General/Vec3.hpp>
#include <General/Palette.hpp>
#include <Interfaces/HistogramInterface.hpp>
#include <OpenGL/Camera.hpp>
#include <OpenGL/GLBuffer.hpp>
#include <OpenGL/BasicDisplayCommand.hpp>
#include <Geometry/ObjectList.hpp>

class QOpenGLFramebufferObject;

class ObjectListDisplayCommand : public poca::opengl::BasicDisplayCommand {
public:
	ObjectListDisplayCommand(poca::geometry::ObjectList*);
	ObjectListDisplayCommand(const ObjectListDisplayCommand&);
	~ObjectListDisplayCommand();

	void execute(poca::core::CommandInfo*);
	poca::core::CommandInfo createCommand(const std::string&, const nlohmann::json&);
	poca::core::Command* copy();

	void freeGPUMemory();
	void generateFeatureBuffer(poca::core::HistogramInterface* = NULL);
	void generateBoundingBoxSelection(const int);

	QString getInfosTriangle(const int) const;

protected:
	void display(poca::opengl::Camera*, const bool, const bool = false);
	void drawElements(poca::opengl::Camera*, const bool = false);
	void drawPicking(poca::opengl::Camera*);

	void displayZoomToBBox(poca::opengl::Camera*, const poca::core::BoundingBox&);
	void createDisplay();
	void sortWrtCameraPosition(const glm::vec3&, const glm::vec3&);

protected:
	poca::geometry::ObjectList* m_objects;

	GLuint m_textureLutID;
	GLfloat m_minOriginalFeature, m_maxOriginalFeature, m_actualValueFeature, m_alphaValue;

	//For localizations
	poca::opengl::PointSingleGLBuffer <poca::core::Vec3mf> m_pointBuffer, m_pointNormalBuffer, m_outlinePointBuffer;
	poca::opengl::LineSingleGLBuffer <poca::core::Vec3mf> m_normalsBuffer;
	poca::opengl::PointSingleGLBuffer <float> m_idLocsBuffer;
	poca::opengl::FeatureSingleGLBuffer <float> m_locsFeatureBuffer, m_outlineLocsFeatureBuffer;

	//For objects
	poca::opengl::TriangleGLBuffer <poca::core::Vec3mf> m_triangleBuffer, m_triangleNormalBuffer;
	poca::opengl::TriangleGLBuffer <float> m_idBuffer;
	poca::opengl::TriangleGLBuffer <float> m_triangleFeatureBuffer;

	//Only used with 2D objects, to display the outline in openGL line mode
	poca::opengl::LineSingleGLBuffer <poca::core::Vec3mf> m_lineBuffer;
	poca::opengl::LineSingleGLBuffer <float> m_lineFeatureBuffer;

	poca::opengl::LineSingleGLBuffer <poca::core::Vec3mf> m_boundingBoxSelection;

	bool m_fill;

	QOpenGLFramebufferObject* m_pickOneObject;
	QImage m_imageObject;
};
#endif // DetectionSetDisplayCommand_h__

