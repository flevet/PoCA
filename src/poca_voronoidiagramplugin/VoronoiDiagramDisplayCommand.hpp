/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      VoronoiDiagramDisplayCommand.hpp
*
* Copyright: Florian Levet (2020-2025)
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
#include <Geometry/VoronoiDiagram.hpp>

class QOpenGLFramebufferObject;

class VoronoiDiagramDisplayCommand : public poca::opengl::BasicDisplayCommand {
public:
	VoronoiDiagramDisplayCommand(poca::geometry::VoronoiDiagram*);
	VoronoiDiagramDisplayCommand(const VoronoiDiagramDisplayCommand&);
	~VoronoiDiagramDisplayCommand();

	void execute(poca::core::CommandInfo*);
	poca::core::CommandInfo createCommand(const std::string&, const nlohmann::json&);
	poca::core::Command* copy();

	void freeGPUMemory();
	void generateFeatureBuffer(poca::core::HistogramInterface* = NULL);
	void generateBoundingBoxSelection(const int);

	QString getInfosTriangle(const int) const;

protected:
	void display(poca::opengl::Camera*, const bool);
	void drawElements(poca::opengl::Camera*);
	void drawPicking(poca::opengl::Camera*);

	void createDisplay();

	void explodeDiagram(const float);

protected:
	poca::geometry::VoronoiDiagram* m_voronoi;

	GLuint m_textureLutID;
	GLfloat m_minOriginalFeature, m_maxOriginalFeature, m_actualValueFeature, m_alphaValue;

	//For localizations
	poca::opengl::PointSingleGLBuffer <poca::core::Vec3mf> m_pointBuffer;
	poca::opengl::PointSingleGLBuffer <float> m_idLocsBuffer;
	poca::opengl::FeatureSingleGLBuffer <float> m_locsFeatureBuffer;

	//For Voronoi polytopes
	poca::opengl::TriangleGLBuffer <poca::core::Vec3mf> m_triangleBuffer;
	poca::opengl::TriangleGLBuffer <float> m_idPolytopeBuffer;
	poca::opengl::TriangleGLBuffer <float> m_triangleFeatureBuffer;
	poca::opengl::LineSingleGLBuffer <poca::core::Vec3mf> m_lineBuffer, m_lineNormalBuffer;
	poca::opengl::LineSingleGLBuffer <float> m_lineFeatureBuffer;

	poca::opengl::LineGLBuffer <poca::core::Vec3mf> m_boundingBoxSelection;

	bool m_displayTriangleSelection;
	poca::opengl::TriangleGLBuffer <poca::core::Vec3mf> m_triangleSelection;
};
#endif // DetectionSetDisplayCommand_h__

