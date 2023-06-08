/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectColocalizationCommands.hpp
*
* Copyright: Florian Levet (2020-2021)
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

#ifndef ObjectColocalizationCommands_h__
#define ObjectColocalizationCommands_h__

#include <OpenGL/BasicDisplayCommand.hpp>
#include <OpenGL/Camera.hpp>

class ObjectColocalization;

class ObjectColocalizationCommands : public poca::opengl::BasicDisplayCommand
{
public:
	ObjectColocalizationCommands(ObjectColocalization*);
	ObjectColocalizationCommands(const ObjectColocalizationCommands&);
	~ObjectColocalizationCommands();

	void execute(poca::core::CommandInfo*);
	poca::core::Command* copy();
	const poca::core::CommandInfos saveParameters() const {
		return poca::core::CommandInfos();
	}
	poca::core::CommandInfo createCommand(const std::string&, const nlohmann::json&);
	//void saveCommands(nlohmann::json&);
	void freeGPUMemory();
	void generateFeatureBuffer(poca::core::HistogramInterface* = NULL);
	void generateBoundingBoxSelection(const int);
	QString getInfosTriangle(const int) const;

protected:
	void display(poca::opengl::Camera*, const bool);
	void drawElements(poca::opengl::Camera*);
	void drawPicking(poca::opengl::Camera*);
	void createDisplay();
	void displayZoomToBBox(poca::opengl::Camera*, const poca::core::BoundingBox&);

protected:
	ObjectColocalization* m_colocalization;

	GLuint m_textureLutID;
	GLfloat m_minOriginalFeature, m_maxOriginalFeature, m_actualValueFeature, m_alphaValue;

	poca::opengl::TriangleSingleGLBuffer <poca::core::Vec3mf> m_triangleBuffer;
	poca::opengl::TriangleSingleGLBuffer <float> m_idBuffer;
	poca::opengl::TriangleSingleGLBuffer <float> m_triangleFeatureBuffer;

	poca::opengl::PointSingleGLBuffer <poca::core::Vec3mf> m_pointBuffer;

	poca::opengl::LineSingleGLBuffer <poca::core::Vec3mf> m_boundingBoxSelection;
};

#endif

