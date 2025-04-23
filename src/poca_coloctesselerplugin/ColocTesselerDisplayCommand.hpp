/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ColocTesselerDisplayCommand.hpp
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
#include <Interfaces/HistogramInterface.hpp>
#include <OpenGL/Camera.hpp>
#include <OpenGL/GLBuffer.hpp>
#include <OpenGL/BasicDisplayCommand.hpp>

#include "ColocTesseler.hpp"

class QOpenGLFramebufferObject;

class ColocTesselerDisplayCommand : public poca::opengl::BasicDisplayCommand {
public:
	ColocTesselerDisplayCommand(ColocTesseler*);
	ColocTesselerDisplayCommand(const ColocTesselerDisplayCommand&);
	~ColocTesselerDisplayCommand();

	void execute(poca::core::CommandInfo*);
	poca::core::Command* copy();
	poca::core::CommandInfo createCommand(const std::string&, const nlohmann::json&);

	void freeGPUMemory();
	void generateLutTexture();
	void generateFeatureBuffer();

protected:
	void display(poca::opengl::Camera*, const bool);
	void drawElements(poca::opengl::Camera*);
	void drawPicking(poca::opengl::Camera*);

	void createDisplay();

protected:
	ColocTesseler* m_colocTesseler;

	GLuint m_textureLutID[2];
	GLfloat m_minOriginalFeature, m_maxOriginalFeature, m_actualValueFeature, m_alphaValue;

	//For localizations
	poca::opengl::PointSingleGLBuffer <poca::core::Vec3mf> m_pointBuffer[2];
	poca::opengl::PointSingleGLBuffer <float> m_idLocsBuffer[2];
	poca::opengl::FeatureSingleGLBuffer <float> m_locsFeatureBuffer[2];
};
#endif // DetectionSetDisplayCommand_h__

