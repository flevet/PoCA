/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      HeatMapDisplayCommand.hpp
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

#ifndef HeatMapDisplayCommand_h__
#define HeatMapDisplayCommand_h__

#include <General/Command.hpp>
#include <Geometry/DetectionSet.hpp>
#include <OpenGL/Camera.hpp>
#include <OpenGL/GLBuffer.hpp>

class QOpenGLFramebufferObject;
class DetectionSetDisplayCommand;

class HeatMapDisplayCommand : public poca::core::Command {
public:
	HeatMapDisplayCommand(poca::geometry::DetectionSet*);
	HeatMapDisplayCommand(const HeatMapDisplayCommand&);
	~HeatMapDisplayCommand();

	void execute(poca::core::CommandInfo*);
	void freeGPUMemory();

	poca::core::Command* copy();
	const poca::core::CommandInfos saveParameters() const { return poca::core::CommandInfos(); }
	poca::core::CommandInfo createCommand(const std::string&, const nlohmann::json&);

protected:
	void display(poca::opengl::Camera*, const bool);
	void createDisplay();
	void updateFBO(const int _w, const int _h);
	void generateLutTexture(poca::core::PaletteInterface*);
	void updateSelectedPoints();

protected:
	poca::geometry::DetectionSet* m_dset;
	float m_minX;

	GLuint m_textureLutID;
	GLfloat m_minOriginalFeature, m_maxOriginalFeature, m_actualValueFeature, m_alphaValue;
	poca::opengl::PointSingleGLBuffer <poca::core::Vec3mf> m_pointBuffer;
	poca::opengl::PointSingleGLBuffer <float> m_selectedPointBuffer;

	poca::core::PaletteInterface* m_palette;

	QOpenGLFramebufferObject* m_fbo;
	int m_wImage, m_hImage;

	DetectionSetDisplayCommand* m_dc;
};

#endif

