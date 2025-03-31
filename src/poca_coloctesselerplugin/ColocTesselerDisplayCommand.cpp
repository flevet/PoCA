/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ColocTesselerDisplayCommand.cpp
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

#include <Windows.h>
#include <gl/glew.h>
#include <gl/GL.h>
#include <QtGui/QOpenGLFramebufferObject>
#include <QtGui/QImage>

#include <General/Engine.hpp>
#include <General/Palette.hpp>
#include <General/Histogram.hpp>
#include <General/MyData.hpp>
#include <General/Misc.h>
#include <Interfaces/HistogramInterface.hpp>
#include <Interfaces/CameraInterface.hpp>
#include <OpenGL/Shader.hpp>

#include "ColocTesselerDisplayCommand.hpp"


ColocTesselerDisplayCommand::ColocTesselerDisplayCommand(ColocTesseler* _coloc) : poca::opengl::BasicDisplayCommand(_coloc, "ColocTesselerDisplayCommand")//, m_fill(true), m_locsDisplay(true), m_polytopesDisplay(true)
{
	m_colocTesseler = _coloc;
	m_textureLutID[0] = m_textureLutID[1] = 0;
}

ColocTesselerDisplayCommand::ColocTesselerDisplayCommand(const ColocTesselerDisplayCommand& _o) : poca::opengl::BasicDisplayCommand(_o)//, m_fill(_o.m_fill)
{
	m_colocTesseler = _o.m_colocTesseler;
}

ColocTesselerDisplayCommand::~ColocTesselerDisplayCommand()
{
}

void ColocTesselerDisplayCommand::execute(poca::core::CommandInfo* _infos)
{
	poca::opengl::BasicDisplayCommand::execute(_infos);
	if (_infos->nameCommand == "updateFeature") {
		generateFeatureBuffer();
	}
	else if (_infos->nameCommand == "display") {
		poca::opengl::Camera* cam = _infos->getParameterPtr<poca::opengl::Camera>("camera");
		bool offscrean = false;
		if (_infos->hasParameter("offscreen"))
			offscrean = _infos->getParameter<bool>("offscreen");
		display(cam, offscrean);
	}
	if (_infos->nameCommand == "pick") {
	}
	else if (hasParameter(_infos->nameCommand)) {
		loadParameters(*_infos);
	}
	else if (_infos->nameCommand == "freeGPU") {
		freeGPUMemory();
	}
}

poca::core::CommandInfo ColocTesselerDisplayCommand::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	if (_nameCommand == "freeGPU" || _nameCommand == "updateFeature")
		return poca::core::CommandInfo(false, _nameCommand);
	return poca::opengl::BasicDisplayCommand::createCommand(_nameCommand, _parameters);
}

poca::core::Command* ColocTesselerDisplayCommand::copy()
{
	return new ColocTesselerDisplayCommand(*this);
}

void ColocTesselerDisplayCommand::display(poca::opengl::Camera* _cam, const bool _offscreen)
{
	if (!m_colocTesseler->isSelected()) return;

	drawElements(_cam);
	if (!_offscreen)
		drawPicking(_cam);
}

void ColocTesselerDisplayCommand::drawElements(poca::opengl::Camera* _cam)
{
	if (m_pointBuffer[0].empty())
		createDisplay();

	glEnable(GL_DEPTH_TEST);

	poca::opengl::Shader* shader = _cam->getShader("simpleShader");

	uint32_t sizeGL = 1;
	 poca::core::Engine* engine = poca::core::Engine::instance();
	poca::core::MyObjectInterface* obj = engine->getObject(m_colocTesseler);
	poca::core::CommandableObject* comObj = dynamic_cast <poca::core::CommandableObject*>(obj);
	if (comObj)
		if (comObj->hasParameter("pointSizeGL"))
			sizeGL = comObj->getParameter<uint32_t>("pointSizeGL");

	for(size_t n = 0; n < 2; n++)
		//_cam->drawSimpleShader<poca::core::Vec3mf, float>(m_textureLutID[n], m_pointBuffer[n], m_locsFeatureBuffer[n], 0.f, 1.f);
		_cam->drawSphereRendering<poca::core::Vec3mf, float>(m_textureLutID[n], m_pointBuffer[n], m_locsFeatureBuffer[n], 0.f, 1.f, sizeGL, false);


	GL_CHECK_ERRORS();
}

void ColocTesselerDisplayCommand::drawPicking(poca::opengl::Camera* _cam)
{
	if (m_pickFBO == NULL)
		updatePickingFBO(_cam->getWidth(), _cam->getHeight());

	if (m_pickFBO == NULL) return;

	glEnable(GL_DEPTH_TEST);
	GLfloat bkColor[4];
	glGetFloatv(GL_COLOR_CLEAR_VALUE, bkColor);
	glClearColor(0.f, 0.f, 0.f, 0.f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	bool success = m_pickFBO->bind();
	if (!success) std::cout << "Problem with binding" << std::endl;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	for (size_t n = 0; n < 2; n++)
		_cam->drawPickingShader<poca::core::Vec3mf, float>(m_pointBuffer[n], m_idLocsBuffer[n], m_locsFeatureBuffer[n], 0.f);

	success = m_pickFBO->release();
	if (!success) std::cout << "Problem with releasing" << std::endl;
	glClearColor(bkColor[0], bkColor[1], bkColor[2], bkColor[3]);
	GL_CHECK_ERRORS();
}

void ColocTesselerDisplayCommand::createDisplay()
{
	freeGPUMemory();

	glGenTextures(2, &m_textureLutID[0]);
	generateLutTexture();

	size_t cpt = 1;
	for (size_t n = 0; n < 2; n++) {
		poca::geometry::VoronoiDiagram* voronoi = m_colocTesseler->voronoiAt(n);
		std::vector <poca::core::Vec3mf> points(voronoi->nbFaces());
		const float* xs = voronoi->getXs(), * ys = voronoi->getYs(), * zs = voronoi->getZs();
		if (zs != NULL)
			for (size_t n = 0; n < voronoi->nbFaces(); n++)
				points[n].set(xs[n], ys[n], zs[n]);
		else
			for (size_t n = 0; n < voronoi->nbFaces(); n++)
				points[n].set(xs[n], ys[n], 0.f);
		std::vector <float> ids(voronoi->nbFaces());
		std::iota(std::begin(ids), std::end(ids), cpt);
		m_pointBuffer[n].generateBuffer(points.size(), 3, GL_FLOAT);
		m_idLocsBuffer[n].generateBuffer(points.size(), 1, GL_FLOAT);
		m_locsFeatureBuffer[n].generateBuffer(points.size(), 1, GL_FLOAT);
		m_pointBuffer[n].updateBuffer(points.data());
		m_idLocsBuffer[n].updateBuffer(ids.data());
		cpt += voronoi->nbFaces();
	}
	generateFeatureBuffer();
}

void ColocTesselerDisplayCommand::freeGPUMemory()
{
	if (m_textureLutID[0] == 0) return;

	glDeleteTextures(2, &m_textureLutID[0]);
	for (size_t n = 0; n < 2; n++) {
		m_pointBuffer[n].freeGPUMemory();
		m_idLocsBuffer[n].freeGPUMemory();
		m_locsFeatureBuffer[n].freeGPUMemory();
	}
	m_textureLutID[0] = m_textureLutID[1] = 0;
}

void ColocTesselerDisplayCommand::generateLutTexture()
{
	poca::core::Palette palettes[2] = { poca::core::Palette(poca::core::Color4uc(0, 0, 0, 255), poca::core::Color4uc(0, 0, 0, 255)), poca::core::Palette(poca::core::Color4uc(0, 0, 0, 255), poca::core::Color4uc(0, 0, 0, 255)) };
	poca::core::Color4B bottomLeft[2] = { poca::core::Color4B(0, 0, 0, 255), poca::core::Color4B(0, 0, 0, 255) }, organizedColors[2] = { poca::core::Color4B(255, 221, 85, 255), poca::core::Color4B(39, 204, 244, 255) }, colocColors[2] = { poca::core::Color4B(255, 0, 0, 255), poca::core::Color4B(0, 0, 255, 255) };
	for (unsigned int n = 0; n < 2; n++) {
		palettes[n].setColor(1. / 4., poca::core::Color4uc(bottomLeft[n][0], bottomLeft[n][1], bottomLeft[n][2], bottomLeft[n][3]));
		palettes[n].setColor(2. / 4., poca::core::Color4uc(organizedColors[n][0], organizedColors[n][1], organizedColors[n][2], organizedColors[n][3]));
		palettes[n].setColor(3. / 4., poca::core::Color4uc(colocColors[n][0], colocColors[n][1], colocColors[n][2], colocColors[n][3]));
	}
	for (unsigned int n = 0; n < 2; n++) {
		//Creation of the texture for the LUT
		unsigned int sizeLut = 512;
		unsigned int cpt = 0;
		std::vector <float> lutValues(sizeLut * 4);
		double stepLut = 1. / (double)(sizeLut - 1);
		for (double val = 0.; val <= 1.; val += stepLut) {
			poca::core::Color4uc c = palettes[n].getColor(val);
			lutValues[cpt++] = (float)c[0] / 255.f;
			lutValues[cpt++] = (float)c[1] / 255.f;
			lutValues[cpt++] = (float)c[2] / 255.f;
			lutValues[cpt++] = (float)c[3] / 255.f;
		}
		glBindTexture(GL_TEXTURE_1D, m_textureLutID[n]);
		glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, sizeLut, 0, GL_RGBA, GL_FLOAT, lutValues.data());
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_1D, 0);
	}
}

void ColocTesselerDisplayCommand::generateFeatureBuffer()
{
	for (size_t n = 0; n < 2; n++) {
		poca::geometry::VoronoiDiagram* voronoi = m_colocTesseler->voronoiAt(n);
		std::vector<float> featureLocs;
		std::vector <unsigned char>& classesLocs = m_colocTesseler->classesLocsAt(n);
		float divisor = 4.f;
		std::transform(classesLocs.begin(), classesLocs.end(), std::back_inserter(featureLocs), [&divisor](auto i) { return  (float)i / divisor; });
		m_locsFeatureBuffer[n].updateBuffer(featureLocs.data());
	}
}

