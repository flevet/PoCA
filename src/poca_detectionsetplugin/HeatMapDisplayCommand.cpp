/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      HeatMapDisplayCommand.cpp
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
#include <glm/gtx/string_cast.hpp>
#include <algorithm>

#include <QtGui/QOpenGLFramebufferObject>
#include <QtWidgets/QMessageBox>

#include <General/Palette.hpp>
#include <General/MyData.hpp>
#include <OpenGL/Shader.hpp>
#include <DesignPatterns/StateSoftwareSingleton.hpp>

#include "HeatMapDisplayCommand.hpp"
#include "DetectionSetDisplayCommand.hpp"

HeatMapDisplayCommand::HeatMapDisplayCommand(poca::geometry::DetectionSet* _ds) :poca::core::Command("HeatMapDisplayCommand"), m_fbo(NULL), m_dc(NULL)
{
	m_dset = _ds;

	poca::core::StateSoftwareSingleton* sss = poca::core::StateSoftwareSingleton::instance();
	const nlohmann::json& parameters = sss->getParameters();
	if (!parameters.contains(name())) {
		
		addCommandInfo(poca::core::CommandInfo(false, "displayHeatmap", false));
		addCommandInfo(poca::core::CommandInfo(false, "radiusHeatmap", 25.f));
		addCommandInfo(poca::core::CommandInfo(false, "intensityHeatmap", 1.f));
		addCommandInfo(poca::core::CommandInfo(false, "radiusHeatmapType", "radiusScreenHeatmap", false, "radiusWorldHeatmap", true));
		addCommandInfo(poca::core::CommandInfo(false, "interpolateHeatmapLUT", true));
	}
	else {
		nlohmann::json param = parameters[name()];
		addCommandInfo(poca::core::CommandInfo(false, "displayHeatmap", param["displayHeatmap"].get<bool>()));
		addCommandInfo(poca::core::CommandInfo(false, "radiusHeatmap", param["radiusHeatmap"].get<float>()));
		addCommandInfo(poca::core::CommandInfo(false, "intensityHeatmap", param["intensityHeatmap"].get<float>()));
		addCommandInfo(poca::core::CommandInfo(false, "interpolateHeatmapLUT", param["interpolateHeatmapLUT"].get<bool>()));
		bool type[] = { param["radiusHeatmapType"]["radiusScreenHeatmap"].get<bool>() , param["radiusHeatmapType"]["radiusWorldHeatmap"].get<bool>() };
		addCommandInfo(poca::core::CommandInfo(false, "radiusHeatmapType", "radiusScreenHeatmap", type[0], "radiusWorldHeatmap", type[1]));
	}
}

HeatMapDisplayCommand::HeatMapDisplayCommand(const HeatMapDisplayCommand& _o) : poca::core::Command(_o)
{
	m_dset = _o.m_dset;
}

HeatMapDisplayCommand::~HeatMapDisplayCommand()
{
	delete m_palette;
}

poca::core::Command* HeatMapDisplayCommand::copy()
{
	return new HeatMapDisplayCommand(*this);
}

void HeatMapDisplayCommand::execute(poca::core::CommandInfo* _infos)
{
	if (_infos->nameCommand == "histogram" || _infos->nameCommand == "updateFeature") {
		updateSelectedPoints();
	}
	else if (_infos->nameCommand == "updatePickingBuffer") {
		int w = _infos->getParameter<int>("width"), h = _infos->getParameter<int>("height");
		updateFBO(w, h);
	}
	else if (_infos->nameCommand == "display") {
		poca::opengl::Camera* cam = _infos->getParameterPtr<poca::opengl::Camera>("camera");
		bool offscrean = false;
		if (_infos->hasParameter("offscreen"))
			offscrean = _infos->getParameter<bool>("offscreen");
		display(cam, offscrean);
	}
	else if (hasCommand(_infos->nameCommand)) {
		loadParameters(*_infos);
	}
	else if (_infos->nameCommand == "changeLUTHeatmap") {
		std::string nameLut = _infos->getParameter<std::string>("changeLUTHeatmap");
		poca::core::Palette pal = poca::core::Palette::getStaticLut(nameLut);
		if (pal.null()) {
			std::cout << "LUT " << nameLut << " does not exist. Aborting." << std::endl;
			return;
		}		
		m_palette->setPalette(&pal);
		generateLutTexture(m_palette);
	}
	if (_infos->nameCommand == "interpolateHeatmapLUT")
		generateLutTexture(m_palette);
}

poca::core::CommandInfo HeatMapDisplayCommand::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	addCommandInfo(poca::core::CommandInfo(false, "radiusHeatmapType", "radiusScreenHeatmap", false, "radiusWorldHeatmap", true));
	
	if (_nameCommand == "displayHeatmap" || _nameCommand == "interpolateHeatmapLUT") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "radiusHeatmap" || _nameCommand == "intensityHeatmap") {
		float val = _parameters.get<float>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "radiusHeatmapType") {
		bool radiusScreenHeatmap, radiusWorldHeatmap;
		bool complete = _parameters.contains("radiusScreenHeatmap");
		if (complete)
			radiusScreenHeatmap = _parameters["radiusScreenHeatmap"].get<bool>();
		complete &= _parameters.contains("radiusWorldHeatmap");
		if (complete) {
			radiusWorldHeatmap = _parameters["radiusWorldHeatmap"].get<bool>();
			return poca::core::CommandInfo(false, _nameCommand, "radiusScreenHeatmap", radiusScreenHeatmap, "radiusWorldHeatmap", radiusWorldHeatmap);
		}
	}
	else if (_nameCommand == "updatePickingBuffer") {
		int w, h;
		bool complete = _parameters.contains("width");
		if (complete)
			w = _parameters["width"].get<int>();
		complete &= _parameters.contains("height");
		if (complete) {
			h = _parameters["height"].get<int>();
			return poca::core::CommandInfo(false, _nameCommand, "width", w, "height", h);
		}
	}
	return poca::core::CommandInfo();
}

void HeatMapDisplayCommand::createDisplay()
{
	freeGPUMemory();

	try {
		glGenTextures(1, &m_textureLutID);
		m_palette = poca::core::Palette::getStaticLutPtr("Heatmap");
		generateLutTexture(m_palette);
		GL_CHECK_ERRORS();

		if (!m_dc) {
			const std::vector <float>& xs = m_dset->getMyData("x")->getOriginalData(), & ys = m_dset->getMyData("y")->getOriginalData();
			m_minX = *std::min_element(xs.begin(), xs.end());
			std::vector <poca::core::Vec3mf> points(xs.size());
			if (m_dset->hasData("z")) {
				const std::vector <float>& zs = m_dset->getMyData("z")->getOriginalData();
				for (size_t n = 0; n < xs.size(); n++)
					points[n].set(xs[n], ys[n], zs[n]);
			}
			else
				for (size_t n = 0; n < xs.size(); n++)
					points[n].set(xs[n], ys[n], 0.f);
			m_pointBuffer.generateBuffer(points.size(), 3, GL_FLOAT);
			m_pointBuffer.updateBuffer(points.data());
			GL_CHECK_ERRORS();
		}

		updateSelectedPoints();
		GL_CHECK_ERRORS();
	}
	catch (std::runtime_error const& e) {
		std::string mess("Error: creating display for command " + name() + " of component " + m_dset->getName() + " failed with error message: " + e.what());
		QMessageBox msgBox;
		msgBox.setText(mess.c_str());
		msgBox.exec();
	}
}

void HeatMapDisplayCommand::updateSelectedPoints()
{
	const std::vector <bool>& selectionLocs = m_dset->getSelection();
	std::vector <float> selection;
	std::transform(selectionLocs.begin(), selectionLocs.end(), std::back_inserter(selection), [](auto i) { return  i ? 1.f : 0.f; });

	if(m_selectedPointBuffer.empty())
		m_selectedPointBuffer.generateBuffer(selection.size(), 1, GL_FLOAT);
	m_selectedPointBuffer.updateBuffer(selection.data());
}

void HeatMapDisplayCommand::generateLutTexture(poca::core::PaletteInterface* _palette)
{
	bool interpolate = getParameter<bool>("interpolateHeatmapLUT");

	poca::core::Color4uc c = _palette->colorAt(0);
	c[3] = 0;
	_palette->setColorAt(0, c);
	unsigned int sizeLut = 512;
	unsigned int cpt = 0;
	std::vector <float> lutValues(sizeLut * 4);
	float stepLut = 1. / (float)(sizeLut - 1);
	for (float val = 0.f; val <= 1.f; val += stepLut) {
		poca::core::Color4uc c = interpolate ? _palette->getColor(val) :_palette->getColorNoInterpolation(val);
		lutValues[cpt++] = (float)c[0] / 255.f;
		lutValues[cpt++] = (float)c[1] / 255.f;
		lutValues[cpt++] = (float)c[2] / 255.f;
		lutValues[cpt++] = (float)c[3] / 255.f;
	}
	glBindTexture(GL_TEXTURE_1D, m_textureLutID);
	glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, sizeLut, 0, GL_RGBA, GL_FLOAT, lutValues.data());
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_1D, 0);
}

void HeatMapDisplayCommand::freeGPUMemory()
{
	if (m_textureLutID == 0) return;
	if (m_textureLutID != 0)
		glDeleteTextures(1, &m_textureLutID);
	m_pointBuffer.freeGPUMemory();
	m_textureLutID = 0;
}

void HeatMapDisplayCommand::updateFBO(const int _w, const int _h)
{
	m_wImage = _w;
	m_hImage = _h;
	if (m_fbo != NULL)
		delete m_fbo;
	m_fbo = new QOpenGLFramebufferObject(m_wImage, m_hImage, QOpenGLFramebufferObject::NoAttachment, GL_TEXTURE_2D, GL_RED);
	glBindTexture(GL_TEXTURE_2D, m_fbo->texture());
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_wImage, m_hImage, 0, GL_RED, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void HeatMapDisplayCommand::display(poca::opengl::Camera* _cam, const bool _offscreen)
{
	if (!m_dset->isSelected()) return;

	if (m_pointBuffer.empty() && m_dc == NULL)
		createDisplay();

	if (m_fbo == NULL) return;

	bool displayHeatmap = getParameter<bool>("displayHeatmap");
	float radiusHeatmap = getParameter<float>("radiusHeatmap");
	float intensityHeatmap = getParameter<float>("intensityHeatmap");
	bool screenRadius = getParameter<bool>("radiusHeatmapType", "radiusScreenHeatmap");

	if (!displayHeatmap) return;

	GLfloat bkColor[4];
	glGetFloatv(GL_COLOR_CLEAR_VALUE, bkColor);

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glClearColor(0.f, 0.f, 0.f, 0.f);
	glBlendFunc(GL_ONE, GL_ONE);
	bool success = m_fbo->bind();
	if (!success) std::cout << "Problem with binding" << std::endl;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	_cam->enableClippingPlanes();
	_cam->drawHeatmap<poca::core::Vec3mf, float>(m_dc ? m_dc->getPointBuffer() : m_pointBuffer, m_selectedPointBuffer, intensityHeatmap, radiusHeatmap, screenRadius, m_minX);
	_cam->disableClippingPlanes();
	success = m_fbo->release();
	if (!success) std::cout << "Problem with releasing" << std::endl;
	GL_CHECK_ERRORS();

	glClearColor(bkColor[0], bkColor[1], bkColor[2], 0.f);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	_cam->drawTextureFBO(m_fbo->texture(), m_textureLutID);
	glDisable(GL_BLEND);

	GL_CHECK_ERRORS();
}
