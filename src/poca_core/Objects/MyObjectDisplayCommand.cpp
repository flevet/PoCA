/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MyObjectDisplayCommand.cpp
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
#include <gl/GL.h>
#include <gl/GLU.h>
#include <fstream>

#include <DesignPatterns/StateSoftwareSingleton.hpp>

#include "MyObjectDisplayCommand.hpp"
#include "MyObject.hpp"

#ifndef NDEBUG
#include <iostream>
#include <cassert>

#define __TO_STR2(x) __EVAL_STR2(x)
#define __EVAL_STR2(x) #x

#define glAssert2(code) do{code; int l = __LINE__;\
   GLuint err = glGetError(); \
                if (err != GL_NO_ERROR)\
                { \
                  std::cerr << "OpenGL error : " << __FILE__ << "\n";\
                  std::cerr << "line : " << l << "\n";\
                  std::cerr << "Source code : " << __TO_STR2(code) << "\n";\
                  std::cerr << "Message : " << (const char*)gluErrorString(err) << "("<<err<<")" << "\n";\
                  assert(false); \
              }\
}while(false)

// -----------------------------------------------------------------------------

#define GL_CHECK_ERRORS2() \
do{ GLuint err = glGetError(); \
                if (err != GL_NO_ERROR)\
                { \
                  std::cerr << "OpenGL error : " << __FILE__ << "\n";\
                  std::cerr << "line : " << __LINE__ << "\n";\
                  std::cerr << "Source code : " << __TO_STR2(code) << "\n";\
                  std::cerr << "Message : " << (const char*)gluErrorString(err) << "("<<err<<")" << "\n";\
                  assert(false); \
              }\
}while(false)

#else

#define __TO_STR(x) __EVAL_STR(x)
#define __EVAL_STR(x) #x

#define glAssert(code) \
    code

//#define GL_CHECK_ERRORS()
#define GL_CHECK_ERRORS() \
do{ GLuint err = glGetError(); \
                if (err != GL_NO_ERROR)\
                { \
                  std::cerr << "OpenGL error : " << __FILE__ << "\n";\
                  std::cerr << "line : " << __LINE__ << "\n";\
                  std::cerr << "Source code : " << __TO_STR(code) << "\n";\
                  std::cerr << "Message : " << (const char*)gluErrorString(err) << "("<<err<<")" << "\n";\
              }\
}while(false)
#endif

MyObjectDisplayCommand::MyObjectDisplayCommand(poca::core::MyObjectInterface* _obj) :poca::core::Command("MyObjectDisplayCommand")
{
	m_object = _obj;

	poca::core::StateSoftwareSingleton* sss = poca::core::StateSoftwareSingleton::instance();
	const nlohmann::json& parameters = sss->getParameters();
	addCommandInfo(poca::core::CommandInfo(false, "zoomFactor", 0.1f));
	addCommandInfo(poca::core::CommandInfo(false, "currentZoom", 1.f));
	addCommandInfo(poca::core::CommandInfo(false, "continuousZoom", false));
	addCommandInfo(poca::core::CommandInfo(false, "smoothPoint", true));
	addCommandInfo(poca::core::CommandInfo(false, "smoothLine", false));
	addCommandInfo(poca::core::CommandInfo(false, "positionScaleBar", 0u));
	addCommandInfo(poca::core::CommandInfo(false, "scaleBarHeight", 10u));
	addCommandInfo(poca::core::CommandInfo(false, "displayScaleBar", true));
	addCommandInfo(poca::core::CommandInfo(false, "colorScaleBar", std::array<unsigned char, 4>{0, 0, 0, 255}));
	addCommandInfo(poca::core::CommandInfo(false, "scaleBarWidth", 1000u));
	addCommandInfo(poca::core::CommandInfo(false, "pointSizeGL", 6u));
	addCommandInfo(poca::core::CommandInfo(false, "lineWidthGL", 1u));
	addCommandInfo(poca::core::CommandInfo(false, "colorBakground", std::array<unsigned char, 4>{255, 255, 255, 255}));

	addCommandInfo(poca::core::CommandInfo(false, "nbGrid", std::array <uint8_t, 3>{ (uint8_t)5, (uint8_t)5, (uint8_t)5 }));
	addCommandInfo(poca::core::CommandInfo(false, "stepGrid", std::array <float, 3>{ 50.f, 50.f, 50.f }));
	addCommandInfo(poca::core::CommandInfo(false, "useNbForGrid", true));
	addCommandInfo(poca::core::CommandInfo(false, "isotropicGrid", true));
	addCommandInfo(poca::core::CommandInfo(false, "antialias", true));
	addCommandInfo(poca::core::CommandInfo(false, "cullFace", true));
	addCommandInfo(poca::core::CommandInfo(false, "clip", true));
	addCommandInfo(poca::core::CommandInfo(false, "fillPolygon", true));
	addCommandInfo(poca::core::CommandInfo(false, "fontDisplay", true));
	addCommandInfo(poca::core::CommandInfo(false, "fontSize", 20.f));

	addCommandInfo(poca::core::CommandInfo(false, "colorSelectedROIs", std::array<unsigned char, 4>{255, 0, 255, 255}));
	addCommandInfo(poca::core::CommandInfo(false, "colorUnselectedROIs", std::array<unsigned char, 4>{255, 0, 0, 255}));
	addCommandInfo(poca::core::CommandInfo(false, "displayROIs", true));
	addCommandInfo(poca::core::CommandInfo(false, "displayROILabels", true));

	addCommandInfo(poca::core::CommandInfo(false, "useSSAO", false));
	addCommandInfo(poca::core::CommandInfo(false, "radiusSSAO", 10.f));
	addCommandInfo(poca::core::CommandInfo(false, "strengthSSAO", 1.f));
	addCommandInfo(poca::core::CommandInfo(false, "useSilhouetteSSAO", false));
	addCommandInfo(poca::core::CommandInfo(false, "useDebugSSAO", false));
	addCommandInfo(poca::core::CommandInfo(false, "currentDebugSSAO", (int)0));
	if (parameters.contains(name())) {
		nlohmann::json param = parameters[name()];
		if (param.contains("zoomFactor"))
			loadParameters(poca::core::CommandInfo(false, "zoomFactor", param["zoomFactor"].get<float>()));
		if (param.contains("currentZoom"))
			loadParameters(poca::core::CommandInfo(false, "currentZoom", param["currentZoom"].get<float>()));
		if (param.contains("continuousZoom"))
			loadParameters(poca::core::CommandInfo(false, "continuousZoom", param["continuousZoom"].get<bool>()));
		if (param.contains("smoothPoint"))
			loadParameters(poca::core::CommandInfo(false, "smoothPoint", param["smoothPoint"].get<bool>()));
		if (param.contains("smoothLine"))
			loadParameters(poca::core::CommandInfo(false, "smoothLine", param["smoothLine"].get<bool>()));
		if (param.contains("positionScaleBar"))
			loadParameters(poca::core::CommandInfo(false, "positionScaleBar", param["positionScaleBar"].get<uint32_t>()));
		if (param.contains("scaleBarHeight"))
			loadParameters(poca::core::CommandInfo(false, "scaleBarHeight", param["positionScaleBar"].get<uint32_t>()));
		if (param.contains("displayScaleBar"))
			loadParameters(poca::core::CommandInfo(false, "displayScaleBar", param["displayScaleBar"].get<bool>()));
		if (param.contains("colorScaleBar"))
			loadParameters(poca::core::CommandInfo(false, "colorScaleBar", param["colorScaleBar"].get<std::array<unsigned char, 4>>()));
		if (param.contains("scaleBarWidth"))
			loadParameters(poca::core::CommandInfo(false, "scaleBarWidth", param["scaleBarWidth"].get<uint32_t>()));
		if (param.contains("pointSizeGL"))
			loadParameters(poca::core::CommandInfo(false, "pointSizeGL", param["pointSizeGL"].get<uint32_t>()));
		if (param.contains("lineWidthGL"))
			loadParameters(poca::core::CommandInfo(false, "lineWidthGL", param["lineWidthGL"].get<uint32_t>()));
		if (param.contains("colorBakground"))
			loadParameters(poca::core::CommandInfo(false, "colorBakground", param["colorBakground"].get<std::array<unsigned char, 4>>()));

		if (param.contains("nbGrid"))
			loadParameters(poca::core::CommandInfo(false, "nbGrid", param["nbGrid"].get<std::array<uint8_t, 3>>()));
		if (param.contains("stepGrid"))
			loadParameters(poca::core::CommandInfo(false, "stepGrid", param["stepGrid"].get<std::array<float, 3>>()));
		if (param.contains("useNbForGrid"))
			loadParameters(poca::core::CommandInfo(false, "useNbForGrid", param["useNbForGrid"].get<bool>()));
		if (param.contains("isotropicGrid"))
			loadParameters(poca::core::CommandInfo(false, "isotropicGrid", param["isotropicGrid"].get<bool>()));
		if (param.contains("antialias"))
			loadParameters(poca::core::CommandInfo(false, "antialias", param["antialias"].get<bool>()));
		if (param.contains("cullFace"))
			loadParameters(poca::core::CommandInfo(false, "cullFace", param["cullFace"].get<bool>()));
		if (param.contains("clip"))
			loadParameters(poca::core::CommandInfo(false, "clip", param["clip"].get<bool>()));
		if (param.contains("fillPolygon"))
			loadParameters(poca::core::CommandInfo(false, "fillPolygon", param["fillPolygon"].get<bool>()));
		if (param.contains("fontDisplay"))
			loadParameters(poca::core::CommandInfo(false, "fontDisplay", param["fontDisplay"].get<bool>()));
		if (param.contains("fontSize"))
			loadParameters(poca::core::CommandInfo(false, "fontSize", param["fontSize"].get<float>()));

		if (param.contains("colorSelectedROIs"))
			loadParameters(poca::core::CommandInfo(false, "colorSelectedROIs", param["colorSelectedROIs"].get<std::array<unsigned char, 4>>()));
		if (param.contains("colorUnselectedROIs"))
			loadParameters(poca::core::CommandInfo(false, "colorUnselectedROIs", param["colorUnselectedROIs"].get<std::array<unsigned char, 4>>()));
		if (param.contains("displayROIs"))
			loadParameters(poca::core::CommandInfo(false, "displayROIs", param["displayROIs"].get<bool>()));
		if (param.contains("displayROILabels"))
			loadParameters(poca::core::CommandInfo(false, "displayROILabels", param["displayROILabels"].get<bool>()));

		if (param.contains("useSSAO"))
			loadParameters(poca::core::CommandInfo(false, "useSSAO", param["useSSAO"].get<bool>()));
		if (param.contains("radiusSSAO"))
			loadParameters(poca::core::CommandInfo(false, "radiusSSAO", param["radiusSSAO"].get<float>()));
		if (param.contains("strengthSSAO"))
			loadParameters(poca::core::CommandInfo(false, "strengthSSAO", param["strengthSSAO"].get<float>()));
		if (param.contains("useSilhouetteSSAO"))
			loadParameters(poca::core::CommandInfo(false, "useSilhouetteSSAO", param["useSilhouetteSSAO"].get<bool>()));
		if (param.contains("useDebugSSAO"))
			loadParameters(poca::core::CommandInfo(false, "useDebugSSAO", param["useDebugSSAO"].get<bool>()));
		if (param.contains("currentDebugSSAO"))
			loadParameters(poca::core::CommandInfo(false, "currentDebugSSAO", param["currentDebugSSAO"].get<int>()));
	}
}

MyObjectDisplayCommand::MyObjectDisplayCommand(const MyObjectDisplayCommand& _o) : Command(_o)
{
	m_object = _o.m_object;
}

MyObjectDisplayCommand::~MyObjectDisplayCommand()
{

}

void MyObjectDisplayCommand::execute(poca::core::CommandInfo* _ci)
{
	loadParameters(*_ci);
	if (_ci->nameCommand == "display")
		display();
}

poca::core::Command* MyObjectDisplayCommand::copy()
{
	return NULL;
}

const poca::core::CommandInfos MyObjectDisplayCommand::saveParameters() const
{
	return poca::core::CommandInfos();
}

void MyObjectDisplayCommand::display() const
{
	/*GL_CHECK_ERRORS2();
	if (hasParameter("smoothPoint")) {
		if (getParameter<bool>("smoothPoint"))
			glEnable(GL_POINT_SMOOTH);
		else
			glDisable(GL_POINT_SMOOTH);
	}
	GL_CHECK_ERRORS2();

	if (hasParameter("smoothLine")) {
		if (getParameter<bool>("smoothLine"))
			glEnable(GL_LINE_SMOOTH);
		else
			glDisable(GL_LINE_SMOOTH);
	}
	GL_CHECK_ERRORS2();

	if (hasParameter("pointSizeGL"))
		glPointSize(getParameter<unsigned int>("pointSizeGL"));
	GL_CHECK_ERRORS2();*/

	//if (hasParameter("lineWidthGL"))
	//	glPointSize(getParameter<unsigned int>("lineWidthGL"));

	//glClear(GL_DEPTH_BUFFER_BIT);
}

poca::core::CommandInfo MyObjectDisplayCommand::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	if (_nameCommand == "zoomFactor") {
		float val = _parameters.get<float>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "currentZoom") {
		float val = _parameters.get<float>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "continuousZoom") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "smoothPoint") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "smoothLine") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "positionScaleBar") {
		uint32_t val = _parameters.get<uint32_t>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "scaleBarHeight") {
		uint32_t val = _parameters.get<uint32_t>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "displayScaleBar") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "colorScaleBar") {
		std::array<unsigned char, 4> val = _parameters.get<std::array<unsigned char, 4>>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "scaleBarWidth") {
		uint32_t val = _parameters.get<uint32_t>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "pointSizeGL") {
		uint32_t val = _parameters.get<uint32_t>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "lineWidthGL") {
		uint32_t val = _parameters.get<uint32_t>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "colorBakground") {
		std::array<unsigned char, 4> val = _parameters.get<std::array<unsigned char, 4>>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "nbGrid") {
		std::array<uint8_t, 3> val = _parameters.get<std::array<uint8_t, 3>>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "stepGrid") {
		std::array<float, 3> val = _parameters.get<std::array<float, 3>>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "useNbForGrid") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "isotropicGrid") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "antialias") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "cullFace") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "fontDisplay") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "fontSize") {
		float val = _parameters.get<float>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "colorSelectedROIs") {
		std::array<unsigned char, 4> val = _parameters.get<std::array<unsigned char, 4>>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "colorUnselectedROIs") {
		std::array<unsigned char, 4> val = _parameters.get<std::array<unsigned char, 4>>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "displayROIs") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "displayROILabels") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "useSSAO") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "radiusSSAO") {
		float val = _parameters.get<float>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "strengthSSAO") {
		float val = _parameters.get<float>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "useSilhouetteSSAO") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "useDebugSSAO") {
		bool val = _parameters.get<bool>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	else if (_nameCommand == "currentDebugSSAO") {
		int val = _parameters.get<int>();
		return poca::core::CommandInfo(false, _nameCommand, val);
	}
	return poca::core::CommandInfo();
}

void MyObjectDisplayCommand::saveCommands(nlohmann::json& _json)
{
	poca::core::Command::saveCommands(_json);
	_json["useSSAO"] = false;
}

