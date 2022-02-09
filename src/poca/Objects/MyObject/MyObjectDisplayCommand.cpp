/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MyObjectDisplayCommand.cpp
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

#include <Windows.h>
#include <gl/GL.h>
#include <fstream>

#include "MyObjectDisplayCommand.hpp"
#include "../../Objects/MyObject/MyObject.hpp"

MyObjectDisplayCommand::MyObjectDisplayCommand(poca::core::MyObjectInterface* _obj, const nlohmann::json& _parameters) :poca::core::Command("MyObjectDisplayCommand")
{
	m_object = _obj;

	if (!_parameters.contains(name())) {
		addCommandInfo(poca::core::CommandInfo("zoomFactor", 0.1f));
		addCommandInfo(poca::core::CommandInfo("currentZoom", 1.f));
		addCommandInfo(poca::core::CommandInfo("continuousZoom", false));
		addCommandInfo(poca::core::CommandInfo("smoothPoint", true));
		addCommandInfo(poca::core::CommandInfo("smoothLine", false));
		addCommandInfo(poca::core::CommandInfo("positionScaleBar", 0u));
		addCommandInfo(poca::core::CommandInfo("scaleBarHeight", 10u));
		addCommandInfo(poca::core::CommandInfo("displayScaleBar", true));
		addCommandInfo(poca::core::CommandInfo("colorScaleBar", std::array<unsigned char, 4>{0, 0, 0, 255}));
		addCommandInfo(poca::core::CommandInfo("scaleBarWidth", 1000u));
		addCommandInfo(poca::core::CommandInfo("pointSizeGL", 6u));
		addCommandInfo(poca::core::CommandInfo("lineWidthGL", 1u));
		addCommandInfo(poca::core::CommandInfo("colorBakground", std::array<unsigned char, 4>{255, 255, 255, 255}));

		addCommandInfo(poca::core::CommandInfo("nbGrid", std::array <uint8_t, 3>{ (uint8_t)5, (uint8_t)5, (uint8_t)5 }));
		addCommandInfo(poca::core::CommandInfo("stepGrid", std::array <float, 3>{ 50.f, 50.f, 50.f }));
		addCommandInfo(poca::core::CommandInfo("useNbForGrid", true));
		addCommandInfo(poca::core::CommandInfo("isotropicGrid", true));
		addCommandInfo(poca::core::CommandInfo("antialias", true));
		addCommandInfo(poca::core::CommandInfo("fontDisplay", true));
		addCommandInfo(poca::core::CommandInfo("fontSize", 20.f));

		addCommandInfo(poca::core::CommandInfo("colorSelectedROIs", std::array<unsigned char, 4>{255, 0, 255, 255}));
		addCommandInfo(poca::core::CommandInfo("colorUnselectedROIs", std::array<unsigned char, 4>{255, 0, 0, 255}));
		addCommandInfo(poca::core::CommandInfo("displayROIs", true));
		addCommandInfo(poca::core::CommandInfo("displayROILabels", true));
	}
	else {
		nlohmann::json param = _parameters[name()];
		addCommandInfo(poca::core::CommandInfo("zoomFactor", param["zoomFactor"].get<float>()));
		addCommandInfo(poca::core::CommandInfo("currentZoom", param["currentZoom"].get<float>()));
		addCommandInfo(poca::core::CommandInfo("continuousZoom", param["continuousZoom"].get<bool>()));
		addCommandInfo(poca::core::CommandInfo("smoothPoint", param["smoothPoint"].get<bool>()));
		addCommandInfo(poca::core::CommandInfo("smoothLine", param["smoothLine"].get<bool>()));
		addCommandInfo(poca::core::CommandInfo("positionScaleBar", param["positionScaleBar"].get<uint32_t>()));
		addCommandInfo(poca::core::CommandInfo("scaleBarHeight", param["positionScaleBar"].get<uint32_t>()));
		addCommandInfo(poca::core::CommandInfo("displayScaleBar", param["displayScaleBar"].get<bool>()));
		addCommandInfo(poca::core::CommandInfo("colorScaleBar", param["colorScaleBar"].get<std::array<unsigned char, 4>>()));
		addCommandInfo(poca::core::CommandInfo("scaleBarWidth", param["scaleBarWidth"].get<uint32_t>()));
		addCommandInfo(poca::core::CommandInfo("pointSizeGL", param["pointSizeGL"].get<uint32_t>()));
		addCommandInfo(poca::core::CommandInfo("lineWidthGL", param["lineWidthGL"].get<uint32_t>()));
		addCommandInfo(poca::core::CommandInfo("colorBakground", param["colorBakground"].get<std::array<unsigned char, 4>>()));

		addCommandInfo(poca::core::CommandInfo("nbGrid", param["nbGrid"].get<std::array<uint8_t, 3>>()));
		addCommandInfo(poca::core::CommandInfo("stepGrid", param["stepGrid"].get<std::array<float, 3>>()));
		addCommandInfo(poca::core::CommandInfo("useNbForGrid", param["useNbForGrid"].get<bool>()));
		addCommandInfo(poca::core::CommandInfo("isotropicGrid", param["isotropicGrid"].get<bool>()));
		addCommandInfo(poca::core::CommandInfo("antialias", param["antialias"].get<bool>()));
		addCommandInfo(poca::core::CommandInfo("fontDisplay", param["fontDisplay"].get<bool>()));
		addCommandInfo(poca::core::CommandInfo("fontSize", param["fontSize"].get<float>()));

		addCommandInfo(poca::core::CommandInfo("colorSelectedROIs", param["colorSelectedROIs"].get<std::array<unsigned char, 4>>()));
		addCommandInfo(poca::core::CommandInfo("colorUnselectedROIs", param["colorUnselectedROIs"].get<std::array<unsigned char, 4>>()));
		addCommandInfo(poca::core::CommandInfo("displayROIs", param["displayROIs"].get<bool>()));
		addCommandInfo(poca::core::CommandInfo("displayROILabels", param["displayROILabels"].get<bool>()));
	}
}

MyObjectDisplayCommand::MyObjectDisplayCommand(const MyObjectDisplayCommand & _o) : Command(_o)
{
	m_object = _o.m_object;
}

MyObjectDisplayCommand::~MyObjectDisplayCommand()
{

}

void MyObjectDisplayCommand::execute(poca::core::CommandInfo * _ci)
{
	loadParameters(*_ci);
	if (_ci->nameCommand == "display")
		display();
}

poca::core::Command * MyObjectDisplayCommand::copy()
{
	return NULL;
}

const poca::core::CommandInfos MyObjectDisplayCommand::saveParameters() const
{
	return poca::core::CommandInfos();
}

void MyObjectDisplayCommand::display() const
{
	bool ok;
	std::any param = getParameter("smoothPoint", ok);
	if (ok && std::any_cast<bool>(param))
		glEnable(GL_POINT_SMOOTH);
	else
		glDisable(GL_POINT_SMOOTH);

	param = getParameter("smoothLine", ok);
	if (ok && std::any_cast<bool>(param))
		glEnable(GL_LINE_SMOOTH);
	else
		glDisable(GL_LINE_SMOOTH);

	param = getParameter("pointSizeGL", ok);
	unsigned int ptSize = std::any_cast<unsigned int>(param);
	glPointSize(ptSize);

	param = getParameter("lineWidthGL", ok);
	unsigned int lineW = std::any_cast<unsigned int>(param);
	glLineWidth(lineW);
}

void MyObjectDisplayCommand::loadParametersFromMacro(poca::core::MyMacro * _macro)
{
	/*bool ok;
	for (poca::core::CommandInfos::iterator it = m_commandInfos.begin(); it != m_commandInfos.end(); it++){
		if (it->first == _macro->getCommand()){
			if (it->first == "zoomFactor"){
				float val = std::stof(_macro->getParameters()[0].second);
				if (ok) it->second[0] = val;
			}
			else if (it->first == "currentZoom"){
				float val = std::stof(_macro->getParameters()[0].second);
				if (ok) it->second[0] = val;
			}
			else if (it->first == "continuousZoom"){
				bool val = _macro->getParameters()[0].second == "yes";
				if (ok) it->second[0] = val;
			}
			else if (it->first == "smoothPoint"){
				float val = _macro->getParameters()[0].second == "yes";
				if (ok) it->second[0] = val;
			}
			else if (it->first == "smoothLine"){
				float val = _macro->getParameters()[0].second == "yes";
				if (ok) it->second[0] = val;
			}
			else if (it->first == "positionScaleBar"){
				unsigned int val = (unsigned int)std::stoul(_macro->getParameters()[0].second);
				if (ok) it->second[0] = val;
			}
			else if (it->first == "scaleBarHeight"){
				unsigned int val = (unsigned int)std::stoul(_macro->getParameters()[0].second);
				if (ok) it->second[0] = val;
			}
			else if (it->first == "displayScaleBar"){
				float val = _macro->getParameters()[0].second == "yes";
				if (ok) it->second[0] = val;
			}
			else if (it->first == "colorScaleBar"){
				unsigned char r = (unsigned char)std::stoul(_macro->getParameters()[0].second);
				if (ok) it->second[0] = r;
				unsigned char g = (unsigned char)std::stoul(_macro->getParameters()[1].second);
				if (ok) it->second[1] = g;
				unsigned char b = (unsigned char)std::stoul(_macro->getParameters()[2].second);
				if (ok) it->second[2] = b;
				unsigned char a = (unsigned char)std::stoul(_macro->getParameters()[3].second);
				if (ok) it->second[3] = a;
			}
			else if (it->first == "scaleBarWidth"){
				unsigned int val = (unsigned int)std::stoul(_macro->getParameters()[0].second);
				if (ok) it->second[0] = val;
			}
			else if (it->first == "pointSizeGL"){
				unsigned int val = (unsigned int)std::stoul(_macro->getParameters()[0].second);
				if (ok) it->second[0] = val;
			}
			else if (it->first == "lineWidthGL"){
				unsigned int val = (unsigned int)std::stoul(_macro->getParameters()[0].second);
				if (ok) it->second[0] = val;
			}
			else if (it->first == "colorBakground"){
				unsigned char r = (unsigned char)std::stoul(_macro->getParameters()[0].second);
				if (ok) it->second[0] = r;
				unsigned char g = (unsigned char)std::stoul(_macro->getParameters()[1].second);
				if (ok) it->second[1] = g;
				unsigned char b = (unsigned char)std::stoul(_macro->getParameters()[2].second);
				if (ok) it->second[2] = b;
				unsigned char a = (unsigned char)std::stoul(_macro->getParameters()[3].second);
				if (ok) it->second[3] = a;
			}
		}
	}*/
}

void MyObjectDisplayCommand::saveCommands(nlohmann::json& _json)
{
	bool ok;
	_json["zoomFactor"] = getCastedParameter<float>("zoomFactor", ok);
	_json["currentZoom"] = getCastedParameter<float>("currentZoom", ok);
	_json["continuousZoom"] = getCastedParameter<bool>("continuousZoom", ok);
	_json["smoothPoint"] = getCastedParameter<bool>("smoothPoint", ok);
	_json["smoothLine"] = getCastedParameter<bool>("smoothLine", ok);
	_json["positionScaleBar"] = getCastedParameter<uint32_t>("positionScaleBar", ok);
	_json["scaleBarHeight"] = getCastedParameter<uint32_t>("scaleBarHeight", ok);
	_json["displayScaleBar"] = getCastedParameter<bool>("displayScaleBar", ok);
	_json["colorScaleBar"] = getCastedParameter<std::array<unsigned char, 4>>("colorScaleBar", ok);
	_json["scaleBarWidth"] = getCastedParameter<uint32_t>("scaleBarWidth", ok);
	_json["pointSizeGL"] = getCastedParameter<uint32_t>("pointSizeGL", ok);
	_json["lineWidthGL"] = getCastedParameter<uint32_t>("lineWidthGL", ok);
	_json["colorBakground"] = getCastedParameter<std::array<unsigned char, 4>>("colorBakground", ok);

	_json["nbGrid"] = getCastedParameter<std::array <uint8_t, 3>>("nbGrid", ok);
	_json["stepGrid"] = getCastedParameter<std::array <float, 3>>("stepGrid", ok);
	_json["useNbForGrid"] = getCastedParameter<bool>("useNbForGrid", ok);
	_json["isotropicGrid"] = getCastedParameter<bool>("isotropicGrid", ok);
	_json["antialias"] = getCastedParameter<bool>("antialias", ok);
	_json["fontDisplay"] = getCastedParameter<bool>("fontDisplay", ok);
	_json["fontSize"] = getCastedParameter<float>("fontSize", ok);

	_json["colorSelectedROIs"] = getCastedParameter<std::array<unsigned char, 4>>("colorSelectedROIs", ok);
	_json["colorUnselectedROIs"] = getCastedParameter<std::array<unsigned char, 4>>("colorUnselectedROIs", ok);
	_json["displayROIs"] = getCastedParameter<bool>("displayROIs", ok);
	_json["displayROILabels"] = getCastedParameter<bool>("displayROILabels", ok);
}

