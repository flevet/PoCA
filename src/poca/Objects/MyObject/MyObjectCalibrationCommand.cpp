/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MyObjectCalibrationCommand.cpp
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

#include <General/Calibration.hpp>

#include "MyObjectCalibrationCommand.hpp"
#include "../../Objects/MyObject/MyObject.hpp"

MyObjectCalibrationCommand::MyObjectCalibrationCommand(MyObject * _obj):poca::core::Command("MyObjectCalibrationCommand")
{
	m_object = _obj;
	addCommandInfo(poca::core::CommandInfo("calibrationPixXY", 1.));
	addCommandInfo(poca::core::CommandInfo("calibrationPixZ", 1.));
	addCommandInfo(poca::core::CommandInfo("calibrationTime", 1.));
	addCommandInfo(poca::core::CommandInfo("calibrationDimemsionUnit", QString("nm")));
	addCommandInfo(poca::core::CommandInfo("calibrationTimeUnit", QString("s")));

	/*poca::core::CommandInfo ci("calibrationParameters",
		"calibrationPixXY", 1.,
		"calibrationPixZ", 1.,
		"calibrationTime", 1.,
		"calibrationDimemsionUnit", QString("nm"),
		"calibrationTimeUnit", QString("s")
	);
	addCommandInfo(ci);*/
}

MyObjectCalibrationCommand::MyObjectCalibrationCommand(const MyObjectCalibrationCommand & _o) : Command(_o)
{
	m_object = _o.m_object;
}

MyObjectCalibrationCommand::~MyObjectCalibrationCommand()
{

}

void MyObjectCalibrationCommand::execute(poca::core::CommandInfo * _ci)
{
	loadParameters(*_ci);
	poca::core::Calibration * cal = poca::core::ObjectCalibrationsSingleton::instance()->getCalibration(m_object);
	poca::core::CommandInfos cp;

	bool ok;
	std::any param = getParameter("calibrationPixXY", ok);
	if (ok) {
		double val = std::any_cast<double>(param);
		cal->setPixelXY(val);
	}
	param = getParameter("calibrationPixZ", ok);
	if (ok) {
		double val = std::any_cast<double>(param);
		cal->setPixelZ(val);
	}
	param = getParameter("calibrationTime", ok);
	if (ok) {
		double val = std::any_cast<double>(param);
		cal->setTime(val);
	}
	param = getParameter("calibrationDimemsionUnit", ok);
	if (ok) {
		QString val = std::any_cast<QString>(param);
		cal->setDimensionUnit(val.toLatin1().data());
	}
	param = getParameter("calibrationTimeUnit", ok);
	if (ok) {
		QString val = std::any_cast<QString>(param);
		cal->setTimeUnit(val.toLatin1().data());
	}
}

poca::core::Command * MyObjectCalibrationCommand::copy()
{
	return NULL;
}

const poca::core::CommandInfos MyObjectCalibrationCommand::saveParameters() const
{
	return poca::core::CommandInfos();
}

void MyObjectCalibrationCommand::loadParametersFromMacro(poca::core::MyMacro * _macro)
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

void MyObjectCalibrationCommand::saveCommands(nlohmann::json& _json)
{
}

