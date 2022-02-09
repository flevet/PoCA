/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      BasicComponent.cpp
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

#include <fstream>
#include <algorithm>

#include "BasicComponent.hpp"

#include "Palette.hpp"
#include "Histogram.hpp"
#include "Command.hpp"
#include "../General/MyData.hpp"

namespace poca::core {

	BasicComponent::BasicComponent(const std::string& _name) :CommandableObject(_name), m_nameComponent(_name), m_log(false), m_nbSelection(0), m_selected(true), m_hilow(false), m_paletteSaved(NULL)
	{
		m_palette = Palette::getStaticLutPtr("HotCold2");
	}

	BasicComponent::BasicComponent(const BasicComponent& _o) : CommandableObject(_o), m_nameComponent(_o.m_nameComponent), m_bbox(_o.m_bbox), m_currentHistogram(_o.m_currentHistogram), m_data(_o.m_data), m_log(_o.m_log), m_nbSelection(_o.m_nbSelection), m_selected(_o.m_selected)
	{
		m_palette = new Palette(*_o.m_palette);
	}

	BasicComponent::~BasicComponent()
	{
		delete m_palette;
		m_palette = NULL;
	}

	void BasicComponent::setData(const std::map <std::string, std::vector <float>>& _data)
	{
		if (!_data.empty()) {
			std::map <std::string, std::vector <float>>::const_iterator it = _data.begin();
			m_selection.resize(it->second.size());
		}
	}

	void BasicComponent::addFeature(const std::string& _nameF, MyData* _dataF)
	{
		m_data[_nameF] = _dataF;
	}

	const unsigned int BasicComponent::memorySize() const
	{
		unsigned int memoryS = 0;
		return memoryS;
	}

	void BasicComponent::forceRegenerateSelection()
	{
		std::fill(m_selection.begin(), m_selection.end(), 1);

		for (std::map <std::string, MyData*>::iterator it = m_data.begin(); it != m_data.end(); it++) {
			MyData* current = it->second;
			Histogram* histogram = current->getHistogram();
			const std::vector<float>& values = histogram->getValues();
			for (int n = 0; n < values.size(); n++)
				m_selection[n] = m_selection[n] && histogram->getCurrentMin() <= values[n] && values[n] <= histogram->getCurrentMax();
		}

		m_nbSelection = 0;
		for (int n = 0; n < m_selection.size(); n++)
			if (m_selection[n])
				m_nbSelection++;
	}

	MyData* BasicComponent::getMyData(const std::string& _type)
	{
		return m_data.find(_type) != m_data.end() ? m_data[_type] : NULL;
	}

	void BasicComponent::deleteFeature(const std::string& _type)
	{
		std::map <std::string, MyData*>::iterator it = m_data.find(_type);
		if (it == m_data.end() || m_data.size() == 1) return;
		m_data.erase(it);
		if(m_currentHistogram == _type)
			m_currentHistogram = m_data.empty() ? "" : m_data.begin()->first;
	}

	std::vector <float>& BasicComponent::getData(const std::string& _type)
	{
		return m_data[_type]->getData();
	}

	const std::vector <float>& BasicComponent::getData(const std::string& _type) const
	{
		return m_data.at(_type)->getData();
	}

	float* BasicComponent::getDataPtr(const std::string& _type)
	{
		return hasData(_type) ? m_data[_type]->getData().data() : nullptr;
	}

	const float* BasicComponent::getDataPtr(const std::string& _type) const
	{
		return hasData(_type) ? m_data.at(_type)->getData().data() : nullptr;
	}

	std::vector <float>& BasicComponent::getOriginalData(const std::string& _type)
	{
		if (hasData(_type))
			return m_data.at(_type)->getOriginalHistogram()->getValues();
		else
			throw std::runtime_error(std::string("data " + _type + " was not found for compoent " + m_nameComponent));
	}

	const std::vector <float>& BasicComponent::getOriginalData(const std::string& _type) const
	{
		if (hasData(_type))
			return m_data.at(_type)->getOriginalHistogram()->getValues();
		else
			throw std::runtime_error(std::string("data " + _type + " was not found for compoent " + m_nameComponent));
	}

	float* BasicComponent::getOriginalDataPtr(const std::string& _type)
	{
		return hasData(_type) ? m_data[_type]->getOriginalData().data() : nullptr;
	}

	const float* BasicComponent::getOriginalDataPtr(const std::string& _type) const
	{
		return hasData(_type) ? m_data.at(_type)->getOriginalData().data() : nullptr;
	}

	const std::map <std::string, MyData*>& BasicComponent::getData() const
	{
		return m_data;
	}

	std::map <std::string, MyData*>& BasicComponent::getData() {
		return m_data;
	}

	const bool BasicComponent::hasData(const std::string& _name)
	{
		bool found = m_data.find(_name) != m_data.end();
		return found;
	}

	const bool BasicComponent::hasData(const std::string& _name) const
	{
		bool found = m_data.find(_name) != m_data.end();
		return found;
	}

	void BasicComponent::executeCommand(CommandInfo* _ci)
	{
			if (_ci->nameCommand == "histogram") {
				std::string type = _ci->getParameter<std::string>("feature");
				MyData* data = getMyData(type);
				if (data != NULL) {
					std::string action = _ci->getParameter<std::string>("action");
					if (action == "log") {
						bool val = _ci->getParameter<bool>("value");
						data->setLog(val);
					}
					else if (action == "changeBoundsCustom") {
						HistogramInterface* hist = getHistogram(type);

						bool changed = false;
						if (_ci->hasParameter("min")) {
							float minV = _ci->getParameter<float>("min");
							hist->setCurrentMin(minV);
							changed = true;
						}
						if (_ci->hasParameter("max")) {
							float maxV = _ci->getParameter<float>("max");
							hist->setCurrentMax(maxV);
							changed = true;
						}

						if(changed)
							forceRegenerateSelection();
					}
					else if (action == "changeHistogramBounds") {
						HistogramInterface* hist = getHistogram(type);

						bool changedMin = false, changedMax = false;
						float minV = FLT_MAX, maxV = FLT_MAX;
						if (_ci->hasParameter("min")) {
							minV = _ci->getParameter<float>("min");
							hist->setCurrentMin(minV);
							changedMin = true;
						}
						if (_ci->hasParameter("max")) {
							maxV = _ci->getParameter<float>("max");
							hist->setCurrentMax(maxV);
							changedMax = true;
						}

						if (changedMin || changedMax) {
							hist->changeHistogramBounds(minV, maxV);
							forceRegenerateSelection();
						}
					}
					else if (action == "displayWithLUT") {
						setCurrentHistogramType(type);
					}
					else if (action == "save") {
						std::string dir = _ci->getParameter<std::string>("dir");
						if (!(dir.back() != '/'))
							dir.append("/");
						HistogramInterface* hist = data->getHistogram();
						std::string nameFile(data->isLog() ? "log_" : "");
						nameFile.append(type).append(".txt");
						nameFile = dir + nameFile;
						std::ofstream fs(nameFile.c_str());
						std::vector <float> values = hist->getValues();
						for (size_t n = 0; n < values.size(); n++)
							fs << values[n] << std::endl;
						fs.close();
						std::cout << "File " << nameFile.c_str() << " has been saved." << std::endl;
					}
					else if (action == "delete")
						deleteFeature(type);
				}
			}
			else if (_ci->nameCommand == "changeLUT") {
				if (!_ci->hasParameter("LUT")) return;
				bool hilowPrec = m_hilow;
				std::string nameLut = _ci->getParameter<std::string>("LUT");
				Palette pal = Palette::getStaticLut(nameLut);
				if (pal.null()) {
					std::cout << "LUT " << nameLut << " does not exist. Aborting." << std::endl;
					return;
				}
				m_palette->setPalette(pal);
				m_hilow = pal.isHiLow();
				if (hilowPrec != m_hilow)
					_ci->addParameter("regenerateFeatureBuffer", bool(true));
			}
			else if (_ci->nameCommand == "selected") {
				bool val = _ci->getParameter<bool>("selected");
				m_selected = val;
			}
			CommandableObject::executeCommand(_ci);
	}

	CommandInfo BasicComponent::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
	{
		if (_nameCommand == "histogram") {
			if (_parameters.contains("feature")) {
				std::string feature = _parameters["feature"].get< std::string>();
				if (_parameters.contains("action")) {
					std::string action = _parameters["action"].get< std::string>();

					if (action == "log") {
						if (_parameters.contains("value")) {
							bool val = _parameters["value"].get<bool>();
							return poca::core::CommandInfo(false, _nameCommand, "feature", feature, "action", action, "value", val);
						}
					}
					else if (action == "changeBoundsCustom") {
						float minV, maxV;
						bool complete = _parameters.contains("min");
						if (complete)
							minV = _parameters["min"].get<float>();
						complete &= _parameters.contains("max");
						if (complete) {
							maxV = _parameters["max"].get<float>();
							return poca::core::CommandInfo(false, _nameCommand, "feature", feature, "action", action, "min", minV, "max", maxV);
						}
					}
					else if (action == "changeHistogramBounds") {
						float minV, maxV;
						bool complete = _parameters.contains("min");
						if (complete)
							minV = _parameters["min"].get<float>();
						complete &= _parameters.contains("max");
						if (complete) {
							maxV = _parameters["max"].get<float>();
							return poca::core::CommandInfo(false, _nameCommand, "feature", feature, "action", action, "min", minV, "max", maxV);
						}
					}
					else if (action == "save") {
						if (_parameters.contains("dir")) {
							std::string val = _parameters["dir"].get<std::string>();
							return poca::core::CommandInfo(false, _nameCommand, "feature", feature, "action", action, "dir", val);
						}
					}
					else if (action == "displayWithLUT")
						return poca::core::CommandInfo(false, _nameCommand, "feature", feature, "action", action);
				}
			}
		}
		else if (_nameCommand == "selected") {
			bool val = _parameters.get<bool>();
			return poca::core::CommandInfo(false, _nameCommand, val);
		}
		else if (_nameCommand == "changeLUT") {
			std::string val = _parameters.get<std::string>();
			return poca::core::CommandInfo(false, _nameCommand, val);
		}

		return CommandableObject::createCommand(_nameCommand, _parameters);
	}

	void BasicComponent::setPalette(Palette* _palette)
	{
		if (m_palette != NULL)
			delete m_palette;
		m_palette = _palette;
	}

	HistogramInterface* BasicComponent::getHistogram(const std::string& _nameHist)
	{
		if (!hasData(_nameHist)) return nullptr;
		return m_data[_nameHist]->getHistogram();
	}

	HistogramInterface* BasicComponent::getHistogram(const std::string& _nameHist) const
	{
		if (!hasData(_nameHist)) return nullptr;
		return m_data.at(_nameHist)->getHistogram();
	}

	const bool BasicComponent::isLogHistogram(const std::string& _type) const
	{
		if (hasData(_type)) return m_data.at(_type)->isLog();
		return false;
	}

	HistogramInterface* BasicComponent::getOriginalHistogram(const std::string& _nameHist)
	{
		if (!hasData(_nameHist)) return nullptr;
		return m_data[_nameHist]->getOriginalHistogram();
	}

	HistogramInterface* BasicComponent::getOriginalHistogram(const std::string& _nameHist) const
	{
		if (!hasData(_nameHist)) return nullptr;
		return m_data.at(_nameHist)->getOriginalHistogram();
	}

	const bool BasicComponent::isCurrentHistogram(const std::string& _type)
	{
		return m_currentHistogram == _type;
	}

	const Color4uc BasicComponent::getColor(const float _pos) const
	{
		return m_palette->getColor(_pos);
	}

	PaletteInterface* BasicComponent::getPalette() const
	{
		return m_palette;
	}

	const std::string BasicComponent::currentHistogramType() const
	{
		return m_currentHistogram;
	}

	void BasicComponent::setCurrentHistogramType(const std::string _type)
	{
		m_currentHistogram = _type;
	}

	HistogramInterface* BasicComponent::getCurrentHistogram()
	{
		return m_data[m_currentHistogram]->getHistogram();
	}

	stringList BasicComponent::getNameData() const
	{
		poca::core::stringList list;
		for (std::map <std::string, poca::core::MyData*>::const_iterator it = getData().begin(); it != getData().end(); it++)
			list.push_back(it->first.c_str());
		return list;
	}

	stringList BasicComponent::getNameData(const std::string& _componentName) const
	{
		poca::core::stringList list;
		for (std::map <std::string, poca::core::MyData*>::const_iterator it = this->getData().begin(); it != this->getData().end(); it++)
			list.push_back(it->first.c_str());
		return list;
	}
}

