/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ImagesList.cpp
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

#include <Interfaces/ImageInterface.hpp>
#include <General/BasicComponentList.hpp>

#include "ImagesList.hpp"

namespace poca::core {
	ImagesList::ImagesList(ImageInterface* _obj, const std::string& _name) :BasicComponentList("ImagesList")
	{
		m_components.push_back(_obj);
		m_currentComponent = 0;
		m_names.push_back(_name);
	}

	ImagesList::~ImagesList() {

	}

	BasicComponentInterface* ImagesList::copy()
	{
		return new ImagesList(*this);
	}

	void ImagesList::copyComponentsPtr(BasicComponentList* _list) {
		BasicComponentList::copyComponentsPtr(_list);
		ImagesList* list = dynamic_cast <ImagesList*>(_list);
		if (list) {
			for (const auto& name : list->m_names)
				m_names.push_back(name);
		}
	}

	void ImagesList::addImage(ImageInterface* _obj, const std::string& _name)
	{
		addComponent(_obj);
		m_names.push_back(_name);
	}

	ImageInterface* ImagesList::currentImage()
	{
		return static_cast<ImageInterface*>(m_components[m_currentComponent]);
	}

	uint32_t ImagesList::currentImageIndex() const
	{
		return m_currentComponent;
	}

	ImageInterface* ImagesList::getImage(const uint32_t _idx)
	{
		return static_cast<ImageInterface*>(m_components[_idx]);
	}

	void ImagesList::eraseImage(const uint32_t _index)
	{
		if (m_components.empty()) return;
		BasicComponentList::eraseComponent(_index);
		m_names.erase(m_names.begin() + _index);
	}

	const std::string& ImagesList::currentName() const
	{
		if (m_currentComponent < m_components.size())
			return m_names[m_currentComponent];
		else
			return std::string("");
	}

	const std::string& ImagesList::getName(const uint32_t _index) const
	{
		if (_index < m_components.size())
			return m_names[_index];
		else
			return std::string("");
	}

	void ImagesList::setName(const uint32_t _index, const std::string& _name)
	{
		if (_index < m_components.size())
			m_names[_index] = _name;
	}

	void ImagesList::setCurrentName(const std::string& _name)
	{
		if (m_currentComponent < m_components.size())
			m_names[m_currentComponent] = _name;
	}
}

