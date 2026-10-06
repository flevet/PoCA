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
	ImagesList::ImagesList(ImageInterface* _im, const std::string& _name) :BasicComponentList("ImagesList", _im)
	{
		m_names.push_back(_name);
		m_labelSources.push_back(-1);
	}

	ImagesList::ImagesList(std::unique_ptr<ImageInterface> _image, const std::string& _name)
		: BasicComponentList("ImagesList",std::unique_ptr<BasicComponent>(std::move(_image)))
	{
		m_names.push_back(_name);
		m_labelSources.push_back(-1);
	}

	ImagesList::ImagesList(const ImagesList& _other) : BasicComponentList(_other),
		m_names(_other.m_names), m_labelSources(_other.m_labelSources)
	{
		m_currentComponent = _other.m_currentComponent;
	}

	ImagesList::~ImagesList() {

	}

	BasicComponentInterface* ImagesList::copy()
	{
		return new ImagesList(*this);
	}

	void ImagesList::copyComponentsPtr(BasicComponentList* _list) {
		const auto offset = m_components.size();
		BasicComponentList::copyComponentsPtr(_list);
		ImagesList* list = dynamic_cast <ImagesList*>(_list);
		if (list) {
			for (const auto& name : list->m_names)
				m_names.push_back(name);
			for (const auto source : list->m_labelSources)
				m_labelSources.push_back(source < 0 ? -1 : source + static_cast<int64_t>(offset));
		}
	}

	void ImagesList::addImage(ImageInterface* _obj, const std::string& _name)
	{
		int64_t source = -1;
		if (_obj->isLabelImage() && m_currentComponent < m_components.size()) {
			const auto* current = static_cast<ImageInterface*>(m_components[m_currentComponent]);
			source = current->isRawImage() ? static_cast<int64_t>(m_currentComponent) : m_labelSources.at(m_currentComponent);
		}
		addComponent(_obj);
		m_names.push_back(_name);
		m_labelSources.push_back(source);
	}

	void ImagesList::associateLabel(uint32_t _label, uint32_t _source)
	{
		if (_label >= m_components.size() || _source >= m_components.size() ||
			!getImage(_label)->isLabelImage() || !getImage(_source)->isRawImage())
			throw std::invalid_argument("Label association requires a LABEL entry and a RAW source entry");
		m_labelSources.at(_label) = _source;
	}

	void ImagesList::addLabelImage(ImageInterface* _image, const std::string& _name, uint32_t _source)
	{
		if (!_image || !_image->isLabelImage() || _source >= m_components.size() || !getImage(_source)->isRawImage())
			throw std::invalid_argument("Invalid source/label image association");
		addImage(_image, _name);
		associateLabel(m_currentComponent, _source);
	}

	void ImagesList::addLabelImage(std::unique_ptr<ImageInterface> _image, const std::string& _name, uint32_t _source)
	{
		if (!_image || !_image->isLabelImage() || _source >= m_components.size() || !getImage(_source)->isRawImage())
			throw std::invalid_argument("Invalid owned source/label image association");
		auto box = boundingBox();
		const auto& added = _image->boundingBox();
		for (int i = 0; i < 3; ++i) box[i] = (std::min)(box[i],added[i]);
		for (int i = 3; i < 6; ++i) box[i] = (std::max)(box[i],added[i]);
		m_components.reserve(m_components.size()+1);
		m_labelSources.reserve(m_labelSources.size()+1);
		m_names.push_back(_name);
		m_labelSources.push_back(_source);
		m_components.push_back(_image.get());
		_image.release();
		m_currentComponent = static_cast<uint32_t>(m_components.size()-1);
		m_bbox = box;
	}

	std::vector<uint32_t> ImagesList::labelsForImage(uint32_t _source) const
	{
		if (_source >= m_components.size()) throw std::out_of_range("Source image index");
		std::vector<uint32_t> labels;
		for (uint32_t i = 0; i < m_components.size(); ++i)
			if (m_labelSources.at(i) == _source && static_cast<ImageInterface*>(m_components[i])->isLabelImage())
				labels.push_back(i);
		return labels;
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

	void ImagesList::eraseComponent(const uint32_t _index)
	{
		if (m_components.empty()) return;
		if (_index >= m_components.size()) throw std::out_of_range("Image index");
		const auto previousCurrent = m_currentComponent;
		BasicComponentList::eraseComponent(_index);
		if (_index < previousCurrent) m_currentComponent = previousCurrent-1;
		m_names.erase(m_names.begin() + _index);
		m_labelSources.erase(m_labelSources.begin() + _index);
		for (auto& source : m_labelSources) {
			if (source == _index) source = -1;
			else if (source > _index) --source;
		}
		if (!m_components.empty() && m_currentComponent >= m_components.size())
			m_currentComponent = static_cast<uint32_t>(m_components.size()-1);
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

