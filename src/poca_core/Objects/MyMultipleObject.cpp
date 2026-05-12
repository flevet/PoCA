/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MyMultipleObject.cpp
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

#define STB_RECT_PACK_IMPLEMENTATION

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <General/Vec4.hpp>
#include <General/Histogram.hpp>
#include <General/MyData.hpp>
#include <General/Misc.h>
#include <Interfaces/CameraInterface.hpp>
#include <Interfaces/BasicComponentInterface.hpp>
#include <General/stb_rect_pack.h>
#include <OpenGL/RenderCommandContext.hpp>

#include "MyMultipleObject.hpp"

namespace {
	bool isComponentFamilyHandled(const poca::core::CommandExecutionResult& _result, const std::string& _componentName)
	{
		if (!_result.has<poca::opengl::RenderedComponentFamilies>())
			return false;
		const std::vector<std::string>& names = _result.get<poca::opengl::RenderedComponentFamilies>().componentNames;
		return std::find(names.begin(), names.end(), _componentName) != names.end();
	}

	void appendUnique(std::vector<size_t>& _values, const size_t _value)
	{
		if (std::find(_values.begin(), _values.end(), _value) == _values.end())
			_values.push_back(_value);
	}

	poca::core::BoundingBox localObjectBoundingBox(poca::core::MyObjectInterface* _object)
	{
		if (_object == nullptr)
			return poca::core::BoundingBox::initBBox();

		poca::core::BoundingBox bbox = poca::core::BoundingBox::initBBox();
		bool hasBBox = false;
		for (poca::core::BasicComponentInterface* component : _object->getComponents()) {
			if (component == nullptr) continue;
			const poca::core::BoundingBox& componentBBox = component->boundingBox();
			for (int ix = 0; ix < 2; ++ix) {
				for (int iy = 0; iy < 2; ++iy) {
					for (int iz = 0; iz < 2; ++iz) {
						const float x = ix == 0 ? componentBBox[0] : componentBBox[3];
						const float y = iy == 0 ? componentBBox[1] : componentBBox[4];
						const float z = iz == 0 ? componentBBox[2] : componentBBox[5];
						if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
							continue;
						bbox.addPointBBox(x, y, z);
						hasBBox = true;
					}
				}
			}
		}
		return hasBBox ? bbox : _object->boundingBox();
	}

}

MyMultipleObject::MyMultipleObject(std::vector<poca::core::MyObjectInterface*> _colors) :MyObject(), m_colors(_colors), m_currentColor(0)
{
	m_internalId = poca::core::NbObjects++;

	recomputeGrid();
}

MyMultipleObject::~MyMultipleObject()
{
	for (poca::core::MyObjectInterface* obj : m_colors)
		delete obj;
}

float MyMultipleObject::getX() const
{
	float x = FLT_MAX;
	for (poca::core::MyObjectInterface* obj : m_colors) {
		float val = obj->getX();
		if (val < x) x = val;
	}
	return x;
}

float MyMultipleObject::getY() const
{
	float y = FLT_MAX;
	for (poca::core::MyObjectInterface* obj : m_colors) {
		float val = obj->getY();
		if (val < y) y = val;
	}
	return y;
}

float MyMultipleObject::getZ() const
{
	float z = FLT_MAX;
	for (poca::core::MyObjectInterface* obj : m_colors) {
		float val = obj->getZ();
		if (val < z) z = val;
	}
	return z;
}

float MyMultipleObject::getWidth() const
{
	float w = -FLT_MAX;
	for (poca::core::MyObjectInterface* obj : m_colors) {
		float val = obj->getWidth();
		if (val > w) w = val;
	}
	return w;
}

float MyMultipleObject::getHeight() const
{
	float h = -FLT_MAX;
	for (poca::core::MyObjectInterface* obj : m_colors) {
		float val = obj->getHeight();
		if (val > h) h = val;
	}
	return h;
}

float MyMultipleObject::getThick() const
{
	float t = -FLT_MAX;
	for (poca::core::MyObjectInterface* obj : m_colors) {
		float val = obj->getThick();
		if (val > t) t = val;
	}
	return t;
}

void MyMultipleObject::setWidth(const float _w)
{
	for (poca::core::MyObjectInterface* obj : m_colors)
		obj->setWidth(_w);
}

void MyMultipleObject::setHeight(const float _h)
{
	for (poca::core::MyObjectInterface* obj : m_colors)
		obj->setHeight(_h);
}

void MyMultipleObject::setThick(const float _t)
{
	for (poca::core::MyObjectInterface* obj : m_colors)
		obj->setThick(_t);
}

void MyMultipleObject::executeCommand(poca::core::CommandInfo* _ci)
{
	poca::core::CommandExecutionContext context;
	executeCommand(_ci, context);
}

void MyMultipleObject::executeCommand(poca::core::CommandInfo* _ci, const poca::core::CommandExecutionContext& _context)
{
	poca::core::CommandExecutionResult result;
	executeCommand(_ci, _context, result);
}

void MyMultipleObject::executeCommand(poca::core::CommandInfo* _ci, const poca::core::CommandExecutionContext& _context, poca::core::CommandExecutionResult& _result)
{
	if (!m_selectedObjectIndices.empty()) {
		for (const size_t index : m_selectedObjectIndices) {
			if (index < m_colors.size() && m_colors[index] != NULL)
				m_colors[index]->executeCommand(_ci, _context, _result);
		}
	}
	else {
		for (poca::core::MyObjectInterface* obj : m_colors) {
			obj->executeCommand(_ci, _context, _result);
		}
	}
	poca::core::MyObject::executeCommand(_ci, _context, _result);
	poca::core::CommandableObject::executeCommand(_ci, _context, _result);
}

const poca::core::BoundingBox MyMultipleObject::boundingBox() const
{
	poca::core::BoundingBox bbox(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
	for (poca::core::MyObjectInterface* obj : m_colors) {
		if (obj == NULL) continue;
		const poca::core::BoundingBox bboxComp = localObjectBoundingBox(obj);
		const glm::mat4& model = obj->getModelMatrix();
		for (int ix = 0; ix < 2; ++ix) {
			for (int iy = 0; iy < 2; ++iy) {
				for (int iz = 0; iz < 2; ++iz) {
					glm::vec4 p(
						ix == 0 ? bboxComp[0] : bboxComp[3],
						iy == 0 ? bboxComp[1] : bboxComp[4],
						iz == 0 ? bboxComp[2] : bboxComp[5],
						1.f);
					p = model * p;
					bbox[0] = std::min(bbox[0], p.x);
					bbox[1] = std::min(bbox[1], p.y);
					bbox[2] = std::min(bbox[2], p.z);
					bbox[3] = std::max(bbox[3], p.x);
					bbox[4] = std::max(bbox[4], p.y);
					bbox[5] = std::max(bbox[5], p.z);
				}
			}
		}
	}
	return bbox;
}

const size_t MyMultipleObject::dimension() const
{
	return m_colors.front()->dimension();
}

void MyMultipleObject::executeCommandOnSpecificComponent(const std::string& _nameComponent, poca::core::CommandInfo* _ci)
{
	poca::core::CommandExecutionContext context;
	executeCommandOnSpecificComponent(_nameComponent, _ci, context);
}

void MyMultipleObject::executeCommandOnSpecificComponent(const std::string& _nameComponent, poca::core::CommandInfo* _ci, const poca::core::CommandExecutionContext& _context)
{
	poca::core::CommandExecutionResult result;
	executeCommandOnSpecificComponent(_nameComponent, _ci, _context, result);
}

void MyMultipleObject::executeCommandOnSpecificComponent(const std::string& _nameComponent, poca::core::CommandInfo* _ci, const poca::core::CommandExecutionContext& _context, poca::core::CommandExecutionResult& _result)
{
	poca::core::BasicComponentInterface* bci = getBasicComponent(_nameComponent);
	if (bci)
		bci->executeCommand(_ci, _context, _result);
	if (_result.has<poca::opengl::ChildObjectRenderingHandled>() && _result.get<poca::opengl::ChildObjectRenderingHandled>().handled)
		return;
	if (!m_selectedObjectIndices.empty()) {
		for (const size_t index : m_selectedObjectIndices) {
			if (index < m_colors.size() && m_colors[index] != NULL)
				m_colors[index]->executeCommandOnSpecificComponent(_nameComponent, _ci, _context, _result);
		}
	}
	else {
		for (poca::core::MyObjectInterface* obj : m_colors)
			obj->executeCommandOnSpecificComponent(_nameComponent, _ci, _context, _result);
	}
}

void MyMultipleObject::executeGlobalCommand(poca::core::CommandInfo* _ci)
{
	poca::core::CommandExecutionContext context;
	executeGlobalCommand(_ci, context);
}

void MyMultipleObject::executeGlobalCommand(poca::core::CommandInfo* _ci, const poca::core::CommandExecutionContext& _context)
{
	poca::core::CommandExecutionResult result;
	executeGlobalCommand(_ci, _context, result);
}

void MyMultipleObject::executeGlobalCommand(poca::core::CommandInfo* _ci, const poca::core::CommandExecutionContext& _context, poca::core::CommandExecutionResult& _result)
{
	//executeCommand(_ci);
	poca::core::MyObject::executeCommand(_ci, _context, _result);
	if (!m_selectedObjectIndices.empty()) {
		for (const size_t index : m_selectedObjectIndices) {
			if (index < m_colors.size() && m_colors[index] != NULL)
				m_colors[index]->executeGlobalCommand(_ci, _context, _result);
		}
	}
	else {
		for (poca::core::MyObjectInterface* obj : m_colors)
			obj->executeGlobalCommand(_ci, _context, _result);
	}
	for (std::vector < poca::core::BasicComponentInterface* >::const_iterator it = m_components.begin(); it != m_components.end(); it++) {
		poca::core::BasicComponentInterface* bc = *it;
		if (isComponentFamilyHandled(_result, bc->getName()))
			continue;
		bc->executeCommand(_ci, _context, _result);
	}
}

void MyMultipleObject::resetModelMatrices(const bool _gridSelected) {
	m_gridSelected = _gridSelected;
	if (!m_gridSelected) {
		const auto& bbox = this->boundingBox();
		float translationX = bbox[0] + (bbox[3] - bbox[0]) / 2.f;
		float translationY = bbox[1] + (bbox[4] - bbox[1]) / 2.f;
		float translationZ = bbox[2] + (bbox[5] - bbox[2]) / 2.f;

		for (auto obj : m_colors) {
			obj->setTranslationVector(glm::vec3(0.f));
			obj->setRotationMatrix(glm::mat4(1.f));
			obj->updateModelMatrixFromTransform();
		}
	}
	else {
		for (auto n = 0; n < m_colors.size(); n++) {
			auto obj = m_colors[n];
			const auto& bboxObj = localObjectBoundingBox(obj);
			const auto& bbox = m_gridBBoxes[n];
			float translationX = bbox[0] + (bbox[3] - bbox[0]) / 2.f;
			float translationY = bbox[1] + (bbox[4] - bbox[1]) / 2.f;
			float translationZ = bbox[2] + (bbox[5] - bbox[2]) / 2.f;

			const glm::vec3 offset(bbox[0] - bboxObj[0], bbox[1] - bboxObj[1], bbox[2] - bboxObj[2]);
			// MyObject::updateModelMatrixFromTransform builds translate(-m_translation).
			// Store the inverse translation so later gizmo rotations/translations keep
			// the multiple-object grid placement instead of overwriting it.
			m_colors[n]->setTranslationVector(-offset);
			m_colors[n]->setRotationMatrix(glm::mat4(1.f));
			m_colors[n]->updateModelMatrixFromTransform();
		}
	}
}

void MyMultipleObject::recomputeGrid()
{
	uint32_t PADDING_BINS = 0;
	size_t total = 0, maxD = 0;
	std::vector<stbrp_rect> rects;
	int cur = 0;
	float startW = 0.f, startH = 0.f;
	m_gridBBoxes.clear();
	for (auto n = 0; n < nbColors(); n++) {
		auto obj = getObject(n);
		const auto& bbox = localObjectBoundingBox(obj);
		//gridBBoxes.emplace_back(bbox.x(), bbox.y(), bbox.z(), bbox.width() + PADDING_BINS, bbox.height() + PADDING_BINS, bbox.thick());
		m_gridBBoxes.emplace_back(startW, 0.f, 0.f, startW + bbox.realWidth(), bbox.realHeight(), bbox.realThick());
		rects.push_back({ cur++, (int)(bbox.realWidth() + PADDING_BINS), (int)(bbox.realHeight() + PADDING_BINS), 0, 0, 0 });
		auto tmp = bbox.realWidth() > bbox.realHeight() ? bbox.realWidth() : bbox.realHeight();
		total += tmp;
		maxD = std::max(maxD, (size_t)tmp);
		//startW += bbox.realWidth();
		std::cout << "GridBBox " << m_gridBBoxes.back() << std::endl;
	}

	Bin bin{ maxD * 6, maxD * 6 };            // start size
	const int MAX_W = total;       // safety caps (tune as needed)
	const int MAX_H = total;

	// Retry with growth until success or we hit caps.
	while (true) {
		// Make a working copy because stbrp_rect gets filled in-place (x,y,was_packed).
		std::cout << " ----------------------- Bin size " << bin.w << ", " << bin.h << std::endl;

		auto work = rects;
		if (try_pack(bin, work)) {
			for (const auto& r : work) {
				float w = m_gridBBoxes[r.id].realWidth(), h = m_gridBBoxes[r.id].realHeight();
				//gridBBoxes[r.id].set(r.x, r.y, -gridBBoxes[r.id].realThick() / 2.f, r.x + w, r.y + h, gridBBoxes[r.id].realThick() / 2.f);
				m_gridBBoxes[r.id].set(r.x, r.y, 0, r.x + w, r.y + h, m_gridBBoxes[r.id].realThick());
			}
			break;
		}


		Bin next = grow(bin, MAX_W, MAX_H, 1.25f);
		if (next.w == bin.w && next.h == bin.h) {
			std::cerr << "Cannot fit within caps " << MAX_W << "x" << MAX_H << "\n";
			return;
		}
		bin = next;
	}
	resetModelMatrices(true);
}

void MyMultipleObject::clearHierarchy()
{
	m_hierarchy.clear();
}

size_t MyMultipleObject::addHierarchyNode(const std::string& _label, const std::string& _levelName, int _parentIndex)
{
	if (_parentIndex >= (int)m_hierarchy.size())
		throw std::out_of_range("Invalid parent hierarchy node index");

	m_hierarchy.push_back({ _label, _levelName, _parentIndex });
	const size_t nodeIndex = m_hierarchy.size() - 1;
	if (_parentIndex >= 0)
		m_hierarchy[_parentIndex].children.push_back(nodeIndex);
	return nodeIndex;
}

void MyMultipleObject::attachObjectToHierarchyNode(const size_t _nodeIndex, const size_t _objectIndex)
{
	if (_nodeIndex >= m_hierarchy.size())
		throw std::out_of_range("Invalid hierarchy node index");
	if (_objectIndex >= m_colors.size())
		throw std::out_of_range("Invalid multiple object child index");

	appendUnique(m_hierarchy[_nodeIndex].objectIndices, _objectIndex);
}

std::vector<size_t> MyMultipleObject::collectObjectIndicesForHierarchyNode(const size_t _nodeIndex, const bool _includeDescendants) const
{
	if (_nodeIndex >= m_hierarchy.size())
		throw std::out_of_range("Invalid hierarchy node index");

	std::vector<size_t> objectIndices = m_hierarchy[_nodeIndex].objectIndices;
	if (!_includeDescendants)
		return objectIndices;

	std::vector<size_t> stack = m_hierarchy[_nodeIndex].children;
	while (!stack.empty()) {
		const size_t childIndex = stack.back();
		stack.pop_back();
		if (childIndex >= m_hierarchy.size())
			continue;

		for (const size_t objectIndex : m_hierarchy[childIndex].objectIndices)
			appendUnique(objectIndices, objectIndex);
		for (const size_t grandChildIndex : m_hierarchy[childIndex].children)
			stack.push_back(grandChildIndex);
	}

	return objectIndices;
}

void MyMultipleObject::setSelectedObjectIndices(const std::vector<size_t>& _indices)
{
	m_selectedObjectIndices.clear();
	for (const size_t index : _indices) {
		if (index < m_colors.size())
			appendUnique(m_selectedObjectIndices, index);
	}
	if (!m_selectedObjectIndices.empty())
		m_currentColor = m_selectedObjectIndices.front();
}
