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

#include <General/Vec4.hpp>
#include <General/Histogram.hpp>
#include <General/MyData.hpp>
#include <General/Misc.h>
#include <Interfaces/CameraInterface.hpp>
#include <General/stb_rect_pack.h>

#include "MyMultipleObject.hpp"

MyMultipleObject::MyMultipleObject(std::vector<poca::core::MyObjectInterface*> _colors) :MyObject(), m_colors(_colors), m_currentColor(0)
{
	m_internalId = poca::core::NbObjects++;

	uint32_t PADDING_BINS = 0;
	size_t total = 0, maxD = 0;
	std::vector<stbrp_rect> rects;
	int cur = 0;
	float startW = 0.f, startH = 0.f;
	for (auto n = 0; n < nbColors(); n++) {
		auto obj = getObject(n);
		const auto& bbox = obj->boundingBox();
		//gridBBoxes.emplace_back(bbox.x(), bbox.y(), bbox.z(), bbox.width() + PADDING_BINS, bbox.height() + PADDING_BINS, bbox.thick());
		m_gridBBoxes.emplace_back(startW, 0.f, 0.f, startW + bbox.realWidth(), bbox.realHeight(), bbox.realThick());
		rects.push_back({ cur++, (int)(bbox.realWidth() + PADDING_BINS), (int)(bbox.realHeight() + PADDING_BINS), 0, 0, 0 });
		auto tmp = bbox.realWidth() > bbox.realHeight() ? bbox.realWidth() : bbox.realHeight();
		total += tmp;
		maxD = std::max(maxD, (size_t)tmp);
		//startW += bbox.realWidth();
		std::cout << "GridBBox " << m_gridBBoxes.back() << std::endl;
	}

	Bin bin{ maxD * 4, maxD * 4};            // start size
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
	for (poca::core::MyObjectInterface* obj : m_colors) {
		/*if (_ci->nameCommand == "display") {
			poca::opengl::Camera* cam = _ci->getParameterPtr<poca::opengl::Camera>("camera");
			poca::core::CommandInfo com(false, "getModelMatrix");
			obj->executeCommand(&com);
			if (com.hasParameter("modelMatrix")) {
				auto mat = com.getParameter<glm::mat4>("modelMatrix");
				cam->setModelMatrix(mat);
			}
		}*/
		obj->executeCommand(_ci);
	}
	poca::core::MyObject::executeCommand(_ci);
	poca::core::CommandableObject::executeCommand(_ci);
}

const poca::core::BoundingBox MyMultipleObject::boundingBox() const
{
	poca::core::BoundingBox bbox(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
	if (!m_gridSelected) {
		for (poca::core::MyObjectInterface* obj : m_colors) {
			poca::core::BoundingBox bboxComp = obj->boundingBox();
			for (size_t i = 0; i < 3; i++)
				bbox[i] = bboxComp[i] < bbox[i] ? bboxComp[i] : bbox[i];
			for (size_t i = 3; i < 6; i++)
				bbox[i] = bboxComp[i] > bbox[i] ? bboxComp[i] : bbox[i];
		}
	}
	else {
		for (const auto& bboxComp : m_gridBBoxes) {
			for (size_t i = 0; i < 3; i++)
				bbox[i] = bboxComp[i] < bbox[i] ? bboxComp[i] : bbox[i];
			for (size_t i = 3; i < 6; i++)
				bbox[i] = bboxComp[i] > bbox[i] ? bboxComp[i] : bbox[i];
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
	poca::core::BasicComponentInterface* bci = getBasicComponent(_nameComponent);
	if (bci)
		bci->executeCommand(_ci);
	for (poca::core::MyObjectInterface* obj : m_colors)
		obj->executeCommandOnSpecificComponent(_nameComponent, _ci);
}

void MyMultipleObject::executeGlobalCommand(poca::core::CommandInfo* _ci)
{
	//executeCommand(_ci);
	poca::core::MyObject::executeCommand(_ci);
	for (poca::core::MyObjectInterface* obj : m_colors)
		obj->executeGlobalCommand(_ci);
	for (std::vector < poca::core::BasicComponentInterface* >::const_iterator it = m_components.begin(); it != m_components.end(); it++) {
		poca::core::BasicComponentInterface* bc = *it;
		bc->executeCommand(_ci);
	}
}

void MyMultipleObject::resetModelMatrices(const bool _gridSelected) {
	m_gridSelected = _gridSelected;
	if (!m_gridSelected) {
		const auto& bbox = this->boundingBox();
		float translationX = bbox[0] + (bbox[3] - bbox[0]) / 2.f;
		float translationY = bbox[1] + (bbox[4] - bbox[1]) / 2.f;
		float translationZ = bbox[2] + (bbox[5] - bbox[2]) / 2.f;

		//auto matrix = glm::translate(glm::mat4(1.f), glm::vec3(-translationX, -translationY, -translationZ));
		auto matrix = glm::translate(glm::mat4(1.f), glm::vec3(bbox[0], bbox[1], bbox[2]));
		for (auto obj : m_colors)
			obj->setModelMatrix(matrix);
	}
	else {
		for (auto n = 0; n < m_colors.size(); n++) {
			auto obj = m_colors[n];
			const auto& bboxObj = obj->boundingBox();
			const auto& bbox = m_gridBBoxes[n];
			float translationX = bbox[0] + (bbox[3] - bbox[0]) / 2.f;
			float translationY = bbox[1] + (bbox[4] - bbox[1]) / 2.f;
			float translationZ = bbox[2] + (bbox[5] - bbox[2]) / 2.f;

			auto matrix = glm::translate(glm::mat4(1.f), glm::vec3(bbox[0] - bboxObj[0], bbox[1] - bboxObj[1], bbox[2] - bboxObj[2]));
			m_colors[n]->setModelMatrix(matrix);
		}
	}
}