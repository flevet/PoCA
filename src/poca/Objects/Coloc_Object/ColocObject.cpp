/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ColocObject.cpp
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

#include <algorithm>

#include <General/Vec4.hpp>
#include <General/Histogram.hpp>
#include <General/MyData.hpp>
#include <General/Misc.h>
#include <Interfaces/CameraInterface.hpp>

#include "ColocObject.hpp"

ColocObject::ColocObject(std::vector<poca::core::MyObjectInterface*> _colors) :MyObject(), m_colors(_colors), m_currentColor(0)
{
	m_internalId = poca::core::NbObjects++;
}

ColocObject::~ColocObject()
{
	for (poca::core::MyObjectInterface* obj : m_colors)
		delete obj;
}

float ColocObject::getX() const
{
	float x = FLT_MAX;
	for (poca::core::MyObjectInterface* obj : m_colors) {
		float val = obj->getX();
		if (val < x) x = val;
	}
	return x;
}

float ColocObject::getY() const
{
	float y = FLT_MAX;
	for (poca::core::MyObjectInterface* obj : m_colors) {
		float val = obj->getY();
		if (val < y) y = val;
	}
	return y;
}

float ColocObject::getZ() const
{
	float z = FLT_MAX;
	for (poca::core::MyObjectInterface* obj : m_colors) {
		float val = obj->getZ();
		if (val < z) z = val;
	}
	return z;
}

float ColocObject::getWidth() const
{
	float w = -FLT_MAX;
	for (poca::core::MyObjectInterface* obj : m_colors) {
		float val = obj->getWidth();
		if (val > w) w = val;
	}
	return w;
}

float ColocObject::getHeight() const
{
	float h = -FLT_MAX;
	for (poca::core::MyObjectInterface* obj : m_colors) {
		float val = obj->getHeight();
		if (val > h) h = val;
	}
	return h;
}

float ColocObject::getThick() const
{
	float t = -FLT_MAX;
	for (poca::core::MyObjectInterface* obj : m_colors) {
		float val = obj->getThick();
		if (val > t) t = val;
	}
	return t;
}

void ColocObject::setWidth(const float _w)
{
	for (poca::core::MyObjectInterface* obj : m_colors)
		obj->setWidth(_w);
}

void ColocObject::setHeight(const float _h)
{
	for (poca::core::MyObjectInterface* obj : m_colors)
		obj->setHeight(_h);
}

void ColocObject::setThick(const float _t)
{
	for (poca::core::MyObjectInterface* obj : m_colors)
		obj->setThick(_t);
}

void ColocObject::executeCommand(poca::core::CommandInfo* _ci)
{
	for (poca::core::MyObjectInterface* obj : m_colors) {
		obj->executeCommand(_ci);
	}
	poca::core::MyObject::executeCommand(_ci);
	poca::core::CommandableObject::executeCommand(_ci);
}

void ColocObject::display(poca::opengl::Camera* _cam, const bool _offscreen)
{

}

const poca::core::BoundingBox ColocObject::boundingBox() const
{
	poca::core::BoundingBox bbox(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
	for (poca::core::MyObjectInterface* obj : m_colors) {
		poca::core::BoundingBox bboxComp = obj->boundingBox();
		for (size_t i = 0; i < 3; i++)
			bbox[i] = bboxComp[i] < bbox[i] ? bboxComp[i] : bbox[i];
		for (size_t i = 3; i < 6; i++)
			bbox[i] = bboxComp[i] > bbox[i] ? bboxComp[i] : bbox[i];
	}
	return bbox;
}

const size_t ColocObject::dimension() const
{
	return m_colors.front()->dimension();
}

void ColocObject::executeCommandOnSpecificComponent(const std::string& _nameComponent, poca::core::CommandInfo* _ci)
{
	poca::core::BasicComponentInterface* bci = getBasicComponent(_nameComponent);
	if (bci)
		bci->executeCommand(_ci);
	for (poca::core::MyObjectInterface* obj : m_colors)
		obj->executeCommandOnSpecificComponent(_nameComponent, _ci);
}

void ColocObject::executeGlobalCommand(poca::core::CommandInfo* _ci)
{
	executeCommand(_ci);
	for (poca::core::MyObjectInterface* obj : m_colors)
		obj->executeGlobalCommand(_ci);
	for (std::vector < poca::core::BasicComponentInterface* >::const_iterator it = m_components.begin(); it != m_components.end(); it++) {
		poca::core::BasicComponentInterface* bc = *it;
		bc->executeCommand(_ci);
	}
}

