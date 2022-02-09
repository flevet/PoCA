/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ColocObject.hpp
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

#ifndef ColocObject_h__
#define ColocObject_h__

#include "DesignPatterns/Subject.hpp"
#include "General/CommandableObject.hpp"
#include "Interfaces/MyObjectInterface.hpp"
#include "General/BasicComponent.hpp"
#include "OpenGL/Camera.hpp"

#include <Objects/MyObject.hpp>

class ColocObject : public poca::core::MyObject {
public:
	ColocObject(std::vector<poca::core::MyObjectInterface*>);
	~ColocObject();

	float getX() const;
	float getY() const;
	float getZ() const;
	float getWidth() const;
	float getHeight() const;
	float getThick() const;

	void setWidth(const float);
	void setHeight(const float);
	void setThick(const float);

	void executeCommand(poca::core::CommandInfo*);

	const poca::core::BoundingBox boundingBox() const;

	const size_t dimension() const;

	virtual void executeCommandOnSpecificComponent(const std::string&, poca::core::CommandInfo*);
	virtual void executeGlobalCommand(poca::core::CommandInfo*);

	const size_t nbColors() const { return m_colors.size(); }
	poca::core::MyObjectInterface* getObject(const size_t _index) { return m_colors[_index]; }
	poca::core::MyObjectInterface* currentObject() { return m_colors[m_currentColor]; }
	size_t currentObjectID() const { return m_currentColor; }
	void setCurrentObject(const size_t _idx) { m_currentColor = _idx; }

protected:
	void display(poca::opengl::Camera*, const bool);

protected:
	std::vector <poca::core::MyObjectInterface*> m_colors;
	size_t m_currentColor;
};

#endif // SMLMObject_h__

