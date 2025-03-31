/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      SMLMObject.cpp
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
#include <QtCore/QDir>
#include <QtCore/QFileInfo>

#include <General/Command.hpp>
#include <Geometry/DetectionSet.hpp>

#include "SMLMObject.hpp"
#include "../../Widgets/MainFilterWidget.hpp"

SMLMObject::SMLMObject() :MyObject()
{
}

SMLMObject::~SMLMObject()
{
	std::cout << "Deleting " << getName() << std::endl;
}

const QString SMLMObject::getCorrectPath(const QString & _path){
	QDir dir(_path);
	if (dir.isAbsolute())
		return _path;
	QString res(this->getDir().c_str());
	if (!res.endsWith("/") && ! _path.startsWith("/") ) res.append("/");
	res.append(_path);
	QFileInfo fileInfo(res);
	return res;
}

poca::geometry::DetectionSet * SMLMObject::getLocalizations() const
{
	for (std::vector < poca::core::BasicComponentInterface * >::const_iterator it = m_components.begin(); it != m_components.end(); it++){
		poca::core::BasicComponentInterface* bc = *it;
		if (bc->getName() == "DetectionSet")
			return dynamic_cast <poca::geometry::DetectionSet *>(bc);
	}
	return NULL;
}

const size_t SMLMObject::dimension() const
{
	if (m_components.size() == 0)
		return 0;
	auto dimension = m_components[0]->dimension();
	for (auto n = 1; n < m_components.size(); n++) {
		auto dimension2 = m_components[n]->dimension();
		if (dimension != dimension2) {
			std::cout << "Weird behavior, dimensions between the component of the object are different, using the biggest dimension found." << std::endl;
			dimension = std::max(dimension, dimension2);
		}
	}
	return dimension;
}

std::ostream & operator<<( std::ostream & _os, const SMLMObject & _obj )
{
	return _os;
}

