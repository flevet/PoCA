/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      LoaderInterface.hpp
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

#ifndef LOADERINTERFACE_H
#define LOADERINTERFACE_H

#include <QObject>
#include <QString>
#include <any>

namespace poca {
	namespace core {
		class CommandInfo;
		class BasicComponentInterface;
		class Engine;
	}
};

//! [0]
class LoaderInterface
{
public:
    virtual ~LoaderInterface() = default;
    virtual poca::core::BasicComponentInterface* loadData(const QString&, poca::core::CommandInfo* = NULL) = 0;
	virtual QStringList extensions() const = 0;
	virtual void setSingletons(poca::core::Engine*) = 0;
};


QT_BEGIN_NAMESPACE

#define LoaderInterface_iid "POCA.LoaderInterface_iid"

Q_DECLARE_INTERFACE(LoaderInterface, LoaderInterface_iid)
QT_END_NAMESPACE

//! [0]
#endif




