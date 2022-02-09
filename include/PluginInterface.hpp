/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      PluginInterface.hpp
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

#ifndef PLUGININTERFACE_H
#define PLUGININTERFACE_H

#include <QtCore/QObject>
#include <any>

class QTabWidget;
class QAction;

//! [0]

namespace poca::core{
	class MyObjectInterface;
	class BasicComponent;
	class MediatorWObjectFWidgetInterface;
    class PluginList;
    class CommandInfo;
    class CommandableObject;
}

/*namespace nlohmann {
    class json;
}*/

class PluginInterface
{
public:
    virtual ~PluginInterface() = default;
    virtual void addGUI(poca::core::MediatorWObjectFWidgetInterface*, QTabWidget*) = 0;
    virtual std::vector <std::pair<QAction*, QString>> getActions() = 0;
    virtual poca::core::MyObjectInterface* actionTriggered(QObject*, poca::core::MyObjectInterface*) = 0;
    virtual void addCommands(poca::core::CommandableObject*) = 0;
    virtual void setPlugins(poca::core::PluginList*) = 0;
    virtual void setSingletons(const std::map <std::string, std::any>&) = 0;
    virtual QString name() const = 0;
    virtual void execute(poca::core::CommandInfo*) = 0;
};


QT_BEGIN_NAMESPACE

#define PluginInterface_iid "POCA.PluginInterface_iid"

Q_DECLARE_INTERFACE(PluginInterface, PluginInterface_iid)
QT_END_NAMESPACE

//! [0]
#endif






