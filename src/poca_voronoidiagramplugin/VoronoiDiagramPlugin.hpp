/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      VoronoiDiagramPlugin.hpp
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

#ifndef VoronoiDiagram_PLUGIN_H
#define VoronoiDiagram_PLUGIN_H

#include <QObject>
#include <QtPlugin>
#include <vector>

#include <Interfaces/MyObjectInterface.hpp>
#include <General/BasicComponent.hpp>
#include <Interfaces/HistogramInterface.hpp>
#include <General/Engine.hpp>
#include <General/Command.hpp>
#include <Geometry/DetectionSet.hpp>

#include "../../include/PluginInterface.hpp"
#include "VoronoiDiagramWidget.hpp"

class VoronoiDiagramConstructionCommand : public poca::core::Command
{
public:
    VoronoiDiagramConstructionCommand(poca::geometry::DetectionSet*);
    VoronoiDiagramConstructionCommand(const VoronoiDiagramConstructionCommand&);
    ~VoronoiDiagramConstructionCommand();

    void execute(poca::core::CommandInfo*);
    poca::core::Command* copy();
    const poca::core::CommandInfos saveParameters() const {
        return poca::core::CommandInfos();
    }
    poca::core::CommandInfo createCommand(const std::string&, const nlohmann::json&);

protected:
    poca::geometry::DetectionSet* m_dset;
};

//! [0]
class VoronoiDiagramPlugin : public QObject, PluginInterface
{
    Q_OBJECT
        Q_PLUGIN_METADATA(IID "POCA.PluginInterface_iid" FILE "voronoidiagramplugin.json")
        Q_INTERFACES(PluginInterface)

public:
    void addGUI(poca::core::MediatorWObjectFWidgetInterface*, QTabWidget*) override;
    std::vector <std::pair<QAction*, QString>> getActions() override;
    poca::core::MyObjectInterface* actionTriggered(QObject*, poca::core::MyObjectInterface*) override;
    void addCommands(poca::core::CommandableObject*);
    void setPlugins(poca::core::PluginList* _plugins) { m_plugins = _plugins; }
    void setSingletons(poca::core::Engine*);
    void execute(poca::core::CommandInfo*);

    void computeVoronoi(poca::core::MyObjectInterface*);

    QString name() const { return "VoronoiDiagramPlugin"; }

protected:
    std::vector <std::pair<QAction*, QString>> m_actions;

    VoronoiDiagramWidget* m_widget;
    QTabWidget* m_parent;

public:
    static nlohmann::json m_parameters;
    static poca::core::PluginList* m_plugins;
};
//! [0]

#endif

