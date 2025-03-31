/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ColocTesselerPlugin.cpp
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

#include <QtWidgets/QTabWidget>
#include <QtWidgets/QAction>

#include <General/Misc.h>
#include <Geometry/VoronoiDiagram.hpp>
#include <Interfaces/DelaunayTriangulationInterface.hpp>
#include <Interfaces/VoronoiDiagramFactoryInterface.hpp>
#include <General/PluginList.hpp>
#include <General/Engine.hpp>
#include <General/Engine.hpp>
#include <DesignPatterns/MacroRecorderSingleton.hpp>
#include <Objects/MyObject.hpp>
#include <General/Engine.hpp>

#include "ColocTesselerPlugin.hpp"
#include "ColocTesselerWidget.hpp"
#include "ColocTesseler.hpp"
#include "ColocTesselerBasicCommands.hpp"
#include "ColocTesselerDisplayCommand.hpp"

static char* colocTesselerIcon[] = {
	/* columns rows colors chars-per-pixel */
	"43 42 79 1 ",
	"  c None",
	". c black",
	"X c #2E7F2E",
	"o c #3E653E",
	"O c #396E39",
	"+ c #3A6D3A",
	"@ c #3C6A3C",
	"# c #347634",
	"$ c #337833",
	"% c #307D30",
	"& c #387038",
	"* c #5D355D",
	"= c #5A3B5A",
	"- c #593C59",
	"; c #5A3C5A",
	": c #6F196F",
	"> c #6C1F6C",
	", c #770C77",
	"< c #760E76",
	"1 c #7A077A",
	"2 c #7B067B",
	"3 c #7D037D",
	"4 c #7E007E",
	"5 c #7E017E",
	"6 c #7F007F",
	"7 c #7F017F",
	"8 c #7E037E",
	"9 c #7C057C",
	"0 c #7C067C",
	"q c #790B79",
	"w c #7A097A",
	"e c #780D78",
	"r c #711671",
	"t c #731473",
	"y c #721672",
	"u c #751075",
	"i c #652765",
	"p c #612F61",
	"a c #662866",
	"s c #6A216A",
	"d c #6B206B",
	"f c #6A226A",
	"g c #6A236A",
	"h c #682668",
	"j c #613061",
	"k c #4E4C4E",
	"l c #465A46",
	"z c #465B46",
	"x c #445C44",
	"c c #4A554A",
	"v c #4C504C",
	"b c #485848",
	"n c #524652",
	"m c #554255",
	"M c #524852",
	"N c #416041",
	"B c #426142",
	"V c #1B9B1B",
	"C c #1A9F1A",
	"Z c #07BB07",
	"A c #06BF06",
	"S c #0BB60B",
	"D c #0EB20E",
	"F c #0CB50C",
	"G c #17A217",
	"H c #16A516",
	"J c #10AE10",
	"K c #288728",
	"L c #298629",
	"P c #2A862A",
	"I c #2C832C",
	"U c #229122",
	"Y c #229222",
	"T c #03C003",
	"R c #00C700",
	"E c #02C502",
	"W c #04C204",
	"Q c #00C800",
	"! c #800080",
	/* pixels */
	".......................................... ",
	".......................................... ",
	"..     RQ !!          !!               R.. ",
	"..     RQz!!          !!               Q.. ",
	"..      Qc!           !!              RQ.. ",
	"..      Q-!           !!              QQ.. ",
	"..       d!           !!              QR.. ",
	"..       2!T          !!             RQ .. ",
	"..       !yR          !!            QQR .. ",
	"..       !aQ          !!  RQQRRQQQQQQQ  .. ",
	"..       !dQQ RRQQQRQR!!-z%CAQQQQQQQQQQ .. ",
	"..       !6RQQQQQQQQQ#!!!!!!!0q!     QQ .. ",
	"..      !!nQRQQ      !!   !!!!!!!!!!!vKK.. ",
	"..      !!YQ         !!         !!!!!!!!.. ",
	"..      !!+L        !!                -d.. ",
	"..    !!!!!!!!!!!  6!!                 Q.. ",
	"..  !!!6m@*!!!!!!!!!!!                 Q.. ",
	"..!!!! ZQQ        !!!!                 Q.. ",
	"..!!   QQ           !!                  .. ",
	"..&WQQQQR           !!                 +.. ",
	"..QQQQQQ             !!               u!.. ",
	"..    RQ             !!             !!!e.. ",
	"..    RQ             !!           !!!0NS.. ",
	"..    RQ             !!         !!!!*SQ .. ",
	"..     QR             !!      !!!!! QQ  .. ",
	"..     QQ             !!    6!!!!  RQ   .. ",
	"..    QQQQ            !!   !!!!   RQQ   .. ",
	"..   QQQQQR            !!!!!!     QR    .. ",
	"..   RQ  RQQR          !!!!      QQ     .. ",
	"..!!iIV    RQQQ        !!       QQR     .. ",
	"..!!!!!!!!62xYDQ       !!      QQR      .. ",
	"..  NM<!!!!!!!!6>py   !!       QQ       .. ",
	"..  QR      !!!6!!!!!!!!      QQ        .. ",
	".. RQQ           X#z=e!!6    RQR        .. ",
	".. RQ              QQQb!!!NRQQQ         .. ",
	".. QR               QC60-!!hHQ          .. ",
	"..RQQ               Rd!!  !!0@          .. ",
	"..RQ                &!!    !!!:   !!!!!!.. ",
	"..QR               Gq!      Me!!!!!!!!!!.. ",
	"..QQ               j!:       J!!!!      .. ",
	".......................................... ",
	".......................................... "
};

ColocTesselerConstructionCommand::ColocTesselerConstructionCommand(poca::core::MyObject* _obj) :poca::core::Command("ColocTesselerConstructionCommand")
{
	m_object = _obj;
}

ColocTesselerConstructionCommand::ColocTesselerConstructionCommand(const ColocTesselerConstructionCommand& _o) : Command(_o)
{
	m_object = _o.m_object;
}

ColocTesselerConstructionCommand::~ColocTesselerConstructionCommand()
{

}

void ColocTesselerConstructionCommand::execute(poca::core::CommandInfo* _ci)
{
	if (m_object == NULL) return;
	if (_ci->nameCommand == "computeColocTesseler") {
		if (m_object->nbColors() < 2) return;
		poca::geometry::VoronoiDiagram* voros[] = { NULL, NULL };
		for (size_t n = 0; n < 2; n++) {
			poca::core::MyObjectInterface* object = m_object->getObject(n);
			poca::core::BasicComponentInterface* bci = object->getBasicComponent("VoronoiDiagram");
			poca::geometry::VoronoiDiagram* voro = dynamic_cast <poca::geometry::VoronoiDiagram*>(bci);
			if (voro == NULL) {
				poca::geometry::VoronoiDiagramFactoryInterface* factoryVoronoi = poca::geometry::createVoronoiDiagramFactory();
				poca::geometry::VoronoiDiagram* voro = factoryVoronoi->createVoronoiDiagram(object, true, ColocTesselerPlugin::m_plugins, false);
				if (voro == NULL) return;
				delete factoryVoronoi;
				voros[n] = voro;
			}
			else
				voros[n] = voro;
		}
		if (voros[0] == NULL || voros[1] == NULL) return;
		ColocTesseler* colocT = new ColocTesseler(voros[0], voros[1]);
		m_object->addBasicComponent(colocT);
		ColocTesselerPlugin::m_plugins->addCommands(colocT);
		m_object->notify("LoadObjCharacteristicsAllWidgets");
		m_object->notifyAll("updateDisplay");
	}

}

poca::core::Command* ColocTesselerConstructionCommand::copy()
{
	return NULL;
}

poca::core::CommandInfo ColocTesselerConstructionCommand::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	if (_nameCommand == "computeColocTesseler") {
		poca::core::CommandInfo ci(false, "computeColocTesseler");
		return ci;
	}
	return poca::core::CommandInfo();
}


poca::core::PluginList* ColocTesselerPlugin::m_plugins = NULL;
nlohmann::json ColocTesselerPlugin::m_parameters;

void ColocTesselerPlugin::addGUI(poca::core::MediatorWObjectFWidgetInterface* _mediator, QTabWidget* _parent)
{
	ColocTesselerWidget* w = new ColocTesselerWidget(_mediator, _parent);
	_mediator->addWidget(w);

	QTabWidget * tabW = poca::core::utils::addSingleTabWidget(_parent, QString("Colocalization"), QString("Coloc-Tesseler"), w);
	w->setParentTab(tabW);
}

std::vector <std::pair<QAction*, QString>> ColocTesselerPlugin::getActions()
{
    QPixmap pixmap(colocTesselerIcon);
    QAction * action = new QAction(QIcon(pixmap), tr("&Coloc-Tesseler"), this);
    action->setStatusTip(tr("Create coloc-tesseler"));

    m_actions.push_back(std::make_pair(action, "Toolbar/2Color"));
	return m_actions;
}

poca::core::MyObjectInterface* ColocTesselerPlugin::actionTriggered(QObject* _sender, poca::core::MyObjectInterface* _obj)
{
	if (_obj == NULL) return NULL;
	QAction* action = static_cast <QAction*>(_sender);
	if (action == m_actions[0].first) {
		if (_obj->nbColors() < 2) return NULL;
		poca::core::CommandInfo command(true, "computeColocTesseler");
		_obj->executeCommand(&command);
	}
	return NULL;
}

void ColocTesselerPlugin::addCommands(poca::core::CommandableObject* _bc)
{
	poca::core::MyObject* obj = dynamic_cast <poca::core::MyObject*>(_bc);
	if (obj && obj->nbColors() > 1) {
		obj->addCommand(new ColocTesselerConstructionCommand(obj));
	}
	ColocTesseler* colocT = dynamic_cast <ColocTesseler*>(_bc);
	if (colocT) {
		colocT->addCommand(new ColocTesselerBasicCommands(colocT));
		colocT->addCommand(new ColocTesselerDisplayCommand(colocT));
	}
}

void ColocTesselerPlugin::setSingletons(poca::core::Engine* _engine)
{
	poca::core::Engine::instance()->setEngineSingleton(_engine); poca::core::Engine::instance()->setAllSingletons();
}

