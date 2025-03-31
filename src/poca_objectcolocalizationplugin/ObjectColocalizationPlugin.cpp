/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectColocalizationPlugin.cpp
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

#include <QtWidgets/QTabWidget>

#include <General/Misc.h>
#include <OpenGL/Helper.h>
#include <General/Engine.hpp>
#include <Geometry/ObjectLists.hpp>
#include <General/PluginList.hpp>
#include <Objects/MyObject.hpp>
#include <General/Engine.hpp>

#include "ObjectColocalizationPlugin.hpp"
#include "ObjectColocalization.hpp"
#include "ObjectColocalizationCommands.hpp"
#include "ObjectColocalizationWidget.hpp"
#include "ObjectColocalizationInfoDialog.hpp"

static char* objColocIcon[] = {
	/* columns rows colors chars-per-pixel */
	"29 29 103 2 ",
	"   c #7E3E0E",
	".  c #385723",
	"X  c #33522F",
	"o  c #314E3C",
	"O  c #345030",
	"+  c #355131",
	"@  c #4D4F1C",
	"#  c #5A4D1B",
	"$  c #435320",
	"%  c #445623",
	"&  c #595926",
	"*  c #715524",
	"=  c #62583C",
	"-  c #466631",
	";  c #53743C",
	":  c #796735",
	">  c #203864",
	",  c #253D69",
	"<  c #354341",
	"1  c #345143",
	"2  c #28416D",
	"3  c #364867",
	"4  c #2E4673",
	"5  c #36507D",
	"6  c #43577C",
	"7  c #496566",
	"8  c #4A6474",
	"9  c #843C0C",
	"0  c #8A4211",
	"q  c #8C481B",
	"w  c #954F1F",
	"e  c #A0592A",
	"r  c #946837",
	"t  c #A96233",
	"y  c #AE6838",
	"u  c #9F6741",
	"i  c #997B49",
	"p  c #A5704D",
	"a  c #BC7647",
	"s  c #A97654",
	"d  c #618349",
	"f  c #73985B",
	"g  c #799D61",
	"h  c #BE8253",
	"j  c #B4886B",
	"k  c #BC957B",
	"l  c #80A568",
	"z  c #82A769",
	"x  c #8EB475",
	"c  c #96BC7B",
	"v  c #99BF7F",
	"b  c #C68153",
	"n  c #C88455",
	"m  c #D08C5E",
	"M  c #CD9A6A",
	"N  c #D69263",
	"B  c #D89465",
	"V  c #D89E70",
	"C  c #EBA87A",
	"Z  c #3D5684",
	"A  c #435B8A",
	"S  c #445E8B",
	"D  c #546788",
	"F  c #4B6593",
	"G  c #6D7D99",
	"H  c #5B74A3",
	"J  c #607AA9",
	"K  c #6D88A9",
	"L  c #7B89A3",
	"P  c #6E88B8",
	"I  c #728CBC",
	"U  c #738DBE",
	"Y  c #7791C2",
	"T  c #7C97C7",
	"R  c #8996AD",
	"E  c #96A1B6",
	"W  c #9DA8BB",
	"Q  c #C3A089",
	"!  c #CDAF9C",
	"~  c #F4B183",
	"^  c #D1B6A4",
	"/  c #D6BFAF",
	"(  c #A0C886",
	")  c #A4CB89",
	"_  c #A9D18E",
	"`  c #DAC5B7",
	"'  c #859BC7",
	"]  c #819CCD",
	"[  c #8EA8DB",
	"{  c #8FAADC",
	"}  c #AFB7C7",
	"|  c #B9C1CE",
	" . c #BDC4D1",
	".. c #CBD0DB",
	"X. c #CED3DD",
	"o. c #E2D1C6",
	"O. c #E8DAD1",
	"+. c #EEE3DD",
	"@. c #D9DDE4",
	"#. c #EAECF0",
	"$. c #F1F3F5",
	"%. c #FDFCFB",
	"&. c white",
	/* pixels */
	"&.&.&.&.` u w s / &.&.&.&.&.O.k j ! &.&.&.&.&.&.&.&.&.&.&.",
	"&.&.&.! 9 9 9 9 9 q Q ^ k w 9 9 9 9 k &.&.&.&.&.&.&.&.&.&.",
	"&.&.O.9 9 b C n w 9 9 9 9 0 0 y n 9 9 O.&.&.&.&.&.&.&.&.&.",
	"&.&.s 9 a ~ ~ ~ ~ B a t a n ~ C B e 9 j &.&.&.&.&.&.&.&.&.",
	"&.o.9 0 C ~ ~ ~ ~ ~ C h r * # @ $ % . . 3 D L W ..&.&.&.&.",
	"&.k 9 y ~ ~ ~ ~ V * % . . . . . . . . . O > > , > , G #.&.",
	"&.p 9 N ~ ~ ~ m & . . . ; d l l c ( f . < I H F 2 > > , #.",
	"&.w 9 B ~ ~ ~ & . - l ) _ _ _ _ _ ( v . O { [ [ { T Z > L ",
	"&.0 9 ~ ~ ~ ~ . . ( ) _ _ _ _ _ _ ) ( . . T [ { { { T > 6 ",
	"&.9 9 ~ ~ ~ ~ & . l ( _ _ _ _ _ _ _ ) . . Y [ [ [ { [ > > ",
	"&.0 9 C ~ ~ ~ i . ; _ _ _ _ _ _ _ _ _ . . I [ [ [ [ [ > > ",
	"&.s 9 n ~ ~ ~ C . . ( ( _ _ _ _ _ _ _ . . I [ [ [ { { > > ",
	"&.Q 9 w ~ ~ ~ ~ : . ; _ _ _ _ _ _ _ v . . T [ [ [ [ [ > , ",
	"&.%.0 0 B ~ ~ ~ V . . _ _ _ _ _ _ _ x . O [ [ [ [ [ T > 6 ",
	"&.&.k 9 w ~ ~ ~ ~ % . c _ _ _ _ _ _ g . o { [ [ [ [ K > G ",
	"&.&.#.0 9 a ~ ~ M . . ) _ _ _ _ _ ( - . 8 { [ H 2 A Z > W ",
	"&.&.&.` 9   t h % . d _ ) ( x g d . . O ] [ 2 > > > > > @.",
	"&.&.&.&.` 0   . . . . . . . . . . . 1 T { 5 > A X.L A E &.",
	"&.&.&.&.&.+.= @ . . . . . . 1 1 7 K { { F > 5 #.&.&.&.&.&.",
	"&.&.&.&.&.$.> > ' { [ [ [ { { { { { { T > > @.&.&.&.&.&.&.",
	"&.&.&.&.&.W > A { [ [ [ { [ { { { { [ 4 > W &.&.&.&.&.&.&.",
	"&.&.&.&.&.D > I [ { [ [ [ [ [ { { { K > 6 &.&.&.&.&.&.&.&.",
	"&.&.&.&.&.4 > ' [ { { [ [ { [ [ [ [ > > | &.&.&.&.&.&.&.&.",
	"&.&.&.&.&.D > H [ [ { { [ { { { [ F > D &.&.&.&.&.&.&.&.&.",
	"&.&.&.&.&...> > Z I { { { { { [ P > , #.&.&.&.&.&.&.&.&.&.",
	"&.&.&.&.&.&.| 4 > > 4 H T [ [ K > > } &.&.&.&.&.&.&.&.&.&.",
	"&.&.&.&.&.&.&.&.W 5 > > > > > > > L &.&.&.&.&.&.&.&.&.&.&.",
	"&.&.&.&.&.&.&.&.&.&. .R S 5 5 D | &.&.&.&.&.&.&.&.&.&.&.&.",
	"&.&.&.&.&.&.&.&.&.&.&.&.&.&.&.&.&.&.&.&.&.&.&.&.&.&.&.&.&."
};

ObjectColocalizationConstructionCommand::ObjectColocalizationConstructionCommand(poca::core::MyObjectInterface* _obj) :poca::core::Command("ObjectColocalizationConstructionCommand")
{
	m_object = _obj;
}

ObjectColocalizationConstructionCommand::ObjectColocalizationConstructionCommand(const ObjectColocalizationConstructionCommand& _o) : Command(_o)
{
	m_object = _o.m_object;
}

ObjectColocalizationConstructionCommand::~ObjectColocalizationConstructionCommand()
{

}

void ObjectColocalizationConstructionCommand::execute(poca::core::CommandInfo* _ci)
{
	if (m_object == NULL || m_object->nbColors() < 2) return;
	if (_ci->nameCommand == "computeObjectColocalization") {
		for (size_t n = 0; n < 2; n++) {
			poca::core::MyObjectInterface* object = m_object->getObject(n);
			poca::core::BasicComponentInterface* bci = object->getBasicComponent("ObjectLists");
			poca::geometry::ObjectLists* obj = dynamic_cast <poca::geometry::ObjectLists*>(bci);
			if (obj == NULL)
				return;
		}
		if (m_object->getObject(0)->dimension() != m_object->getObject(1)->dimension()) {
			std::cout << "The two datasets have different dimensions, overlaps cannot be computed!" << std::endl;
			return;
		}
		ObjectColocalization* coloc = new ObjectColocalization(m_object->getObject(0), m_object->getObject(1),
			ObjectColocalizationPlugin::m_parameters["ObjectColocalizationPlugin"]["samplingEnabled"].get<bool>(),
			ObjectColocalizationPlugin::m_parameters["ObjectColocalizationPlugin"]["distanceSampling"].get<float>(),
			ObjectColocalizationPlugin::m_parameters["ObjectColocalizationPlugin"]["subdivisionSampling"].get<uint32_t>(),
			ObjectColocalizationPlugin::m_parameters["ObjectColocalizationPlugin"]["minNbPoints"].get<uint32_t>());
		m_object->addBasicComponent(coloc);
		ObjectColocalizationPlugin::m_plugins->addCommands(coloc);
		m_object->notify("LoadObjCharacteristicsAllWidgets");
		m_object->notifyAll("updateDisplay");
	}

}

poca::core::Command* ObjectColocalizationConstructionCommand::copy()
{
	return NULL;
}

poca::core::CommandInfo ObjectColocalizationConstructionCommand::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
{
	if (_nameCommand == "computeObjectColocalization") {
		return poca::core::CommandInfo(false, _nameCommand);
	}
	return poca::core::CommandInfo();
}

nlohmann::json ObjectColocalizationPlugin::m_parameters;
poca::core::PluginList* ObjectColocalizationPlugin::m_plugins = NULL;

void ObjectColocalizationPlugin::addGUI(poca::core::MediatorWObjectFWidgetInterface* _mediator, QTabWidget* _parent)
{
	std::string nameStr = name().toLatin1().data();
	m_parameters[nameStr]["samplingEnabled"] = true;
	m_parameters[nameStr]["distanceSampling"] = 0.5f;
	m_parameters[nameStr]["subdivisionSampling"] = (uint32_t)2;
	m_parameters[nameStr]["minNbPoints"] = (uint32_t)5;
	
	const nlohmann::json& parameters = poca::core::Engine::instance()->getGlobalParameters();
	if (parameters.contains(nameStr)) {
		nlohmann::json param = parameters[nameStr];
		if (param.contains("samplingEnabled"))
			m_parameters[nameStr]["samplingEnabled"] = param["samplingEnabled"].get<bool>();
		if (param.contains("distanceSampling"))
			m_parameters[nameStr]["distanceSampling"] = param["distanceSampling"].get<float>();
		if (param.contains("subdivisionSampling"))
			m_parameters[nameStr]["subdivisionSampling"] = param["subdivisionSampling"].get<uint32_t>();
		if (param.contains("minNbPoints"))
			m_parameters[nameStr]["minNbPoints"] = param["minNbPoints"].get<uint32_t>();
	}

	m_parent = NULL;
	int pos = -1;
	for (int n = 0; n < _parent->count(); n++)
		if (_parent->tabText(n) == "Colocalization")
			pos = n;
	if (pos != -1) {
		m_parent = static_cast <QTabWidget*>(_parent->widget(pos));
	}
	else {
		pos = _parent->addTab(new QTabWidget, QObject::tr("Colocalization"));
		m_parent = static_cast <QTabWidget*>(_parent->widget(pos));
	}

	ObjectColocalizationWidget* w = new ObjectColocalizationWidget(_mediator, m_parent);
	_mediator->addWidget(w);
	int index = m_parent->addTab(w, QObject::tr("Object"));
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
	m_parent->setTabVisible(index, false);
#endif
}

std::vector <std::pair<QAction*, QString>> ObjectColocalizationPlugin::getActions()
{
	if (m_actions.empty()) {
		QPixmap pixmap(objColocIcon);
		QAction* action = new QAction(QIcon(pixmap), tr("&Object colocalization"), this);
		action->setStatusTip(tr("Compute object colocalization"));

		QAction* action2 = new QAction(tr("Parameters"), this);
		action2->setStatusTip(tr("Set parameters"));

		m_actions.push_back(std::make_pair(action, "Toolbar/2Color"));
		m_actions.push_back(std::make_pair(action2, "Plugins/Object colocalization"));
	}

	return m_actions;
}

poca::core::MyObjectInterface* ObjectColocalizationPlugin::actionTriggered(QObject* _sender, poca::core::MyObjectInterface* _obj)
{
	QAction* action = static_cast <QAction*>(_sender);
	std::string nameStr = name().toLatin1().data();
	if (action == m_actions[0].first) {
		/*if (_obj == NULL || _obj->nbColors() < 2) return NULL;
		for (size_t n = 0; n < 2; n++) {
			poca::core::MyObjectInterface* object = _obj->getObject(n);
			poca::core::BasicComponentInterface* bci = object->getBasicComponent("ObjectList");
			poca::geometry::ObjectList* obj = dynamic_cast <poca::geometry::ObjectList*>(bci);
			if (obj == NULL)
				return NULL;
		}
		if (_obj->getObject(0)->dimension() != _obj->getObject(1)->dimension()) {
			std::cout << "The two datasets have different dimensions, overlaps cannot be computed!" << std::endl;
			return NULL;
		}
		ObjectColocalization* coloc = new ObjectColocalization(_obj->getObject(0), _obj->getObject(1), 
			m_parameters[nameStr]["samplingEnabled"].get<bool>(), 
			m_parameters[nameStr]["distanceSampling"].get<float>(), 
			m_parameters[nameStr]["subdivisionSampling"].get<uint32_t>(), 
			m_parameters[nameStr]["minNbPoints"].get<uint32_t>());
		_obj->addBasicComponent(coloc);
		m_plugins->addCommands(coloc);
		_obj->notify("LoadObjCharacteristicsAllWidgets");
		_obj->notifyAll("updateDisplay");*/

		_obj->executeCommand(&poca::core::CommandInfo(true, "computeObjectColocalization"));
	}
	if (action == m_actions[1].first) {
		ObjectColocalizationInfoDialog* dial = new ObjectColocalizationInfoDialog(m_parameters[nameStr]["samplingEnabled"].get<bool>(),
			m_parameters[nameStr]["distanceSampling"].get<float>(),
			m_parameters[nameStr]["subdivisionSampling"].get<uint32_t>(),
			m_parameters[nameStr]["minNbPoints"].get<uint32_t>());
		dial->setModal(true);
		if (dial->exec() == QDialog::Accepted) {
			m_parameters[nameStr]["samplingEnabled"] = dial->isSamplingEnabled();
			m_parameters[nameStr]["distanceSampling"] = dial->getDistance();
			m_parameters[nameStr]["subdivisionSampling"] = dial->getNbSubdivision();
			m_parameters[nameStr]["minNbPoints"] = dial->getMinNbPoints();
		}
		delete dial;
	}
	return NULL;
}

void ObjectColocalizationPlugin::addCommands(poca::core::CommandableObject* _bc)
{
	ObjectColocalization* coloc = dynamic_cast <ObjectColocalization*>(_bc);
	if (coloc) {
		coloc->addCommand(new ObjectColocalizationCommands(coloc));
	}

	poca::core::MyObject* obj = dynamic_cast <poca::core::MyObject*>(_bc);
	if (obj) {
		obj->addCommand(new ObjectColocalizationConstructionCommand(obj));
	}
}

void ObjectColocalizationPlugin::setSingletons(poca::core::Engine* _engine)
{
	poca::core::Engine::instance()->setEngineSingleton(_engine); poca::core::Engine::instance()->setAllSingletons();
}

void ObjectColocalizationPlugin::execute(poca::core::CommandInfo* _com)
{
	if (_com->nameCommand == "saveParameters") {
		if (!_com->hasParameter("file")) return;
		nlohmann::json* json = _com->getParameterPtr<nlohmann::json>("file");

		std::string nameStr = name().toLatin1().data();
		(*json)[nameStr] = m_parameters[nameStr];
	}
}

