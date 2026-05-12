/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DelaunayTriangulationWidget.cpp
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

#include <QtWidgets/QDockWidget>
#include <QtWidgets/QVBoxLayout>
#include <iostream>

#include <Interfaces/CommandableObjectInterface.hpp>
#include <General/BasicComponent.hpp>
#include <Factory/ObjectListFactory.hpp>
#include <General/CommandableObject.hpp>
#include <Plot/Icons.hpp>
#include <Plot/Misc.h>
#include <Widgets/ButtonLayer.hpp>
#include <General/Engine.hpp>

#include "DelaunayTriangulationWidget.hpp"

DelaunayTriangulationWidget::DelaunayTriangulationWidget(poca::core::MediatorWObjectFWidgetInterface* _mediator, QWidget* _parent)
{
	m_parentTab = (QTabWidget*)_parent;
	m_mediator = _mediator;
	m_object = NULL;

	this->setObjectName("DelaunayTriangulationWidget");
	this->addActionToObserve("LoadObjCharacteristicsAllWidgets");
	this->addActionToObserve("LoadObjCharacteristicsDelaunayTriangulationWidget");
	this->addActionToObserve("UpdateHistogramDelaunayTriangulationWidget");

	int maxSize = 20;
	nlohmann::json buttonLayerConfig = {
		{ "iconSize", maxSize },
		{ "maxPalettesOnFirstLine", 9 },
		{ "palettes", { { { "name", "HotCold2" } }, { { "name", "InvFire" } }, { { "name", "Fire" } }, { { "name", "Ice" } }, { { "name", "AllRedColorBlind" } }, { { "name", "AllGreenColorBlind" } }, { { "name", "AllBlue" } }, { { "name", "AllWhite" } }, { { "name", "AllBlack" } }, { { "name", "AllRed" } }, { { "name", "AllGreen" } }, { { "name", "AllOrange" } }, { { "name", "AllTomato" } }, { { "name", "AllCyan" } }, { { "name", "Random" } } } },
		{ "actions", {
			{ { "identifier", "transferCells" }, { "icon", "export" }, { "tooltip", "Transfer cells" } },
			{ { "identifier", "invertSelection" }, { "icon", "invert" }, { "tooltip", "Create objects" } },
			{ { "identifier", "bboxSelection" }, { "icon", "bbox" }, { "tooltip", "Toggle bbox selection" }, { "checkable", true }, { "checked", true } },
			{ { "identifier", "createObjects" }, { "icon", "object" }, { "tooltip", "Create objects" } },
			{ { "identifier", "fill" }, { "icon", "fill" }, { "tooltip", "Toggle fill/line" }, { "checkable", true }, { "checked", true } },
			{ { "identifier", "display" }, { "icon", "display" }, { "tooltip", "Toggle display" }, { "checkable", true }, { "checked", true } }
		} }
	};
	auto buttonLayer = poca::qt::generateButtonLayer(buttonLayerConfig, this);
	m_lutsWidget = buttonLayer;
	m_lutsWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	m_lutsWidget2 = new QWidget;
	m_lutsWidget2->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	m_lutsWidget2->hide();
	for (const auto& palette : buttonLayerConfig["palettes"]) {
		std::string name = palette["name"].get<std::string>();
		QPushButton* button = buttonLayer->paletteButton(QString::fromStdString(name));
		if (button != nullptr) {
			m_lutButtons.push_back(std::make_pair(button, name));
			QObject::connect(button, SIGNAL(pressed()), this, SLOT(actionNeeded()));
		}
	}
	m_transferCellsButton = buttonLayer->button("transferCells");
	m_invertSelectionButton = buttonLayer->button("invertSelection");
	m_bboxSelectionButton = buttonLayer->button("bboxSelection");
	m_creationFlteredObjectsButton = buttonLayer->button("createObjects");
	m_fillButton = buttonLayer->button("fill");
	m_displayButton = buttonLayer->button("display");
	QObject::connect(m_transferCellsButton, SIGNAL(released()), this, SLOT(actionNeeded()));
	QObject::connect(m_invertSelectionButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));
	QObject::connect(m_bboxSelectionButton, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));
	QObject::connect(m_creationFlteredObjectsButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));
	QObject::connect(m_fillButton, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));
	QObject::connect(m_displayButton, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));

	m_delaunayTriangulationFilteringWidget = new QWidget;
	m_delaunayTriangulationFilteringWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	m_emptyWidget = new QWidget;
	m_emptyWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

	m_cutDistanceEdit = new QLineEdit("50");
	m_cutDistanceEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_cboxApplyCutDistance = new QCheckBox("Cut distance");
	m_cboxApplyCutDistance->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_cboxApplyCutDistance->setChecked(true);
	m_minLocLEdit = new QLineEdit("10");
	m_minLocLEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_cboxApplyMinLocs = new QCheckBox("Min # locs");
	m_cboxApplyMinLocs->setChecked(true);
	m_cboxApplyMinLocs->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_maxLocLEdit = new QLineEdit("10");
	m_maxLocLEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_cboxApplyMaxLocs = new QCheckBox("Max # locs");
	m_cboxApplyMaxLocs->setChecked(false);
	m_cboxApplyMaxLocs->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_minAreaLEdit = new QLineEdit("10");
	m_minAreaLEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_cboxApplyMinArea = new QCheckBox("Min area");
	m_cboxApplyMinArea->setChecked(false);
	m_cboxApplyMinArea->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_maxAreaLEdit = new QLineEdit("10");
	m_maxAreaLEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_cboxApplyMaxArea = new QCheckBox("Max area");
	m_cboxApplyMaxArea->setChecked(false);
	m_cboxApplyMaxArea->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_applyCutDButton = new QPushButton("Apply distance");
	m_applyCutDButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	QObject::connect(m_applyCutDButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));
	m_cboxInROIs = new QCheckBox("In ROIs");
	m_cboxInROIs->setChecked(false);
	m_cboxInROIs->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);

	int lineCount = 0, columnCount = 0;
	QGridLayout* layoutFilter = new QGridLayout;
	layoutFilter->addWidget(m_cboxApplyCutDistance, lineCount, columnCount++, 1, 1);
	layoutFilter->addWidget(m_cutDistanceEdit, lineCount, columnCount++, 1, 1);
	layoutFilter->addWidget(m_cboxApplyMinLocs, lineCount, columnCount++, 1, 1);
	layoutFilter->addWidget(m_minLocLEdit, lineCount, columnCount++, 1, 1);
	layoutFilter->addWidget(m_cboxApplyMaxLocs, lineCount, columnCount++, 1, 1);
	layoutFilter->addWidget(m_maxLocLEdit, lineCount++, columnCount++, 1, 1);
	columnCount = 2;
	layoutFilter->addWidget(m_applyCutDButton, lineCount, 0, 1, 1);
	layoutFilter->addWidget(m_cboxInROIs, lineCount, 1, 1, 1);
	layoutFilter->addWidget(m_cboxApplyMinArea, lineCount, columnCount++, 1, 1);
	layoutFilter->addWidget(m_minAreaLEdit, lineCount, columnCount++, 1, 1);
	layoutFilter->addWidget(m_cboxApplyMaxArea, lineCount, columnCount++, 1, 1);
	layoutFilter->addWidget(m_maxAreaLEdit, lineCount, columnCount++, 1, 1);
	QWidget* widgetFilter = new QWidget;
	widgetFilter->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	widgetFilter->setLayout(layoutFilter);

	QVBoxLayout* layout = new QVBoxLayout;
	layout->setContentsMargins(1, 1, 1, 1);
	layout->setSpacing(1);
	layout->addWidget(m_lutsWidget);
	layout->addWidget(m_lutsWidget2);
	layout->addWidget(m_delaunayTriangulationFilteringWidget);
	layout->addWidget(widgetFilter);
	layout->addWidget(m_emptyWidget);
	this->setLayout(layout);
}

DelaunayTriangulationWidget::~DelaunayTriangulationWidget()
{

}

void DelaunayTriangulationWidget::actionNeeded()
{
	poca::core::MyObjectInterface* obj = m_object->currentObject();
	poca::core::BasicComponentInterface* bc = obj->getBasicComponent("DelaunayTriangulation");
	if (!bc) return;
	poca::core::CommandableObject* delaunay = dynamic_cast <poca::core::CommandableObject*>(bc);

	QObject* sender = QObject::sender();
	bool found = false;
	for (size_t n = 0; n < m_lutButtons.size() && !found; n++) {
		found = (m_lutButtons[n].first == sender);
		if (found) {
			QString paletteName = m_lutButtons[n].first->property("poca_palette").toString();
			delaunay->executeCommand(true, "changeLUT", "LUT", paletteName.isEmpty() ? m_lutButtons[n].second : paletteName.toStdString());
			for (poca::plot::FilterHistogramWidget* histW : m_histWidgets)
				histW->redraw();
			m_object->notifyAll("updateDisplay");
		}
	}
	if (sender == m_creationFlteredObjectsButton) {
		float cutD = std::numeric_limits < float >::max(), minArea = 0.f, maxArea = std::numeric_limits < float >::max();
		size_t minLocs = 3, maxLocs = std::numeric_limits <size_t>::max();
		bool ok = false;
		float val = 0.f;
		val = this->getCutDistance(&ok);
		if (ok)	cutD = val;
		val = this->getMinNbLocs(&ok);
		if (ok)	minLocs = (size_t)val;
		val = this->getMaxNbLocs(&ok);
		if (ok)	maxLocs = (size_t)val;
		val = this->getMinArea(&ok);
		if (ok)	minArea = val;
		val = this->getMaxArea(&ok);
		if (ok)	maxArea = val;
		bool inROIs = m_cboxInROIs->isChecked();
		delaunay->executeCommand(true, "objectCreationParameters",
				"useDistance", m_cboxApplyCutDistance->isChecked(),
				"useMinLocs", m_cboxApplyMinLocs->isChecked(),
				"useMaxLocs", m_cboxApplyMaxLocs->isChecked(),
				"useMinArea", m_cboxApplyMinArea->isChecked(),
				"useMaxArea", m_cboxApplyMaxArea->isChecked(),
				"minLocs", minLocs, "maxLocs", maxLocs,
				"minArea", minArea, "maxArea", maxArea,
				"cutDistance", cutD, "inROIs", inROIs);
		delaunay->executeCommand(true, "createFilteredObjects",
			"useDistance", m_cboxApplyCutDistance->isChecked(),
			"useMinLocs", m_cboxApplyMinLocs->isChecked(),
			"useMaxLocs", m_cboxApplyMaxLocs->isChecked(),
			"useMinArea", m_cboxApplyMinArea->isChecked(),
			"useMaxArea", m_cboxApplyMaxArea->isChecked(),
			"minLocs", minLocs, "maxLocs", maxLocs,
			"minArea", minArea, "maxArea", maxArea,
			"cutDistance", cutD, "inROIs", inROIs);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_applyCutDButton) {
		float cutD = std::numeric_limits < float >::max();
		bool ok = false;
		float val = 0.f;
		val = this->getCutDistance(&ok);
		if (ok)	cutD = val;
		delaunay->executeCommand(true, "objectCreationParameters", "cutDistance", cutD);
		delaunay->executeCommand(true, "applyCutDistance");
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_invertSelectionButton) {
		delaunay->executeCommand(true, "invertSelection");
		m_object->notifyAll("updateDisplay");
	}
}

void DelaunayTriangulationWidget::actionNeeded(bool _val)
{
	poca::core::MyObjectInterface* obj = m_object->currentObject();
	poca::core::BasicComponentInterface* bc = obj->getBasicComponent("DelaunayTriangulation");
	if (!bc) return;
	poca::core::CommandableObject* delau = dynamic_cast <poca::core::CommandableObject*>(bc);
	QObject* sender = QObject::sender();
	if (sender == m_displayButton) {
		delau->executeCommand(true, "selected", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	if (sender == m_fillButton) {
		delau->executeCommand(true, "fill", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	if (sender == m_bboxSelectionButton) {
		delau->executeCommand(true, "bboxSelection");
		m_object->notifyAll("updateDisplay");
	}
}

void DelaunayTriangulationWidget::performAction(poca::core::MyObjectInterface* _obj, poca::core::CommandInfo* _ci)
{
	if (_obj == NULL) {
		update(NULL, "");
		return;
	}
	poca::core::MyObjectInterface* obj = _obj->currentObject();
	bool actionDone = false;
	if (_ci->nameCommand == "histogram" || _ci->nameCommand == "changeLUT" || _ci->nameCommand == "selected" || _ci->nameCommand == "fill") {
		poca::core::BasicComponentInterface* bc = obj->getBasicComponent("DelaunayTriangulation");
		poca::core::Engine::instance()->executeCommand(bc, _ci);
		actionDone = true;
	}
	if (actionDone) {
		_obj->notifyAll("LoadObjCharacteristicsDelaunayTriangulationWidget");
		_obj->notifyAll("updateDisplay");
	}
}

void DelaunayTriangulationWidget::update(poca::core::SubjectInterface* _subject, const poca::core::CommandInfo& _aspect)
{
	poca::core::MyObjectInterface* obj = dynamic_cast <poca::core::MyObjectInterface*> (_subject);
	poca::core::MyObjectInterface* objOneColor = obj->currentObject();
	m_object = obj;

	bool visible = (objOneColor != NULL && objOneColor->hasBasicComponent("DelaunayTriangulation"));
	//Delaunay is always created, but may not have commands attached to it, if no commands attached, don't show the widget
	if (visible) {
		poca::core::BasicComponentInterface* bci = objOneColor->getBasicComponent("DelaunayTriangulation");
		visible = bci->nbCommands() != 0;
	}
	
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
	auto index = m_parentTab->currentIndex();
	m_parentTab->setTabVisible(m_parentTab->indexOf(this), visible);
	m_parentTab->setCurrentIndex(index);
#endif

	if (_aspect == "LoadObjCharacteristicsAllWidgets" || _aspect == "LoadObjCharacteristicsDelaunayTriangulationWidget") {

		poca::core::BasicComponentInterface* bci = objOneColor->getBasicComponent("DelaunayTriangulation");
		if (!bci) return;
		poca::core::stringList nameData = bci->getNameData();

		QVBoxLayout* layout = NULL;
		//First time we load data -> no hist widget was created before
		if (m_histWidgets.empty()) {
			layout = new QVBoxLayout;
			layout->setContentsMargins(1, 1, 1, 1);
			layout->setSpacing(1);
			for (size_t n = 0; n < nameData.size(); n++) {
				m_histWidgets.push_back(new poca::plot::FilterHistogramWidget(m_mediator, "DelaunayTriangulation", this));
				layout->addWidget(m_histWidgets[n]);
			}
			m_delaunayTriangulationFilteringWidget->setLayout(layout);
		}
		else if (nameData.size() > m_histWidgets.size()) {
			layout = dynamic_cast <QVBoxLayout*>(m_delaunayTriangulationFilteringWidget->layout());
			//Here, we need to add some hist widgets because this loc data has more features than the one loaded before
			for (size_t n = m_histWidgets.size(); n < nameData.size(); n++) {
				m_histWidgets.push_back(new poca::plot::FilterHistogramWidget(m_mediator, "DelaunayTriangulation", this));
				layout->addWidget(m_histWidgets[n]);
			}
		}
		else if (nameData.size() < m_histWidgets.size()) {
			//Here, wee have less feature to display than hist widgets available, we hide the ones that are unecessary
			for (size_t n = 0; n < m_histWidgets.size(); n++)
				m_histWidgets[n]->setVisible(n < nameData.size());
		}

		int cpt = 0;

		std::vector <float> ts, bins;
		for (std::string type : nameData) {
			poca::core::HistogramInterface* hist = bci->getHistogram(type);
			if (hist != NULL)
				m_histWidgets[cpt++]->setInfos(type.c_str(), hist, bci->isLogHistogram(type), bci->isCurrentHistogram(type) ? bci->getPalette() : NULL);
		}

		m_delaunayTriangulationFilteringWidget->updateGeometry();

		this->blockSignals(true);
		if (bci->hasParameter("fill")) {
			bool val = bci->getParameter<bool>("fill");
			m_fillButton->blockSignals(true);
			m_fillButton->setChecked(val);
			m_fillButton->blockSignals(false);
		}
		if (bci->hasParameter("bboxSelection")) {
			bool val = bci->getParameter<bool>("bboxSelection");
			m_bboxSelectionButton->blockSignals(true);
			m_bboxSelectionButton->setChecked(val);
			m_bboxSelectionButton->blockSignals(false);
		}
		if (bci->hasParameter("objectCreationParameters", "useDistance"))
			m_cboxApplyCutDistance->setChecked(bci->getParameter<bool>("objectCreationParameters", "useDistance"));
		if (bci->hasParameter("objectCreationParameters", "useMinLocs"))
			m_cboxApplyMinLocs->setChecked(bci->getParameter<bool>("objectCreationParameters", "useMinLocs"));
		if (bci->hasParameter("objectCreationParameters", "useMaxLocs"))
			m_cboxApplyMaxLocs->setChecked(bci->getParameter<bool>("objectCreationParameters", "useMaxLocs"));
		if (bci->hasParameter("objectCreationParameters", "useMinArea"))
			m_cboxApplyMinArea->setChecked(bci->getParameter<bool>("objectCreationParameters", "useMinArea"));
		if (bci->hasParameter("objectCreationParameters", "useMaxArea"))
			m_cboxApplyMaxArea->setChecked(bci->getParameter<bool>("objectCreationParameters", "useMaxArea"));
		if (bci->hasParameter("objectCreationParameters", "minLocs"))
			m_minLocLEdit->setText(QString::number(bci->getParameter<size_t>("objectCreationParameters", "minLocs")));
		if (bci->hasParameter("objectCreationParameters", "maxLocs"))
			m_maxLocLEdit->setText(QString::number(bci->getParameter<size_t>("objectCreationParameters", "maxLocs")));
		if (bci->hasParameter("objectCreationParameters", "minArea"))
			m_minAreaLEdit->setText(QString::number(bci->getParameter<float>("objectCreationParameters", "minArea")));
		if (bci->hasParameter("objectCreationParameters", "maxArea"))
			m_maxAreaLEdit->setText(QString::number(bci->getParameter<float>("objectCreationParameters", "maxArea")));
		if (bci->hasParameter("objectCreationParameters", "cutDistance"))
			m_cutDistanceEdit->setText(QString::number(bci->getParameter<float>("objectCreationParameters", "cutDistance")));
		if (bci->hasParameter("objectCreationParameters", "inROIs"))
			m_cboxInROIs->setChecked(bci->getParameter<bool>("objectCreationParameters", "inROIs"));

		bool selected = bci->isSelected();
		m_displayButton->blockSignals(true);
		m_displayButton->setChecked(selected);
		m_displayButton->blockSignals(false);

		this->blockSignals(false);
	}
}

void DelaunayTriangulationWidget::executeMacro(poca::core::MyObjectInterface* _wobj, poca::core::CommandInfo* _ci)
{
	this->performAction(_wobj, _ci);
}

const float DelaunayTriangulationWidget::getCutDistance(bool* _ok) const
{
	return m_cutDistanceEdit->text().toFloat(_ok);
}

const float DelaunayTriangulationWidget::getMinNbLocs(bool* _ok) const
{
	return m_minLocLEdit->text().toFloat(_ok);
}

const float DelaunayTriangulationWidget::getMaxNbLocs(bool* _ok) const
{
	return m_maxLocLEdit->text().toFloat(_ok);
}

const float DelaunayTriangulationWidget::getMinArea(bool* _ok) const
{
	return m_minAreaLEdit->text().toFloat(_ok);
}

const float DelaunayTriangulationWidget::getMaxArea(bool* _ok) const
{
	return m_maxAreaLEdit->text().toFloat(_ok);
}

