/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      NearestLocsMultiColorWidget.cpp
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

#include <QtWidgets/QDockWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QMessageBox>
#include <iostream>
#include <fstream>

#include <Interfaces/MyObjectInterface.hpp>
#include <Interfaces/CommandableObjectInterface.hpp>
#include <General/BasicComponent.hpp>
#include <Factory/ObjectListFactory.hpp>
#include <General/CommandableObject.hpp>
#include <Geometry/ObjectLists.hpp>
#include <Plot/Icons.hpp>
#include <Plot/Misc.h>

#include "NearestLocsMultiColorWidget.hpp"
#include "NearestLocsMultiColorCommands.hpp"

NearestLocsMultiColorWidget::NearestLocsMultiColorWidget(poca::core::MediatorWObjectFWidgetInterface* _mediator, QWidget* _parent) :QTabWidget(_parent)
{
	m_parentTab = (QTabWidget*)_parent;
	m_mediator = _mediator;

	this->setObjectName("NearestLocsMultiColorWidget");
	this->addActionToObserve("LoadObjCharacteristicsAllWidgets");
	this->addActionToObserve("LoadObjCharacteristicsNearestLocsMultiColorWidget");

	QFont fontLegend("Helvetica", 9);
	fontLegend.setBold(true);
	QColor background = QWidget::palette().color(QWidget::backgroundRole());
	
	QHBoxLayout* layoutActions = new QHBoxLayout;
	int maxSize = 20;
	m_displayCentroidsButton = new QPushButton();
	m_displayCentroidsButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_displayCentroidsButton->setMaximumSize(QSize(maxSize, maxSize));
	m_displayCentroidsButton->setIcon(QIcon(QPixmap(poca::plot::brushIcon)));
	m_displayCentroidsButton->setToolTip("Toggle display to centroids");
	m_displayCentroidsButton->setCheckable(true);
	m_displayCentroidsButton->setChecked(true);
	layoutActions->addWidget(m_displayCentroidsButton, 0, Qt::AlignRight);
	QObject::connect(m_displayCentroidsButton, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));
	m_displayOutlinesButton = new QPushButton();
	m_displayOutlinesButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_displayOutlinesButton->setMaximumSize(QSize(maxSize, maxSize));
	m_displayOutlinesButton->setIcon(QIcon(QPixmap(poca::plot::brushIcon)));
	m_displayOutlinesButton->setToolTip("Toggle display to outlines");
	m_displayOutlinesButton->setCheckable(true);
	m_displayOutlinesButton->setChecked(true);
	layoutActions->addWidget(m_displayOutlinesButton, 0, Qt::AlignRight);
	QObject::connect(m_displayOutlinesButton, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));
	m_transferSelectedObjectsButton = new QPushButton();
	m_transferSelectedObjectsButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_transferSelectedObjectsButton->setMaximumSize(QSize(maxSize, maxSize));
	m_transferSelectedObjectsButton->setIcon(QIcon(QPixmap("./images/duplicate.png")));
	m_transferSelectedObjectsButton->setToolTip("Transfer selected objects");
	layoutActions->addWidget(m_transferSelectedObjectsButton, 0, Qt::AlignRight);
	QObject::connect(m_transferSelectedObjectsButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));
	m_saveDistancesButton = new QPushButton();
	m_saveDistancesButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_saveDistancesButton->setMaximumSize(QSize(maxSize, maxSize));
	m_saveDistancesButton->setIcon(QIcon(QPixmap(poca::plot::saveIcon)));
	m_saveDistancesButton->setToolTip("Save distances");
	layoutActions->addWidget(m_saveDistancesButton, 0, Qt::AlignRight);
	QObject::connect(m_saveDistancesButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));
	QWidget* emptyW = new QWidget;
	emptyW->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	layoutActions->addWidget(emptyW);

	//QGroupBox* m_FitEllipsoidGBox = new QGroupBox(tr("K-Ripley"));
	this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	m_inROIsCB = new QCheckBox("In ROIs");
	m_inROIsCB->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_inROIsCB->setChecked(true);
	m_computeButton = new QPushButton("Compute");
	m_computeButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	QObject::connect(m_computeButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));
	QHBoxLayout* layout = new QHBoxLayout;
	layout->addWidget(m_inROIsCB);
	layout->addWidget(m_computeButton);

	QLabel* labl = new QLabel("Reference");
	labl->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_bgroup = new QButtonGroup;
	QHBoxLayout* layout3 = new QHBoxLayout;
	layout3->addWidget(labl);
	for (auto n = 0; n < 2; n++) {
		m_referencesCB[n] = new QCheckBox(QString("color ").append(QString::number(n + 1)));
		m_referencesCB[n]->setChecked(n == 0 ? false : true);
		m_referencesCB[n]->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
		m_bgroup->addButton(m_referencesCB[n], n);
		layout3->addWidget(m_referencesCB[n]);
	}

	m_maxDistanceCB = new QCheckBox("Max distance:");
	m_maxDistanceCB->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_maxDistanceCB->setChecked(false);
	m_maxDistanceEdit = new QLineEdit;
	m_maxDistanceEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	QHBoxLayout* layout4 = new QHBoxLayout;
	layout4->addWidget(m_maxDistanceCB);
	layout4->addWidget(m_maxDistanceEdit);

	m_histoToCentroids = new poca::plot::QCPHistogram();
	m_histoToCentroids->setMaximumHeight(200);
	m_histoToCentroids->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	m_histoToCentroids->xAxis->setUpperEnding(QCPLineEnding::esSpikeArrow);
	m_histoToCentroids->yAxis->setUpperEnding(QCPLineEnding::esSpikeArrow);
	m_histoToCentroids->legend->setTextColor(Qt::black);
	m_histoToCentroids->legend->setFont(fontLegend);
	m_histoToCentroids->legend->setBrush(Qt::NoBrush);
	m_histoToCentroids->legend->setBorderPen(Qt::NoPen);
	m_histoToCentroids->setBackground(background);
	m_histoToCentroids->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);

	m_histoToOutlines = new poca::plot::QCPHistogram();
	m_histoToOutlines->setMaximumHeight(200);
	m_histoToOutlines->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	m_histoToOutlines->xAxis->setUpperEnding(QCPLineEnding::esSpikeArrow);
	m_histoToOutlines->yAxis->setUpperEnding(QCPLineEnding::esSpikeArrow);
	m_histoToOutlines->legend->setTextColor(Qt::black);
	m_histoToOutlines->legend->setFont(fontLegend);
	m_histoToOutlines->legend->setBrush(Qt::NoBrush);
	m_histoToOutlines->legend->setBorderPen(Qt::NoPen);
	m_histoToOutlines->setBackground(background);
	m_histoToOutlines->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);

	m_tableInfos = new QTableWidget;
	m_tableInfos->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

	QVBoxLayout* layout2 = new QVBoxLayout;
	layout2->addLayout(layoutActions);
	layout2->addLayout(layout3);
	layout2->addLayout(layout4);
	layout2->addLayout(layout);
	layout2->addWidget(m_histoToCentroids);
	layout2->addWidget(m_histoToOutlines);
	layout2->addWidget(m_tableInfos);

	this->setLayout(layout2);
}

NearestLocsMultiColorWidget::~NearestLocsMultiColorWidget()
{

}

void NearestLocsMultiColorWidget::actionNeeded()
{
	if (m_object && m_object->nbColors() == 1)
		return;
	poca::geometry::ObjectLists* objs[2] = { NULL, NULL };
	for (auto n = 0; n < 2; n++) {
		poca::core::MyObjectInterface* obj = m_object->getObject(n);
		poca::core::BasicComponentInterface* bc = obj->getBasicComponent("ObjectLists");
		objs[n] = dynamic_cast <poca::geometry::ObjectLists*>(bc);
	}
	if (objs[0] == NULL || objs[1] == NULL) {
		QMessageBox msgBox;
		msgBox.setText("Objects in both color are required.");
		msgBox.exec();
		return;
	}
	QObject* sender = QObject::sender();
	bool found = false;
	
	if (sender == m_computeButton) {
		bool inROIs = m_inROIsCB->isChecked();
		auto idx = m_bgroup->checkedId();
		float maxD = DBL_MAX;
		if (m_maxDistanceCB->isChecked()) {
			QString text = m_maxDistanceEdit->text();
			if (!text.isEmpty()) {
				bool ok;
				float tmp = text.toDouble(&ok);
				if (ok)
					maxD = tmp;
			}
		}
		poca::core::CommandInfo com(true, "computeNearestNeighMulticolor", "inROIs", inROIs, "reference", (uint32_t)idx, "maxDistance", maxD);
		m_object->executeCommand(&com);
		return;
	}
	else if (sender == m_transferSelectedObjectsButton) {
		poca::core::CommandInfo com(true, "transferSelectedObjectsNearestNeighMulticolor");
		m_object->executeCommand(&com);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_saveDistancesButton) {
		QString name = m_object->getName().c_str(), filename(m_object->getDir().c_str());
		if (!filename.endsWith("/"))
			filename.append("/");
		filename.append(name).append("_distances.txt");
		filename = QFileDialog::getSaveFileName(NULL, QObject::tr("Save distances..."), filename, QString("Text files (*.txt)"), 0, QFileDialog::DontUseNativeDialog);
		if (filename.isEmpty()) return;

		poca::core::CommandInfo com(true, "saveDistancesNearestNeighMulticolor", "path", filename.toStdString());
		m_object->executeCommand(&com);
	}
}

void NearestLocsMultiColorWidget::actionNeeded(bool _val)
{
	QObject* sender = QObject::sender();
	if (sender == m_displayCentroidsButton) {
		poca::core::CommandInfo com(true, "displayCentroidsNearestNeighMulticolor", _val);
		m_object->executeCommand(&com);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_displayOutlinesButton) {
		poca::core::CommandInfo com(true, "displayOutlinesNearestNeighMulticolor", _val);
		m_object->executeCommand(&com);
		m_object->notifyAll("updateDisplay");
		return;
	}
}

void NearestLocsMultiColorWidget::performAction(poca::core::MyObjectInterface* _obj, poca::core::CommandInfo* _ci)
{
}

void NearestLocsMultiColorWidget::update(poca::core::SubjectInterface* _subject, const poca::core::CommandInfo& _aspect)
{
	poca::core::MyObjectInterface* obj = dynamic_cast <poca::core::MyObjectInterface*> (_subject);
	m_object = obj;

	bool visible = (obj != NULL && obj->nbColors() > 1);
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
	auto index = m_parentTab->currentIndex();
	m_parentTab->setTabVisible(m_parentTab->indexOf(this), visible);
	m_parentTab->setCurrentIndex(index);
#endif

	if (_aspect == "LoadObjCharacteristicsAllWidgets" || _aspect == "LoadObjCharacteristicsNearestLocsMultiColorWidget") {
		poca::core::CommandableObject* comObject = dynamic_cast <poca::core::CommandableObject*>(m_object);
		NearestLocsMultiColorCommands* nlmcom = comObject->getCommand<NearestLocsMultiColorCommands>();
		if (nlmcom == NULL) return;

		const std::vector <uint32_t>& idx = nlmcom->getIndexObjects();
		const std::vector <float>& dtc = nlmcom->getDistanceToCentroids();
		const std::vector <float>& dto = nlmcom->getDistanceToOutlines();

		if (idx.empty()) return;

		if (nlmcom->hasParameter("inROIs")) {
			bool val = nlmcom->getParameter<bool>("inROIs");
			m_inROIsCB->setChecked(val);
		}
		if (nlmcom->hasParameter("reference")) {
			uint32_t val = nlmcom->getParameter<uint32_t>("reference");
			m_referencesCB[val]->setChecked(true);
		}

		if (nlmcom->hasParameter("displayCentroidsNearestNeighMulticolor")) {
			bool val = nlmcom->getParameter<bool>("displayCentroidsNearestNeighMulticolor");
			m_displayCentroidsButton->setChecked(true);
		}

		if (nlmcom->hasParameter("displayOutlinesNearestNeighMulticolor")) {
			bool val = nlmcom->getParameter<bool>("displayOutlinesNearestNeighMulticolor");
			m_displayOutlinesButton->setChecked(true);
		}

		QStringList headerInfos;
		headerInfos << "Index object" << "distanceToCentoids" << "distanceToOutlines";
		m_tableInfos->setColumnCount(headerInfos.size());
		m_tableInfos->setHorizontalHeaderLabels(headerInfos);
		m_tableInfos->setRowCount(dtc.size());
		for (auto n = 0; n < dtc.size(); n++) {
			m_tableInfos->setItem(n, 0, new QTableWidgetItem(QString::number(idx[n])));
			m_tableInfos->setItem(n, 1, new QTableWidgetItem(QString::number(dtc[n])));
			m_tableInfos->setItem(n, 2, new QTableWidgetItem(QString::number(dto[n])));
		}

		m_histoToCentroids->setInfos(QString("centroids"), nlmcom->getHistogramCentroids(), nlmcom->getPalette());
		m_histoToCentroids->update();

		m_histoToOutlines->setInfos(QString("outlines"), nlmcom->getHistogramOutlines(), nlmcom->getPalette());
		m_histoToOutlines->update();
	}
}

void NearestLocsMultiColorWidget::executeMacro(poca::core::MyObjectInterface* _wobj, poca::core::CommandInfo* _ci)
{
	this->performAction(_wobj, _ci);
}