/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DBSCANWidget.cpp
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

#include <QtWidgets/QDockWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <iostream>
#include <fstream>

#include <Interfaces/MyObjectInterface.hpp>
#include <Interfaces/CommandableObjectInterface.hpp>
#include <General/BasicComponent.hpp>
#include <Plot/QCPHistogram.hpp>
#include <Plot/Icons.hpp>
#include <General/Palette.hpp>

#include "DBSCANWidget.hpp"
#include "DBSCANCommand.hpp"

DBSCANWidget::DBSCANWidget(poca::core::MediatorWObjectFWidgetInterface* _mediator, QWidget* _parent)
{
	m_parentTab = (QTabWidget*)_parent;
	m_mediator = _mediator;
	m_object = NULL;

	this->setObjectName("DBSCANWidget");
	this->addActionToObserve("LoadObjCharacteristicsAllWidgets");
	this->addActionToObserve("LoadObjCharacteristicsDBSCANWidget");

	m_distanceDBScanLbl = new QLabel("Distance:");
	m_distanceDBScanLbl->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_leditDistanceDBScan = new QLineEdit("50");
	m_leditDistanceDBScan->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	QLabel* minDDBScanLbl = new QLabel("Min # locs:");
	minDDBScanLbl->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_leditMinDDBScan = new QLineEdit("20");
	m_leditMinDDBScan->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_buttonDBScan = new QPushButton("DBScan");
	m_buttonDBScan->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	QObject::connect(m_buttonDBScan, SIGNAL(pressed()), this, SLOT(actionNeeded()));
	m_buttonExportDBSCANRes = new QPushButton("Export results");
	m_buttonExportDBSCANRes->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	QLabel* minNbPtsClusterDBSCAN = new QLabel("Min # locs in cluster:");
	minNbPtsClusterDBSCAN->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_leditMinPtsPerCluster = new QLineEdit("15");
	m_leditMinPtsPerCluster->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);

	int maxSize = 20;
	m_displayButton = new QPushButton();
	m_displayButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_displayButton->setMaximumSize(QSize(maxSize, maxSize));
	m_displayButton->setIcon(QIcon(QPixmap(poca::plot::brushIcon)));
	m_displayButton->setToolTip("Toggle display");
	m_displayButton->setCheckable(true);
	m_displayButton->setChecked(true);
	QObject::connect(m_displayButton, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));

	m_creationFlteredObjectsButton = new QPushButton();
	m_creationFlteredObjectsButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_creationFlteredObjectsButton->setMaximumSize(QSize(maxSize, maxSize));
	m_creationFlteredObjectsButton->setIcon(QIcon(QPixmap(poca::plot::objectIcon)));
	m_creationFlteredObjectsButton->setToolTip("Create objects");
	QObject::connect(m_creationFlteredObjectsButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));

	m_customPlotDBSCAN = new poca::plot::QCPHistogram();
	m_customPlotDBSCAN->setMaximumHeight(150);
	m_customPlotDBSCAN->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	m_customPlotDBSCAN->xAxis->setUpperEnding(QCPLineEnding::esSpikeArrow);
	m_customPlotDBSCAN->yAxis->setUpperEnding(QCPLineEnding::esSpikeArrow);
	m_customPlotDBSCAN->legend->setTextColor(Qt::black);
	QFont fontLegend("Helvetica", 9);
	fontLegend.setBold(true);
	m_customPlotDBSCAN->legend->setFont(fontLegend);
	m_customPlotDBSCAN->legend->setBrush(Qt::NoBrush);
	m_customPlotDBSCAN->legend->setBorderPen(Qt::NoPen);
	QColor background = QWidget::palette().color(QWidget::backgroundRole());
	m_customPlotDBSCAN->setBackground(background);
	m_customPlotDBSCAN->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);
	m_tableObjs = new QTableWidget;
	m_tableObjs->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	QStringList tableHeader;
	tableHeader << "Size" << "# detections" << "Major axis" << "Minor axis";
	m_tableObjs->setColumnCount(tableHeader.size());
	m_tableObjs->setHorizontalHeaderLabels(tableHeader);
	QGridLayout* layoutDBScan = new QGridLayout;
	uint32_t columnCount = 0;
	layoutDBScan->addWidget(m_distanceDBScanLbl, 0, 0, 1, 1);
	layoutDBScan->addWidget(m_leditDistanceDBScan, 0, 1, 1, 1);
	layoutDBScan->addWidget(minDDBScanLbl, 0, 2, 1, 1);
	layoutDBScan->addWidget(m_leditMinDDBScan, 0, 3, 1, 1);
	layoutDBScan->addWidget(m_buttonDBScan, 0, 4, 1, 1);
	layoutDBScan->addWidget(m_displayButton, 1, 0, 1, 1);
	layoutDBScan->addWidget(m_creationFlteredObjectsButton, 1, 1, 1, 1);
	layoutDBScan->addWidget(minNbPtsClusterDBSCAN, 1, 3, 1, 1);
	layoutDBScan->addWidget(m_leditMinPtsPerCluster, 1, 4, 1, 1);

	layoutDBScan->addWidget(m_customPlotDBSCAN, 4, 0, 1, 5);
	layoutDBScan->addWidget(m_tableObjs, 5, 0, 1, 5);

	this->setLayout(layoutDBScan);
	this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

DBSCANWidget::~DBSCANWidget()
{

}

void DBSCANWidget::actionNeeded()
{
	QObject* sender = QObject::sender();
	bool found = false;
	if (sender == m_buttonDBScan) {
		bool ok;
		float radius = m_leditDistanceDBScan->text().toFloat(&ok);
		if (!ok) return;
		uint32_t minNb = m_leditMinDDBScan->text().toUInt(&ok);
		if (!ok) return;
		uint32_t minNbForCluster = m_leditMinPtsPerCluster->text().toUInt(&ok);
		if (!ok) return;
		m_object->executeCommandOnSpecificComponent("DetectionSet", &poca::core::CommandInfo(true, "DBSCAN", "radius", radius, "min", minNb, "minNbForCluster", minNbForCluster));
		m_object->notifyAll("LoadObjCharacteristicsDBSCANWidget");
		m_object->notifyAll("updateDisplay"); 
	}
	else if (sender == m_creationFlteredObjectsButton) {
		m_object->executeCommandOnSpecificComponent("DetectionSet", &poca::core::CommandInfo(true, "createDBSCANObjects",
			"myObject", m_object));
		m_object->notifyAll("updateDisplay");
	}
}

void DBSCANWidget::actionNeeded(bool _val)
{
	QObject* sender = QObject::sender();
	if (sender == m_displayButton) {
		m_object->executeCommandOnSpecificComponent("DetectionSet", &poca::core::CommandInfo(true, "displayDBSCAN", _val));
		m_object->notifyAll("updateDisplay");
		return;
	}
}

void DBSCANWidget::performAction(poca::core::MyObjectInterface* _obj, poca::core::CommandInfo* _ci)
{
}

void DBSCANWidget::update(poca::core::SubjectInterface* _subject, const poca::core::CommandInfo& _aspect)
{
	poca::core::MyObjectInterface* obj = dynamic_cast <poca::core::MyObjectInterface*> (_subject);
	poca::core::MyObjectInterface* objOneColor = obj->currentObject();
	m_object = obj;

	bool visible = (objOneColor != NULL && objOneColor->hasBasicComponent("DetectionSet"));
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
	m_parentTab->setTabVisible(m_parentTab->indexOf(this), visible);
#endif
	if (_aspect == "LoadObjCharacteristicsAllWidgets" || _aspect == "LoadObjCharacteristicsDBSCANWidget") {

		poca::core::BasicComponent* bci = obj->getBasicComponent("DetectionSet");
		if (!bci) return;
		updateDBSCANResults();
	}
}

void DBSCANWidget::executeMacro(poca::core::MyObjectInterface* _wobj, poca::core::CommandInfo* _ci)
{
	this->performAction(_wobj, _ci);
}

void DBSCANWidget::updateDBSCANResults()
{
	poca::core::CommandInfo ci(false, "getDBSCANCommand");
	m_object->executeCommandOnSpecificComponent("DetectionSet", &ci);

	if (ci.json.empty())
		return;

	DBSCANCommand* command = ci.getParameterPtr< DBSCANCommand>("dbscanCommand");
	//Creation of the histogram of the clusters size
	unsigned int nbBins = 50;
	double* bins = new double[nbBins], * ts = new double[nbBins], minH = DBL_MAX, maxH = 0.;
	memset(bins, 0, nbBins * sizeof(double));
	const std::vector <float>& values = command->getSizeClusters();
	const std::vector <float>& majors = command->getMajorAxisClusters();
	const std::vector <float>& minors = command->getMinorAxisClusters();
	const std::vector <uint32_t>& nbLocsClusters = command->getNbLocsClusters();
	unsigned int nbClusters = command->nbClusters();
	for (unsigned int n = 0; n < nbClusters; n++) {
		if (values[n] > maxH) maxH = values[n];
		if (values[n] < minH) minH = values[n];
	}
	double step = (maxH - minH) / (double)nbBins;
	for (unsigned int n = 0; n < nbClusters; n++) {
		unsigned int index = floor((values[n] - minH) / step);
		if (index < nbBins)
			bins[index] = bins[index] + 1.;
	}
	for (unsigned int n = 0; n < nbBins; n++)
		ts[n] = minH + (double)n * step;

	m_customPlotDBSCAN->setInfos("size", command->getHistogram(), poca::core::Palette::getStaticLutPtr("AllGreen"));
	m_customPlotDBSCAN->update();

	m_tableObjs->clear();
	QStringList tableHeader;
	tableHeader << "Size" << "# detections" << "Major axis" << "Minor axis";
	m_tableObjs->setHorizontalHeaderLabels(tableHeader);
	m_tableObjs->setRowCount(nbClusters);
	if (nbClusters > 0) {
		for (int i = 0; i < nbClusters; i++) {
			int y = 0;
			m_tableObjs->setItem(i, y++, new QTableWidgetItem(QString::number(values[i])));
			m_tableObjs->setItem(i, y++, new QTableWidgetItem(QString::number(nbLocsClusters[i])));
			m_tableObjs->setItem(i, y++, new QTableWidgetItem(QString::number(majors[i])));
			m_tableObjs->setItem(i, y++, new QTableWidgetItem(QString::number(minors[i])));
		}
	}
	m_tableObjs->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

	delete[] bins;
	delete[] ts;
}


