/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      KRipleyWidget.cpp
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
#include <QtWidgets/QLabel>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <iostream>
#include <fstream>

#include <Interfaces/MyObjectInterface.hpp>
#include <Interfaces/CommandableObjectInterface.hpp>
#include <General/BasicComponent.hpp>
#include <Plot/qcustomplot.h>
#include <General/CommandableObject.hpp>

#include "KRipleyWidget.hpp"

KRipleyWidget::KRipleyWidget(poca::core::MediatorWObjectFWidgetInterface* _mediator, QWidget* _parent) :m_lsSelected(false)
{
	m_parentTab = (QTabWidget*)_parent;
	m_mediator = _mediator;
	m_object = NULL;

	this->setObjectName("KRipleyWidget");
	this->addActionToObserve("LoadObjCharacteristicsAllWidgets");
	this->addActionToObserve("LoadObjCharacteristicsKRipleyWidget");

	QFont fontLegend("Helvetica", 9);
	fontLegend.setBold(true);
	QColor background = QWidget::palette().color(QWidget::backgroundRole());
	int maxSize = 20;

	this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	m_minKRipleyLbl = new QLabel("Min radius:");
	m_minKRipleyLbl->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	m_minKRipleyEdit = new QLineEdit("10");
	m_minKRipleyEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	m_maxKRipleyLbl = new QLabel("Max radius:");
	m_maxKRipleyLbl->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	m_maxKRipleyEdit = new QLineEdit("200");
	m_maxKRipleyEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	m_stepKRipleyLbl = new QLabel("Step radius:");
	m_stepKRipleyLbl->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	m_stepKRipleyEdit = new QLineEdit("10");
	m_stepKRipleyEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	m_cboxLsDisplayKRipley = new QCheckBox("Display L function");
	m_cboxLsDisplayKRipley->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	m_cboxLsDisplayKRipley->setChecked(false);
	QObject::connect(m_cboxLsDisplayKRipley, SIGNAL(toggled(bool)), this, SLOT(toggleRipleyFunctionDisplay(bool)));
	m_buttonKRipley = new QPushButton("K-Ripley");
	m_buttonKRipley->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	QObject::connect(m_buttonKRipley, SIGNAL(pressed()), this, SLOT(actionNeeded()));
	m_customPlotKRipley = new QCustomPlot();
	m_customPlotKRipley->setMinimumHeight(200);
	m_customPlotKRipley->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	m_customPlotKRipley->xAxis->setUpperEnding(QCPLineEnding::esSpikeArrow);
	m_customPlotKRipley->yAxis->setUpperEnding(QCPLineEnding::esSpikeArrow);
	m_customPlotKRipley->moveLayer(m_customPlotKRipley->layer("grid"), m_customPlotKRipley->layer("main"), QCustomPlot::limAbove);
	m_customPlotKRipley->legend->setTextColor(Qt::black);
	m_customPlotKRipley->legend->setFont(fontLegend);
	m_customPlotKRipley->legend->setBrush(Qt::NoBrush);
	m_customPlotKRipley->legend->setBorderPen(Qt::NoPen);
	m_customPlotKRipley->setBackground(background);
	m_customPlotKRipley->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);
	m_resKRipleyLbl = new QLabel("");
	m_resKRipleyLbl->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_buttonExportKRipleyRes = new QPushButton("Export results");
	m_buttonExportKRipleyRes->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	QObject::connect(m_buttonExportKRipleyRes, SIGNAL(pressed()), this, SLOT(exportKRipleyResults()));
	QWidget* empty = new QWidget;
	empty->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	QGridLayout* layoutKripley = new QGridLayout;
	layoutKripley->addWidget(m_minKRipleyLbl, 0, 0, 1, 1);
	layoutKripley->addWidget(m_minKRipleyEdit, 0, 1, 1, 1);
	layoutKripley->addWidget(m_maxKRipleyLbl, 0, 2, 1, 1);
	layoutKripley->addWidget(m_maxKRipleyEdit, 0, 3, 1, 1);
	layoutKripley->addWidget(m_buttonKRipley, 0, 4, 1, 1);
	layoutKripley->addWidget(m_stepKRipleyLbl, 1, 0, 1, 1);
	layoutKripley->addWidget(m_stepKRipleyEdit, 1, 1, 1, 1);
	layoutKripley->addWidget(m_cboxLsDisplayKRipley, 1, 3, 1, 1);
	layoutKripley->addWidget(m_buttonExportKRipleyRes, 1, 4, 1, 1);
	layoutKripley->addWidget(m_customPlotKRipley, 2, 0, 1, 5);
	layoutKripley->addWidget(m_resKRipleyLbl, 3, 0, 1, 5);
	layoutKripley->addWidget(empty, 4, 0, 1, 5);
	this->setLayout(layoutKripley);
	this->setMinimumHeight(300);
}

KRipleyWidget::~KRipleyWidget()
{

}

void KRipleyWidget::actionNeeded()
{
	poca::core::MyObjectInterface* obj = m_object->currentObject();
	poca::core::BasicComponentInterface* bc = obj->getBasicComponent("DetectionSet");
	if (!bc) return;
	poca::core::CommandableObject* dset = dynamic_cast <poca::core::CommandableObject*>(bc);

	QObject* sender = QObject::sender();
	bool found = false;
	if (sender == m_buttonKRipley) {
		bool ok;
		float minRadius = m_minKRipleyEdit->text().toFloat(&ok);
		if (!ok) return;
		float maxRadius = m_maxKRipleyEdit->text().toFloat(&ok);
		if (!ok) return;
		float step = m_stepKRipleyEdit->text().toFloat(&ok);
		if (!ok) return;
		dset->executeCommand(true, "kripley", "minRadius", minRadius, "maxRadius", maxRadius, "step", step);
		m_object->notifyAll("LoadObjCharacteristicsKRipleyWidget");
	}
}

void KRipleyWidget::performAction(poca::core::MyObjectInterface* _obj, poca::core::CommandInfo* _ci)
{
}

void KRipleyWidget::update(poca::core::SubjectInterface* _subject, const poca::core::CommandInfo& _aspect)
{
	poca::core::MyObjectInterface* obj = dynamic_cast <poca::core::MyObjectInterface*> (_subject);
	poca::core::MyObjectInterface* objOneColor = obj->currentObject();
	m_object = obj;

	bool visible = (objOneColor != NULL && objOneColor->hasBasicComponent("DetectionSet"));
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
	auto index = m_parentTab->currentIndex();
	m_parentTab->setTabVisible(m_parentTab->indexOf(this), visible);
	m_parentTab->setCurrentIndex(index);
#endif
	if (_aspect == "LoadObjCharacteristicsAllWidgets" || _aspect == "LoadObjCharacteristicsKRipleyWidget") {

		poca::core::BasicComponentInterface* bci = obj->getBasicComponent("DetectionSet");
		if (!bci) return;
		setKripleyCurveDisplay();
	}
}

void KRipleyWidget::executeMacro(poca::core::MyObjectInterface* _wobj, poca::core::CommandInfo* _ci)
{
	this->performAction(_wobj, _ci);
}

void KRipleyWidget::toggleRipleyFunctionDisplay(bool _val)
{
	m_lsSelected = _val;
	setKripleyCurveDisplay();
}

void KRipleyWidget::setKripleyCurveDisplay()
{
	poca::core::CommandInfo ci(false, m_lsSelected ? "getKRipleyResultsLs" : "getKRipleyResultsKs");
	m_object->executeCommandOnSpecificComponent("DetectionSet" , &ci);

	if (ci.json.empty())
		return;
	size_t nbBins = ci.getParameter <size_t>("nbSteps");
	float* ts = ci.getParameterPtr <float>("ts");
	float* values = ci.getParameterPtr <float>("values");
	float* lValues = ci.getParameterPtr <float>("ls");

	unsigned int indexMaxY = 0, indexMaxYForL = 0;
	float maxValue = -FLT_MAX, minValue = FLT_MAX;
	QVector<double> x1(nbBins), y1(nbBins);
	for (unsigned int j = 0; j < nbBins; j++) {
		x1[j] = ts[j];
		y1[j] = values[j];
		if (y1[j] > maxValue) maxValue = y1[j];
		if (y1[j] < minValue) minValue = y1[j];
		if (y1[j] > y1[indexMaxY]) indexMaxY = j;
	
		if (lValues[j] > lValues[indexMaxYForL]) indexMaxYForL = j;
	}

	QCustomPlot* customPlot = m_customPlotKRipley;
	customPlot->clearGraphs();
	customPlot->clearItems();

	if (indexMaxY > 0 && indexMaxY < nbBins - 1) {
		QCPItemLine* arrow = new QCPItemLine(customPlot);
		QPen penLines(Qt::black);
		penLines.setWidth(1);
		arrow->setPen(penLines);
		arrow->start->setCoords(ts[indexMaxY], 0);
		arrow->end->setCoords(ts[indexMaxY], maxValue);
		m_resKRipleyLbl->setText(QString("Radius of maximum aggregation: %1").arg(ts[indexMaxY]));
	}
	if (indexMaxYForL > 0 && indexMaxYForL < nbBins - 1)
		m_resKRipleyLbl->setText(QString("Radius of maximum aggregation: %1").arg(ts[indexMaxYForL]));
	else
		m_resKRipleyLbl->setText("No radius of maximum aggregation was found");

	unsigned int currentGraph = 0;
	customPlot->legend->clearItems();
	customPlot->legend->setVisible(true);
	customPlot->legend->setFont(QFont("Helvetica", 9));
	customPlot->addGraph();
	customPlot->graph(currentGraph)->setPen(QPen(Qt::blue));
	customPlot->graph(currentGraph)->setName(m_lsSelected ? "L Ripley" : "K Ripley");
	customPlot->graph(currentGraph)->setData(x1, y1);
	customPlot->yAxis->setRange(minValue, maxValue);
	customPlot->xAxis->setRange(ts[0], ts[nbBins - 1]);
	customPlot->replot();
	customPlot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);

	customPlot->replot();
}

void KRipleyWidget::exportKRipleyResults()
{
	poca::core::CommandInfo ci(false, "getKRipleyResults");
	m_object->executeCommandOnSpecificComponent("DetectionSet", &ci);

	if (ci.json.empty())
		return;
	size_t nbSteps = ci.getParameter <size_t>("nbSteps");
	float* ks = ci.getParameterPtr <float>("ks");
	float* ts = ci.getParameterPtr <float>("ts");
	float* ls = ci.getParameterPtr <float>("ls");
	
	QString nameXls(m_object->getDir().c_str());
	nameXls.append("./KRipley_results.xls");
	nameXls = QFileDialog::getSaveFileName(NULL, QObject::tr("Save stats..."), nameXls, QObject::tr("Stats files (*.xls)"), 0, QFileDialog::DontUseNativeDialog);
	if (nameXls.isEmpty()) return;
	std::ofstream fs(nameXls.toLatin1().data());
	if (!fs) {
		std::cout << "System failed to open " << nameXls.toLatin1().data() << std::endl;
		return;
	}
	else
		std::cout << "Saving stats in file " << nameXls.toLatin1().data() << std::endl;

	fs << "Radius\tK value\tL value" << std::endl;
	for (int i = 0; i < nbSteps; i++)
		fs << ts[i] << "\t" << ks[i] << "\t" << ls[i] << std::endl;
	fs.close();
}

