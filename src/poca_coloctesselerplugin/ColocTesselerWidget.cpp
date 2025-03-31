/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ColocTesselerWidget.cpp
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
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QSlider>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QHeaderView>


#include <iostream>

#include <Interfaces/MyObjectInterface.hpp>
#include <Interfaces/CommandableObjectInterface.hpp>
#include <General/BasicComponent.hpp>
#include <Factory/ObjectListFactory.hpp>
#include <Geometry/VoronoiDiagram.hpp>
#include <Plot/Icons.hpp>
#include <Plot/Misc.h>

#include "ColocTesselerWidget.hpp"
#include "ColocTesseler.hpp"

ColocTesselerWidget::ColocTesselerWidget(poca::core::MediatorWObjectFWidgetInterface* _mediator, QWidget* _parent) :QTabWidget(_parent)
{
	m_parentTab = (QTabWidget*)_parent;
	m_mediator = _mediator;
	m_object = NULL;

	this->setObjectName("ColocTesselerWidget");
	this->addActionToObserve("LoadObjCharacteristicsAllWidgets");
	this->addActionToObserve("LoadObjCharacteristicsColocTesselerWidget");

	m_actionsGBox = new QGroupBox(tr("Parameters"));
	m_actionsGBox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	QHBoxLayout* layoutActions = new QHBoxLayout;
	m_correctionCbox = new QCheckBox("Correction");
	m_correctionCbox->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_correctionCbox->setChecked(true);
	layoutActions->addWidget(m_correctionCbox);
	QObject::connect(m_correctionCbox, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));

	QLabel* factorDensityLbl[2];
	for (size_t n = 0; n < 2; n++) {
		factorDensityLbl[n] = new QLabel(QString("Color %1 factor").arg(n + 1));
		m_factorDensityEdit[n] = new QLineEdit("1");
		layoutActions->addWidget(factorDensityLbl[n]);
		layoutActions->addWidget(m_factorDensityEdit[n]);
	}
	m_applyFactorButton = new QPushButton("Apply");
	m_applyFactorButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	layoutActions->addWidget(m_applyFactorButton);
	QObject::connect(m_applyFactorButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));

	QWidget* emptyActions = new QWidget;
	emptyActions->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	layoutActions->addWidget(emptyActions);

	int maxSize = 20;
	m_displayButton = new QPushButton();
	m_displayButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_displayButton->setMaximumSize(QSize(maxSize, maxSize));
	m_displayButton->setIcon(QIcon(QPixmap(poca::plot::brushIcon)));
	m_displayButton->setToolTip("Toggle display");
	m_displayButton->setCheckable(true);
	m_displayButton->setChecked(true);
	layoutActions->addWidget(m_displayButton, 0, Qt::AlignRight);
	QObject::connect(m_displayButton, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));

	m_saveButton = new QPushButton();
	m_saveButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_saveButton->setMaximumSize(QSize(maxSize, maxSize));
	m_saveButton->setIcon(QIcon(QPixmap(poca::plot::saveIcon)));
	m_saveButton->setToolTip("Save pair densities");
	layoutActions->addWidget(m_saveButton, 0, Qt::AlignRight);
	QObject::connect(m_saveButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));

	m_inROIsCbox = new QCheckBox("In ROIs");
	m_inROIsCbox->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_inROIsCbox->setChecked(false);
	QObject::connect(m_inROIsCbox, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));

	QVBoxLayout* layoutactWithROI = new QVBoxLayout;
	layoutactWithROI->addLayout(layoutActions);
	layoutactWithROI->addWidget(m_inROIsCbox);

	m_actionsGBox->setLayout(layoutactWithROI);

	QGroupBox* heatmapGBox = new QGroupBox("Scatter plot");
	m_minRadiusEdit = new QLineEdit("1");
	m_minRadiusEdit->setEnabled(false);
	m_minRadiusEdit->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_minRadiusEdit->setMaximumSize(QSize(maxSize * 2, maxSize));
	m_currentRadiusEdit = new QLineEdit("5");
	m_currentRadiusEdit->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_currentRadiusEdit->setMaximumSize(QSize(maxSize * 2, maxSize));
	QObject::connect(m_currentRadiusEdit, SIGNAL(returnPressed()), SLOT(actionNeeded()));
	m_maxRadiusEdit = new QLineEdit("100");
	m_maxRadiusEdit->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_maxRadiusEdit->setMaximumSize(QSize(maxSize * 2, maxSize));
	QObject::connect(m_maxRadiusEdit, SIGNAL(returnPressed()), SLOT(actionNeeded()));
	m_radiusSlider = new QSlider(Qt::Horizontal);
	m_radiusSlider->setMinimum(1);
	m_radiusSlider->setMaximum(100);
	m_radiusSlider->setSliderPosition(5);
	m_radiusSlider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	QObject::connect(m_radiusSlider, SIGNAL(valueChanged(int)), SLOT(actionNeeded(int)));
	QLabel* curIntensityLbl = new QLabel("Intensity:");
	curIntensityLbl->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_intensityEdit = new QLineEdit("1");
	m_intensityEdit->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_intensityEdit->setMaximumSize(QSize(maxSize * 2, maxSize));
	QObject::connect(m_intensityEdit, SIGNAL(returnPressed()), SLOT(actionNeeded()));
	
	QLineEdit* intensityMinEdit= new QLineEdit("0.001");
	intensityMinEdit->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	intensityMinEdit->setEnabled(false);
	QLineEdit* intensityMaxEdit = new QLineEdit("1");
	intensityMaxEdit->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	intensityMaxEdit->setEnabled(false);
	m_intensitySlider = new QSlider(Qt::Horizontal);
	m_intensitySlider->setMinimum(1);
	m_intensitySlider->setMaximum(1000);
	m_intensitySlider->setSliderPosition(1000);
	m_intensitySlider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	QObject::connect(m_intensitySlider, SIGNAL(valueChanged(int)), SLOT(actionNeeded(int)));

	QLabel* curRadiusLbl = new QLabel("Current radius (pix):");
	curRadiusLbl->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	QWidget* emptyCurRadiusW = new QWidget;
	emptyCurRadiusW->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	m_logHeatMapCBox = new QCheckBox("log");
	m_logHeatMapCBox->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_logHeatMapCBox->setChecked(true);
	QObject::connect(m_logHeatMapCBox, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));
	m_scatterGL = new poca::plot::ScatterplotGL(this);
	m_scatterGL->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	m_scatterGL->setMinimumHeight(400);
	QHBoxLayout* layoutH1 = new QHBoxLayout;
	layoutH1->addWidget(m_minRadiusEdit);
	layoutH1->addWidget(m_radiusSlider);
	layoutH1->addWidget(m_maxRadiusEdit);

	QHBoxLayout* layoutH3 = new QHBoxLayout;
	layoutH3->addWidget(intensityMinEdit);
	layoutH3->addWidget(m_intensitySlider);
	layoutH3->addWidget(intensityMaxEdit);

	QHBoxLayout* layoutH2 = new QHBoxLayout;
	m_lutHeatmapButtons.push_back(std::make_pair(new QPushButton(), std::string("HotCold2")));
	m_lutHeatmapButtons.push_back(std::make_pair(new QPushButton(), std::string("InvFire")));
	m_lutHeatmapButtons.push_back(std::make_pair(new QPushButton(), std::string("Fire")));
	m_lutHeatmapButtons.push_back(std::make_pair(new QPushButton(), std::string("Ice")));
	m_lutHeatmapButtons.push_back(std::make_pair(new QPushButton(), std::string("Heatmap")));
	for (size_t n = 0; n < m_lutHeatmapButtons.size(); n++) {
		m_lutHeatmapButtons[n].first->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
		m_lutHeatmapButtons[n].first->setMaximumSize(QSize(maxSize, maxSize));
		QImage im = poca::core::generateImage(maxSize, maxSize, &poca::core::Palette::getStaticLut(m_lutHeatmapButtons[n].second));
		QPixmap pix = QPixmap::fromImage(im);
		QIcon icon(pix);
		m_lutHeatmapButtons[n].first->setIcon(icon);
		layoutH2->addWidget(m_lutHeatmapButtons[n].first, 0, Qt::AlignLeft);

		QObject::connect(m_lutHeatmapButtons[n].first, SIGNAL(pressed()), this, SLOT(actionNeeded()));
	}
	layoutH2->addWidget(emptyCurRadiusW);
	layoutH2->addWidget(curIntensityLbl, Qt::AlignRight);
	layoutH2->addWidget(m_intensityEdit, Qt::AlignRight);
	layoutH2->addWidget(curRadiusLbl, Qt::AlignRight);
	layoutH2->addWidget(m_currentRadiusEdit, Qt::AlignRight);
	layoutH2->addWidget(m_logHeatMapCBox, Qt::AlignRight);
	QVBoxLayout* layoutHeatmap = new QVBoxLayout;
	layoutHeatmap->addLayout(layoutH1);
	layoutHeatmap->addLayout(layoutH3);
	layoutHeatmap->addLayout(layoutH2);
	layoutHeatmap->addWidget(m_scatterGL);
	heatmapGBox->setLayout(layoutHeatmap);

	m_tableColoc = new MyTableWidget;
	m_tableColoc->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

	QVBoxLayout* layout = new QVBoxLayout;
	layout->setContentsMargins(1, 1, 1, 1);
	layout->setSpacing(1);
	layout->addWidget(m_actionsGBox);
	layout->addWidget(heatmapGBox);
	layout->addWidget(m_tableColoc);
	
	this->setLayout(layout);
}

ColocTesselerWidget::~ColocTesselerWidget()
{

}

void ColocTesselerWidget::actionNeeded()
{
	QObject* sender = QObject::sender();
	bool found = false;
	for (size_t n = 0; n < m_lutHeatmapButtons.size() && !found; n++) {
		found = (m_lutHeatmapButtons[n].first == sender);
		if (found) {
			m_scatterGL->setPalette(&poca::core::Palette::getStaticLut(m_lutHeatmapButtons[n].second));
			m_scatterGL->update();
			return;
		}
	}
	if (sender == m_applyFactorButton) {
		bool ok;
		float factors[2];
		for (size_t n = 0; n < 2; n++) {
			factors[n] = getDensityFactor(n, &ok);
			if (!ok) return;
		}
		m_object->executeCommandOnSpecificComponent("ColocTesseler", &poca::core::CommandInfo(true, "threshold", "color1", factors[0], "color2", factors[1]));
		m_object->executeCommandOnSpecificComponent("ColocTesseler", &poca::core::CommandInfo(true, "computeCoefficients"));
		m_object->executeCommandOnSpecificComponent("ColocTesseler", &poca::core::CommandInfo(true, "updateFeature"));
		m_object->notifyAll("updateDisplay");
		m_object->notifyAll("LoadObjCharacteristicsColocTesselerWidget");
		return;
	}
	else if (sender == m_maxRadiusEdit) {
		bool ok;
		int val = m_maxRadiusEdit->text().toInt(&ok);
		if (!ok) return;
		m_radiusSlider->blockSignals(true);
		m_radiusSlider->setMaximum(val);
		m_radiusSlider->blockSignals(false);
	}
	else if (sender == m_currentRadiusEdit) {
		bool ok;
		int val = m_currentRadiusEdit->text().toInt(&ok);
		if (!ok) return;
		m_radiusSlider->blockSignals(true);
		m_radiusSlider->setSliderPosition(val);
		m_radiusSlider->blockSignals(false);
		m_scatterGL->setRadiusHeatmap(val);
		m_scatterGL->update();
	}
	else if (sender == m_intensityEdit) {
		bool ok;
		float val = m_intensityEdit->text().toFloat(&ok);
		if (!ok) return;
		m_scatterGL->setIntensityHeatmap(val);
		m_scatterGL->update();
	}
	else if (sender == m_saveButton) {
		QString name = m_object->getName().c_str(), filename(m_object->getDir().c_str());
		if (!filename.endsWith("/"))
			filename.append("/");
		filename.append(name);
		int index = filename.lastIndexOf(".");
		if (index != -1)
			filename = filename.left(index - 1);
		filename.append("_pairDensities.txt");

		filename = QFileDialog::getSaveFileName(NULL, QObject::tr("Save pair-densities..."), filename, QString("Pair-densities files (*.txt)"), 0, QFileDialog::DontUseNativeDialog);
		if (filename.isEmpty()) return;

		m_object->executeCommandOnSpecificComponent("ColocTesseler", &poca::core::CommandInfo(true, "savePairDensities", "filename", std::string(filename.toLatin1().data())));
	}
}

void ColocTesselerWidget::actionNeeded(bool _val)
{
	QObject* sender = QObject::sender();
	if (sender == m_displayButton) {
		m_object->executeCommandOnSpecificComponent("ColocTesseler", &poca::core::CommandInfo(true, "selected", _val));
		m_object->notifyAll("updateDisplay");
		return;
	}
	if (sender == m_correctionCbox) {
		m_object->executeCommandOnSpecificComponent("ColocTesseler", &poca::core::CommandInfo(true, "correction", _val));
		m_object->notifyAll("updateDisplay");
		return;
	}
	if (sender == m_inROIsCbox) {
		m_object->executeCommandOnSpecificComponent("ColocTesseler", &poca::core::CommandInfo(true, "inROIs", _val));
		m_object->notifyAll("updateDisplay");
		return;
	}
	if (sender == m_logHeatMapCBox) {
		poca::core::BasicComponentInterface* bci = m_object->getBasicComponent("ColocTesseler");
		if (!bci) return;
		ColocTesseler* coloc = dynamic_cast <ColocTesseler*>(bci);
		if (m_logHeatMapCBox->isChecked())
			m_scatterGL->setScatterPlot(coloc->scattergramLogPtrAt(0), coloc->scattergramLogPtrAt(1), true);
		else
			m_scatterGL->setScatterPlot(coloc->scattergramPtrAt(0), coloc->scattergramPtrAt(1), false);
		m_scatterGL->update();
		return;
	}
}

void ColocTesselerWidget::actionNeeded(int _val)
{
	QObject* sender = QObject::sender();
	if (sender == m_radiusSlider) {
		m_currentRadiusEdit->blockSignals(true);
		m_currentRadiusEdit->setText(QString::number(_val));
		m_currentRadiusEdit->blockSignals(false);
		m_scatterGL->setRadiusHeatmap(_val);
		m_scatterGL->update();
	}
	if (sender == m_intensitySlider) {
		float val = (float)_val / 1000.f;
		m_intensityEdit->blockSignals(true);
		m_intensityEdit->setText(QString::number(val));
		m_intensityEdit->blockSignals(false);
		m_scatterGL->setIntensityHeatmap(val);
		m_scatterGL->update();
	}
}

void ColocTesselerWidget::actionNeeded(const QString& _action)
{
	QObject* sender = QObject::sender();
	if (sender == m_scatterGL) {
		if (_action == "applyThresholdonClick") {
			float threshs[2] = { m_scatterGL->getThresholdX(), m_scatterGL->getThresholdY() };
			for (size_t n = 0; n < 2; n++) {
				m_factorDensityEdit[n]->blockSignals(true);
				m_factorDensityEdit[n]->setText(QString::number(threshs[n]));
				m_factorDensityEdit[n]->blockSignals(false);
			}
			m_object->executeCommandOnSpecificComponent("ColocTesseler", &poca::core::CommandInfo(true, "threshold", "color1", threshs[0], "color2", threshs[1]));
			m_object->executeCommandOnSpecificComponent("ColocTesseler", &poca::core::CommandInfo(true, "ColocTesseler", "computeCoefficients"));
			m_object->executeCommandOnSpecificComponent("ColocTesseler", &poca::core::CommandInfo(true, "ColocTesseler", "updateFeature"));
			m_object->notifyAll("updateDisplay");
			m_object->notifyAll("LoadObjCharacteristicsColocTesselerWidget");
		}
	}
}

void ColocTesselerWidget::performAction(poca::core::MyObjectInterface* _obj, poca::core::CommandInfo* _ci)
{
	if (_obj == NULL) {
		update(NULL, "");
		return;
	}
	bool actionDone = false;
	if (_ci->nameCommand == "selected") {
		poca::core::BasicComponentInterface* bc = _obj->getBasicComponent("ColocTesseler");
		bc->executeCommand(_ci);
		actionDone = true;
	}
	if (actionDone) {
		_obj->notifyAll("LoadObjCharacteristicsColocTesselerWidget");
		_obj->notifyAll("updateDisplay");
	}
}

void ColocTesselerWidget::update(poca::core::SubjectInterface* _subject, const poca::core::CommandInfo& _aspect)
{
	poca::core::MyObjectInterface* obj = dynamic_cast <poca::core::MyObjectInterface*> (_subject);
	m_object = obj;

	bool visible = (obj != NULL && obj->hasBasicComponent("ColocTesseler"));

	int pos = -1;
	for (int n = 0; n < m_parentTab->count() && pos == -1; n++)
		if (m_parentTab->tabText(n) == "Coloc-Tesseler")
			pos = n;
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
	auto index = m_parentTab->currentIndex();
	m_parentTab->setTabVisible(m_parentTab->indexOf(this), visible);
	m_parentTab->setCurrentIndex(index);
#endif
	if (_aspect == "LoadObjCharacteristicsAllWidgets" || _aspect == "LoadObjCharacteristicsColocTesselerWidget") {

		poca::core::BasicComponentInterface* bci = obj->getBasicComponent("ColocTesseler");
		if (!bci) return;
		if (bci->hasParameter("correction")) {
			bool val = bci->getParameter<bool>("correction");
			m_correctionCbox->blockSignals(true);
			m_correctionCbox->setChecked(val);
			m_correctionCbox->blockSignals(false);
		}
		bool selected = bci->isSelected();
		m_displayButton->blockSignals(true);
		m_displayButton->setChecked(selected);
		m_displayButton->blockSignals(false);

		ColocTesseler* coloc = dynamic_cast <ColocTesseler*>(bci);
		if(m_logHeatMapCBox->isChecked())
			m_scatterGL->setScatterPlot(coloc->scattergramLogPtrAt(0), coloc->scattergramLogPtrAt(1), true);
		else
			m_scatterGL->setScatterPlot(coloc->scattergramPtrAt(0), coloc->scattergramPtrAt(1), false);

		const std::array <float, 2>& spearmans = coloc->getSpearmans(), & manders = coloc->getManders();
		const std::array<std::vector <float>, 2>& spearmans2 = coloc->getSpearmans2(), & manders2 = coloc->getManders2();

		int nbRows = spearmans2[0].size(), currentLine = 0, currentColumn = 0;
		m_tableColoc->clear();
		QStringList tableHeader;
		tableHeader << "" << "Spearmann 1" << "Spearmann 2" << "Manders 1" << "Manders 2";
		m_tableColoc->setColumnCount(tableHeader.size());
		m_tableColoc->setHorizontalHeaderLabels(tableHeader);
		m_tableColoc->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
		m_tableColoc->setRowCount(nbRows);
		for (auto n = 0; n < nbRows; n++) {
			currentColumn = 0;
			m_tableColoc->setItem(currentLine, currentColumn++, new QTableWidgetItem("ROI " + QString::number(n + 1)));
			m_tableColoc->setItem(currentLine, currentColumn++, new QTableWidgetItem(QString::number(spearmans2[0][n])));
			m_tableColoc->setItem(currentLine, currentColumn++, new QTableWidgetItem(QString::number(spearmans2[1][n])));
			m_tableColoc->setItem(currentLine, currentColumn++, new QTableWidgetItem(QString::number(manders2[0][n])));
			m_tableColoc->setItem(currentLine, currentColumn++, new QTableWidgetItem(QString::number(manders2[1][n])));
			currentLine++;
		}
	}
}

void ColocTesselerWidget::executeMacro(poca::core::MyObjectInterface* _wobj, poca::core::CommandInfo* _ci)
{
	this->performAction(_wobj, _ci);
}

const float ColocTesselerWidget::getDensityFactor(size_t _idx, bool* _ok) const
{
	return m_factorDensityEdit[_idx]->text().toFloat(_ok);
}

