/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      VoronoiDiagramWidget.cpp
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
#include <CGAL/boost/graph/copy_face_graph.h>
#include <CGAL/boost/graph/helpers.h> // is_valid_polygon_mesh

#include <Interfaces/MyObjectInterface.hpp>
#include <Interfaces/CommandableObjectInterface.hpp>
#include <General/BasicComponent.hpp>
#include <Factory/ObjectListFactory.hpp>
#include <Geometry/VoronoiDiagram.hpp>
#include <General/CommandableObject.hpp>
#include <Plot/Icons.hpp>
#include <Plot/Misc.h>
#include <Geometry/ObjectListMesh.hpp>
#include <Geometry/ObjectLists.hpp>
#include <General/PluginList.hpp>
#include <General/Engine.hpp>

#include "VoronoiDiagramWidget.hpp"
#include "VoronoiDiagramPlugin.hpp"

VoronoiDiagramWidget::VoronoiDiagramWidget(poca::core::MediatorWObjectFWidgetInterface* _mediator, QWidget* _parent)
{
	m_parentTab = (QTabWidget*)_parent;
	m_mediator = _mediator;

	this->setObjectName("VoronoiDiagramWidget");
	this->addActionToObserve("LoadObjCharacteristicsAllWidgets");
	this->addActionToObserve("LoadObjCharacteristicsVoronoiDiagramWidget");
	this->addActionToObserve("UpdateHistogramVoronoiDiagramWidget");

	m_lutsWidget = new QWidget;
	m_lutsWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	QHBoxLayout* layoutLuts = new QHBoxLayout;
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("HotCold2")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("InvFire")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("Fire")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("Ice")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("AllRedColorBlind")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("AllGreenColorBlind")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("AllBlue")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("AllWhite")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("AllBlack")));
	int maxSize = 20;
	for (size_t n = 0; n < m_lutButtons.size(); n++) {
		m_lutButtons[n].first->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
		m_lutButtons[n].first->setMaximumSize(QSize(maxSize, maxSize));
		QImage im = poca::core::generateImage(maxSize, maxSize, &poca::core::Palette::getStaticLut(m_lutButtons[n].second));
		QPixmap pix = QPixmap::fromImage(im);
		QIcon icon(pix);
		m_lutButtons[n].first->setIcon(icon);
		layoutLuts->addWidget(m_lutButtons[n].first);

		QObject::connect(m_lutButtons[n].first, SIGNAL(pressed()), this, SLOT(actionNeeded()));
	}
	QWidget* emptyLuts = new QWidget;
	emptyLuts->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	layoutLuts->addWidget(emptyLuts);

	m_transferCellsButton = new QPushButton();
	m_transferCellsButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_transferCellsButton->setMaximumSize(QSize(maxSize, maxSize));
	m_transferCellsButton->setIcon(QIcon(QPixmap(poca::plot::exportIcon)));
	m_transferCellsButton->setToolTip("Transfer cells");
	layoutLuts->addWidget(m_transferCellsButton, 0, Qt::AlignRight);
	QObject::connect(m_transferCellsButton, SIGNAL(released()), this, SLOT(actionNeeded()));
	
	m_invertSelectionButton = new QPushButton();
	m_invertSelectionButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_invertSelectionButton->setMaximumSize(QSize(maxSize, maxSize));
	m_invertSelectionButton->setIcon(QIcon(QPixmap(poca::plot::invertIcon)));
	m_invertSelectionButton->setToolTip("Invert selection");
	layoutLuts->addWidget(m_invertSelectionButton, 0, Qt::AlignRight);
	QObject::connect(m_invertSelectionButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));

	m_bboxSelectionButton = new QPushButton();
	m_bboxSelectionButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_bboxSelectionButton->setMaximumSize(QSize(maxSize, maxSize));
	m_bboxSelectionButton->setIcon(QIcon(QPixmap(poca::plot::bboxIcon)));
	m_bboxSelectionButton->setToolTip("Toggle bbox selection");
	m_bboxSelectionButton->setCheckable(true);
	m_bboxSelectionButton->setChecked(true);
	layoutLuts->addWidget(m_bboxSelectionButton, 0, Qt::AlignRight);
	QObject::connect(m_bboxSelectionButton, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));

	m_creationFlteredObjectsButton = new QPushButton();
	m_creationFlteredObjectsButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_creationFlteredObjectsButton->setMaximumSize(QSize(maxSize, maxSize));
	m_creationFlteredObjectsButton->setIcon(QIcon(QPixmap(poca::plot::objectIcon)));
	m_creationFlteredObjectsButton->setToolTip("Create objects");
	layoutLuts->addWidget(m_creationFlteredObjectsButton, 0, Qt::AlignRight);
	QObject::connect(m_creationFlteredObjectsButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));

	m_pointRenderButton = new QPushButton();
	m_pointRenderButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_pointRenderButton->setMaximumSize(QSize(maxSize, maxSize));
	m_pointRenderButton->setIcon(QIcon(QPixmap(poca::plot::pointRenderingIcon)));
	m_pointRenderButton->setToolTip("Render points");
	m_pointRenderButton->setCheckable(true);
	m_pointRenderButton->setChecked(true);
	layoutLuts->addWidget(m_pointRenderButton, 0, Qt::AlignRight);
	QObject::connect(m_pointRenderButton, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));

	m_polyRenderButton = new QPushButton();
	m_polyRenderButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_polyRenderButton->setMaximumSize(QSize(maxSize, maxSize));
	m_polyRenderButton->setIcon(QIcon(QPixmap(poca::plot::polytopeRenderingIcon)));
	m_polyRenderButton->setToolTip("Render polytopes");
	m_polyRenderButton->setCheckable(true);
	m_polyRenderButton->setChecked(true);
	layoutLuts->addWidget(m_polyRenderButton, 0, Qt::AlignRight);
	QObject::connect(m_polyRenderButton, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));

	m_fillButton = new QPushButton();
	m_fillButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_fillButton->setMaximumSize(QSize(maxSize, maxSize));
	m_fillButton->setIcon(QIcon(QPixmap(poca::plot::fillIcon)));
	m_fillButton->setToolTip("Toggle fill/line");
	m_fillButton->setCheckable(true);
	m_fillButton->setChecked(true);
	layoutLuts->addWidget(m_fillButton, 0, Qt::AlignRight);
	QObject::connect(m_fillButton, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));

	m_displayButton = new QPushButton();
	m_displayButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_displayButton->setMaximumSize(QSize(maxSize, maxSize));
	m_displayButton->setIcon(QIcon(QPixmap(poca::plot::brushIcon)));
	m_displayButton->setToolTip("Toggle display");
	m_displayButton->setCheckable(true);
	m_displayButton->setChecked(true);
	layoutLuts->addWidget(m_displayButton, 0, Qt::AlignRight);
	QObject::connect(m_displayButton, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));

	m_lutsWidget->setLayout(layoutLuts);

	m_buttonsWidget = new QWidget;
	m_buttonsWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	QHBoxLayout* layoutButtons = new QHBoxLayout;
	QWidget* emptyButtons = new QWidget;
	emptyButtons->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	layoutButtons->addWidget(emptyButtons);

	m_saveAsSVGButton = new QPushButton();
	m_saveAsSVGButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_saveAsSVGButton->setMaximumSize(QSize(maxSize, maxSize));
	m_saveAsSVGButton->setIcon(QIcon(QPixmap(poca::plot::saveIcon)));
	m_saveAsSVGButton->setToolTip("Save voronoi as SVG");
	layoutButtons->addWidget(m_saveAsSVGButton, 0, Qt::AlignRight);
	QObject::connect(m_saveAsSVGButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));


	QLabel* sizePointLbl = new QLabel();
	sizePointLbl->setMaximumSize(QSize(maxSize, maxSize));
	sizePointLbl->setPixmap(QPixmap(poca::plot::pointSizeIcon).scaled(maxSize, maxSize, Qt::KeepAspectRatio));
	layoutButtons->addWidget(sizePointLbl, 0, Qt::AlignRight);
	m_sizePointSpn = new QSpinBox;
	m_sizePointSpn->setRange(1, 100);
	m_sizePointSpn->setValue(1);
	m_sizePointSpn->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	QObject::connect(m_sizePointSpn, SIGNAL(valueChanged(int)), this, SLOT(actionNeeded(int)));
	layoutButtons->addWidget(m_sizePointSpn, 0, Qt::AlignRight);
	m_buttonsWidget->setLayout(layoutButtons);

	m_voronoiFilteringWidget = new QWidget;
	m_voronoiFilteringWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	m_emptyWidget = new QWidget;
	m_emptyWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

	m_cutDistanceEdit = new QLineEdit("50");
	m_cutDistanceEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_cboxApplyCutDistance = new QCheckBox("Cut distance");
	m_cboxApplyCutDistance->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_cboxApplyCutDistance->setChecked(true);
	m_factorEdit = new QLineEdit("1");
	m_factorEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
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
	m_cboxInROIs = new QCheckBox("In ROIs");
	m_cboxInROIs->setChecked(false);
	m_cboxInROIs->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_applyFactorButton = new QPushButton("Apply factor");
	m_applyFactorButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	QObject::connect(m_applyFactorButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));

	int lineCount = 0, columnCount = 0;
	QGridLayout* layoutFilter = new QGridLayout;
	layoutFilter->addWidget(m_cboxApplyCutDistance, lineCount, columnCount++, 1, 1);
	layoutFilter->addWidget(m_cutDistanceEdit, lineCount, columnCount++, 1, 1);
	layoutFilter->addWidget(m_cboxApplyMinLocs, lineCount, columnCount++, 1, 1);
	layoutFilter->addWidget(m_minLocLEdit, lineCount, columnCount++, 1, 1);
	layoutFilter->addWidget(m_cboxApplyMaxLocs, lineCount, columnCount++, 1, 1);
	layoutFilter->addWidget(m_maxLocLEdit, lineCount++, columnCount++, 1, 1);
	columnCount = 0;
	layoutFilter->addWidget(m_cboxInROIs, lineCount, columnCount++, 1, 1);
	columnCount++;
	layoutFilter->addWidget(m_cboxApplyMinArea, lineCount, columnCount++, 1, 1);
	layoutFilter->addWidget(m_minAreaLEdit, lineCount, columnCount++, 1, 1);
	layoutFilter->addWidget(m_cboxApplyMaxArea, lineCount, columnCount++, 1, 1);
	layoutFilter->addWidget(m_maxAreaLEdit, lineCount++, columnCount++, 1, 1);
	columnCount = 0;
	layoutFilter->addWidget(m_applyFactorButton, lineCount, columnCount++, 1, 1);
	layoutFilter->addWidget(m_factorEdit, lineCount, columnCount++, 1, 1);
	QWidget* widgetFilter = new QWidget;
	widgetFilter->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	widgetFilter->setLayout(layoutFilter);

	QRegExp rx2("[0-9.]+");
	QValidator* validator2 = new QRegExpValidator(rx2, this);
	m_dockVoronoiCharateristics = new QDockWidget("Characteristics");
	m_dockVoronoiCharateristics->setFeatures(QDockWidget::DockWidgetFloatable | QDockWidget::DockWidgetClosable);
	m_dockVoronoiCharateristics->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_btnApplyCharacteristics = new QPushButton("Compute voronoi characteristics");
	m_btnApplyCharacteristics->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	QObject::connect(m_btnApplyCharacteristics, SIGNAL(clicked()), this, SLOT(actionNeeded()));
	m_cboxCumulativeCurves = new QCheckBox("Cumulative curves");
	m_cboxCumulativeCurves->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	QObject::connect(m_cboxCumulativeCurves, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));
	m_cboxROIsCharacteristics = new QCheckBox("In ROIs");
	m_cboxROIsCharacteristics->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_cboxROIsCharacteristics->setChecked(false);
	m_bntBorderLocs = new QPushButton("Select border locs");
	m_bntBorderLocs->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	QObject::connect(m_bntBorderLocs, SIGNAL(clicked()), this, SLOT(actionNeeded()));
	QLabel* lblEnv = new QLabel("Feature enveloppe [0-1]:");
	lblEnv->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_leditEnveloppeFeature = new QLineEdit("0.999");
	m_leditEnveloppeFeature->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_leditEnveloppeFeature->setValidator(validator2);
	QLabel* lblBinsCharac = new QLabel("# bins:");
	lblBinsCharac->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_leditNbBinsCharacteristics = new QLineEdit("100");
	m_leditNbBinsCharacteristics->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_leditNbBinsCharacteristics->setValidator(validator2);
	QLabel* lblDegreePoly = new QLabel("Degree polynome:");
	lblDegreePoly->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_leditDegreePolynome = new QLineEdit("3");
	m_leditDegreePolynome->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_leditDegreePolynome->setValidator(validator2);
	m_customPlotVoronoiCharacteristics = new QCustomPlot();
	m_customPlotVoronoiCharacteristics->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_customPlotVoronoiCharacteristics->setMinimumHeight(400);
	QGridLayout* layoutVoronoiCharacteristics = new QGridLayout;
	layoutVoronoiCharacteristics->addWidget(m_btnApplyCharacteristics, 0, 0, 1, 1);
	layoutVoronoiCharacteristics->addWidget(m_cboxCumulativeCurves, 0, 1, 1, 1);
	layoutVoronoiCharacteristics->addWidget(m_cboxROIsCharacteristics, 0, 2, 1, 1);
	layoutVoronoiCharacteristics->addWidget(m_bntBorderLocs, 0, 3, 1, 1);
	layoutVoronoiCharacteristics->addWidget(lblEnv, 1, 0, 1, 1);
	layoutVoronoiCharacteristics->addWidget(m_leditEnveloppeFeature, 1, 1, 1, 1);
	layoutVoronoiCharacteristics->addWidget(lblDegreePoly, 1, 2, 1, 1);
	layoutVoronoiCharacteristics->addWidget(m_leditDegreePolynome, 1, 3, 1, 1);
	layoutVoronoiCharacteristics->addWidget(lblBinsCharac, 2, 0, 1, 1);
	layoutVoronoiCharacteristics->addWidget(m_leditNbBinsCharacteristics, 2, 1, 1, 1);
	layoutVoronoiCharacteristics->addWidget(m_customPlotVoronoiCharacteristics, 3, 0, 1, 4);
	QWidget* vcharacteristicsDW = new QWidget;
	vcharacteristicsDW->setLayout(layoutVoronoiCharacteristics);
	m_dockVoronoiCharateristics->setWidget(vcharacteristicsDW);

	QVBoxLayout* layout = new QVBoxLayout;
	layout->setContentsMargins(1, 1, 1, 1);
	layout->setSpacing(1);
	layout->addWidget(m_lutsWidget);
	layout->addWidget(m_buttonsWidget);
	layout->addWidget(m_voronoiFilteringWidget);
	layout->addWidget(widgetFilter);
	layout->addWidget(m_dockVoronoiCharateristics);
	layout->addWidget(m_emptyWidget);
	this->setLayout(layout);
}

VoronoiDiagramWidget::~VoronoiDiagramWidget()
{

}

void VoronoiDiagramWidget::actionNeeded()
{
	poca::core::MyObjectInterface* obj = m_object->currentObject();
	poca::core::BasicComponentInterface* bc = obj->getBasicComponent("VoronoiDiagram");
	if (!bc) return;

	poca::core::Engine* engine = poca::core::Engine::instance();

	QObject* sender = QObject::sender();
	bool found = false;
	for (size_t n = 0; n < m_lutButtons.size() && !found; n++) {
		found = (m_lutButtons[n].first == sender);
		if (found) {
			engine->executeCommand(bc, true, "changeLUT", "LUT", m_lutButtons[n].second);
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
		if (ok && m_cboxApplyCutDistance->isChecked())	cutD = val;
		val = this->getMinNbLocs(&ok);
		if (ok && m_cboxApplyMinLocs->isChecked())	minLocs = (size_t)val;
		val = this->getMaxNbLocs(&ok);
		if (ok && m_cboxApplyMaxLocs->isChecked())	maxLocs = (size_t)val;
		val = this->getMinArea(&ok);
		if (ok && m_cboxApplyMinArea->isChecked())	minArea = val;
		val = this->getMaxArea(&ok);
		if (ok && m_cboxApplyMaxArea->isChecked())	maxArea = val;
		bool inROIs = m_cboxInROIs->isChecked();

		engine->executeCommand(bc, true, "createFilteredObjects",
			"cutDistance", cutD,
			"minLocs", minLocs,
			"maxLocs", maxLocs,
			"minArea", minArea,
			"maxArea", maxArea,
			"inROIs", inROIs);

		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_applyFactorButton) {
		bool ok;
		float factor = getDensityFactor(&ok);
		if (!ok) return;
		bool inROIs = m_cboxInROIs->isChecked();
		engine->executeCommand(bc, true, "densityFactor", "factor", factor, "inROIs", inROIs);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_invertSelectionButton) {
		engine->executeCommand(bc, true, "invertSelection");
		m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_transferCellsButton) {
		poca::geometry::VoronoiDiagram3D* voro3D = dynamic_cast<poca::geometry::VoronoiDiagram3D*>(bc);
		if (voro3D == NULL) return;
		if (!voro3D->hasCells()) return;
		const auto& polyhedrons = voro3D->getPolyhedrons();
		std::vector < Surface_mesh_3_double> meshes;
		for (const auto& poly : polyhedrons) {
			meshes.push_back(Surface_mesh_3_double());
			CGAL::copy_face_graph(poly, meshes.back());
			assert(CGAL::is_valid_polygon_mesh(meshes.back()));
		}
		poca::geometry::ObjectListMesh* newomesh = new poca::geometry::ObjectListMesh(meshes);
		VoronoiDiagramPlugin::m_plugins->addCommands(newomesh);
		poca::core::CommandInfo ci;
		if (!obj->hasBasicComponent("ObjectLists")) {
			poca::geometry::ObjectLists* objsList = new poca::geometry::ObjectLists(newomesh, ci, "VoronoiDiagramPlugin");
			VoronoiDiagramPlugin::m_plugins->addCommands(objsList);
			obj->addBasicComponent(objsList);
		}
		else {
			poca::geometry::ObjectLists* objsList = dynamic_cast<poca::geometry::ObjectLists*>(obj->getBasicComponent("ObjectLists"));
			if (objsList)
				objsList->addObjectList(newomesh, ci, "ObjectListPlugin");
		}
		m_object->notify("LoadObjCharacteristicsAllWidgets");
		m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_btnApplyCharacteristics) {
		bool ok;
		float tmp = m_leditEnveloppeFeature->text().toFloat(&ok), env = ok ? tmp : 0.999;
		unsigned int tmp2 = m_leditNbBinsCharacteristics->text().toUInt(&ok), nbBins = ok ? tmp2 : 100;
		unsigned int tmp3 = m_leditDegreePolynome->text().toUInt(&ok), degreePoly = ok ? tmp3 : 3;
		bool onROIs = m_cboxROIsCharacteristics->isChecked();
		engine->executeCommand(bc, true, "voronoiCharacteristics", "env", env, "onROIs", onROIs, "nbBins", nbBins, "degreePolynome", degreePoly);
		m_object->notify("LoadObjCharacteristicsVoronoiDiagramWidget");
		m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_bntBorderLocs) {
		poca::geometry::VoronoiDiagram* voro = (poca::geometry::VoronoiDiagram*)bc;
		const std::vector <bool>& selection = voro->borderLocalizations();
		auto count = std::count(selection.begin(), selection.end(), true);
		std::cout << "# of border cells: " << count << " / " << voro->nbFaces() << std::endl;
		voro->setSelection(selection);
		engine->executeCommand(bc, false, "updateFeature");
	}
	else if (sender == m_saveAsSVGButton) {
		QString name = m_object->getName().c_str(), filename(m_object->getDir().c_str());
		if (!filename.endsWith("/"))
			filename.append("/");
		filename.append(name);
		QString extension = name.indexOf(".") != -1 ? name.right(name.size() - name.indexOf(".")) : ".svg";
		filename = QFileDialog::getSaveFileName(NULL, QObject::tr("Save voronoi as SVG..."), filename, QString("svg files (*svg)"), 0, QFileDialog::DontUseNativeDialog);
		if (filename.isEmpty()) return;
		engine->executeCommand(bc, true, "saveAsSVG", "filename", filename.toStdString());
	}
}

void VoronoiDiagramWidget::actionNeeded(bool _val)
{
	poca::core::MyObjectInterface* obj = m_object->currentObject();
	poca::core::BasicComponentInterface* bc = obj->getBasicComponent("VoronoiDiagram");
	if (!bc) return;
	poca::core::Engine* engine = poca::core::Engine::instance();

	QObject* sender = QObject::sender();
	if (sender == m_displayButton) {
		engine->executeCommand(bc, true, "selected", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	if (sender == m_fillButton) {
		engine->executeCommand(bc, true, "fill", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	if (sender == m_pointRenderButton) {
		engine->executeCommand(bc, true, "pointRendering", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	if (sender == m_polyRenderButton) {
		engine->executeCommand(bc, true, "polytopeRendering", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	if (sender == m_bboxSelectionButton) {
		engine->executeCommand(bc, true, "bboxSelection", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
}

void VoronoiDiagramWidget::actionNeeded(int _val)
{
	poca::core::MyObjectInterface* obj = m_object->currentObject();
	poca::core::BasicComponentInterface* bc = obj->getBasicComponent("VoronoiDiagram");
	if (!bc) return;
	poca::core::Engine* engine = poca::core::Engine::instance();

	QObject* sender = QObject::sender();
	if (sender == m_sizePointSpn) {
		unsigned int valD = this->pointSize();
		engine->executeCommand(bc, true, "pointSizeGL", valD);
		m_object->notifyAll("updateDisplay");
	}
}

void VoronoiDiagramWidget::performAction(poca::core::MyObjectInterface* _obj, poca::core::CommandInfo* _ci)
{
	if (_obj == NULL) {
		update(NULL, "");
		return;
	}
	poca::core::Engine* engine = poca::core::Engine::instance();
	poca::core::MyObjectInterface* obj = _obj->currentObject();
	bool actionDone = false;
	if (_ci->nameCommand == "histogram" || _ci->nameCommand == "changeLUT" || _ci->nameCommand == "selected" || _ci->nameCommand == "fill" || _ci->nameCommand == "pointRendering" || _ci->nameCommand == "polytopeRendering") {
		poca::core::BasicComponentInterface* bc = obj->getBasicComponent("VoronoiDiagram");
		engine->executeCommand(bc, _ci);
		actionDone = true;
	}

	if (actionDone) {
		_obj->notifyAll("LoadObjCharacteristicsVoronoiDiagramWidget");
		_obj->notifyAll("updateDisplay");
	}
}

void VoronoiDiagramWidget::update(poca::core::SubjectInterface* _subject, const poca::core::CommandInfo& _aspect)
{
	poca::core::MyObjectInterface* obj = dynamic_cast <poca::core::MyObjectInterface*> (_subject);
	poca::core::MyObjectInterface* objOneColor = obj->currentObject();
	m_object = obj;

	bool visible = (objOneColor != NULL && objOneColor->hasBasicComponent("VoronoiDiagram"));
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
	auto index = m_parentTab->currentIndex();
	m_parentTab->setTabVisible(m_parentTab->indexOf(this), visible);
	m_parentTab->setCurrentIndex(index);
#endif

	if (_aspect == "LoadObjCharacteristicsAllWidgets" || _aspect == "LoadObjCharacteristicsVoronoiDiagramWidget") {

		poca::core::BasicComponentInterface* bci = objOneColor->getBasicComponent("VoronoiDiagram");
		if (!bci) return;
		poca::core::stringList nameData = bci->getNameData();

		QVBoxLayout* layout = NULL;
		//First time we load data -> no hist widget was created before
		if (m_histWidgets.empty()) {
			layout = new QVBoxLayout;
			layout->setContentsMargins(1, 1, 1, 1);
			layout->setSpacing(1);
			for (size_t n = 0; n < nameData.size(); n++) {
				m_histWidgets.push_back(new poca::plot::FilterHistogramWidget(m_mediator, "VoronoiDiagram", this));
				layout->addWidget(m_histWidgets[n]);
			}
			m_voronoiFilteringWidget->setLayout(layout);
		}
		else if (nameData.size() > m_histWidgets.size()) {
			layout = dynamic_cast <QVBoxLayout*>(m_voronoiFilteringWidget->layout());
			//Here, we need to add some hist widgets because this loc data has more features than the one loaded before
			for (size_t n = m_histWidgets.size(); n < nameData.size(); n++) {
				m_histWidgets.push_back(new poca::plot::FilterHistogramWidget(m_mediator, "VoronoiDiagram", this));
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

		m_voronoiFilteringWidget->updateGeometry();

		if (bci->hasParameter("pointRendering")) {
			bool val = bci->getParameter<bool>("pointRendering");
			m_pointRenderButton->blockSignals(true);
			m_pointRenderButton->setChecked(val);
			m_pointRenderButton->blockSignals(false);
		}
		if (bci->hasParameter("polytopeRendering")) {
			bool val = bci->getParameter<bool>("polytopeRendering");
			m_polyRenderButton->blockSignals(true);
			m_polyRenderButton->setChecked(val);
			m_polyRenderButton->blockSignals(false);
		}
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

		bool selected = bci->isSelected();
		m_displayButton->blockSignals(true);
		m_displayButton->setChecked(selected);
		m_displayButton->blockSignals(false);

		poca::geometry::VoronoiDiagram3D* voro3D = dynamic_cast<poca::geometry::VoronoiDiagram3D*>(bci);
		bool displayTransferCells = voro3D && voro3D->hasCells();
		m_transferCellsButton->blockSignals(true);
		m_transferCellsButton->setVisible(displayTransferCells);
		m_transferCellsButton->blockSignals(false);
	}
}

void VoronoiDiagramWidget::executeMacro(poca::core::MyObjectInterface* _wobj, poca::core::CommandInfo* _ci)
{
	this->performAction(_wobj, _ci);
}

const float VoronoiDiagramWidget::getDensityFactor(bool* _ok) const
{
	return m_factorEdit->text().toFloat(_ok);
}

const float VoronoiDiagramWidget::getCutDistance(bool* _ok) const
{
	return m_cutDistanceEdit->text().toFloat(_ok);
}

const float VoronoiDiagramWidget::getMinNbLocs(bool* _ok) const
{
	return m_minLocLEdit->text().toFloat(_ok);
}

const float VoronoiDiagramWidget::getMaxNbLocs(bool* _ok) const
{
	return m_maxLocLEdit->text().toFloat(_ok);
}

const float VoronoiDiagramWidget::getMinArea(bool* _ok) const
{
	return m_minAreaLEdit->text().toFloat(_ok);
}

const float VoronoiDiagramWidget::getMaxArea(bool* _ok) const
{
	return m_maxAreaLEdit->text().toFloat(_ok);
}

