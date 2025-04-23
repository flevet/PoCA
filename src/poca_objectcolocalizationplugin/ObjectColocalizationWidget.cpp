/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectColocalizationWidget.cpp
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
#include <fstream>

#include <Interfaces/MyObjectInterface.hpp>
#include <Interfaces/CommandableObjectInterface.hpp>
#include <General/BasicComponent.hpp>
#include <Factory/ObjectListFactory.hpp>
#include <General/CommandableObject.hpp>
#include <Geometry/ObjectLists.hpp>
#include <Plot/Icons.hpp>
#include <Plot/Misc.h>
#include <General/MyData.hpp>

#include "ObjectColocalizationWidget.hpp"
#include "ObjectColocalization.hpp"

ObjectColocalizationWidget::ObjectColocalizationWidget(poca::core::MediatorWObjectFWidgetInterface* _mediator, QWidget* _parent) :QTabWidget(_parent)
{
	m_parentTab = (QTabWidget*)_parent;
	m_mediator = _mediator;

	this->setObjectName("ObjectColocalizationWidget");
	this->addActionToObserve("LoadObjCharacteristicsAllWidgets");
	this->addActionToObserve("LoadObjCharacteristicsObjectColocalizationWidget");

	m_lutsWidget = new QWidget;
	m_lutsWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	m_buttonsWidget = new QWidget;
	m_buttonsWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	QHBoxLayout* layoutLuts = new QHBoxLayout, * layoutButtons = new QHBoxLayout;
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("HotCold2")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("InvFire")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("Fire")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("Ice")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("AllRedColorBlind")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("AllGreenColorBlind")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("AllBlue")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("AllWhite")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("AllBlack")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("AllRed")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("AllGreen")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("AllOrange")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("AllTomato")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("AllCyan")));
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("AllTurquoise")));
	int maxSize = 20;
	for (size_t n = 0; n < m_lutButtons.size(); n++) {
		m_lutButtons[n].first->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
		m_lutButtons[n].first->setMaximumSize(QSize(maxSize, maxSize));
		QImage im = poca::core::generateImage(maxSize, maxSize, &poca::core::Palette::getStaticLut(m_lutButtons[n].second));
		QPixmap pix = QPixmap::fromImage(im);
		QIcon icon(pix);
		m_lutButtons[n].first->setIcon(icon);
		if (n < 9)
			layoutLuts->addWidget(m_lutButtons[n].first);
		else
			layoutButtons->addWidget(m_lutButtons[n].first);

		QObject::connect(m_lutButtons[n].first, SIGNAL(pressed()), this, SLOT(actionNeeded()));
	}
	m_hilowButton = std::make_pair(new QPushButton(), std::string("HiLo"));
	m_hilowButton.first->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_hilowButton.first->setMaximumSize(QSize(2 * maxSize, maxSize));
	m_hilowButton.first->setIcon(QPixmap(poca::plot::hiLowIcon3));
	m_hilowButton.first->setIconSize(QSize(40, 20));;
	layoutLuts->addWidget(m_hilowButton.first);
	QObject::connect(m_hilowButton.first, SIGNAL(pressed()), this, SLOT(actionNeeded()));

	QWidget* emptyLuts = new QWidget;
	emptyLuts->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	layoutLuts->addWidget(emptyLuts);

	m_pointRenderButton = new QPushButton();
	m_pointRenderButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_pointRenderButton->setMaximumSize(QSize(maxSize, maxSize));
	m_pointRenderButton->setIcon(QIcon(QPixmap(poca::plot::pointRenderingIcon)));
	m_pointRenderButton->setToolTip("Render points");
	m_pointRenderButton->setCheckable(true);
	m_pointRenderButton->setChecked(true);
	layoutLuts->addWidget(m_pointRenderButton, 0, Qt::AlignRight);
	QObject::connect(m_pointRenderButton, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));

	m_shapeRenderButton = new QPushButton();
	m_shapeRenderButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_shapeRenderButton->setMaximumSize(QSize(maxSize, maxSize));
	m_shapeRenderButton->setIcon(QIcon(QPixmap(poca::plot::polytopeRenderingIcon)));
	m_shapeRenderButton->setToolTip("Render shapes");
	m_shapeRenderButton->setCheckable(true);
	m_shapeRenderButton->setChecked(true);
	layoutLuts->addWidget(m_shapeRenderButton, 0, Qt::AlignRight);
	QObject::connect(m_shapeRenderButton, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));

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

	QWidget* emptyButtons = new QWidget;
	emptyButtons->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	layoutButtons->addWidget(emptyButtons);

	m_exportButton = new QPushButton();
	m_exportButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_exportButton->setMaximumSize(QSize(maxSize, maxSize));
	m_exportButton->setIcon(QIcon(QPixmap(poca::plot::exportIcon)));
	m_exportButton->setToolTip("Export objects");
	layoutButtons->addWidget(m_exportButton, 0, Qt::AlignRight);
	QObject::connect(m_exportButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));

	m_selectionButton = new QPushButton();
	m_selectionButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_selectionButton->setMaximumSize(QSize(maxSize, maxSize));
	m_selectionButton->setIcon(QIcon(QPixmap(poca::plot::selectionIcon)));
	m_selectionButton->setToolTip("Toggle picking");
	m_selectionButton->setCheckable(true);
	m_selectionButton->setChecked(true);
	layoutButtons->addWidget(m_selectionButton, 0, Qt::AlignRight);
	QObject::connect(m_selectionButton, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));

	m_bboxSelectionButton = new QPushButton();
	m_bboxSelectionButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_bboxSelectionButton->setMaximumSize(QSize(maxSize, maxSize));
	m_bboxSelectionButton->setIcon(QIcon(QPixmap(poca::plot::bboxIcon)));
	m_bboxSelectionButton->setToolTip("Toggle bbox selection");
	m_bboxSelectionButton->setCheckable(true);
	m_bboxSelectionButton->setChecked(true);
	layoutButtons->addWidget(m_bboxSelectionButton, 0, Qt::AlignRight);
	QObject::connect(m_bboxSelectionButton, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));

	m_lutsWidget->setLayout(layoutLuts);
	m_buttonsWidget->setLayout(layoutButtons);

	m_delaunayTriangulationFilteringWidget = new QWidget;
	m_delaunayTriangulationFilteringWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	
	m_tableObjects = new QTableWidget;
	m_tableObjects->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	QHeaderView* headerGoods = m_tableObjects->horizontalHeader();
	connect(headerGoods, SIGNAL(sectionClicked(int)), m_tableObjects, SLOT(sortByColumn(int)));
	connect(m_tableObjects, SIGNAL(itemSelectionChanged()), this, SLOT(actionNeeded()));
	m_tableInfos = new QTableWidget;
	m_tableInfos->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	QTabWidget * tabW = new QTabWidget;
	tabW->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	tabW->addTab(m_tableObjects, "Overlap objects");
	tabW->addTab(m_tableInfos, "Overlap infos");

	m_emptyWidget = new QWidget;
	m_emptyWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

	QVBoxLayout* layout = new QVBoxLayout;
	layout->setContentsMargins(1, 1, 1, 1);
	layout->setSpacing(1);
	layout->addWidget(m_lutsWidget);
	layout->addWidget(m_buttonsWidget);
	layout->addWidget(m_delaunayTriangulationFilteringWidget);
	layout->addWidget(tabW);

	this->setLayout(layout);
}

ObjectColocalizationWidget::~ObjectColocalizationWidget()
{

}

void ObjectColocalizationWidget::actionNeeded()
{
	poca::core::BasicComponentInterface* bc = m_object->getBasicComponent("ObjectColocalization");
	if (!bc) return;
	ObjectColocalization* coloc = dynamic_cast <ObjectColocalization*>(bc);
	if (!coloc) return;
	poca::core::CommandableObject* colocCommand = static_cast <poca::core::CommandableObject*>(bc);
	poca::core::CommandableObject* objList = dynamic_cast <poca::core::CommandableObject*>(coloc->getObjectsOverlap());

	QObject* sender = QObject::sender();
	bool found = false;
	for (size_t n = 0; n < m_lutButtons.size() && !found; n++) {
		found = (m_lutButtons[n].first == sender);
		if (found) {
			if (m_hilowButton.first->isChecked()) {
				m_hilowButton.first->blockSignals(true);
				m_hilowButton.first->setChecked(false);
				m_hilowButton.first->blockSignals(false);
			}
			objList->executeCommand(true, "changeLUT", "LUT", m_lutButtons[n].second);
			colocCommand->executeCommand(true, "changeLUT", "LUT", m_lutButtons[n].second);
			for (poca::plot::FilterHistogramWidget* histW : m_histWidgets)
				histW->redraw();
			m_object->notifyAll("updateDisplay");
		}
	}
	if (sender == m_hilowButton.first) {
		objList->executeCommand(true, "changeLUT", m_hilowButton.second);
		colocCommand->executeCommand(true, "changeLUT", m_hilowButton.second);
		for (poca::plot::FilterHistogramWidget* histW : m_histWidgets)
			histW->redraw();
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_exportButton) {
		QString filename, separator(","), extension(".csv");
		filename = m_object->getDir().c_str();
		if (!filename.endsWith("/")) filename.append("/");
		filename.append(m_object->getName().c_str());
		if (filename.contains("."))
			filename = filename.mid(0, filename.lastIndexOf("."));
		filename.append(extension);
		filename = QFileDialog::getSaveFileName(this, QObject::tr("Save stats..."), filename, QObject::tr("Stats files (*.csv)"), 0, QFileDialog::DontUseNativeDialog);
		std::vector <QString> headers, data;
		for (int n = 0; n < m_tableObjects->columnCount(); n++)
			headers.push_back(m_tableObjects->horizontalHeaderItem(n)->text());
		for (int i = 0; i < m_tableObjects->rowCount(); i++) {
			for (int j = 0; j < m_tableObjects->columnCount(); j++)
				data.push_back(m_tableObjects->item(i, j)->text());
		}
		std::ofstream fs;
		fs.open(filename.toLatin1().data());
		if (!fs.is_open()) {
			std::cout << "Failed to open file " << filename.toLatin1().data() << std::endl;
			return;
		}

		size_t nbColumns = headers.size(), cpt = 0;
		for (size_t n = 0; n < nbColumns; n++) {
			fs << headers[n].toLatin1().data();
			if (n < nbColumns - 1)
				fs << separator.toLatin1().data();
		}
		fs << std::endl;

		for (size_t n = 0; n < data.size(); n++) {
			fs << data[n].toLatin1().data();
			if (cpt < nbColumns - 1) {
				fs << separator.toLatin1().data();
				cpt++;
			}
			else {
				fs << std::endl;
				cpt = 0;
			}
		}
		fs.close();
		std::cout << "File " << filename.toLatin1().data() << " was written" << std::endl;

		filename.insert(filename.lastIndexOf("."), "_overlapInfos");
		fs.open(filename.toLatin1().data());
		if (!fs.is_open()) {
			std::cout << "Failed to open file " << filename.toLatin1().data() << std::endl;
			return;
		}

		headers.clear();
		data.clear();
		for (int n = 0; n < m_tableInfos->columnCount(); n++) {
			headers.push_back(m_tableInfos->horizontalHeaderItem(n)->text());
		}
		for (int i = 0; i < m_tableInfos->rowCount(); i++) {
			for (int j = 0; j < m_tableInfos->columnCount(); j++) {
				data.push_back(m_tableInfos->item(i, j)->text());
			}
		}

		nbColumns = headers.size();
		cpt = 0;
		for (size_t n = 0; n < nbColumns; n++) {
			fs << headers[n].toLatin1().data();
			if (n < nbColumns - 1)
				fs << separator.toLatin1().data();
		}
		fs << std::endl;

		for (size_t n = 0; n < data.size(); n++) {
			fs << data[n].toLatin1().data();
			if (cpt < nbColumns - 1) {
				fs << separator.toLatin1().data();
				cpt++;
			}
			else {
				fs << std::endl;
				cpt = 0;
			}
		}
		fs.close();
		std::cout << "File " << filename.toLatin1().data() << " was written" << std::endl;
}
	else if (sender == m_tableObjects) {
		std::set <int> selectedRows;
		QList<QTableWidgetSelectionRange> ranges = m_tableObjects->selectedRanges();
		for (QTableWidgetSelectionRange range : ranges)
			for (int n = 0; n < range.rowCount(); n++)
				selectedRows.insert(range.topRow() + n);
		if (selectedRows.empty()) return;
		if (selectedRows.size() == 1) {
			coloc->executeCommand(false, "setIDObjectPicked", *selectedRows.begin());
			m_object->notify("updateInfosObjectOverlap");
			m_object->notifyAll("updateDisplay");
		}
		else {
			if (m_selectionButton->isChecked())
				m_selectionButton->setChecked(false);
			std::vector <bool> selection(m_tableObjects->rowCount(), false);
			for (int idx : selectedRows)
				selection[idx] = true;
			coloc->getObjectsOverlap()->setSelection(selection);
			objList->executeCommand(false, "setIDObjectPicked", -1);
			objList->executeCommand(false, "updateFeature");
			m_object->notifyAll("updateDisplay");
		}
	}
}

void ObjectColocalizationWidget::actionNeeded(bool _val)
{
	poca::core::BasicComponentInterface* bc = m_object->getBasicComponent("ObjectColocalization");
	if (!bc) return;
	poca::core::CommandableObject* coloc = dynamic_cast <poca::core::CommandableObject*>(bc);

	QObject* sender = QObject::sender();
	if (sender == m_displayButton) {
		coloc->executeCommand(true, "selected", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_pointRenderButton) {
		coloc->executeCommand(true, "pointRendering", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_shapeRenderButton) {
		coloc->executeCommand(true, "shapeRendering", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_fillButton) {
		coloc->executeCommand(true, "fill", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_bboxSelectionButton) {
		coloc->executeCommand(true, "bboxSelection", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_selectionButton) {
		coloc->executeCommand(true, "togglePicking", _val);
		return;
	}
}

void ObjectColocalizationWidget::performAction(poca::core::MyObjectInterface* _obj, poca::core::CommandInfo* _ci)
{
	if (_obj == NULL) {
		update(NULL, "");
		return;
	}
	m_object = _obj;
	bool actionDone = false;
	if (_ci->nameCommand == "histogram" || _ci->nameCommand == "changeLUT") {
		std::string action = _ci->getParameter<std::string>("action");
		if (action == "save")
			_ci->addParameter("dir", _obj->getDir());
		poca::core::BasicComponentInterface* bc = m_object->getBasicComponent("ObjectColocalization");
		if (!bc) return;
		ObjectColocalization* coloc = dynamic_cast <ObjectColocalization*>(bc);
		if (!coloc) return;
		coloc->getObjectsOverlap()->executeCommand(_ci);
		actionDone = true;
	}
	if (_ci->nameCommand == "histogram" || _ci->nameCommand == "changeLUT" ||_ci->nameCommand == "selected" || _ci->nameCommand == "pointRendering" || _ci->nameCommand == "shapeRendering" || _ci->nameCommand == "fill" || _ci->nameCommand == "delaunayRendering" || _ci->nameCommand == "selectedDelaunayRendering") {
		poca::core::BasicComponentInterface* bc = m_object->getBasicComponent("ObjectColocalization");
		bc->executeCommand(_ci);
		actionDone = true;
	}
	/*else if (_ci->nameCommand == "histogram" || _ci->nameCommand == "changeLUT") {
		if (_ci->nameCommand == "histogram") {
			std::string action = _ci->getParameter<std::string>("action");
			if (action == "save")
				_ci->addParameter("dir", _obj->getDir());
		}
		poca::core::BasicComponentInterface* bc = obj->getBasicComponent("ObjectColocalization");
		if (!bc) return;
		ObjectColocalization* coloc = dynamic_cast <ObjectColocalization*>(bc);
		if (!coloc) return;
		coloc->getObjectsOverlap()->executeCommand(_ci);
		actionDone = true;
	}*/
	else if (_ci->nameCommand == "selectObject") {
		int id = _ci->getParameter<int>("id");
		if(id < m_tableObjects->rowCount())
			m_tableObjects->selectRow(id);
	}
	else if (_ci->nameCommand == "updatePickedObject") {
		poca::core::CommandInfo ci3(false, "getObjectPickedID");
		poca::core::BasicComponentInterface* bc = _obj->getBasicComponent("ObjectColocalization");
		if (bc != NULL) {
			bc->executeCommand(&ci3);
			if (!ci3.json.empty() && ci3.hasParameter("id")) {
				int id = ci3.getParameter<int>("id");
				if (id != -1)
					m_tableObjects->selectRow(id);
			}
		}
	}
	if (actionDone) {
		_obj->notifyAll("LoadObjCharacteristicsObjectColocalizationWidget");
		_obj->notifyAll("updateDisplay");
	}
}

void ObjectColocalizationWidget::update(poca::core::SubjectInterface* _subject, const poca::core::CommandInfo& _aspect)
{
	poca::core::MyObjectInterface* obj = dynamic_cast <poca::core::MyObjectInterface*> (_subject);
	m_object = obj;

	bool visible = (obj != NULL && obj->hasBasicComponent("ObjectColocalization"));
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
	auto index = m_parentTab->currentIndex();
	m_parentTab->setTabVisible(m_parentTab->indexOf(this), visible);
	m_parentTab->setCurrentIndex(index);
#endif

	if (_aspect == "LoadObjCharacteristicsAllWidgets" || _aspect == "LoadObjCharacteristicsObjectColocalizationWidget") {
		poca::core::BasicComponentInterface* bci = obj->getBasicComponent("ObjectColocalization");
		if (!bci) return;
		ObjectColocalization* coloc = dynamic_cast <ObjectColocalization*>(bci);
		if (!coloc) return;
		poca::geometry::ObjectListInterface* objList = coloc->getObjectsOverlap();
		if (objList == NULL) return;

		poca::core::stringList nameData = objList->getNameData();

		QVBoxLayout* layout = NULL;
		//First time we load data -> no hist widget was created before
		if (m_histWidgets.empty()) {
			layout = new QVBoxLayout;
			//std::cout << "Spacing = " << layout->spacing() << std::endl;
			layout->setContentsMargins(1, 1, 1, 1);
			layout->setSpacing(1);
			for (size_t n = 0; n < nameData.size(); n++) {
				m_histWidgets.push_back(new poca::plot::FilterHistogramWidget(m_mediator, "ObjectList", this));
				layout->addWidget(m_histWidgets[n]);
			}
			m_delaunayTriangulationFilteringWidget->setLayout(layout);
		}
		else if (nameData.size() > m_histWidgets.size()) {
			layout = dynamic_cast <QVBoxLayout*>(m_delaunayTriangulationFilteringWidget->layout());
			//Here, we need to add some hist widgets because this loc data has more features than the one loaded before
			for (size_t n = m_histWidgets.size(); n < nameData.size(); n++) {
				m_histWidgets.push_back(new poca::plot::FilterHistogramWidget(m_mediator, "ObjectList", this));
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
			poca::core::HistogramInterface* hist = objList->getHistogram(type);
			if (hist != NULL)
				m_histWidgets[cpt++]->setInfos(type.c_str(), hist, objList->isLogHistogram(type), objList->isCurrentHistogram(type) ? objList->getPalette() : NULL);
		}

		m_tableObjects->setSortingEnabled(false);
		QStringList tableHeader2;
		tableHeader2 << "ID";
		for (std::string type : nameData) {
			tableHeader2 << type.c_str();
		}
		m_tableObjects->setColumnCount(tableHeader2.size());
		m_tableObjects->setHorizontalHeaderLabels(tableHeader2);
		m_tableObjects->setRowCount((int)objList->nbElements());
		int columnCount = 0;
		for (size_t rowCount = 0; rowCount < objList->nbElements(); rowCount++)
			m_tableObjects->setItem((int)rowCount, (int)columnCount, new QTableWidgetItem(QString::number(rowCount + 1)));
		columnCount++;
		for (std::string type : nameData) {
			if (type != "nbLocs") {
				if (objList->hasData(type)) {
					const std::vector <float>& values = objList->getMyData(type)->getData<float>();
					for (size_t rowCount = 0; rowCount < values.size(); rowCount++)
						m_tableObjects->setItem((int)rowCount, (int)columnCount, new QTableWidgetItem(QString::number(values[rowCount])));
				}
			}
			else {
				//Since we may have added locs to have a better approximation of the overlap objects
				//we cannot use the nbLocs computed from the ObjectList
				//we therefore get the proper number from m_origPointsInOverlapObjects
				const std::vector <uint32_t>& range = coloc->getLocsOverlapObjects().getFirstElements();
				for (size_t rowCount = 0; rowCount < coloc->getLocsOverlapObjects().nbElements(); rowCount++)
					m_tableObjects->setItem((int)rowCount, (int)columnCount, new QTableWidgetItem(QString::number(range[rowCount + 1] - range[rowCount])));
			}
			columnCount++;
		}

		/*QHeaderView* headerGoods = m_tableObjects->horizontalHeader();
		//SortIndicator is a triangle indicator next to the horizontal title bar text
		headerGoods->setSortIndicator(0, Qt::AscendingOrder);
		headerGoods->setSortIndicatorShown(true);
		m_tableObjects->setSortingEnabled(true);*/

		const poca::core::MyArrayUInt32& infosColor1 = coloc->getInfosColor1();
		const poca::core::MyArrayUInt32& infosColor2 = coloc->getInfosColor2();
		size_t maxNbOverlaps = 0;
		for (size_t n = 0; n < infosColor1.nbElements(); n++)
			if (infosColor1.nbElementsObject(n) > maxNbOverlaps)
				maxNbOverlaps = infosColor1.nbElementsObject(n);
		for (size_t n = 0; n < infosColor2.nbElements(); n++)
			if (infosColor2.nbElementsObject(n) > maxNbOverlaps)
				maxNbOverlaps = infosColor2.nbElementsObject(n);
		QStringList headerInfos;
		headerInfos << "Color" << "id" << "# unique objects" << "# overlaps";
		for (size_t n = 0; n < maxNbOverlaps; n++)
			headerInfos << QString("id %1").arg(n + 1);
		m_tableInfos->setColumnCount(headerInfos.size());
		m_tableInfos->setHorizontalHeaderLabels(headerInfos);
		m_tableInfos->setRowCount((int)(infosColor1.nbElements() + infosColor2.nbElements()));
		size_t rowCount = 0;
		for (size_t n = 0; n < infosColor1.nbElements(); n++, rowCount++) {
			int columnCount = 0;
			m_tableInfos->setItem((int)rowCount, (int)columnCount++, new QTableWidgetItem("1"));
			m_tableInfos->setItem((int)rowCount, (int)columnCount++, new QTableWidgetItem(QString::number(n + 1)));
			std::set <uint32_t> uniqueObjects;
			for (uint32_t i = 0; i < infosColor1.nbElementsObject(n); i++)
				uniqueObjects.insert(infosColor1.elementIObject(n, i));
			m_tableInfos->setItem((int)rowCount, (int)columnCount++, new QTableWidgetItem(QString::number(uniqueObjects.size())));
			m_tableInfos->setItem((int)rowCount, (int)columnCount++, new QTableWidgetItem(QString::number(infosColor1.nbElementsObject(n))));
			for (uint32_t i = 0; i < maxNbOverlaps; i++)
				m_tableInfos->setItem((int)rowCount, (int)columnCount++, i < infosColor1.nbElementsObject(n) ? new QTableWidgetItem(QString::number(infosColor1.elementIObject(n, i) + 1)) : new QTableWidgetItem(""));
			/*for(uint32_t i = 0; i < infosColor1.nbElementsObject(n); i++)
				m_tableInfos->setItem((int)rowCount, (int)columnCount++, new QTableWidgetItem(QString::number(infosColor1.elementIObject(n, i) + 1)));*/
		}
		for (size_t n = 0; n < infosColor2.nbElements(); n++, rowCount++) {
			int columnCount = 0;
			m_tableInfos->setItem((int)rowCount, (int)columnCount++, new QTableWidgetItem("2"));
			m_tableInfos->setItem((int)rowCount, (int)columnCount++, new QTableWidgetItem(QString::number(n + 1)));
			std::set <uint32_t> uniqueObjects;
			for (uint32_t i = 0; i < infosColor2.nbElementsObject(n); i++)
				uniqueObjects.insert(infosColor2.elementIObject(n, i));
			m_tableInfos->setItem((int)rowCount, (int)columnCount++, new QTableWidgetItem(QString::number(uniqueObjects.size())));
			m_tableInfos->setItem((int)rowCount, (int)columnCount++, new QTableWidgetItem(QString::number(infosColor2.nbElementsObject(n))));
			for (uint32_t i = 0; i < maxNbOverlaps; i++)
				m_tableInfos->setItem((int)rowCount, (int)columnCount++, i < infosColor2.nbElementsObject(n) ? new QTableWidgetItem(QString::number(infosColor2.elementIObject(n, i) + 1)) : new QTableWidgetItem(""));
			//for (uint32_t i = 0; i < infosColor2.nbElementsObject(n); i++)
			//	m_tableInfos->setItem((int)rowCount, (int)columnCount++, new QTableWidgetItem(QString::number(infosColor2.elementIObject(n, i) + 1)));
		}

		m_delaunayTriangulationFilteringWidget->updateGeometry();

		if (bci->hasParameter("pointRendering")) {
			bool val = bci->getParameter<bool>("pointRendering");
			m_pointRenderButton->blockSignals(true);
			m_pointRenderButton->setChecked(val);
			m_pointRenderButton->blockSignals(false);
		}
		if (bci->hasParameter("shapeRendering")) {
			bool val = bci->getParameter<bool>("shapeRendering");
			m_shapeRenderButton->blockSignals(true);
			m_shapeRenderButton->setChecked(val);
			m_shapeRenderButton->blockSignals(false);
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

	}
}

void ObjectColocalizationWidget::executeMacro(poca::core::MyObjectInterface* _wobj, poca::core::CommandInfo* _ci)
{
	this->performAction(_wobj, _ci);
}