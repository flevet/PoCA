/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectListWidget.cpp
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
#include <General/CommandableObject.hpp>
#include <General/Histogram.hpp>
#include <Plot/Icons.hpp>
#include <Plot/Misc.h>

#include "ObjectListWidget.hpp"

static char* duplicateCentroidsIcon[] = {
	/* columns rows colors chars-per-pixel */
	"37 36 2 1 ",
	"  c None",
	". c black",
	/* pixels */
	"                                     ",
	"                                     ",
	"                                     ",
	"                                     ",
	"                   .                 ",
	"                   ..                ",
	"                   ...               ",
	"                   .....             ",
	"      ...................            ",
	"      ....................           ",
	"      ....................           ",
	"      ...................            ",
	"      ....         .....             ",
	"      ....         ...               ",
	"      ....         ..                ",
	"      ....         .                 ",
	"      ....                           ",
	"      ....                           ",
	"      ....                           ",
	"      .... ..                        ",
	"          ......                     ",
	"         .........                   ",
	"        .... .......                 ",
	"       ....    ......                ",
	"      ....       .....               ",
	"     ....          ....              ",
	"     ...      ...   ....             ",
	"     ...     .....   ....            ",
	"     ...     .....   ....            ",
	"     .....    ...     ....           ",
	"     ..........        ...           ",
	"      ..............  ...            ",
	"         ................            ",
	"              ...........            ",
	"                    ....             ",
	"                                     "
};


ObjectListWidget::ObjectListWidget(poca::core::MediatorWObjectFWidgetInterface* _mediator, QWidget* _parent)
{
	m_parentTab = (QTabWidget*)_parent;
	m_mediator = _mediator;

	this->setObjectName("ObjectListWidget");
	this->addActionToObserve("LoadObjCharacteristicsAllWidgets");
	this->addActionToObserve("LoadObjCharacteristicsObjectListWidget");
	this->addActionToObserve("UpdateHistogramObjectListWidget");

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
	m_lutButtons.push_back(std::make_pair(new QPushButton(), std::string("Random")));
	int maxSize = 20;
	for (size_t n = 0; n < m_lutButtons.size(); n++) {
		m_lutButtons[n].first->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
		m_lutButtons[n].first->setMaximumSize(QSize(maxSize, maxSize));
		QImage im = poca::core::generateImage(maxSize, maxSize, &poca::core::Palette::getStaticLut(m_lutButtons[n].second));
		QPixmap pix = QPixmap::fromImage(im);
		QIcon icon(pix);
		m_lutButtons[n].first->setIcon(icon);
		if(n < 9)
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

	m_outlinePointRenderButton = new QPushButton();
	m_outlinePointRenderButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_outlinePointRenderButton->setMaximumSize(QSize(maxSize, maxSize));
	m_outlinePointRenderButton->setIcon(QIcon(QPixmap(poca::plot::outlinePointRenderingIcon)));
	m_outlinePointRenderButton->setToolTip("Render outline points");
	m_outlinePointRenderButton->setCheckable(true);
	m_outlinePointRenderButton->setChecked(true);
	layoutLuts->addWidget(m_outlinePointRenderButton, 0, Qt::AlignRight);
	QObject::connect(m_outlinePointRenderButton, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));

	m_ellipsoidRenderButton = new QPushButton();
	m_ellipsoidRenderButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_ellipsoidRenderButton->setMaximumSize(QSize(maxSize, maxSize));
	m_ellipsoidRenderButton->setIcon(QIcon(QPixmap(poca::plot::sphereIcon)));
	m_ellipsoidRenderButton->setToolTip("Render size as ellipsoid");
	m_ellipsoidRenderButton->setCheckable(true);
	m_ellipsoidRenderButton->setChecked(true);
	layoutLuts->addWidget(m_ellipsoidRenderButton, 0, Qt::AlignRight);
	QObject::connect(m_ellipsoidRenderButton, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));

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

	m_duplicateCentroidsButton = new QPushButton();
	m_duplicateCentroidsButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_duplicateCentroidsButton->setMaximumSize(QSize(maxSize, maxSize));
	m_duplicateCentroidsButton->setIcon(QIcon(QPixmap("./images/duplicate.png")));
	m_duplicateCentroidsButton->setToolTip("Duplicate centroids");
	layoutButtons->addWidget(m_duplicateCentroidsButton, 0, Qt::AlignRight);
	QObject::connect(m_duplicateCentroidsButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));

	m_duplicateSelectedObjectsButton = new QPushButton();
	m_duplicateSelectedObjectsButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_duplicateSelectedObjectsButton->setMaximumSize(QSize(maxSize, maxSize));
	m_duplicateSelectedObjectsButton->setIcon(QIcon(QPixmap("./images/duplicate.png")));
	m_duplicateSelectedObjectsButton->setToolTip("Duplicate selected objects");
	layoutButtons->addWidget(m_duplicateSelectedObjectsButton, 0, Qt::AlignRight);
	QObject::connect(m_duplicateSelectedObjectsButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));

	m_exportButton = new QPushButton();
	m_exportButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_exportButton->setMaximumSize(QSize(maxSize, maxSize));
	m_exportButton->setIcon(QIcon(QPixmap(poca::plot::exportIcon)));
	m_exportButton->setToolTip("Save stats objects");
	layoutButtons->addWidget(m_exportButton, 0, Qt::AlignRight);
	QObject::connect(m_exportButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));

	m_exportLocsButton = new QPushButton();
	m_exportLocsButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_exportLocsButton->setMaximumSize(QSize(maxSize, maxSize));
	m_exportLocsButton->setIcon(QIcon(QPixmap(poca::plot::exportIcon)));
	m_exportLocsButton->setToolTip("Save locs objects");
	layoutButtons->addWidget(m_exportLocsButton, 0, Qt::AlignRight);
	QObject::connect(m_exportLocsButton, SIGNAL(pressed()), this, SLOT(actionNeeded()));

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

	m_emptyWidget = new QWidget;
	m_emptyWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

	QVBoxLayout* layout = new QVBoxLayout;
	layout->setContentsMargins(1, 1, 1, 1);
	layout->setSpacing(1);
	layout->addWidget(m_lutsWidget);
	layout->addWidget(m_buttonsWidget);
	layout->addWidget(m_delaunayTriangulationFilteringWidget);
	layout->addWidget(m_tableObjects);
	this->setLayout(layout);
}

ObjectListWidget::~ObjectListWidget()
{

}

void ObjectListWidget::actionNeeded()
{
	poca::core::MyObjectInterface* obj = m_object->currentObject();
	poca::core::BasicComponentInterface* bc = obj->getBasicComponent("ObjectList");
	if (!bc) return;
	poca::core::CommandableObject* objList = dynamic_cast <poca::core::CommandableObject*>(bc);

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
			for(poca::plot::FilterHistogramWidget* histW : m_histWidgets)
				histW->redraw();
			m_object->notifyAll("updateDisplay");
		}
	}
	if (sender == m_hilowButton.first) {
		objList->executeCommand(true, "changeLUT", "LUT", m_hilowButton.second);
		for (poca::plot::FilterHistogramWidget* histW : m_histWidgets)
			histW->redraw();
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if(sender == m_exportButton){
		QString filename, separator(","), extension(".csv");
		filename = m_object->getDir().c_str();
		if (!filename.endsWith("/")) filename.append("/");
		filename.append(m_object->getName().c_str());
		if (filename.contains("."))
			filename = filename.mid(0, filename.lastIndexOf("."));
		filename.append("_statsObjs").append(extension);
		filename = QFileDialog::getSaveFileName(this, QObject::tr("Save stats..."), filename, QObject::tr("Stats files (*.csv)"), 0, QFileDialog::DontUseNativeDialog);
		objList->executeCommand(true, "saveStatsObjs", "filename", filename.toStdString(), "separator", separator.toStdString());
	}
	else if (sender == m_exportLocsButton) {
		QString filename, separator(","), extension(".csv");
		filename = m_object->getDir().c_str();
		if (!filename.endsWith("/")) filename.append("/");
		filename.append(m_object->getName().c_str());
		if (filename.contains("."))
			filename = filename.mid(0, filename.lastIndexOf("."));
		filename.append("_locsObjs").append(extension);
		filename = QFileDialog::getSaveFileName(this, QObject::tr("Save locs objects..."), filename, QObject::tr("Info files (*.csv)"), 0, QFileDialog::DontUseNativeDialog);
		objList->executeCommand(true, "saveLocsObjs", "filename", filename.toStdString(), "separator", separator.toStdString());
	}
	else if(sender == m_tableObjects){
		std::set <int> selectedRows;
		QList<QTableWidgetSelectionRange> ranges = m_tableObjects->selectedRanges();
		for (QTableWidgetSelectionRange range : ranges)
			for (int n = 0; n < range.rowCount(); n++)
				selectedRows.insert(range.topRow() + n);
		if (selectedRows.empty()) return;
		if (selectedRows.size() == 1) {
			objList->executeCommand(false, "setIDObjectPicked", *selectedRows.begin());
			m_object->notify("updateInfosObject");
			m_object->notifyAll("updateDisplay");
		}
		else {
			if (m_selectionButton->isChecked())
				m_selectionButton->setChecked(false);
			std::vector <bool> selection(m_tableObjects->rowCount(), false);
			for (int idx : selectedRows)
				selection[idx] = true;
			poca::core::MyObjectInterface* obj = m_object->currentObject();
			poca::core::BasicComponentInterface* bci = obj->getBasicComponent("ObjectList");
			if (bci == NULL) return;
			bci->setSelection(selection);
			objList->executeCommand(false, "setIDObjectPicked", -1);
			objList->executeCommand(false, "updateFeature");
			m_object->notifyAll("updateDisplay");
		}
	}
	else if (sender == m_duplicateCentroidsButton) {
		poca::core::CommandInfo ci(true, "duplicateCentroids");
		objList->executeCommand(&ci);
		if (ci.hasParameter("object")) {
			poca::core::MyObjectInterface* obj = ci.getParameterPtr<poca::core::MyObjectInterface>("object");
			emit(transferNewObjectCreated(obj));
		}
	}
	else if (sender == m_duplicateSelectedObjectsButton) {
		std::set <int> selectedRows;
		QList<QTableWidgetSelectionRange> ranges = m_tableObjects->selectedRanges();
		for (QTableWidgetSelectionRange range : ranges)
			for (int n = 0; n < range.rowCount(); n++)
				selectedRows.insert(range.topRow() + n);
		if (selectedRows.empty()) return;
		poca::core::CommandInfo ci(true, "duplicateSelectedObjects", "selection", selectedRows);
		objList->executeCommand(&ci);
		if (ci.hasParameter("object")) {
			poca::core::MyObjectInterface* obj = ci.getParameterPtr<poca::core::MyObjectInterface>("object");
			emit(transferNewObjectCreated(obj));
		}
	}
}

void ObjectListWidget::actionNeeded(int _val)
{
	poca::core::MyObjectInterface* obj = m_object->currentObject();
	poca::core::BasicComponentInterface* bc = obj->getBasicComponent("ObjectList");
	if (!bc) return;
	poca::core::CommandableObject* comObj = dynamic_cast <poca::core::CommandableObject*>(bc);

	QObject* sender = QObject::sender();
	if (sender == m_sizePointSpn) {
		unsigned int valD = this->pointSize();
		comObj->executeCommand(true, "pointSizeGL", valD);
		m_object->notifyAll("updateDisplay");
	}
}

void ObjectListWidget::actionNeeded(bool _val)
{
	poca::core::MyObjectInterface* obj = m_object->currentObject();
	poca::core::BasicComponentInterface* bc = obj->getBasicComponent("ObjectList");
	if (!bc) return;
	poca::core::CommandableObject* objList = dynamic_cast <poca::core::CommandableObject*>(bc);

	QObject* sender = QObject::sender();
	if (sender == m_displayButton) {
		objList->executeCommand(true, "selected", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_fillButton) {
		objList->executeCommand(true, "fill", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_pointRenderButton) {
		objList->executeCommand(true, "pointRendering", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_outlinePointRenderButton) {
		objList->executeCommand(true, "outlinePointRendering", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_shapeRenderButton) {
		objList->executeCommand(true, "shapeRendering", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_bboxSelectionButton) {
		objList->executeCommand(true, "bboxSelection", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
	else if (sender == m_selectionButton) {
		objList->executeCommand(true, "togglePicking", _val);
		return;
	}
	else if (sender == m_ellipsoidRenderButton) {
		objList->executeCommand(true, "ellipsoidRendering", _val);
		m_object->notifyAll("updateDisplay");
		return;
	}
}

void ObjectListWidget::performAction(poca::core::MyObjectInterface* _obj, poca::core::CommandInfo* _ci)
{
	if (_obj == NULL) {
		update(NULL, "");
		return;
	}
	poca::core::MyObjectInterface* obj = _obj->currentObject();
	bool actionDone = false;
	if (_ci->nameCommand == "histogram" || _ci->nameCommand == "changeLUT" || _ci->nameCommand == "selected" || _ci->nameCommand == "fill" || _ci->nameCommand == "hilow" || _ci->nameCommand == "pointRendering" || _ci->nameCommand == "shapeRendering") {
		if (_ci->nameCommand == "histogram") {
			std::string action = _ci->getParameter<std::string>("action");
			if (action == "save")
				_ci->addParameter("dir", _obj->getDir());
		}	
		poca::core::BasicComponentInterface* bc = obj->getBasicComponent("ObjectList");
		bc->executeCommand(_ci);
		actionDone = true;
	}
	else if (_ci->nameCommand == "selectObject") {
		int id = _ci->getParameter<int>("id");
		m_tableObjects->selectRow(id);
	}
	else if (_ci->nameCommand == "updatePickedObject") {
		poca::core::CommandInfo ci3(false, "getObjectPickedID");
		poca::core::BasicComponentInterface* bc = obj->getBasicComponent("ObjectList");
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
		_obj->notifyAll("LoadObjCharacteristicsObjectListWidget");
		_obj->notifyAll("updateDisplay");
	}
}

void ObjectListWidget::update(poca::core::SubjectInterface* _subject, const poca::core::CommandInfo& _aspect)
{
	poca::core::MyObjectInterface* obj = dynamic_cast <poca::core::MyObjectInterface*> (_subject);
	poca::core::MyObjectInterface* objOneColor = obj->currentObject();
	m_object = obj;

	bool visible = (objOneColor != NULL && objOneColor->hasBasicComponent("ObjectList"));
	QTabWidget * tab = m_parentTab->findChild <QTabWidget*>("ObjectList");
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
	auto index = m_parentTab->currentIndex();
	m_parentTab->setTabVisible(m_parentTab->indexOf(this), visible);
	m_parentTab->setCurrentIndex(index);
#endif

	if (_aspect == "LoadObjCharacteristicsAllWidgets" || _aspect == "LoadObjCharacteristicsObjectListWidget") {

		poca::core::BasicComponentInterface* bci = objOneColor->getBasicComponent("ObjectList");
		if (!bci) return;
		poca::core::stringList nameData = bci->getNameData();

		QVBoxLayout* layout = NULL;
		//First time we load data -> no hist widget was created before
		if (m_histWidgets.empty()) {
			layout = new QVBoxLayout;
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
			poca::core::HistogramInterface* hist = bci->getHistogram(type);
			if (hist != NULL)
				m_histWidgets[cpt++]->setInfos(type.c_str(), hist, bci->isLogHistogram(type), bci->isCurrentHistogram(type) ? bci->getPalette() : NULL);
		}

		m_tableObjects->setSortingEnabled(false);
		QStringList tableHeader2;
		tableHeader2 << "ID";
		for (std::string type : nameData) {
			tableHeader2 << type.c_str();
		}
		m_tableObjects->setColumnCount(tableHeader2.size());
		m_tableObjects->setHorizontalHeaderLabels(tableHeader2);
		m_tableObjects->setRowCount((int)bci->nbElements());
		int columnCount = 0;
		for (size_t rowCount = 0; rowCount < bci->nbElements(); rowCount++)
			m_tableObjects->setItem((int)rowCount, (int)columnCount, new SortableFloatItem(QString::number(rowCount + 1)));
		columnCount++;
		for (std::string type : nameData) {
			poca::core::Histogram<float>* hist = dynamic_cast<poca::core::Histogram<float>*>(bci->getOriginalHistogram(type));
			if (hist == NULL) continue;
			const std::vector <float>& values = hist->getValues();
			for (size_t rowCount = 0; rowCount < values.size(); rowCount++)
				m_tableObjects->setItem((int)rowCount, (int)columnCount, new SortableFloatItem(QString::number(values[rowCount])));
			columnCount++;
		}

		QHeaderView* headerGoods = m_tableObjects->horizontalHeader();
		//SortIndicator is a triangle indicator next to the horizontal title bar text
		headerGoods->setSortIndicator(0, Qt::AscendingOrder);
		headerGoods->setSortIndicatorShown(true);
		m_tableObjects->setSortingEnabled(true);

		m_delaunayTriangulationFilteringWidget->updateGeometry();

		if (bci->hasParameter("pointRendering")) {
			bool val = bci->getParameter<bool>("pointRendering");
			m_pointRenderButton->blockSignals(true);
			m_pointRenderButton->setChecked(val);
			m_pointRenderButton->blockSignals(false);
		}
		if (bci->hasParameter("outlinePointRendering")) {
			bool val = bci->getParameter<bool>("outlinePointRendering");
			m_outlinePointRenderButton->blockSignals(true);
			m_outlinePointRenderButton->setChecked(val);
			m_outlinePointRenderButton->blockSignals(false);
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

		if (bci->hasParameter("pointSizeGL")) {
			uint32_t val = bci->getParameter<uint32_t>("pointSizeGL");
			m_sizePointSpn->blockSignals(true);
			m_sizePointSpn->setValue(val);
			m_sizePointSpn->blockSignals(false);
		}

		if (bci->hasParameter("ellipsoidRendering")) {
			bool val = bci->getParameter<bool>("ellipsoidRendering");
			m_ellipsoidRenderButton->blockSignals(true);
			m_ellipsoidRenderButton->setChecked(val);
			m_ellipsoidRenderButton->blockSignals(false);
		}

		bool selected = bci->isSelected();
		m_displayButton->blockSignals(true);
		m_displayButton->setChecked(selected);
		m_displayButton->blockSignals(false);
	}
}

void ObjectListWidget::executeMacro(poca::core::MyObjectInterface* _wobj, poca::core::CommandInfo* _ci)
{
	this->performAction(_wobj, _ci);
}

