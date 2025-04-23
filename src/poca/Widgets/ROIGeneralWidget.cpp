/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ROIGeneralWidget.cpp
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

#include <QtWidgets/QGroupBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QColorDialog>
#include <fstream>
#include <sstream>
#include <array>

#include <Interfaces/MyObjectInterface.hpp>
#include <Interfaces/ROIInterface.hpp>
#include <General/CommandableObject.hpp>

#include "ROIGeneralWidget.hpp"

ROIGeneralWidget::ROIGeneralWidget(poca::core::MediatorWObjectFWidget* _mediator, QWidget* _parent/*= 0*/) :QGroupBox("ROIs")
{
	m_parentTab = (QTabWidget *)_parent;
	m_mediator = _mediator;

	this->setObjectName("ROIGeneralWidget");
	this->addActionToObserve("LoadObjCharacteristicsAllWidgets");
	this->addActionToObserve("LoadObjCharacteristicsROIWidget");
	this->addActionToObserve("addOneROI");
	this->addActionToObserve("UpdateSelectionROIs");

	//QWidget * generalWidget = new QWidget;

	QWidget * groupROIs = new QWidget();
	groupROIs->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	QLabel* pixXLbl = new QLabel("Pixel X/Y calibration:");
	pixXLbl->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_pixXYEdit = new QLineEdit("1");
	m_pixXYEdit->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);

	m_loadROIsBtn = new QPushButton("Load ROIs");
	m_loadROIsBtn->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_saveROIsBtn = new QPushButton("Save ROIs");
	m_saveROIsBtn->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_discardSelectedROIsBtn = new QPushButton("Discard selected ROIs");
	m_discardSelectedROIsBtn->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_discardAllROIsBtn = new QPushButton("Discard all ROIs");
	m_discardAllROIsBtn->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_cboxDisplayROIs = new QCheckBox("Display ROIs");
	m_cboxDisplayROIs->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_cboxDisplayROIs->setChecked(true);
	m_cboxDisplayLabelROIs = new QCheckBox("Display labels ROIs");
	m_cboxDisplayLabelROIs->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_cboxDisplayLabelROIs->setChecked(true);
	m_colorSelectedROILbl = new QLabel("Selected ROI color:");
	m_colorSelectedROILbl->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_colorSelectedROIBtn = new QPushButton();
	m_colorSelectedROIBtn->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_colorSelectedROIBtn->setStyleSheet("background-color: rgb(255, 0, 0);"
		"border-style: outset;"
		"border-width: 2px;"
		"border-radius: 5px;"
		"border-color: black;"
		"font: 12px;"
		"min-width: 5em;"
		"padding: 3px;"
	);
	m_colorUnselectedROILbl = new QLabel("Unselected ROI color:");
	m_colorUnselectedROILbl->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_colorUnselectedROIBtn = new QPushButton();
	m_colorUnselectedROIBtn->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_colorUnselectedROIBtn->setStyleSheet("background-color: rgb(255, 255, 0);"
		"border-style: outset;"
		"border-width: 2px;"
		"border-radius: 5px;"
		"border-color: black;"
		"font: 12px;"
		"min-width: 5em;"
		"padding: 3px;"
	);
	QGridLayout * layoutROI = new QGridLayout;
	layoutROI->addWidget(pixXLbl, 0, 0, 1, 1);
	layoutROI->addWidget(m_pixXYEdit, 0, 1, 1, 1);
	layoutROI->addWidget(m_loadROIsBtn, 1, 0, 1, 1);
	layoutROI->addWidget(m_saveROIsBtn, 1, 1, 1, 1);
	layoutROI->addWidget(m_discardSelectedROIsBtn, 1, 2, 1, 1);
	layoutROI->addWidget(m_discardAllROIsBtn, 1, 3, 1, 1);
	layoutROI->addWidget(m_cboxDisplayROIs, 2, 2, 1, 1);
	layoutROI->addWidget(m_cboxDisplayLabelROIs, 2, 3, 1, 1);
	layoutROI->addWidget(m_colorSelectedROILbl, 3, 0, 1, 1);
	layoutROI->addWidget(m_colorSelectedROIBtn, 3, 1, 1, 1);
	layoutROI->addWidget(m_colorUnselectedROILbl, 3, 2, 1, 1);
	layoutROI->addWidget(m_colorUnselectedROIBtn, 3, 3, 1, 1);

	groupROIs->setLayout(layoutROI);
	groupROIs->setVisible(true);

	m_tableW = new QTableWidget;
	m_tableW->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_tableW->setSelectionBehavior(QAbstractItemView::SelectRows);
	m_tableW->setSelectionMode(QAbstractItemView::MultiSelection);
	QStringList tableHeader2;
	tableHeader2 << "Name" << "Type" << "Area" << "Perimeter";
	m_tableW->setColumnCount(tableHeader2.size());
	m_tableW->setHorizontalHeaderLabels(tableHeader2);

	QVBoxLayout * layoutROIs = new QVBoxLayout;
	layoutROIs->addWidget(groupROIs);
	layoutROIs->addWidget(m_tableW);
	this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	this->setLayout(layoutROIs);

	//this->addTab(generalWidget, tr("General"));

	QObject::connect(m_loadROIsBtn, SIGNAL(clicked()), this, SLOT(actionNeeded()));
	QObject::connect(m_saveROIsBtn, SIGNAL(clicked()), this, SLOT(actionNeeded()));
	QObject::connect(m_discardSelectedROIsBtn, SIGNAL(clicked()), this, SLOT(actionNeeded()));
	QObject::connect(m_discardAllROIsBtn, SIGNAL(clicked()), this, SLOT(actionNeeded()));
	QObject::connect(m_colorSelectedROIBtn, SIGNAL(clicked()), this, SLOT(actionNeeded()));
	QObject::connect(m_colorUnselectedROIBtn, SIGNAL(clicked()), this, SLOT(actionNeeded()));
	QObject::connect(m_cboxDisplayROIs, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));
	QObject::connect(m_cboxDisplayLabelROIs, SIGNAL(toggled(bool)), this, SLOT(actionNeeded(bool)));

	QObject::connect(m_tableW, SIGNAL(itemClicked(QTableWidgetItem *)), this, SLOT(actionNeeded(QTableWidgetItem *)));
}

ROIGeneralWidget::~ROIGeneralWidget()
{

}

void ROIGeneralWidget::actionNeeded()
{ 
	QObject * sender = QObject::sender();
	if (sender == m_loadROIsBtn){
		QString dir(m_object->getDir().c_str());
		QString name = QFileDialog::getOpenFileName(this, QObject::tr("Load ROIs..."), dir, QObject::tr("ROIs files (*.txt *.rgn)"), 0, QFileDialog::DontUseNativeDialog);
		if (name.isEmpty()) return;
		std::cout << name.toLatin1().data() << std::endl;
		//m_object->loadROIs(name.toLatin1().data(), getXY());
		bool ok;
		float cal = m_pixXYEdit->text().toFloat(&ok);
		if (!ok) return;
		m_object->executeCommand(&poca::core::CommandInfo(true, "loadROIs", "filename", name.toStdString(), "calibrationXY", cal));
		m_object->notify("LoadObjCharacteristicsROIWidget");
	}
	else if (sender == m_saveROIsBtn){
		std::vector <poca::core::ROIInterface*>& ROIs = m_object->getROIs();
		if (ROIs.empty()) return;
		QString dir(m_object->getDir().c_str()), name(m_object->getName().c_str());
		name.replace(name.size() - 4, 4, "_rois.txt");
		if (!dir.endsWith("/")) dir.append("/");
		dir.append(name);
		name = QFileDialog::getSaveFileName(this, QObject::tr("Save ROIs..."), dir, QObject::tr("ROIs files (*.txt)"), 0, QFileDialog::DontUseNativeDialog);

		std::cout << name.toLatin1().data() << std::endl;
		m_object->executeCommand(&poca::core::CommandInfo(true, "saveROIs", "filename", name.toStdString()));
		//m_object->saveROIs(name.toLatin1().data());
	}
	else if (sender == m_discardSelectedROIsBtn){
		std::vector <poca::core::ROIInterface*>& ROIs = m_object->getROIs();
		std::vector < std::vector< poca::core::ROIInterface* >::iterator > toDelete;
		int cpt = 0;
		for (std::vector < poca::core::ROIInterface* >::iterator it = ROIs.begin(); it != ROIs.end(); it++, cpt++) {
			poca::core::ROIInterface* ROI = *it;
			if (ROI->selected())
				toDelete.push_back(it);
		}
		for (int n = toDelete.size() - 1; n >= 0; n--) {
			poca::core::ROIInterface* ROI = *toDelete[n];
			delete ROI;
			ROIs.erase(toDelete[n]);
		}
		loadROIsIntoTable(ROIs);
		m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_discardAllROIsBtn){
		std::vector <poca::core::ROIInterface*>& ROIs = m_object->getROIs();
		for (poca::core::ROIInterface* ROI : ROIs)
			delete ROI;
		ROIs.clear();
		loadROIsIntoTable(ROIs);
		m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_colorSelectedROIBtn) {
		std::array <unsigned char, 4> rgba{ 255, 255, 255, 255 };
		QColor colorTmp = QColorDialog::getColor(QColor((int)rgba[0], (int)rgba[1], (int)rgba[2]));
		if (colorTmp.isValid()) {
			rgba[0] = colorTmp.red();
			rgba[1] = colorTmp.green();
			rgba[2] = colorTmp.blue();
			m_object->executeCommand(&poca::core::CommandInfo(true, "colorSelectedROIs", rgba));
			setColorROIs(m_colorSelectedROIBtn, rgba[0], rgba[1], rgba[2]);
			m_object->notifyAll("updateDisplay");
		}
	}
	else if (sender == m_colorUnselectedROIBtn) {
		std::array <unsigned char, 4> rgba{ 0, 0, 0, 255 };
		QColor colorTmp = QColorDialog::getColor(QColor((int)rgba[0], (int)rgba[1], (int)rgba[2]));
		if (colorTmp.isValid()) {
			rgba[0] = colorTmp.red();
			rgba[1] = colorTmp.green();
			rgba[2] = colorTmp.blue();
			m_object->executeCommand(&poca::core::CommandInfo(true, "colorUnselectedROIs", rgba));
			setColorROIs(m_colorUnselectedROIBtn, rgba[0], rgba[1], rgba[2]);
			m_object->notifyAll("updateDisplay");
		}
	}
}

void ROIGeneralWidget::actionNeeded(bool _val)
{
	QObject* sender = QObject::sender();
	if (sender == m_cboxDisplayROIs) {
		m_object->executeCommand(&poca::core::CommandInfo(true, "displayROIs", _val));
		m_object->notifyAll("updateDisplay");
	}
	else if (sender == m_cboxDisplayLabelROIs) {
		m_object->executeCommand(&poca::core::CommandInfo(true, "displayROILabels", _val));
		m_object->notifyAll("updateDisplay");
	}
}

void ROIGeneralWidget::actionNeeded(QTableWidgetItem * _item)
{
	QObject * sender = QObject::sender();
	if (sender == m_tableW){
		m_object->resetROIsSelection();
		std::vector < int > selectionIndexes;
		QModelIndexList list = m_tableW->selectionModel()->selectedRows();
		std::vector <poca::core::ROIInterface*>& ROIs = m_object->getROIs();
		for (int n = 0; n < list.size(); n++)
			ROIs[list.at(n).row()]->setSelected(true);
		m_object->notifyAll("updateDisplay");
	}
}

void ROIGeneralWidget::loadROIsIntoTable(const std::vector <poca::core::ROIInterface*>& _ROIs)
{
	//Adding inside the tabWidget
	m_tableW->clear();
	QStringList tableHeader2;
	tableHeader2 << "Name" << "Type" << QString("Area") << QString("Perimeter");
	m_tableW->setColumnCount(tableHeader2.size());
	m_tableW->setHorizontalHeaderLabels(tableHeader2);
	m_tableW->setRowCount(_ROIs.size());

	int cpt = 0;
	for (std::vector <poca::core::ROIInterface*>::const_iterator it = _ROIs.begin(); it != _ROIs.end(); it++, cpt++){
		poca::core::ROIInterface* ROI = *it;
		int y = 0;
		m_tableW->setItem(cpt, y++, new QTableWidgetItem(ROI->getName().c_str()));
		m_tableW->setItem(cpt, y++, new QTableWidgetItem(ROI->getType().c_str()));
		float area = ROI->getFeature("area");
		m_tableW->setItem(cpt, y++, new QTableWidgetItem(QString::number(area)));
		float perimeter = ROI->getFeature("perimeter");
		m_tableW->setItem(cpt, y++, new QTableWidgetItem(QString::number(perimeter)));
	}
}

void ROIGeneralWidget::addLastROI(poca::core::MyObjectInterface* _obj)
{
	std::vector <poca::core::ROIInterface*>& ROIs = m_object->getROIs();

	poca::core::ROIInterface* ROI = ROIs.back();
	int size = m_tableW->rowCount(), y = 0;
	m_tableW->setRowCount(size + 1);
	m_tableW->setItem(size, y++, new QTableWidgetItem(ROI->getName().c_str()));
	m_tableW->setItem(size, y++, new QTableWidgetItem(ROI->getType().c_str()));
	double area = ROI->getFeature("area");
	m_tableW->setItem(size, y++, new QTableWidgetItem(QString::number(area)));
	double perimeter = ROI->getFeature("perimeter");
	m_tableW->setItem(size, y++, new QTableWidgetItem(QString::number(perimeter)));
}

void ROIGeneralWidget::performAction(poca::core::MyObjectInterface* _obj, poca::core::CommandInfo * _ci)
{
	if (_obj != NULL) {
		m_object = _obj;
		m_object->executeCommand(_ci);
		m_object->notifyAll("updateDisplay");
	}
}

void ROIGeneralWidget::update(poca::core::SubjectInterface* _subject, const poca::core::CommandInfo & _aspect)
{
	poca::core::MyObjectInterface* obj = dynamic_cast <poca::core::MyObjectInterface*> (_subject);
	if (obj == NULL) return;
	m_object = obj;
	if (_aspect == "LoadObjCharacteristicsAllWidgets" || _aspect == "LoadObjCharacteristicsROIWidget"){
		this->blockSignals(true);
		std::vector <poca::core::ROIInterface*>& ROIs = m_object->getROIs();
		this->loadROIsIntoTable(ROIs);
		this->updateSelectionROIsFromObject(obj);

		poca::core::CommandableObject* comObj = dynamic_cast <poca::core::CommandableObject*>(m_object);
		if (comObj) {
			if (comObj->hasParameter("colorSelectedROIs")) {
				std::array <unsigned char, 4> rgba = comObj->getParameter< std::array <unsigned char, 4>>("colorSelectedROIs");
				setColorROIs(m_colorSelectedROIBtn, rgba[0], rgba[1], rgba[2]);
			}
			if (comObj->hasParameter("colorUnselectedROIs")) {
				std::array <unsigned char, 4> rgba = comObj->getParameter< std::array <unsigned char, 4>>("colorUnselectedROIs");
				setColorROIs(m_colorUnselectedROIBtn, rgba[0], rgba[1], rgba[2]);
			}
		}
		this->blockSignals(false);
	}
	else if (_aspect == "addOneROI"){
		this->addLastROI(obj);
		this->updateSelectionROIsFromObject(obj);
	}
	else if (_aspect == "UpdateSelectionROIs")
		this->updateSelectionROIsFromObject(obj);
}

void ROIGeneralWidget::updateSelectionROIsFromObject(poca::core::MyObjectInterface* _obj)
{
	/*m_tableW->blockSignals(true);
	m_tableW->clearSelection();
	poca::core::CommandInfo ci("MyObject", "getSelectionROIs", std::vector < std::any >() = {});
	_obj->executeCommand(&ci);
	std::vector < int > * selectionROIs = std::any_cast< std::vector < int > * >(ci.second[0]);
	for (int n = 0; n < selectionROIs->size(); n++)
		m_tableW->selectRow((*selectionROIs)[n]);
	m_tableW->setFocus();
	m_tableW->blockSignals(false);*/
}

void ROIGeneralWidget::executeMacro(poca::core::MyObjectInterface* _wobj, poca::core::CommandInfo * _macro)
{
	this->performAction(_wobj, _macro);
}

void ROIGeneralWidget::setColorROIs(QPushButton* _btn, const unsigned char _r, const unsigned char _g, const unsigned char _b)
{
	_btn->setStyleSheet("background-color: rgb(" + QString::number((int)_r) + ", " + QString::number((int)_g) + ", " + QString::number((int)_b) + ");"
		"border-style: outset;"
		"border-width: 2px;"
		"border-radius: 5px;"
		"border-color: black;"
		"font: 12px;"
		"min-width: 5em;"
		"padding: 3px;"
	);
}

const float ROIGeneralWidget::getXY() const
{
	bool ok;
	float val = m_pixXYEdit->text().toFloat(&ok);
	return ok ? val : 1.f;
}