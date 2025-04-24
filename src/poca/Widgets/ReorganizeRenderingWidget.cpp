/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ReorganizeRenderingWidget.cpp
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
#include <QtWidgets/QMessageBox>
#include <iostream>
#include <fstream>

#include <Interfaces/MyObjectInterface.hpp>
#include <Interfaces/CommandableObjectInterface.hpp>
#include <General/BasicComponent.hpp>
#include <General/CommandableObject.hpp>
#include <General/Histogram.hpp>
#include <General/MyData.hpp>

#include "ReorganizeRenderingWidget.hpp"

ReorganizeRenderingWidget::ReorganizeRenderingWidget(poca::core::MediatorWObjectFWidgetInterface* _mediator, QWidget* _parent) :QGroupBox(tr("ReorganizeRendering"))
{
	m_parentTab = (QTabWidget*)_parent;
	m_mediator = _mediator;
	m_object = NULL;

	this->setObjectName("ReorganizeRenderingWidget");
	this->addActionToObserve("LoadObjCharacteristicsAllWidgets");
	this->addActionToObserve("LoadObjCharacteristicsReorganizeRenderingWidget");

	QFont fontLegend("Helvetica", 9);
	fontLegend.setBold(true);
	QColor background = QWidget::palette().color(QWidget::backgroundRole());
	int maxSize = 20;

	//QGroupBox* m_FitEllipsoidGBox = new QGroupBox(tr("K-Ripley"));
	this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	m_basicComponentsList = new QListWidget;
	m_basicComponentsList->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_commandsList = new QListWidget;
	m_commandsList->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	QGridLayout* layout = new QGridLayout;
	layout->addWidget(m_basicComponentsList, 0, 0, 1, 1);
	layout->addWidget(m_commandsList, 0, 1, 1, 1);
	this->setLayout(layout);
	this->setMinimumHeight(150);
	this->setMaximumHeight(800);
	//this->setLayout(layoutFitEllipsoid);
	//this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
	m_basicComponentsList->setDragDropMode(QAbstractItemView::InternalMove);
	m_commandsList->setDragDropMode(QAbstractItemView::InternalMove);
	QObject::connect(m_basicComponentsList, SIGNAL(currentItemChanged(QListWidgetItem *, QListWidgetItem *)), this, SLOT(updateCommands(QListWidgetItem *, QListWidgetItem *)));
	QObject::connect(m_basicComponentsList->model(), SIGNAL(rowsMoved(const QModelIndex&, int, int, const QModelIndex&, int)), this, SLOT(rowsMoved(const QModelIndex&, int, int, const QModelIndex&, int)));
	QObject::connect(m_commandsList->model(), SIGNAL(rowsMoved(const QModelIndex&, int, int, const QModelIndex&, int)), this, SLOT(rowsMoved(const QModelIndex&, int, int, const QModelIndex&, int)));
}

ReorganizeRenderingWidget::~ReorganizeRenderingWidget()
{

}

void ReorganizeRenderingWidget::actionNeeded()
{
	QObject* sender = QObject::sender();
	bool found = false;
}

void ReorganizeRenderingWidget::actionNeeded(int _idx)
{
	if (_idx == -1) return;
	QObject* sender = QObject::sender();
	bool found = false;
}

void ReorganizeRenderingWidget::performAction(poca::core::MyObjectInterface* _obj, poca::core::CommandInfo* _ci)
{
}

void ReorganizeRenderingWidget::update(poca::core::SubjectInterface* _subject, const poca::core::CommandInfo& _aspect)
{
	poca::core::MyObjectInterface* obj = dynamic_cast <poca::core::MyObjectInterface*> (_subject);
	if (obj == NULL)
		return;
	obj = obj->currentObject();
	if (obj == NULL)
		return;
	//if (!obj->hasBasicComponent("DetectionSet"))
	//	return;

	m_basicComponentsList->clear();
	m_commandsList->clear();

	m_object = obj;

	if (m_object->nbBasicComponents() < 1) return;

	if (_aspect == "LoadObjCharacteristicsAllWidgets" || _aspect == "LoadObjCharacteristicsReorganizeRenderingWidget") {

		for (size_t n = 0; n < m_object->nbBasicComponents(); n++)
			m_basicComponentsList->addItem(m_object->getBasicComponent(n)->getName().c_str());
		poca::core::BasicComponentInterface* bci = obj->getComponents()[0];
		for(auto command : bci->getCommands())
			m_commandsList->addItem(command->name().c_str());
	}
}

void ReorganizeRenderingWidget::executeMacro(poca::core::MyObjectInterface* _wobj, poca::core::CommandInfo* _ci)
{
	this->performAction(_wobj, _ci);
}

void ReorganizeRenderingWidget::updateCommands(QListWidgetItem* current, QListWidgetItem* previous)
{
	if (current == NULL) return;
	poca::core::MyObjectInterface* obj = m_object->currentObject();
	if (obj == NULL)
		return;
	poca::core::BasicComponentInterface* bci = obj->getBasicComponent(current->text().toStdString());
	m_commandsList->clear();
	for (auto command : bci->getCommands())
		m_commandsList->addItem(command->name().c_str());
}

void ReorganizeRenderingWidget::rowsMoved(const QModelIndex& sourceParent, int sourceStart, int sourceEnd, const QModelIndex& destinationParent, int destinationRow)
{
	poca::core::MyObjectInterface* obj = m_object->currentObject();
	if (obj == NULL)
		return;
	std::cout << sourceStart << ", " << sourceEnd << ", " << destinationRow << std::endl;
	if (destinationRow > sourceEnd)
		destinationRow = destinationRow - 1;
	QObject* sender = QObject::sender();
	if (sender == m_basicComponentsList->model()) {
		obj->reorganizeComponents(sourceEnd, destinationRow);
	}
	if (sender == m_commandsList->model()) {
		poca::core::BasicComponentInterface* bci = NULL;
		QListWidgetItem* current = m_basicComponentsList->currentItem();
		if (current == NULL)
			bci = obj->getComponents()[0];
		else
			bci = obj->getBasicComponent(current->text().toStdString());
		bci->reorganizeCommands(sourceEnd, destinationRow);
	}
	m_object->notify("LoadObjCharacteristicsReorganizeRenderingWidget");
	m_object->notifyAll("updateDisplay");
}

