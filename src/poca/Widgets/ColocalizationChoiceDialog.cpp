/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ColocalizationChoiceDialog.cpp
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

#include <QtWidgets/QPushButton>
#include <QtWidgets/QBoxLayout>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <QtCore/QString>
#include <QtWidgets/QComboBox>

#include <iostream>

#include "ColocalizationChoiceDialog.hpp"

ColocalizationChoiceDialog::ColocalizationChoiceDialog(const std::vector < std::pair < QString, MdiChild* > > & _datasets, QWidget * _parent, Qt::WindowFlags _f) :QDialog(_parent, _f), m_datasets(_datasets)
{
	std::vector <QLabel*> labels;
	QLabel * label = new QLabel("Dataset 1:");
	label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	QComboBox * comboDat = new QComboBox;
	for (unsigned int n = 0; n < _datasets.size(); n++)
		comboDat->addItem(_datasets.at(n).first);
	comboDat->setCurrentIndex(0);
	comboDat->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	labels.push_back(label);
	m_comboDats.push_back(comboDat);

	label = new QLabel("Dataset 2:");
	label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	comboDat = new QComboBox;
	for (unsigned int n = 0; n < _datasets.size(); n++)
		comboDat->addItem(_datasets.at(n).first);
	comboDat->setCurrentIndex(1);
	comboDat->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	labels.push_back(label);
	m_comboDats.push_back(comboDat);

	label = new QLabel("Dataset 3:");
	label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	comboDat = new QComboBox;
	for (unsigned int n = 0; n < _datasets.size(); n++)
		comboDat->addItem(_datasets.at(n).first);
	comboDat->addItem("");
	comboDat->setCurrentIndex(_datasets.size());
	comboDat->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	labels.push_back(label);
	m_comboDats.push_back(comboDat);

	label = new QLabel("Dataset 4:");
	label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	comboDat = new QComboBox;
	for (unsigned int n = 0; n < _datasets.size(); n++)
		comboDat->addItem(_datasets.at(n).first);
	comboDat->addItem("");
	comboDat->setCurrentIndex(_datasets.size());
	comboDat->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	labels.push_back(label);
	m_comboDats.push_back(comboDat);

	QGridLayout * layoutData = new QGridLayout;
	for (size_t n = 0; n < m_comboDats.size(); n++) {
		layoutData->addWidget(labels[n], n, 0, 1, 1);
		layoutData->addWidget(m_comboDats[n], n, 1, 1, 1);
	}
	QWidget * widDatasets = new QWidget;
	widDatasets->setLayout(layoutData);

	QPushButton * closeBtn = new QPushButton("Ok", this);
	QPushButton * cancelBtn = new QPushButton("Cancel", this);
	QHBoxLayout * layoutButton = new QHBoxLayout;
	layoutButton->addWidget(closeBtn);
	layoutButton->addWidget(cancelBtn);
	QWidget * widButton = new QWidget;
	widButton->setLayout(layoutButton);

	QVBoxLayout * layout = new QVBoxLayout;
	layout->addWidget(widDatasets);
	layout->addWidget(widButton);

	this->setLayout(layout);
	this->setWindowTitle("Colocalization");
	QPoint p = QCursor::pos();
	this->setGeometry(p.x(), p.y(), sizeHint().width(), sizeHint().height());

	QObject::connect(closeBtn, SIGNAL(clicked()), this, SLOT(accept()));
	QObject::connect(cancelBtn, SIGNAL(clicked()), this, SLOT(reject()));
}

ColocalizationChoiceDialog::~ColocalizationChoiceDialog()
{

}

void ColocalizationChoiceDialog::changeChosenDataset(int _index)
{
}

MdiChild* ColocalizationChoiceDialog::getIdObject(const unsigned int _whichObject) const
{
	if (_whichObject >= m_comboDats.size()) return NULL;
	return m_datasets.at(m_comboDats[_whichObject]->currentIndex()).second;
}

const uint32_t ColocalizationChoiceDialog::nbColors() const
{
	uint32_t nbColors = 0;
	for (QComboBox* c : m_comboDats) {
		if (!c->currentText().isEmpty())
			nbColors++;
	}
	return nbColors;
}

std::vector <MdiChild*> ColocalizationChoiceDialog::getObjects() const
{
	std::vector < MdiChild*> data;
	for (QComboBox* c : m_comboDats) {
		std::cout << c->currentText().toLatin1().data() << " - " << c->currentIndex() << std::endl;
		if (c->currentIndex() < m_datasets.size())
			data.push_back(m_datasets.at(c->currentIndex()).second);
	}
	return data;
}

