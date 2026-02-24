/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MergeDatasetsChoiceDialog.cpp
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

#include <QtWidgets/QPushButton>
#include <QtWidgets/QBoxLayout>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <QtCore/QString>
#include <QtWidgets/QComboBox>

#include <iostream>

#include "MergeDatasetsChoiceDialog.hpp"

MergeDatasetsChoiceDialog::MergeDatasetsChoiceDialog(const std::vector < std::pair < QString, MdiChild* > > & _datasets, QWidget * _parent, Qt::WindowFlags _f) :QDialog(_parent, _f), m_datasets(_datasets)
{
	// Create widgets
	m_allDatasetsList = new QListWidget;
	m_datasetsToMergeList = new QListWidget;
	QPushButton* transferButton = new QPushButton("Transfer Selected");
	m_cboxGrid = new QCheckBox("Grid");
	m_cboxGrid->setChecked(true);

	// Fill left list with some example data
	for (int i = 0; i < (int)m_datasets.size(); ++i) {
		auto* it = new QListWidgetItem(m_datasets[i].first);
		it->setData(Qt::UserRole, i);          // store original index
		m_allDatasetsList->addItem(it);
	}

	// Enable multi-selection (Ctrl/Maj)
	m_allDatasetsList->setSelectionMode(QAbstractItemView::ExtendedSelection);

	QPushButton* closeBtn = new QPushButton("Ok", this);
	QPushButton* cancelBtn = new QPushButton("Cancel", this);
	QHBoxLayout* layoutButton = new QHBoxLayout;
	layoutButton->addWidget(closeBtn);
	layoutButton->addWidget(cancelBtn);

	// Layout
	QHBoxLayout* hLayout = new QHBoxLayout;
	QVBoxLayout* vLayout = new QVBoxLayout;
	QVBoxLayout* leftLayout = new QVBoxLayout;
	QVBoxLayout* rightLayout = new QVBoxLayout;

	leftLayout->addWidget(m_allDatasetsList);
	leftLayout->addWidget(transferButton);
	leftLayout->addWidget(m_cboxGrid);
	rightLayout->addWidget(m_datasetsToMergeList);

	hLayout->addLayout(leftLayout);
	hLayout->addLayout(rightLayout);

	vLayout->addLayout(hLayout);
	vLayout->addLayout(layoutButton);

	// Connect button
	QObject::connect(transferButton, &QPushButton::clicked, [&]() {
		QList<QListWidgetItem*> selectedItems = m_allDatasetsList->selectedItems();

		if (!selectedItems.empty()) {
			for (QListWidgetItem* item : selectedItems) {
				int row = m_allDatasetsList->row(item);
				QListWidgetItem* taken = m_allDatasetsList->takeItem(row); // removes without deleting
				m_datasetsToMergeList->addItem(taken);                      // reuses same item (keeps UserRole)
			}
		}
		else {
			while (m_allDatasetsList->count() > 0) {
				QListWidgetItem* taken = m_allDatasetsList->takeItem(0);
				m_datasetsToMergeList->addItem(taken);
			}
		}
	});

	this->setLayout(vLayout);
	this->setWindowTitle("Merge datasets");
	QPoint p = QCursor::pos();
	this->setGeometry(p.x(), p.y(), sizeHint().width(), sizeHint().height());

	QObject::connect(closeBtn, SIGNAL(clicked()), this, SLOT(accept()));
	QObject::connect(cancelBtn, SIGNAL(clicked()), this, SLOT(reject()));
}

MergeDatasetsChoiceDialog::~MergeDatasetsChoiceDialog()
{

}

std::vector <MdiChild*> MergeDatasetsChoiceDialog::getObjects() const
{
	std::vector<MdiChild*> data;
	data.reserve(m_datasetsToMergeList->count());

	for (int i = 0; i < m_datasetsToMergeList->count(); ++i) {
		QListWidgetItem* item = m_datasetsToMergeList->item(i);
		const int idx = item->data(Qt::UserRole).toInt();

		if (0 <= idx && idx < (int)m_datasets.size())
			data.push_back(m_datasets[idx].second);
	}

	return data;
}

