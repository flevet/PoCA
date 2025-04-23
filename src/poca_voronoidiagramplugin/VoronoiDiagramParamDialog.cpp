/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      VoronoiDiagramParamDialog.cpp
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
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QGroupBox>

#include "VoronoiDiagramParamDialog.hpp"

VoronoiDiagramParamDialog::VoronoiDiagramParamDialog(const bool _showTab, const bool _createCells, const bool _sphere, QWidget * _parent, Qt::WindowFlags _f) :QDialog(_parent, _f)
{
	QGroupBox* gBoxShow = new QGroupBox(tr("Misc"));
	gBoxShow->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_cboxDisplay = new QCheckBox("Show Voronoi tab");
	m_cboxDisplay->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_cboxDisplay->setChecked(_showTab);
	m_cboxVoronoiOnSphere = new QCheckBox("Voronoi on sphere");
	m_cboxVoronoiOnSphere->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_cboxVoronoiOnSphere->setChecked(_sphere);
	QHBoxLayout* ltab = new QHBoxLayout;
	ltab->addWidget(m_cboxDisplay);
#ifndef NO_PYTHON
	ltab->addWidget(m_cboxVoronoiOnSphere);
#endif
	gBoxShow->setLayout(ltab);

	QGroupBox* gBoxCells = new QGroupBox(tr("Cells"));
	gBoxCells->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_cbox = new QCheckBox("Create cells");
	m_cbox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_cbox->setChecked(_createCells);
	QHBoxLayout* lcells = new QHBoxLayout;
	lcells->addWidget(m_cbox);
	gBoxCells->setLayout(lcells);

	QPushButton * closeBtn = new QPushButton("Ok", this);
	QPushButton * cancelBtn = new QPushButton("Cancel", this);
	QHBoxLayout * layoutButton = new QHBoxLayout;
	layoutButton->addWidget(closeBtn);
	layoutButton->addWidget(cancelBtn);
	QWidget * widButton = new QWidget;
	widButton->setLayout(layoutButton);

	QVBoxLayout * layout = new QVBoxLayout;
	layout->addWidget(gBoxShow);
	layout->addWidget(gBoxCells);
	layout->addWidget(widButton);

	this->setLayout(layout);
	this->setWindowTitle("Parameters");
	QPoint p = QCursor::pos();
	this->setGeometry(p.x(), p.y(), sizeHint().width(), sizeHint().height());

	QObject::connect(closeBtn, SIGNAL(clicked()), this, SLOT(accept()));
	QObject::connect(cancelBtn, SIGNAL(clicked()), this, SLOT(reject()));
}

VoronoiDiagramParamDialog::~VoronoiDiagramParamDialog()
{

}
