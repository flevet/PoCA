/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ColocalizationChoiceDialog.cpp
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
#include <QtWidgets/QButtonGroup>

#include "DelaunayTriangulationParamDialog.hpp"

DelaunayTriangulationParamDialog::DelaunayTriangulationParamDialog(const bool _3D, const bool _sphere, QWidget * _parent, Qt::WindowFlags _f) :QDialog(_parent, _f)
{
	QGroupBox* gBoxShow = new QGroupBox(tr("Delaunay"));
	gBoxShow->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_cboxDelaunay3D = new QCheckBox("Delaunay3D");
	m_cboxDelaunay3D->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_cboxDelaunay3D->setChecked(_3D);
	m_cboxDelaunayOnSphere = new QCheckBox("Delaunay on sphere");
	m_cboxDelaunayOnSphere->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_cboxDelaunayOnSphere->setChecked(_sphere);
	QButtonGroup* group = new QButtonGroup;
	group->addButton(m_cboxDelaunay3D);
	group->addButton(m_cboxDelaunayOnSphere);
	QHBoxLayout* ltab = new QHBoxLayout;
	ltab->addWidget(m_cboxDelaunay3D);
	ltab->addWidget(m_cboxDelaunayOnSphere);
	gBoxShow->setLayout(ltab);

	QPushButton * closeBtn = new QPushButton("Ok", this);
	QPushButton * cancelBtn = new QPushButton("Cancel", this);
	QHBoxLayout * layoutButton = new QHBoxLayout;
	layoutButton->addWidget(closeBtn);
	layoutButton->addWidget(cancelBtn);
	QWidget * widButton = new QWidget;
	widButton->setLayout(layoutButton);

	QVBoxLayout * layout = new QVBoxLayout;
	layout->addWidget(gBoxShow);
	layout->addWidget(widButton);

	this->setLayout(layout);
	this->setWindowTitle("Parameters");
	QPoint p = QCursor::pos();
	this->setGeometry(p.x(), p.y(), sizeHint().width(), sizeHint().height());

	QObject::connect(closeBtn, SIGNAL(clicked()), this, SLOT(accept()));
	QObject::connect(cancelBtn, SIGNAL(clicked()), this, SLOT(reject()));
}

DelaunayTriangulationParamDialog::~DelaunayTriangulationParamDialog()
{

}
