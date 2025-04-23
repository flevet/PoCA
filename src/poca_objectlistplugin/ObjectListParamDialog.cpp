/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectListParamDialog.cpp
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
#include <QtWidgets/QGroupBox>

#include <Interfaces/ObjectListFactoryInterface.hpp>

#include "ObjectListParamDialog.hpp"

ObjectListParamDialog::ObjectListParamDialog(const std::string _typeObject, QWidget * _parent, Qt::WindowFlags _f) :QDialog(_parent, _f)
{
	QGroupBox* gBoxShow = new QGroupBox(tr("Type"));
	gBoxShow->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_cboxTriangulation = new QCheckBox("Triangulation");
	m_cboxTriangulation->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_cboxConvexHull = new QCheckBox("Convex hull");
	m_cboxConvexHull->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_cboxPoisson = new QCheckBox("Poisson reconstruction");
	m_cboxPoisson->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_cboxAlpha = new QCheckBox("Alpha shape");
	m_cboxPoisson->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	QVBoxLayout* ltab = new QVBoxLayout;
	ltab->addWidget(m_cboxTriangulation);
	ltab->addWidget(m_cboxConvexHull);
	ltab->addWidget(m_cboxPoisson);
	ltab->addWidget(m_cboxAlpha);
	gBoxShow->setLayout(ltab);

	m_bgroup = new QButtonGroup;
	m_bgroup->addButton(m_cboxTriangulation, poca::geometry::ObjectListFactoryInterface::TRIANGULATION);
	m_bgroup->addButton(m_cboxConvexHull, poca::geometry::ObjectListFactoryInterface::CONVEX_HULL);
	m_bgroup->addButton(m_cboxPoisson, poca::geometry::ObjectListFactoryInterface::POISSON_SURFACE);
	m_bgroup->addButton(m_cboxAlpha, poca::geometry::ObjectListFactoryInterface::ALPHA_SHAPE);

	int typeId = poca::geometry::ObjectListFactoryInterface::getTypeId(_typeObject);
	switch (typeId) {
	case poca::geometry::ObjectListFactoryInterface::TRIANGULATION:
		m_cboxTriangulation->setChecked(true);
		break;
	case poca::geometry::ObjectListFactoryInterface::CONVEX_HULL:
		m_cboxConvexHull->setChecked(true);
		break;
	case poca::geometry::ObjectListFactoryInterface::POISSON_SURFACE:
		m_cboxPoisson->setChecked(true);
		break;
	case poca::geometry::ObjectListFactoryInterface::ALPHA_SHAPE:
		m_cboxAlpha->setChecked(true);
		break;
	}

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

ObjectListParamDialog::~ObjectListParamDialog()
{

}

const std::string ObjectListParamDialog::typeObject() const
{
	return poca::geometry::ObjectListFactoryInterface::getTypeStr(m_bgroup->checkedId());
}
