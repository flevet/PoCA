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
#include <QtWidgets/QLineEdit>

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
	m_cboxMesh = new QCheckBox("Mesh");
	m_cboxMesh->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	QVBoxLayout* ltab = new QVBoxLayout;
	ltab->addWidget(m_cboxTriangulation);
	ltab->addWidget(m_cboxConvexHull);
	ltab->addWidget(m_cboxPoisson);
	ltab->addWidget(m_cboxAlpha);
	ltab->addWidget(m_cboxMesh);
	gBoxShow->setLayout(ltab);

	m_bgroup = new QButtonGroup;
	m_bgroup->addButton(m_cboxTriangulation, poca::geometry::ObjectListFactoryInterface::TRIANGULATION);
	m_bgroup->addButton(m_cboxConvexHull, poca::geometry::ObjectListFactoryInterface::CONVEX_HULL);
	m_bgroup->addButton(m_cboxPoisson, poca::geometry::ObjectListFactoryInterface::POISSON_SURFACE);
	m_bgroup->addButton(m_cboxAlpha, poca::geometry::ObjectListFactoryInterface::ALPHA_SHAPE);
	m_bgroup->addButton(m_cboxMesh, poca::geometry::ObjectListFactoryInterface::MESH);

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
	case poca::geometry::ObjectListFactoryInterface::MESH:
		m_cboxMesh->setChecked(true);
		break;
	}

	QGroupBox* gBoxPoisson = new QGroupBox(tr("Poisson surface reconstruction"));
	gBoxPoisson->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	QLabel* lblAngle = new QLabel("Min angle:");
	lblAngle->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_leditAngle = new QLineEdit(QString::number(20));
	m_leditAngle->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	QHBoxLayout* langle = new QHBoxLayout;
	langle->addWidget(lblAngle);
	langle->addWidget(m_leditAngle);
	QLabel* lblRadius = new QLabel("Radius:");
	lblRadius->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_leditRadius = new QLineEdit(QString::number(100));
	m_leditRadius->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	QHBoxLayout* lradius = new QHBoxLayout;
	lradius->addWidget(lblRadius);
	lradius->addWidget(m_leditRadius);
	QLabel* lblDistance = new QLabel("Distance:");
	lblDistance->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_leditDistance = new QLineEdit(QString::number(0.25));
	m_leditDistance->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	QHBoxLayout* ldistance = new QHBoxLayout;
	ldistance->addWidget(lblDistance);
	ldistance->addWidget(m_leditDistance);
	QLabel* lblFactorAvgSpacing = new QLabel("Factor average spacing:");
	lblFactorAvgSpacing->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_leditFactorAvgSpacing = new QLineEdit(QString::number(1));
	m_leditFactorAvgSpacing->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	QHBoxLayout* lavgspacing = new QHBoxLayout;
	lavgspacing->addWidget(lblFactorAvgSpacing);
	lavgspacing->addWidget(m_leditFactorAvgSpacing);

	QVBoxLayout* layoutPoisson = new QVBoxLayout;
	layoutPoisson->addLayout(langle);
	layoutPoisson->addLayout(lradius);
	layoutPoisson->addLayout(ldistance);
	layoutPoisson->addLayout(lavgspacing);
	gBoxPoisson->setLayout(layoutPoisson);

	QGroupBox* gBoxMesh = new QGroupBox(tr("Mesh"));
	gBoxMesh->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	QLabel* lblTargetLength = new QLabel("Target length:");
	lblTargetLength->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_leditTargetLength = new QLineEdit(QString::number(20));
	m_leditTargetLength->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	QHBoxLayout* ltarget = new QHBoxLayout;
	ltarget->addWidget(lblTargetLength);
	ltarget->addWidget(m_leditTargetLength);
	QLabel* lblIterations = new QLabel("Iterations:");
	lblIterations->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_leditIterations = new QLineEdit(QString::number(5));
	m_leditIterations->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	QHBoxLayout* literations = new QHBoxLayout;
	literations->addWidget(lblIterations);
	literations->addWidget(m_leditIterations);
	QVBoxLayout* layoutMesh = new QVBoxLayout;
	layoutMesh->addLayout(ltarget);
	layoutMesh->addLayout(literations);
	gBoxMesh->setLayout(layoutMesh);

	QPushButton * closeBtn = new QPushButton("Ok", this);
	QPushButton * cancelBtn = new QPushButton("Cancel", this);
	QHBoxLayout * layoutButton = new QHBoxLayout;
	layoutButton->addWidget(closeBtn);
	layoutButton->addWidget(cancelBtn);
	QWidget * widButton = new QWidget;
	widButton->setLayout(layoutButton);

	QVBoxLayout * layout = new QVBoxLayout;
	layout->addWidget(gBoxShow);
	layout->addWidget(gBoxPoisson);
	layout->addWidget(gBoxMesh);
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

const double ObjectListParamDialog::getAngle() const
{
	bool ok;
	double val = m_leditAngle->text().toDouble(&ok);
	return ok ? val : 20.;
}

const double ObjectListParamDialog::getRadius() const
{
	bool ok;
	double val = m_leditRadius->text().toDouble(&ok);
	return ok ? val : 100;
}

const double ObjectListParamDialog::getDistance() const
{
	bool ok;
	double val = m_leditDistance->text().toDouble(&ok);
	return ok ? val : 0.25;
}

const double ObjectListParamDialog::getFactorAverageSpacing() const
{
	bool ok;
	double val = m_leditFactorAvgSpacing->text().toDouble(&ok);
	return ok ? val : 1;
}

const double ObjectListParamDialog::getMeshTargetLength() const
{
	bool ok;
	double val = m_leditTargetLength->text().toDouble(&ok);
	return ok ? val : 20;
}

const int ObjectListParamDialog::getMeshIterations() const
{
	bool ok;
	int val = m_leditIterations->text().toInt(&ok);
	return ok ? val : 5;
}