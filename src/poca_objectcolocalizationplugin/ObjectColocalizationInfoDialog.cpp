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

#include "ObjectColocalizationInfoDialog.hpp"

ObjectColocalizationInfoDialog::ObjectColocalizationInfoDialog(const bool _sampling, const float _d, const uint32_t _nbSub, const uint32_t _minNbPoints, QWidget * _parent, Qt::WindowFlags _f) :QDialog(_parent, _f)
{
	m_cbox = new QCheckBox("Subsample objects contour");
	m_cbox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	m_cbox->setChecked(_sampling);

	QGroupBox* gBox2D = new QGroupBox(tr("2D"));
	gBox2D->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	QLabel * lbl1 = new QLabel("Sampling distance:");
	lbl1->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_ledit2D = new QLineEdit(QString::number(_d));
	m_ledit2D->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	QHBoxLayout* l2D = new QHBoxLayout;
	l2D->addWidget(lbl1);
	l2D->addWidget(m_ledit2D);
	gBox2D->setLayout(l2D);

	QGroupBox* gBox3D = new QGroupBox(tr("3D"));
	gBox3D->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	QLabel* lbl2 = new QLabel("# triangle subdivision:");
	lbl2->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_ledit3D = new QLineEdit(QString::number(_nbSub));
	m_ledit3D->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	QHBoxLayout* l3D = new QHBoxLayout;
	l3D->addWidget(lbl2);
	l3D->addWidget(m_ledit3D);
	gBox3D->setLayout(l3D);

	QGroupBox* gBoxParam = new QGroupBox(tr("Parameters"));
	gBoxParam->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	QLabel* lblNbLocs = new QLabel("Min # of points:");
	lblNbLocs->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_leditMinLocs = new QLineEdit(QString::number(_minNbPoints));
	m_leditMinLocs->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	QHBoxLayout* lparams = new QHBoxLayout;
	lparams->addWidget(lblNbLocs);
	lparams->addWidget(m_leditMinLocs);
	gBoxParam->setLayout(lparams);

	// Elle est simplement composé d'un bouton "Fermer"
	QPushButton * closeBtn = new QPushButton("Ok", this);
	QPushButton * cancelBtn = new QPushButton("Cancel", this);
	QHBoxLayout * layoutButton = new QHBoxLayout;
	layoutButton->addWidget(closeBtn);
	layoutButton->addWidget(cancelBtn);
	QWidget * widButton = new QWidget;
	widButton->setLayout(layoutButton);

	QVBoxLayout * layout = new QVBoxLayout;
	layout->addWidget(m_cbox);
	layout->addWidget(gBox2D);
	layout->addWidget(gBox3D);
	layout->addWidget(gBoxParam);
	layout->addWidget(widButton);

	this->setLayout(layout);
	this->setWindowTitle("Infos");
	QPoint p = QCursor::pos();
	this->setGeometry(p.x(), p.y(), sizeHint().width(), sizeHint().height());

	//QObject::connect(m_comboDat1, SIGNAL(currentIndexChanged(int)), this, SLOT(changeChosenDataset(int)));
	//QObject::connect(m_comboDat2, SIGNAL(currentIndexChanged(int)), this, SLOT(changeChosenDataset(int)));

	QObject::connect(closeBtn, SIGNAL(clicked()), this, SLOT(accept()));
	QObject::connect(cancelBtn, SIGNAL(clicked()), this, SLOT(reject()));
}

ObjectColocalizationInfoDialog::~ObjectColocalizationInfoDialog()
{

}

const float ObjectColocalizationInfoDialog::getDistance() const
{ 
	bool ok;
	float val = m_ledit2D->text().toFloat(&ok);
	return ok ? val : 0.5f; 
}

const uint32_t ObjectColocalizationInfoDialog::getNbSubdivision() const
{ 
	bool ok;
	uint32_t val = m_ledit3D->text().toUInt(&ok);
	return ok ? val : 0;
}

const uint32_t ObjectColocalizationInfoDialog::getMinNbPoints() const
{
	bool ok;
	uint32_t val = m_leditMinLocs->text().toUInt(&ok);
	return ok ? val : 5;
}