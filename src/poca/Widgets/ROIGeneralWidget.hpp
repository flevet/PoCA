/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ROIGeneralWidget.hpp
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

#ifndef ROIWidget_h__
#define ROIWidget_h__

#include <QtWidgets/QWidget>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>

#include <DesignPatterns/Observer.hpp>
#include <DesignPatterns/MediatorWObjectFWidget.hpp>
#include <General/Command.hpp>

namespace poca::core {
	class ROIInterface;
}

class ROIGeneralWidget : public QGroupBox, public poca::core::ObserverForMediator{
	Q_OBJECT

public:
	ROIGeneralWidget(poca::core::MediatorWObjectFWidget *, QWidget* = 0);
	~ROIGeneralWidget();

	void performAction(poca::core::MyObjectInterface*, poca::core::CommandInfo*);
	void update(poca::core::SubjectInterface*, const poca::core::CommandInfo&);
	void executeMacro(poca::core::MyObjectInterface*, poca::core::CommandInfo*);

protected:
	void loadROIsIntoTable(const std::vector <poca::core::ROIInterface *> &);
	void addLastROI(poca::core::MyObjectInterface*);
	void updateSelectionROIsFromObject(poca::core::MyObjectInterface*);
	void saveROIs(poca::core::MyObjectInterface*);
	void setColorROIs(QPushButton*, const unsigned char, const unsigned char, const unsigned char);

	const float getXY() const;

	protected slots:
	void actionNeeded();
	void actionNeeded(bool);
	void actionNeeded(QTableWidgetItem *);

protected:
	QTabWidget * m_parentTab;

	QTableWidget * m_tableW;
	QPushButton * m_loadROIsBtn, *m_saveROIsBtn, *m_discardSelectedROIsBtn, *m_discardAllROIsBtn;
	QLabel* m_colorSelectedROILbl, * m_colorUnselectedROILbl;
	QPushButton* m_colorSelectedROIBtn, * m_colorUnselectedROIBtn;
	QCheckBox* m_cboxDisplayROIs, * m_cboxDisplayLabelROIs;
	QLineEdit* m_pixXYEdit;


	poca::core::MyObjectInterface* m_object;
	poca::core::MediatorWObjectFWidget * m_mediator;
};

#endif // ROIWidget_h__

