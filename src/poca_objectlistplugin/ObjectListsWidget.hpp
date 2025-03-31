/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectListsWidget.hpp
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

#ifndef ObjectListsWidget_h__
#define ObjectListsWidget_h__

#include <QtWidgets/QTabWidget>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QDialog>

#include <Plot/FilterHistogramWidget.hpp>
#include <DesignPatterns/Observer.hpp>
#include <General/Command.hpp>
#include <General/Palette.hpp>

class QPushButton;
class QLineEdit;
class QLabel;

namespace poca::geometry{
	class ObjectLists;
}

class ObjectListsParamDialog : public QDialog {
	Q_OBJECT

public:
	ObjectListsParamDialog(poca::geometry::ObjectLists*, QWidget* = 0, Qt::WindowFlags = 0);
	~ObjectListsParamDialog();

	inline const QString description() const { return m_ledit->text(); }

protected slots:

protected:
	poca::geometry::ObjectLists* m_objs;
	QLineEdit* m_ledit;
	QLabel* m_tedit;
};

class SortableFloatItem : public QTableWidgetItem
{
public:
	SortableFloatItem(const QTableWidgetItem& other): QTableWidgetItem(other) {}
	SortableFloatItem(const QIcon& icon, const QString& text, int type = Type): QTableWidgetItem(icon, text, type) {}
	SortableFloatItem(const QString& text, int type = Type): QTableWidgetItem(text, type) {}
	SortableFloatItem(int type = Type): QTableWidgetItem(type) {}

	bool operator< (const QTableWidgetItem& other) const
	{
		// TODO: To be safe, check weather conversion to int is possible.
		return (this->text().toFloat() < other.text().toFloat());
	}
};

//! [0]
class ObjectListsWidget : public QWidget, public poca::core::ObserverForMediator {
	Q_OBJECT

public:
	ObjectListsWidget(poca::core::MediatorWObjectFWidgetInterface*, QWidget* = 0);
	~ObjectListsWidget();

	void performAction(poca::core::MyObjectInterface*, poca::core::CommandInfo*);
	void update(poca::core::SubjectInterface*, const poca::core::CommandInfo&);
	void executeMacro(poca::core::MyObjectInterface*, poca::core::CommandInfo*);

protected:
	inline int pointSize() const { return m_sizePointSpn->value(); }

protected slots:
	void actionNeeded();
	void actionNeeded(int);
	void actionNeeded(bool);

	void changeListObject(QAbstractButton*);

signals:
	void transferNewObjectCreated(poca::core::MyObjectInterface*);

protected:
	QTabWidget* m_parentTab;
	poca::core::MediatorWObjectFWidgetInterface* m_mediator;

	QWidget* m_lutsWidget, * m_buttonsWidget, * m_delaunayTriangulationFilteringWidget, * m_emptyWidget;
	std::vector <std::pair<QPushButton*, std::string>> m_lutButtons;
	std::pair<QPushButton*, std::string> m_hilowButton;
	std::vector <poca::plot::FilterHistogramWidget*> m_histWidgets;
	QPushButton* m_displayButton, * m_fillButton, * m_pointRenderButton, * m_outlinePointRenderButton, * m_shapeRenderButton, * m_bboxSelectionButton, 
		* m_exportButton, * m_exportLocsButton, * m_selectionButton, * m_duplicateCentroidsButton, * m_duplicateSelectedObjectsButton,
		* m_ellipsoidRenderButton, * m_parametersButton, * m_eraseObjectButton, * m_saveSVGButton, * m_saveOBJButton, * m_cullfaceButton;
	QWidget* m_widgetObjectMesh;
	QPushButton* m_computeSkeletonsButton, * m_skeletonRenderButton, * m_linkToSkeletonRenderButton;
	QSpinBox* m_sizePointSpn;

	QTableWidget* m_tableObjects;

	QWidget* m_widgetList;
	std::vector <QPushButton*> m_listButtons;
	QButtonGroup* m_listButtonsGroup;

	poca::core::MyObjectInterface* m_object;
};

//! [0]
#endif

