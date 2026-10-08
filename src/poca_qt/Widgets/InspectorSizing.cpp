/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#include "InspectorSizing.hpp"
#include <QtWidgets/QWidget>
#include <QtWidgets/QLabel>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTabBar>
#include <QtWidgets/QStackedWidget>

namespace poca::qt {
	void makeHorizontallyShrinkable(QWidget* widget)
	{
		QSizePolicy policy = widget->sizePolicy();
		policy.setHorizontalPolicy(QSizePolicy::Ignored);
		widget->setSizePolicy(policy);
		widget->setMinimumWidth(0);
	}

	void configureInspectorCombo(QComboBox* combo)
	{
		combo->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
		combo->setMinimumContentsLength(12);
		makeHorizontallyShrinkable(combo);
	}

	void configureInspectorLabel(QLabel* label, bool wordWrap)
	{
		label->setWordWrap(wordWrap);
		makeHorizontallyShrinkable(label);
	}

	void configureInspectorTabs(QTabWidget* tabs)
	{
		makeHorizontallyShrinkable(tabs);
		makeHorizontallyShrinkable(tabs->tabBar());
		tabs->tabBar()->setUsesScrollButtons(true);
		tabs->setElideMode(Qt::ElideRight);
		for (auto* child : tabs->findChildren<QTabWidget*>()) {
			makeHorizontallyShrinkable(child);
			makeHorizontallyShrinkable(child->tabBar());
			child->tabBar()->setUsesScrollButtons(true);
			child->setElideMode(Qt::ElideRight);
		}
		for (auto* stack : tabs->findChildren<QStackedWidget*>())
			makeHorizontallyShrinkable(stack);
	}
}