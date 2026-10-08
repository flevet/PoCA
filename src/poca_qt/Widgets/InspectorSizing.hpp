/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#ifndef InspectorSizing_hpp__
#define InspectorSizing_hpp__

class QWidget;
class QLabel;
class QComboBox;
class QTabWidget;

namespace poca::qt {
	// Inspector content may be narrower than its hint; vertical policy is preserved.
	void makeHorizontallyShrinkable(QWidget*);
	// Twelve characters leave useful text beside an inspector row label.
	void configureInspectorCombo(QComboBox*);
	// Identifiers clip; explanatory text may wrap. Callers maintain dynamic tooltips.
	void configureInspectorLabel(QLabel*, bool wordWrap = false);
	// Apply after plugin insertion; only tab/stack containers are traversed.
	void configureInspectorTabs(QTabWidget*);
}
#endif