/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#ifndef PlacementPreviewWidget_hpp__
#define PlacementPreviewWidget_hpp__
#include <QtWidgets/QWidget>
#include <QtCore/QRectF>
#include <QtCore/QPointF>
#include <QtCore/QStringList>
#include <General/DatasetPlacement.hpp>
namespace poca::core {
	struct PlacementPreviewItem { QRectF rect; int group{ -1 }; };
	struct PlacementPreviewGroup { QPointF center; float radius{ 0.f }; QString label; };
	class PlacementPreviewWidget : public QWidget {
	public:
		explicit PlacementPreviewWidget(QWidget* parent = nullptr) : QWidget(parent) { setMinimumSize(520,360); }
		void setPreview(const std::vector<PlacementPreviewItem>& items, const std::vector<PlacementPreviewGroup>& groups) {
			m_items = items; m_groups = groups; update();
		}
	protected:
		void paintEvent(QPaintEvent*) override;
	private:
		std::vector<PlacementPreviewItem> m_items;
		std::vector<PlacementPreviewGroup> m_groups;
	};
	// Qt adapter; the calculations and descriptors remain independent of widgets.
	inline std::vector<BoundingBox> computeCirclePlacement(const std::vector<BoundingBox>& base,
		const std::vector<std::vector<size_t>>& groups, const QStringList& labels,
		std::vector<PlacementPreviewGroup>* preview = nullptr) {
		std::vector<std::string> names;
		for (const auto& label : labels) names.push_back(label.toStdString());
		std::vector<PlacementCircle> circles;
		auto result = computeCirclePlacement(base,groups,names,&circles);
		if (preview) {
			preview->clear();
			for (const auto& circle : circles)
				preview->push_back({QPointF(circle.x,circle.y),circle.radius,QString::fromStdString(circle.label)});
		}
		return result;
	}
}
#endif
