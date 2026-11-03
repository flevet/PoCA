/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#include "PlacementPreviewWidget.hpp"
#include <QtGui/QPainter>
#include <QtGui/QPen>
#include <QtGui/QBrush>
#include <algorithm>
namespace poca::core {
	void PlacementPreviewWidget::paintEvent(QPaintEvent*) {
		QPainter painter(this);
		painter.fillRect(rect(), palette().base());
		if (m_items.empty() && m_groups.empty()) {
			painter.drawText(rect(), Qt::AlignCenter, QObject::tr("No spatial footprints available for preview."));
			return;
		}
		QRectF bounds;
		bool hasBounds = false;
		for (const auto& item : m_items) {
			bounds = hasBounds ? bounds.united(item.rect) : item.rect;
			hasBounds = true;
		}
		for (const auto& group : m_groups) {
			QRectF circle(group.center.x() - group.radius, group.center.y() - group.radius, group.radius * 2.f, group.radius * 2.f);
			bounds = hasBounds ? bounds.united(circle) : circle;
			hasBounds = true;
		}
		if (!hasBounds || bounds.width() <= 0.0 || bounds.height() <= 0.0)
			return;
		const float margin = 20.f;
		const double scale = std::min((width() - 2.0 * margin) / bounds.width(), (height() - 2.0 * margin) / bounds.height());
		auto mapRect = [&](const QRectF& r) {
			return QRectF(margin + (r.left() - bounds.left()) * scale, height() - margin - (r.bottom() - bounds.top()) * scale, r.width() * scale, r.height() * scale).normalized();
		};
		auto mapPoint = [&](const QPointF& p) {
			return QPointF(margin + (p.x() - bounds.left()) * scale, height() - margin - (p.y() - bounds.top()) * scale);
		};
		QPen groupPen(Qt::DashLine);
		groupPen.setWidth(2);
		painter.setPen(groupPen);
		painter.setBrush(Qt::NoBrush);
		for (const auto& group : m_groups) {
			const QPointF c = mapPoint(group.center);
			const double r = group.radius * scale;
			painter.drawEllipse(c, r, r);
			painter.drawText(c + QPointF(4, -4), group.label);
		}
		for (size_t i = 0; i < m_items.size(); ++i) {
			QColor color = QColor::fromHsv(int((i * 47) % 360), 170, 220, 110);
			painter.setPen(QPen(color.darker(160), 1));
			painter.setBrush(QBrush(color));
			painter.drawRect(mapRect(m_items[i].rect));
		}
	}

}
