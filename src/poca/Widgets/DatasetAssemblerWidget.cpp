/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DatasetAssemblerWidget.cpp
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

#include <algorithm>
#include <fstream>
#include <cmath>
#include <map>
#include <limits>
#include <set>
#include <utility>

#include <QtWidgets/QAbstractItemView>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDialog>
#include <QtWidgets/QDialogButtonBox>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QTableWidgetItem>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QTreeWidget>
#include <QtWidgets/QTreeWidgetItem>
#include <QtWidgets/QVBoxLayout>
#include <QtCore/QDir>
#include <QtCore/QDirIterator>
#include <QtCore/QFileInfo>
#include <QtCore/QRegularExpression>
#include <QtCore/QSet>
#include <QtCore/QSignalBlocker>
#include <QtGui/QColor>
#include <QtGui/QPainter>
#include <QtGui/QPen>
#include <QtGui/QBrush>
#include <QtGui/QMouseEvent>
#include <QtGui/QOpenGLContext>
#include <QtGui/QOpenGLFunctions>

#include <General/Engine.hpp>
#include <General/Misc.h>
#include <General/stb_rect_pack.h>
#include <Interfaces/MyObjectInterface.hpp>
#include <Interfaces/BasicComponentInterface.hpp>
#include <Objects/MyMultipleObject.hpp>
#include <tinytiffreader.h>

#include "../Widgets/DatasetAssemblerWidget.hpp"

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#else
#include <sys/sysinfo.h>
#endif

#ifndef GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX
#define GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX 0x9049
#endif
#ifndef GL_TEXTURE_FREE_MEMORY_ATI
#define GL_TEXTURE_FREE_MEMORY_ATI 0x87FC
#endif

namespace {
	QString buildDatasetName(const QString& rootFolder, const QString& key, const bool prefixRootName, const QString& separator)
	{
		const QString rootName = QFileInfo(rootFolder).fileName();
		if (prefixRootName && !rootName.isEmpty() && !key.isEmpty())
			return rootName + separator + key;
		if (prefixRootName && !rootName.isEmpty())
			return rootName;
		if (!key.isEmpty())
			return key;
		return rootName;
	}

	QTableWidgetItem* createTextItem(const QString& text = QString())
	{
		auto* item = new QTableWidgetItem(text);
		item->setFlags(item->flags() | Qt::ItemIsEditable);
		return item;
	}

	QTableWidgetItem* createCheckItem(const bool checked)
	{
		auto* item = new QTableWidgetItem;
		item->setFlags((item->flags() | Qt::ItemIsUserCheckable | Qt::ItemIsEnabled) & ~Qt::ItemIsEditable);
		item->setCheckState(checked ? Qt::Checked : Qt::Unchecked);
		return item;
	}

	bool isTiffFilename(const QString& filename)
	{
		return filename.endsWith(".tif", Qt::CaseInsensitive) || filename.endsWith(".tiff", Qt::CaseInsensitive);
	}

	uint64_t safeMul(const uint64_t a, const uint64_t b)
	{
		if (a == 0 || b == 0)
			return 0;
		if (a > std::numeric_limits<uint64_t>::max() / b)
			return std::numeric_limits<uint64_t>::max();
		return a * b;
	}


	struct PlacementPreviewItem {
		QRectF rect;
		int group{ -1 };
	};

	struct PlacementPreviewGroup {
		QPointF center;
		float radius{ 0.f };
		QString label;
	};

	class PlacementPreviewWidget : public QWidget {
	public:
		PlacementPreviewWidget(QWidget* _parent = nullptr) : QWidget(_parent) { setMinimumSize(520, 360); }
		void setPreview(const std::vector<PlacementPreviewItem>& _items, const std::vector<PlacementPreviewGroup>& _groups) {
			m_items = _items; m_groups = _groups; m_dragItem = -1; m_dragGroup = -1; updateTransform(); update();
		}
		const std::vector<PlacementPreviewItem>& items() const { return m_items; }
		const std::vector<PlacementPreviewGroup>& groups() const { return m_groups; }
	protected:
		void resizeEvent(QResizeEvent*) override { updateTransform(); }
		void paintEvent(QPaintEvent*) override {
			QPainter painter(this);
			painter.fillRect(rect(), palette().base());
			if (m_items.empty() && m_groups.empty()) {
				painter.drawText(rect(), Qt::AlignCenter, QObject::tr("Press Preview to compute a placement."));
				return;
			}
			updateTransform();
			QPen groupPen(Qt::DashLine);
			groupPen.setWidth(2);
			painter.setPen(groupPen);
			painter.setBrush(Qt::NoBrush);
			for (const auto& group : m_groups) {
				const QPointF c = mapPoint(group.center);
				const double r = group.radius * m_scale;
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
		void mousePressEvent(QMouseEvent* _event) override {
			m_lastWorld = unmapPoint(_event->pos());
			m_dragItem = -1; m_dragGroup = -1;
			for (int i = int(m_items.size()) - 1; i >= 0; --i)
				if (m_items[i].rect.contains(m_lastWorld)) { m_dragItem = i; return; }
			for (int g = int(m_groups.size()) - 1; g >= 0; --g) {
				const double dx = m_lastWorld.x() - m_groups[g].center.x();
				const double dy = m_lastWorld.y() - m_groups[g].center.y();
				if (std::sqrt(dx * dx + dy * dy) <= m_groups[g].radius) { m_dragGroup = g; return; }
			}
		}
		void mouseMoveEvent(QMouseEvent* _event) override {
			if (m_dragItem < 0 && m_dragGroup < 0) return;
			const QPointF world = unmapPoint(_event->pos());
			const QPointF delta = world - m_lastWorld;
			m_lastWorld = world;
			if (m_dragGroup >= 0) {
				m_groups[m_dragGroup].center += delta;
				for (auto& item : m_items)
					if (item.group == m_dragGroup) item.rect.translate(delta);
			}
			else if (m_dragItem >= 0) { m_items[m_dragItem].rect.translate(delta); updateGroupCircle(m_items[m_dragItem].group); }
			updateTransform(); update();
		}
		void mouseReleaseEvent(QMouseEvent*) override { m_dragItem = -1; m_dragGroup = -1; }
	private:
		QRectF bounds() const {
			QRectF b; bool has = false;
			for (const auto& item : m_items) { b = has ? b.united(item.rect) : item.rect; has = true; }
			for (const auto& group : m_groups) { QRectF c(group.center.x() - group.radius, group.center.y() - group.radius, group.radius * 2.f, group.radius * 2.f); b = has ? b.united(c) : c; has = true; }
			if (!has || b.width() <= 0.0 || b.height() <= 0.0) return QRectF(0, 0, 1, 1);
			return b.adjusted(-1, -1, 1, 1);
		}
		void updateTransform() {
			m_bounds = bounds(); const float margin = 20.f;
			m_scale = std::min((width() - 2.0 * margin) / m_bounds.width(), (height() - 2.0 * margin) / m_bounds.height());
			if (!std::isfinite(m_scale) || m_scale <= 0.0) m_scale = 1.0;
		}
		QRectF mapRect(const QRectF& r) const { return QRectF(20.0 + (r.left() - m_bounds.left()) * m_scale, height() - 20.0 - (r.bottom() - m_bounds.top()) * m_scale, r.width() * m_scale, r.height() * m_scale).normalized(); }
		QPointF mapPoint(const QPointF& p) const { return QPointF(20.0 + (p.x() - m_bounds.left()) * m_scale, height() - 20.0 - (p.y() - m_bounds.top()) * m_scale); }
		QPointF unmapPoint(const QPointF& p) const { return QPointF((p.x() - 20.0) / m_scale + m_bounds.left(), (height() - 20.0 - p.y()) / m_scale + m_bounds.top()); }
		void updateGroupCircle(const int _group) {
			if (_group < 0 || _group >= int(m_groups.size())) return;
			QPointF center(0.0, 0.0); int count = 0;
			for (const auto& item : m_items) if (item.group == _group) { center += item.rect.center(); ++count; }
			if (count == 0) return;
			center /= double(count);
			double radius = 1.0;
			for (const auto& item : m_items) if (item.group == _group) {
				const QPointF corners[4] = { item.rect.topLeft(), item.rect.topRight(), item.rect.bottomLeft(), item.rect.bottomRight() };
				for (const QPointF& corner : corners) { const double dx = corner.x() - center.x(), dy = corner.y() - center.y(); radius = std::max(radius, std::sqrt(dx * dx + dy * dy) + 2.0); }
			}
			m_groups[_group].center = center; m_groups[_group].radius = float(radius);
		}
		std::vector<PlacementPreviewItem> m_items;
		std::vector<PlacementPreviewGroup> m_groups;
		QRectF m_bounds;
		double m_scale{ 1.0 };
		QPointF m_lastWorld;
		int m_dragItem{ -1 }, m_dragGroup{ -1 };
	};


	poca::core::BoundingBox localObjectBBox(poca::core::MyObjectInterface* _object)
	{
		if (_object == nullptr) {
			poca::core::BoundingBox empty;
			empty.set(0.f, 0.f, 0.f, 1.f, 1.f, 1.f);
			return empty;
		}

		poca::core::BoundingBox bbox = poca::core::BoundingBox::initBBox();
		bool hasBBox = false;
		for (poca::core::BasicComponentInterface* component : _object->getComponents()) {
			if (component == nullptr)
				continue;
			const poca::core::BoundingBox& componentBBox = component->boundingBox();
			for (int ix = 0; ix < 2; ++ix)
				for (int iy = 0; iy < 2; ++iy)
					for (int iz = 0; iz < 2; ++iz) {
						const float x = ix == 0 ? componentBBox[0] : componentBBox[3];
						const float y = iy == 0 ? componentBBox[1] : componentBBox[4];
						const float z = iz == 0 ? componentBBox[2] : componentBBox[5];
						if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
							continue;
						bbox.addPointBBox(x, y, z);
						hasBBox = true;
					}
		}

		if (!hasBBox)
			bbox = _object->boundingBox();
		if (!std::isfinite(bbox.realWidth()) || bbox.realWidth() <= 0.f || !std::isfinite(bbox.realHeight()) || bbox.realHeight() <= 0.f)
			bbox.set(0.f, 0.f, 0.f, 1.f, 1.f, 1.f);
		return bbox;
	}

	std::vector<poca::core::BoundingBox> basePlacementBBoxes(MyMultipleObject* _object)
	{
		std::vector<poca::core::BoundingBox> result;
		if (_object == nullptr)
			return result;
		const auto& gridBBoxes = _object->getGridBBoxes();
		for (size_t i = 0; i < _object->nbColors(); ++i) {
			poca::core::BoundingBox bbox = localObjectBBox(_object->getObject(i));
			if (i < gridBBoxes.size()) {
				const auto& gb = gridBBoxes[i];
				bbox.set(gb.x(), gb.y(), gb.z(), gb.x() + std::max(1.f, bbox.realWidth()), gb.y() + std::max(1.f, bbox.realHeight()), gb.z() + bbox.realThick());
			}
			else
				bbox.set(0.f, 0.f, 0.f, std::max(1.f, bbox.realWidth()), std::max(1.f, bbox.realHeight()), bbox.realThick());
			result.push_back(bbox);
		}
		return result;
	}

	std::vector<poca::core::BoundingBox> computeRegularGridPlacement(const std::vector<poca::core::BoundingBox>& _base, int _columns, int _rows)
	{
		std::vector<poca::core::BoundingBox> result = _base;
		if (_base.empty()) return result;
		_columns = std::max(1, _columns);
		_rows = std::max(1, _rows);
		float cellSide = 1.f;
		for (const auto& b : _base)
			cellSide = std::max(cellSide, std::max(b.realWidth(), b.realHeight()));
		for (size_t i = 0; i < result.size(); ++i) {
			const int col = int(i % _columns);
			const int row = int(i / _columns);
			const float x = col * cellSide + (cellSide - _base[i].realWidth()) * 0.5f;
			const float y = row * cellSide + (cellSide - _base[i].realHeight()) * 0.5f;
			result[i].set(x, y, 0.f, x + _base[i].realWidth(), y + _base[i].realHeight(), _base[i].realThick());
		}
		return result;
	}

	std::vector<poca::core::BoundingBox> computePackPlacement(const std::vector<poca::core::BoundingBox>& _base, double _initialMultiplier)
	{
		std::vector<poca::core::BoundingBox> result = _base;
		if (_base.empty()) return result;
		size_t total = 0, maxD = 1;
		std::vector<stbrp_rect> rects;
		for (size_t i = 0; i < _base.size(); ++i) {
			const int w = std::max(1, int(std::ceil(_base[i].realWidth())));
			const int h = std::max(1, int(std::ceil(_base[i].realHeight())));
			rects.push_back({ int(i), w, h, 0, 0, 0 });
			const size_t d = size_t(std::max(w, h));
			total += d;
			maxD = std::max(maxD, d);
		}
		Bin bin{ std::max(1, int(maxD * std::max(1.0, _initialMultiplier))), std::max(1, int(maxD * std::max(1.0, _initialMultiplier))) };
		const int maxW = std::max(bin.w, int(std::max<size_t>(total, maxD)));
		const int maxH = std::max(bin.h, int(std::max<size_t>(total, maxD)));
		while (true) {
			auto work = rects;
			if (try_pack(bin, work)) {
				for (const auto& r : work)
					result[r.id].set(float(r.x), float(r.y), 0.f, float(r.x) + _base[r.id].realWidth(), float(r.y) + _base[r.id].realHeight(), _base[r.id].realThick());
				break;
			}
			Bin next = grow(bin, maxW, maxH, 1.25f);
			if (next.w == bin.w && next.h == bin.h)
				break;
			bin = next;
		}
		return result;
	}

	std::vector<poca::core::BoundingBox> computeCirclePlacement(const std::vector<poca::core::BoundingBox>& _base, const std::vector<std::vector<size_t>>& _groups, const QStringList& _groupLabels, std::vector<PlacementPreviewGroup>* _previewGroups = nullptr)
	{
		std::vector<poca::core::BoundingBox> result = _base;
		if (_base.empty()) return result;
		const float padding = 8.f;
		std::vector<std::vector<poca::geometry::PackingCircle>> circlesGroups;
		for (const auto& group : _groups) {
			if (group.empty()) continue;
			circlesGroups.push_back({});
			auto& circles = circlesGroups.back();
			for (size_t id : group) {
				if (id >= _base.size()) continue;
				const float radius = 0.5f * std::sqrt(_base[id].realWidth() * _base[id].realWidth() + _base[id].realHeight() * _base[id].realHeight()) + padding;
				circles.push_back({ radius, radius, std::max(1.f, radius), int(id) });
			}
			poca::geometry::packCirclesFast(circles, 800, 0.0015f, 0.75f);
		}
		std::vector<poca::geometry::PackingCircle> groupCircles;
		std::vector<poca::core::BoundingBox> groupBounds;
		for (size_t g = 0; g < circlesGroups.size(); ++g) {
			poca::core::BoundingBox bbox = poca::core::BoundingBox::initBBox();
			for (const auto& c : circlesGroups[g]) {
				const size_t id = size_t(c.id); if (id >= _base.size()) continue;
				const float x = c.x - _base[id].realWidth() * 0.5f, y = c.y - _base[id].realHeight() * 0.5f;
				bbox.addPointBBox(x, y, 0.f); bbox.addPointBBox(x + _base[id].realWidth(), y + _base[id].realHeight(), 0.f);
			}
			groupBounds.push_back(bbox);
			const float radius = 0.5f * std::sqrt(bbox.realWidth() * bbox.realWidth() + bbox.realHeight() * bbox.realHeight()) + padding;
			groupCircles.push_back({ radius, radius, std::max(1.f, radius), int(g) });
		}
		poca::geometry::packCirclesFast(groupCircles, 800, 0.0015f, 0.75f);
		if (_previewGroups) _previewGroups->clear();
		for (const auto& cg : groupCircles) {
			const int g = cg.id;
			if (g < 0 || g >= int(circlesGroups.size())) continue;
			const auto& circles = circlesGroups[g];
			const auto& bbox = groupBounds[g];
			const float gx = cg.x - (bbox.x() + bbox.realWidth() * 0.5f);
			const float gy = cg.y - (bbox.y() + bbox.realHeight() * 0.5f);
			if (_previewGroups) _previewGroups->push_back({ QPointF(cg.x, cg.y), cg.r, g >= 0 && g < _groupLabels.size() ? _groupLabels[g] : QString("Group %1").arg(g + 1) });
			for (const auto& c : circles) {
				const size_t id = size_t(c.id); if (id >= _base.size()) continue;
				const float x = c.x - _base[id].realWidth() * 0.5f + gx;
				const float y = c.y - _base[id].realHeight() * 0.5f + gy;
				result[id].set(x, y, 0.f, x + _base[id].realWidth(), y + _base[id].realHeight(), _base[id].realThick());
			}
		}
		return result;
	}

	std::vector<poca::core::BoundingBox> computeOriginalPlacement(const std::vector<poca::core::BoundingBox>& _base, const bool _centered)
	{
		std::vector<poca::core::BoundingBox> result = _base;
		if (!_centered || result.empty()) return result;
		poca::core::BoundingBox englobing = poca::core::BoundingBox::initBBox();
		for (const auto& b : _base) { englobing.addPointBBox(b[0], b[1], b[2]); englobing.addPointBBox(b[3], b[4], b[5]); }
		const float cx = englobing[0] + englobing.realWidth() * 0.5f;
		const float cy = englobing[1] + englobing.realHeight() * 0.5f;
		for (auto& b : result) {
			const float w = b.realWidth(), h = b.realHeight(), t = b.realThick();
			b.set(cx - w * 0.5f, cy - h * 0.5f, b.z(), cx + w * 0.5f, cy + h * 0.5f, b.z() + t);
		}
		return result;
	}


	void buildPreviewItems(const std::vector<poca::core::BoundingBox>& _bboxes, std::vector<PlacementPreviewItem>& _items)
	{
		_items.clear();
		for (const auto& b : _bboxes)
			_items.push_back({ QRectF(b.x(), b.y(), b.realWidth(), b.realHeight()), -1 });
	}

	class DatasetPlacementDialog : public QDialog {
	public:
		DatasetPlacementDialog(MyMultipleObject* _object, const std::vector<DatasetAssemblerWidget::AssembledDatasetInfo>& _infos, QWidget* _parent = nullptr)
			: QDialog(_parent), m_object(_object), m_infos(_infos), m_baseBBoxes(basePlacementBBoxes(_object)) {
			setWindowTitle(QObject::tr("Dataset placement"));
			m_methodCombo = new QComboBox(this);
			m_methodCombo->addItem(QObject::tr("Grid"));
			m_methodCombo->addItem(QObject::tr("Pack"));
			m_methodCombo->addItem(QObject::tr("Circle groups"));
			m_methodCombo->addItem(QObject::tr("Original"));
			m_columnsSpin = new QSpinBox(this); m_columnsSpin->setRange(1, std::max(1, int(m_baseBBoxes.size()))); m_columnsSpin->setValue(std::max(1, int(std::ceil(std::sqrt(double(std::max<size_t>(1, m_baseBBoxes.size())))))));
			m_rowsSpin = new QSpinBox(this); m_rowsSpin->setRange(1, std::max(1, int(m_baseBBoxes.size()))); m_rowsSpin->setValue(std::max(1, int(std::ceil(double(std::max<size_t>(1, m_baseBBoxes.size())) / double(m_columnsSpin->value())))));
			m_packMultiplierSpin = new QDoubleSpinBox(this); m_packMultiplierSpin->setRange(1.0, 20.0); m_packMultiplierSpin->setSingleStep(0.5); m_packMultiplierSpin->setValue(6.0);
			m_levelSpin = new QSpinBox(this); m_levelSpin->setRange(0, maxHierarchyDepth()); m_levelSpin->setValue(std::min(1, maxHierarchyDepth()));
			m_originalCenteredCheck = new QCheckBox(QObject::tr("Centered"), this);
			m_originalCenteredCheck->setChecked(false);
			m_preview = new PlacementPreviewWidget(this);
			m_columnsLabel = new QLabel(QObject::tr("Columns"), this);
			m_rowsLabel = new QLabel(QObject::tr("Rows"), this);
			m_packMultiplierLabel = new QLabel(QObject::tr("Initial pack multiplier"), this);
			m_levelLabel = new QLabel(QObject::tr("Circle hierarchy level (0 = root)"), this);
			QGridLayout* controls = new QGridLayout;
			controls->addWidget(new QLabel(QObject::tr("Method"), this), 0, 0); controls->addWidget(m_methodCombo, 0, 1);
			controls->addWidget(m_columnsLabel, 1, 0); controls->addWidget(m_columnsSpin, 1, 1);
			controls->addWidget(m_rowsLabel, 1, 2); controls->addWidget(m_rowsSpin, 1, 3);
			controls->addWidget(m_packMultiplierLabel, 2, 0); controls->addWidget(m_packMultiplierSpin, 2, 1);
			controls->addWidget(m_levelLabel, 2, 2); controls->addWidget(m_levelSpin, 2, 3);
			controls->addWidget(m_originalCenteredCheck, 3, 0, 1, 4, Qt::AlignCenter);
			QPushButton* previewButton = new QPushButton(QObject::tr("Preview"), this);
			QDialogButtonBox* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
			QHBoxLayout* bottom = new QHBoxLayout; bottom->addWidget(previewButton); bottom->addStretch(1); bottom->addWidget(buttons);
			QVBoxLayout* layout = new QVBoxLayout; layout->addLayout(controls); layout->addWidget(m_preview); layout->addLayout(bottom); setLayout(layout);
			connect(previewButton, &QPushButton::released, this, &DatasetPlacementDialog::updatePreview);
			connect(m_methodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) { updateControlsEnabled(); updatePreview(); });
			connect(m_originalCenteredCheck, &QCheckBox::toggled, this, [this](bool) { updatePreview(); });
			connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
			connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
			updateControlsEnabled();
			updatePreview();
		}
		std::vector<poca::core::BoundingBox> selectedPlacement() {
			std::vector<poca::core::BoundingBox> result = m_baseBBoxes;
			const auto& items = m_preview->items();
			for (size_t i = 0; i < result.size() && i < items.size(); ++i)
				result[i].set(float(items[i].rect.left()), float(items[i].rect.top()), result[i].z(), float(items[i].rect.right()), float(items[i].rect.bottom()), result[i].z() + result[i].realThick());
			return result;
		}
	private:
		int maxHierarchyDepth() const {
			int maxDepth = 1;
			for (const auto& info : m_infos) maxDepth = std::max(maxDepth, int(info.hierarchySegments.size()) + 1);
			return maxDepth;
		}
		std::vector<std::vector<size_t>> groupsForLevel(int _level, QStringList* _labels = nullptr) const {
			std::map<QString, std::vector<size_t>> groups;
			for (size_t i = 0; i < m_infos.size(); ++i) {
				QString key;
				if (_level <= 1)
					key = !m_infos[i].hierarchySegments.empty() ? m_infos[i].hierarchySegments.front() : (m_object != nullptr ? QString::fromStdString(m_object->getName()) : QStringLiteral("Objects"));
				else
					key = m_infos[i].hierarchySegments.size() >= (_level - 1) ? m_infos[i].hierarchySegments[_level - 2] : m_infos[i].datasetKey;
				if (key.isEmpty())
					key = m_infos[i].objectName;
				groups[key].push_back(i);
			}
			std::vector<std::vector<size_t>> result;
			if (_labels != nullptr)
				_labels->clear();
			for (const auto& kv : groups) {
				result.push_back(kv.second);
				if (_labels != nullptr)
					*_labels << kv.first;
			}
			return result;
		}
		void updateControlsEnabled() {
			const int method = m_methodCombo->currentIndex();
			m_columnsLabel->setEnabled(method == 0);
			m_columnsSpin->setEnabled(method == 0);
			m_rowsLabel->setEnabled(method == 0);
			m_rowsSpin->setEnabled(method == 0);
			m_packMultiplierLabel->setEnabled(method == 1);
			m_packMultiplierSpin->setEnabled(method == 1);
			m_levelLabel->setEnabled(method == 2);
			m_levelSpin->setEnabled(method == 2);
			m_originalCenteredCheck->setEnabled(method == 3);
		}
		std::vector<poca::core::BoundingBox> computeSelectedPlacement(std::vector<PlacementPreviewGroup>* _groups) const {
			if (m_methodCombo->currentIndex() == 0)
				return computeRegularGridPlacement(m_baseBBoxes, m_columnsSpin->value(), m_rowsSpin->value());
			if (m_methodCombo->currentIndex() == 1)
				return computePackPlacement(m_baseBBoxes, m_packMultiplierSpin->value());
			if (m_methodCombo->currentIndex() == 3)
				return computeOriginalPlacement(m_baseBBoxes, m_originalCenteredCheck->isChecked());
			QStringList labels;
			return computeCirclePlacement(m_baseBBoxes, groupsForLevel(m_levelSpin->value(), &labels), labels, _groups);
		}
		void updatePreview() {
			std::vector<PlacementPreviewGroup> groups;
			const auto bboxes = computeSelectedPlacement(&groups);
			std::vector<PlacementPreviewItem> items;
			buildPreviewItems(bboxes, items);
			for (auto& item : items) {
				for (int g = 0; g < int(groups.size()); ++g) {
					const QPointF c = item.rect.center();
					const double dx = c.x() - groups[g].center.x();
					const double dy = c.y() - groups[g].center.y();
					if (std::sqrt(dx * dx + dy * dy) <= groups[g].radius) { item.group = g; break; }
				}
			}
			m_preview->setPreview(items, groups);
		}
		MyMultipleObject* m_object{ nullptr };
		std::vector<DatasetAssemblerWidget::AssembledDatasetInfo> m_infos;
		std::vector<poca::core::BoundingBox> m_baseBBoxes;
		QComboBox* m_methodCombo{ nullptr };
		QSpinBox* m_columnsSpin{ nullptr };
		QSpinBox* m_rowsSpin{ nullptr };
		QDoubleSpinBox* m_packMultiplierSpin{ nullptr };
		QSpinBox* m_levelSpin{ nullptr };
		QLabel* m_columnsLabel{ nullptr };
		QLabel* m_rowsLabel{ nullptr };
		QLabel* m_packMultiplierLabel{ nullptr };
		QLabel* m_levelLabel{ nullptr };
		QCheckBox* m_originalCenteredCheck{ nullptr };
		PlacementPreviewWidget* m_preview{ nullptr };
	};

}

DatasetAssemblerWidget::DatasetAssemblerWidget(QWidget* _parent)
	: QWidget(_parent)
{
	setObjectName("DatasetAssemblerWidget");

	m_rootsList = new QListWidget(this);
	m_rootsList->setSelectionMode(QAbstractItemView::ExtendedSelection);

	m_addFolderButton = new QPushButton("Add folder(s)", this);
	m_removeFolderButton = new QPushButton("Remove selected", this);
	connect(m_addFolderButton, SIGNAL(released()), this, SLOT(onAddFolder()));
	connect(m_removeFolderButton, SIGNAL(released()), this, SLOT(onRemoveFolder()));

	QHBoxLayout* rootsButtonsLayout = new QHBoxLayout;
	rootsButtonsLayout->addWidget(m_addFolderButton);
	rootsButtonsLayout->addWidget(m_removeFolderButton);
	rootsButtonsLayout->addStretch(1);

	QGroupBox* rootsGroup = new QGroupBox("Root folders", this);
	QVBoxLayout* rootsLayout = new QVBoxLayout;
	rootsLayout->addLayout(rootsButtonsLayout);
	rootsLayout->addWidget(m_rootsList);
	rootsGroup->setLayout(rootsLayout);

	m_rulesTable = new QTableWidget(this);
	m_rulesTable->setColumnCount(6);
	QStringList headers;
	headers << "On" << "Req" << "Label" << "Relative folder" << "Filename regex" << "Key group";
	m_rulesTable->setHorizontalHeaderLabels(headers);
	m_rulesTable->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
	m_rulesTable->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Stretch);
	m_rulesTable->verticalHeader()->setVisible(false);
	m_rulesTable->setSelectionBehavior(QAbstractItemView::SelectRows);
	m_rulesTable->setSelectionMode(QAbstractItemView::SingleSelection);

	m_addRuleButton = new QPushButton("Add rule", this);
	m_removeRuleButton = new QPushButton("Remove rule", this);
	m_importJsonButton = new QPushButton("Import JSON", this);
	m_exportJsonButton = new QPushButton("Export JSON", this);
	connect(m_addRuleButton, SIGNAL(released()), this, SLOT(onAddRule()));
	connect(m_removeRuleButton, SIGNAL(released()), this, SLOT(onRemoveRule()));
	connect(m_importJsonButton, SIGNAL(released()), this, SLOT(onImportJson()));
	connect(m_exportJsonButton, SIGNAL(released()), this, SLOT(onExportJson()));
	connect(m_rulesTable, SIGNAL(itemChanged(QTableWidgetItem*)), this, SLOT(onRulesChanged()));

	QHBoxLayout* rulesButtonsLayout = new QHBoxLayout;
	rulesButtonsLayout->addWidget(m_addRuleButton);
	rulesButtonsLayout->addWidget(m_removeRuleButton);
	rulesButtonsLayout->addWidget(m_importJsonButton);
	rulesButtonsLayout->addWidget(m_exportJsonButton);
	rulesButtonsLayout->addStretch(1);

	QGroupBox* namingGroup = new QGroupBox("Naming", this);
	m_prefixRootNameCBox = new QCheckBox("Prefix dataset names with root folder name", this);
	m_prefixRootNameCBox->setChecked(true);
	m_choosePlacementButton = new QPushButton("Choose placement", this);
	m_choosePlacementButton->setCheckable(true);
	m_choosePlacementButton->setChecked(false);
	m_choosePlacementButton->setToolTip(tr("Open the dataset placement dialog before applying the multiple-object grid."));
	QLabel* separatorLabel = new QLabel("Separator", this);
	m_nameSeparatorEdit = new QLineEdit("_", this);
	m_nameSeparatorEdit->setMaximumWidth(60);
	QHBoxLayout* namingLayout = new QHBoxLayout;
	namingLayout->addWidget(m_prefixRootNameCBox);
	namingLayout->addStretch(1);
	namingLayout->addWidget(separatorLabel);
	namingLayout->addWidget(m_nameSeparatorEdit);
	namingGroup->setLayout(namingLayout);

	QGroupBox* rulesGroup = new QGroupBox("Component rules", this);
	QVBoxLayout* rulesLayout = new QVBoxLayout;
	rulesLayout->addLayout(rulesButtonsLayout);
	rulesLayout->addWidget(m_rulesTable);
	rulesLayout->addWidget(namingGroup);
	rulesGroup->setLayout(rulesLayout);

	m_previewTree = new QTreeWidget(this);
	m_previewTree->setColumnCount(2);
	m_previewTree->setHeaderLabels(QStringList() << "Hierarchy" << "Type");
	m_previewTree->header()->setSectionResizeMode(0, QHeaderView::Stretch);
	m_previewTree->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
	m_previewTree->setMinimumHeight(180);

	m_logEdit = new QTextEdit(this);
	m_logEdit->setReadOnly(true);
	m_logEdit->setMinimumHeight(180);

	m_previewButton = new QPushButton("Preview", this);
	m_assembleButton = new QPushButton("Assemble datasets", this);
	connect(m_previewButton, SIGNAL(released()), this, SLOT(onPreview()));
	connect(m_assembleButton, SIGNAL(released()), this, SLOT(onAssemble()));

	m_loadingProgressBar = new QProgressBar(this);
	m_loadingProgressBar->setMinimum(0);
	m_loadingProgressBar->setValue(0);
	m_loadingProgressBar->setTextVisible(true);
	m_loadingProgressBar->setFormat(tr("Loading datasets: %v / %m"));
	m_loadingProgressBar->setVisible(false);

	QHBoxLayout* actionsLayout = new QHBoxLayout;
	actionsLayout->addStretch(1);
	actionsLayout->addWidget(m_choosePlacementButton);
	actionsLayout->addWidget(m_previewButton);
	actionsLayout->addWidget(m_assembleButton);

	QVBoxLayout* mainLayout = new QVBoxLayout;
	mainLayout->addWidget(rootsGroup);
	mainLayout->addWidget(rulesGroup);
	mainLayout->addWidget(new QLabel("Hierarchy preview", this));
	mainLayout->addWidget(m_previewTree);
	mainLayout->addWidget(new QLabel("Log", this));
	mainLayout->addWidget(m_logEdit);
	mainLayout->addWidget(m_loadingProgressBar);
	mainLayout->addLayout(actionsLayout);
	setLayout(mainLayout);

	setRulesToTable({ defaultRule() });
	refreshRulesFeedback();
}

void DatasetAssemblerWidget::loadParameters(const nlohmann::json& _json)
{
	const std::string nameStr = objectName().toStdString();
	if (!_json.contains(nameStr))
		return;

	const nlohmann::json& json = _json[nameStr];

	m_rootsList->clear();
	if (json.contains("rootFolders")) {
		try {
			std::vector<std::string> roots = json["rootFolders"].get<std::vector<std::string>>();
			for (const std::string& root : roots)
				m_rootsList->addItem(root.c_str());
		}
		catch (nlohmann::json::exception&) {}
	}

	if (json.contains("lastRootPath")) {
		try { m_lastRootPath = json["lastRootPath"].get<std::string>().c_str(); }
		catch (nlohmann::json::exception&) {}
	}
	if (json.contains("prefixRootName")) {
		try { m_prefixRootNameCBox->setChecked(json["prefixRootName"].get<bool>()); }
		catch (nlohmann::json::exception&) {}
	}
	if (json.contains("choosePlacement")) {
		try { m_choosePlacementButton->setChecked(json["choosePlacement"].get<bool>()); }
		catch (nlohmann::json::exception&) {}
	}
	if (json.contains("nameSeparator")) {
		try { m_nameSeparatorEdit->setText(json["nameSeparator"].get<std::string>().c_str()); }
		catch (nlohmann::json::exception&) {}
	}
	if (json.contains("rules")) {
		std::vector<DatasetRule> rules;
		try {
			for (const auto& jsonRule : json["rules"]) {
				DatasetRule rule;
				if (jsonRule.contains("enabled")) rule.enabled = jsonRule["enabled"].get<bool>();
				if (jsonRule.contains("required")) rule.required = jsonRule["required"].get<bool>();
				if (jsonRule.contains("label")) rule.label = jsonRule["label"].get<std::string>().c_str();
				if (jsonRule.contains("relativeFolder")) rule.relativeFolder = jsonRule["relativeFolder"].get<std::string>().c_str();
				if (jsonRule.contains("regex")) rule.regex = jsonRule["regex"].get<std::string>().c_str();
				if (jsonRule.contains("keyCaptureGroup")) rule.keyCaptureGroup = jsonRule["keyCaptureGroup"].get<int>();
				rules.push_back(rule);
			}
		}
		catch (nlohmann::json::exception&) {}
		if (!rules.empty())
			setRulesToTable(rules);
	}
	refreshRulesFeedback();
}

void DatasetAssemblerWidget::saveParameters(nlohmann::json& _json) const
{
	const std::string nameStr = objectName().toStdString();
	nlohmann::json& json = _json[nameStr];
	json["lastRootPath"] = m_lastRootPath.toStdString();
	json["prefixRootName"] = m_prefixRootNameCBox->isChecked();
	json["nameSeparator"] = m_nameSeparatorEdit->text().toStdString();
	json["choosePlacement"] = m_choosePlacementButton->isChecked();

	std::vector<std::string> roots;
	for (int row = 0; row < m_rootsList->count(); ++row)
		roots.push_back(m_rootsList->item(row)->text().toStdString());
	json["rootFolders"] = roots;

	std::vector<nlohmann::json> rulesJson;
	for (const DatasetRule& rule : rulesFromTable()) {
		nlohmann::json jsonRule;
		jsonRule["enabled"] = rule.enabled;
		jsonRule["required"] = rule.required;
		jsonRule["label"] = rule.label.toStdString();
		jsonRule["relativeFolder"] = rule.relativeFolder.toStdString();
		jsonRule["regex"] = rule.regex.toStdString();
		jsonRule["keyCaptureGroup"] = rule.keyCaptureGroup;
		rulesJson.push_back(jsonRule);
	}
	json["rules"] = rulesJson;
}

void DatasetAssemblerWidget::onAddFolder()
{
	QString startPath = m_lastRootPath.isEmpty() ? QDir::currentPath() : m_lastRootPath;
	const QString folder = QFileDialog::getExistingDirectory(this, tr("Select root folder"), startPath, QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
	if (folder.isEmpty())
		return;

	m_lastRootPath = folder;
	const QString absFolder = QFileInfo(folder).absoluteFilePath();
	for (int row = 0; row < m_rootsList->count(); ++row)
		if (m_rootsList->item(row)->text() == absFolder)
			return;
	m_rootsList->addItem(absFolder);
	refreshRulesFeedback();
}

void DatasetAssemblerWidget::onRemoveFolder()
{
	qDeleteAll(m_rootsList->selectedItems());
	refreshRulesFeedback();
}

void DatasetAssemblerWidget::onAddRule()
{
	const int row = m_rulesTable->rowCount();
	m_rulesTable->insertRow(row);

	const DatasetRule rule = defaultRule();
	m_rulesTable->setItem(row, 0, createCheckItem(rule.enabled));
	m_rulesTable->setItem(row, 1, createCheckItem(rule.required));
	m_rulesTable->setItem(row, 2, createTextItem(rule.label));
	m_rulesTable->setItem(row, 3, createTextItem(rule.relativeFolder));
	m_rulesTable->setItem(row, 4, createTextItem(rule.regex));
	m_rulesTable->setItem(row, 5, createTextItem(QString::number(rule.keyCaptureGroup)));
	refreshRulesFeedback();
}

void DatasetAssemblerWidget::onRemoveRule()
{
	const int row = m_rulesTable->currentRow();
	if (row >= 0)
		m_rulesTable->removeRow(row);
	refreshRulesFeedback();
}

void DatasetAssemblerWidget::onImportJson()
{
	const QString startPath = m_lastRootPath.isEmpty() ? QDir::currentPath() : m_lastRootPath;
	const QString filename = QFileDialog::getOpenFileName(this, tr("Import assembler settings"), startPath, tr("JSON files (*.json)"));
	if (filename.isEmpty())
		return;

	try {
		std::ifstream fs(filename.toStdString());
		if (!fs)
			throw std::runtime_error("Could not open file");
		nlohmann::json json;
		fs >> json;
		loadParameters(json);
		appendLog(QString("Imported assembler settings from %1").arg(filename));
	}
	catch (const std::exception& e) {
		QMessageBox::warning(this, tr("Assembler"), tr("Failed to import JSON: %1").arg(e.what()));
	}
}

void DatasetAssemblerWidget::onExportJson()
{
	const QString startPath = m_lastRootPath.isEmpty() ? QDir::currentPath() : m_lastRootPath;
	const QString filename = QFileDialog::getSaveFileName(this, tr("Export assembler settings"), startPath + "/dataset_assembler.json", tr("JSON files (*.json)"));
	if (filename.isEmpty())
		return;

	try {
		nlohmann::json json;
		saveParameters(json);
		std::ofstream fs(filename.toStdString());
		if (!fs)
			throw std::runtime_error("Could not open file");
		fs << json.dump(2);
		appendLog(QString("Exported assembler settings to %1").arg(filename));
	}
	catch (const std::exception& e) {
		QMessageBox::warning(this, tr("Assembler"), tr("Failed to export JSON: %1").arg(e.what()));
	}
}

void DatasetAssemblerWidget::onPreview()
{
	m_logEdit->clear();

	const std::vector<DatasetRule> rules = rulesFromTable();
	QStringList roots;
	for (int row = 0; row < m_rootsList->count(); ++row)
		roots << m_rootsList->item(row)->text();

	QStringList errors, warnings;
	if (!validateConfiguration(rules, roots, errors, warnings)) {
		for (const QString& error : errors)
			appendLog(QString("Error: %1").arg(error));
		for (const QString& warning : warnings)
			appendLog(QString("Warning: %1").arg(warning));
		QMessageBox::warning(this, tr("Assembler"), tr("The assembler configuration is invalid. See the log for details."));
		return;
	}

	populatePreviewTree(roots, rules);

	const bool prefixRootName = m_prefixRootNameCBox->isChecked();
	const QString separator = m_nameSeparatorEdit->text().isEmpty() ? "_" : m_nameSeparatorEdit->text();
	int validDatasetCount = 0, skippedDatasetCount = 0;

	for (const QString& rootFolder : roots) {
		appendLog(QString("Scanning root folder: %1").arg(rootFolder));
		const ScanResult scan = scanRootFolder(rootFolder, rules);
		for (const QString& message : scan.messages)
			appendLog(message);
		appendLog(QString("Matched %1 file(s) across %2 dataset key(s).").arg(scan.matchedFiles).arg(scan.datasets.size()));

		for (auto it = scan.datasets.begin(); it != scan.datasets.end(); ++it) {
			const QString datasetKey = it.key();
			const DatasetEntry& entry = it.value();

			bool missingRequired = false;
			QStringList matchedLabels, missingLabels;
			for (int ruleIndex = 0; ruleIndex < (int)rules.size(); ++ruleIndex) {
				const DatasetRule& rule = rules[ruleIndex];
				if (!rule.enabled)
					continue;
				if (entry.filesByRule.contains(ruleIndex))
					matchedLabels << ruleDisplayName(rule, ruleIndex);
				else if (rule.required) {
					missingRequired = true;
					missingLabels << ruleDisplayName(rule, ruleIndex);
				}
			}

			const QString objectName = buildDatasetName(rootFolder, datasetKey, prefixRootName, separator);
			if (missingRequired || entry.filesByRule.isEmpty()) {
				appendLog(QString("Skip [%1] -> %2 | matched: %3 | missing required: %4")
					.arg(datasetKey)
					.arg(objectName.isEmpty() ? datasetKey : objectName)
					.arg(matchedLabels.join(", "))
					.arg(missingLabels.join(", ")));
				++skippedDatasetCount;
				continue;
			}

			appendLog(QString("Ready [%1] -> %2 | %3 component(s): %4")
				.arg(datasetKey)
				.arg(objectName.isEmpty() ? datasetKey : objectName)
				.arg(entry.filesByRule.size())
				.arg(matchedLabels.join(", ")));
			++validDatasetCount;
		}
	}

	appendLog(QString("Preview summary: %1 dataset(s) ready, %2 skipped.").arg(validDatasetCount).arg(skippedDatasetCount));
}

void DatasetAssemblerWidget::onAssemble()
{
	m_logEdit->clear();
	if (m_loadingProgressBar != nullptr) {
		m_loadingProgressBar->setVisible(false);
		m_loadingProgressBar->setValue(0);
	}

	const std::vector<DatasetRule> rules = rulesFromTable();
	QStringList roots;
	for (int row = 0; row < m_rootsList->count(); ++row)
		roots << m_rootsList->item(row)->text();

	QStringList errors, warnings;
	if (!validateConfiguration(rules, roots, errors, warnings)) {
		for (const QString& error : errors)
			appendLog(QString("Error: %1").arg(error));
		for (const QString& warning : warnings)
			appendLog(QString("Warning: %1").arg(warning));
		QMessageBox::warning(this, tr("Assembler"), tr("The assembler configuration is invalid. See the log for details."));
		return;
	}

	populatePreviewTree(roots, rules);

	ImageMemoryEstimate totalEstimate;
	std::vector<std::pair<QString, ScanResult>> rootScans;
	for (const QString& rootFolder : roots) {
		const ScanResult scan = scanRootFolder(rootFolder, rules);
		const ImageMemoryEstimate estimate = estimateImageMemoryForScan(scan);
		totalEstimate.cpuBytes += estimate.cpuBytes;
		totalEstimate.gpuBytes += estimate.gpuBytes;
		totalEstimate.imageFiles += estimate.imageFiles;
		totalEstimate.unreadableImageFiles += estimate.unreadableImageFiles;
		totalEstimate.messages << estimate.messages;
		rootScans.push_back(std::make_pair(rootFolder, scan));
	}

	bool useOutOfCore = false;
	bool usePyramidalRendering = false;
	if (!confirmImageMemoryPolicy(totalEstimate, useOutOfCore, usePyramidalRendering))
		return;

	poca::core::Engine* engine = poca::core::Engine::instance();

	//engine->setVerbose(true);
	engine->addVerboseType("debugPyramidalRendering");

	std::vector<poca::core::MyObjectInterface*> objects;
	std::vector<AssembledDatasetInfo> assembledInfos;
	const bool prefixRootName = m_prefixRootNameCBox->isChecked();
	const QString separator = m_nameSeparatorEdit->text().isEmpty() ? "_" : m_nameSeparatorEdit->text();

	int totalLoadSteps = 0;
	for (const auto& rootScan : rootScans) {
		const ScanResult& scan = rootScan.second;
		for (auto it = scan.datasets.begin(); it != scan.datasets.end(); ++it) {
			const DatasetEntry& entry = it.value();

			bool missingRequired = false;
			for (int ruleIndex = 0; ruleIndex < (int)rules.size(); ++ruleIndex) {
				const DatasetRule& rule = rules[ruleIndex];
				if (!rule.enabled || !rule.required)
					continue;
				if (!entry.filesByRule.contains(ruleIndex)) {
					missingRequired = true;
					break;
				}
			}

			if (!missingRequired && !entry.filesByRule.isEmpty())
				totalLoadSteps += entry.filesByRule.size();
		}
	}

	int loadedSteps = 0;
	if (m_loadingProgressBar != nullptr) {
		m_loadingProgressBar->setRange(0, std::max(1, totalLoadSteps));
		m_loadingProgressBar->setValue(0);
		m_loadingProgressBar->setFormat(tr("Loading datasets: %v / %m"));
		m_loadingProgressBar->setVisible(totalLoadSteps > 0);
	}
	QApplication::processEvents();

	auto advanceLoadingProgress = [&]() {
		++loadedSteps;
		if (m_loadingProgressBar != nullptr)
			m_loadingProgressBar->setValue(std::min(loadedSteps, std::max(1, totalLoadSteps)));
		QApplication::processEvents();
	};
	auto finishLoadingProgress = [&]() {
		if (m_loadingProgressBar != nullptr) {
			m_loadingProgressBar->setValue(std::max(1, totalLoadSteps));
			m_loadingProgressBar->setVisible(false);
		}
		QApplication::processEvents();
	};

	for (const auto& rootScan : rootScans) {
		const QString& rootFolder = rootScan.first;
		const ScanResult& scan = rootScan.second;
		appendLog(QString("Scanning root folder: %1").arg(rootFolder));
		for (const QString& message : scan.messages)
			appendLog(message);

		for (auto it = scan.datasets.begin(); it != scan.datasets.end(); ++it) {
			const QString datasetKey = it.key();
			const DatasetEntry& entry = it.value();

			bool missingRequired = false;
			for (int ruleIndex = 0; ruleIndex < (int)rules.size(); ++ruleIndex) {
				const DatasetRule& rule = rules[ruleIndex];
				if (!rule.enabled || !rule.required)
					continue;
				if (!entry.filesByRule.contains(ruleIndex)) {
					missingRequired = true;
					appendLog(QString("Dataset [%1] skipped, missing required rule [%2].").arg(datasetKey).arg(ruleDisplayName(rule, ruleIndex)));
				}
			}
			if (missingRequired || entry.filesByRule.isEmpty())
				continue;

			poca::core::CommandInfo firstLoadInfo(false, "open", "path", entry.filesByRule.begin().value().toStdString(), "outOfCore", useOutOfCore, "pyramidalRendering", usePyramidalRendering);
			poca::core::MyObjectInterface* object = engine->loadDataAndCreateObject(entry.filesByRule.begin().value(), &firstLoadInfo);
			advanceLoadingProgress();
			if (object == nullptr) {
				appendLog(QString("Failed to create object for dataset [%1] from %2").arg(datasetKey).arg(entry.filesByRule.begin().value()));
				continue;
			}

			bool valid = true;
			auto fileIt = entry.filesByRule.begin();
			++fileIt;
			for (; fileIt != entry.filesByRule.end(); ++fileIt) {
				poca::core::CommandInfo addInfo(false, "open", "path", fileIt.value().toStdString(), "outOfCore", useOutOfCore, "pyramidalRendering", usePyramidalRendering);
				const bool added = engine->loadDataAndAddToObject(fileIt.value(), object, &addInfo);
				advanceLoadingProgress();
				if (!added) {
					appendLog(QString("Failed to add component %1 to dataset [%2]").arg(fileIt.value()).arg(datasetKey));
					valid = false;
					break;
				}
			}

			if (!valid) {
				delete object;
				continue;
			}

			const QString objectName = buildDatasetName(rootFolder, datasetKey, prefixRootName, separator);
			if (!objectName.isEmpty())
				object->setName(objectName.toStdString());
			objects.push_back(object);
			assembledInfos.push_back({ rootFolder, datasetKey, objectName.isEmpty() ? datasetKey : objectName, hierarchySegmentsForDatasetFolder(rootFolder, entry.datasetFolder), object });
			appendLog(QString("Created dataset [%1] with %2 component(s).").arg(objectName.isEmpty() ? datasetKey : objectName).arg(entry.filesByRule.size()));
		}
	}

	if (objects.empty()) {
		finishLoadingProgress();
		QMessageBox::information(this, tr("Assembler"), tr("No dataset could be assembled with the current rules."));
		return;
	}

	poca::core::MyObjectInterface* createdObject = objects.size() == 1 ? objects.front() : engine->generateMultipleObject(objects);
	if (createdObject == nullptr) {
		finishLoadingProgress();
		QMessageBox::warning(this, tr("Assembler"), tr("The datasets were created but the final object could not be assembled."));
		return;
	}

	MyMultipleObject* multipleObject = dynamic_cast<MyMultipleObject*>(createdObject);
	if (multipleObject != nullptr) {
		populateHierarchy(multipleObject, assembledInfos);
		if (m_choosePlacementButton != nullptr && m_choosePlacementButton->isChecked()) {
			finishLoadingProgress();
			DatasetPlacementDialog dialog(multipleObject, assembledInfos, this);
			if (dialog.exec() == QDialog::Accepted) {
				multipleObject->setGridBBoxes(dialog.selectedPlacement());
				multipleObject->resetModelMatrices(true);
				appendLog(tr("Custom dataset placement applied."));
			}
			else {
				appendLog(tr("Custom placement canceled; keeping default packing."));
			}
		}
	}

	finishLoadingProgress();
	appendLog(QString("Created %1 object(s).").arg(objects.size()));
	emit transferNewObjectCreated(createdObject);
}

void DatasetAssemblerWidget::onRulesChanged()
{
	refreshRulesFeedback();
}

void DatasetAssemblerWidget::appendLog(const QString& _text)
{
	m_logEdit->append(_text);
}

std::vector<DatasetAssemblerWidget::DatasetRule> DatasetAssemblerWidget::rulesFromTable() const
{
	std::vector<DatasetRule> rules;
	for (int row = 0; row < m_rulesTable->rowCount(); ++row) {
		DatasetRule rule;
		if (m_rulesTable->item(row, 0) != nullptr)
			rule.enabled = m_rulesTable->item(row, 0)->checkState() == Qt::Checked;
		if (m_rulesTable->item(row, 1) != nullptr)
			rule.required = m_rulesTable->item(row, 1)->checkState() == Qt::Checked;
		if (m_rulesTable->item(row, 2) != nullptr)
			rule.label = m_rulesTable->item(row, 2)->text();
		if (m_rulesTable->item(row, 3) != nullptr)
			rule.relativeFolder = m_rulesTable->item(row, 3)->text();
		if (m_rulesTable->item(row, 4) != nullptr)
			rule.regex = m_rulesTable->item(row, 4)->text();
		if (m_rulesTable->item(row, 5) != nullptr)
			rule.keyCaptureGroup = std::max(0, m_rulesTable->item(row, 5)->text().toInt());
		if (!rule.regex.trimmed().isEmpty())
			rules.push_back(rule);
	}
	return rules;
}

void DatasetAssemblerWidget::setRulesToTable(const std::vector<DatasetRule>& _rules)
{
	m_rulesTable->setRowCount(0);
	for (const DatasetRule& rule : _rules) {
		const int row = m_rulesTable->rowCount();
		m_rulesTable->insertRow(row);
		m_rulesTable->setItem(row, 0, createCheckItem(rule.enabled));
		m_rulesTable->setItem(row, 1, createCheckItem(rule.required));
		m_rulesTable->setItem(row, 2, createTextItem(rule.label));
		m_rulesTable->setItem(row, 3, createTextItem(rule.relativeFolder));
		m_rulesTable->setItem(row, 4, createTextItem(rule.regex));
		m_rulesTable->setItem(row, 5, createTextItem(QString::number(rule.keyCaptureGroup)));
	}
	refreshRulesFeedback();
}

DatasetAssemblerWidget::DatasetRule DatasetAssemblerWidget::defaultRule() const
{
	DatasetRule rule;
	rule.label = "raw";
	rule.regex = "(.*)";
	return rule;
}

bool DatasetAssemblerWidget::validateConfiguration(const std::vector<DatasetRule>& _rules, const QStringList& _roots, QStringList& _errors, QStringList& _warnings) const
{
	if (_rules.empty())
		_errors << "Please define at least one rule with a regex.";
	if (_roots.isEmpty())
		_errors << "Please add at least one root folder.";

	QSet<QString> enabledLabels;
	bool hasEnabledRule = false;
	for (int index = 0; index < (int)_rules.size(); ++index) {
		const DatasetRule& rule = _rules[index];
		if (!rule.enabled)
			continue;

		hasEnabledRule = true;
		const QString displayName = ruleDisplayName(rule, index);
		const QString trimmedRegex = rule.regex.trimmed();
		if (trimmedRegex.isEmpty())
			_errors << QString("Rule [%1] has an empty regex.").arg(displayName);

		const QRegularExpression regex(trimmedRegex);
		if (!regex.isValid())
			_errors << QString("Rule [%1] has an invalid regex: %2").arg(displayName).arg(regex.errorString());
		else if (rule.keyCaptureGroup > regex.captureCount())
			_errors << QString("Rule [%1] requests key capture group %2 but the regex only defines %3 capture group(s).")
				.arg(displayName).arg(rule.keyCaptureGroup).arg(regex.captureCount());

		const QString labelKey = rule.label.trimmed().toLower();
		if (!labelKey.isEmpty()) {
			if (enabledLabels.contains(labelKey))
				_warnings << QString("Rule label [%1] is used more than once.").arg(rule.label.trimmed());
			enabledLabels.insert(labelKey);
		}
	}

	if (!hasEnabledRule)
		_errors << "At least one rule must be enabled.";

	for (const QString& root : _roots) {
		const QFileInfo info(root);
		if (!info.exists() || !info.isDir())
			_errors << QString("Root folder does not exist: %1").arg(root);
	}

	return _errors.isEmpty();
}

DatasetAssemblerWidget::ScanResult DatasetAssemblerWidget::scanRootFolder(const QString& _rootFolder, const std::vector<DatasetRule>& _rules) const
{
	ScanResult result;
	const QStringList datasetFolders = discoverDatasetFolders(_rootFolder, _rules);
	result.messages << QString("Discovered %1 dataset folder(s) under root [%2].").arg(datasetFolders.size()).arg(_rootFolder);

	for (const QString& datasetFolder : datasetFolders) {
		const QString datasetFolderName = QFileInfo(datasetFolder).fileName().isEmpty() ? datasetFolder : QFileInfo(datasetFolder).fileName();
		for (int ruleIndex = 0; ruleIndex < (int)_rules.size(); ++ruleIndex) {
			const DatasetRule& rule = _rules[ruleIndex];
			if (!rule.enabled)
				continue;

			QDir baseDir(datasetFolder);
			if (!rule.relativeFolder.trimmed().isEmpty())
				baseDir = QDir(baseDir.filePath(rule.relativeFolder.trimmed()));

			if (!baseDir.exists()) {
				result.messages << QString("Dataset folder [%1], rule [%2] skipped, folder does not exist: %3")
					.arg(datasetFolderName).arg(ruleDisplayName(rule, ruleIndex)).arg(baseDir.absolutePath());
				continue;
			}

			const QRegularExpression regex(rule.regex);
			if (!regex.isValid()) {
				result.messages << QString("Rule [%1] skipped, invalid regex: %2").arg(ruleDisplayName(rule, ruleIndex)).arg(rule.regex);
				continue;
			}

			int matchedForRule = 0;
			QDirIterator it(baseDir.absolutePath(), QDir::Files | QDir::NoDotAndDotDot, QDirIterator::NoIteratorFlags);
			while (it.hasNext()) {
				const QString absPath = it.next();
				const QFileInfo info(absPath);
				const QRegularExpressionMatch match = regex.match(info.fileName());
				if (!match.hasMatch())
					continue;

				QString key = match.captured(rule.keyCaptureGroup);
				if (key.isEmpty())
					key = datasetFolderName;

				DatasetEntry& entry = result.datasets[key];
				if (entry.datasetFolder.isEmpty())
					entry.datasetFolder = datasetFolder;
				if (entry.filesByRule.contains(ruleIndex)) {
					result.messages << QString("Duplicate match ignored for dataset [%1], rule [%2]: %3").arg(key).arg(ruleDisplayName(rule, ruleIndex)).arg(absPath);
					continue;
				}
				entry.filesByRule[ruleIndex] = absPath;
				++matchedForRule;
				++result.matchedFiles;
			}

			if (matchedForRule > 0)
				result.messages << QString("Dataset folder [%1], rule [%2] matched %3 file(s).").arg(datasetFolderName).arg(ruleDisplayName(rule, ruleIndex)).arg(matchedForRule);
		}
	}
	return result;
}

void DatasetAssemblerWidget::refreshRulesFeedback()
{
	QSignalBlocker blocker(m_rulesTable);
	QSet<QString> labels;
	for (int row = 0; row < m_rulesTable->rowCount(); ++row) {
		QTableWidgetItem* labelItem = m_rulesTable->item(row, 2);
		QTableWidgetItem* regexItem = m_rulesTable->item(row, 4);
		QTableWidgetItem* groupItem = m_rulesTable->item(row, 5);
		QTableWidgetItem* enabledItem = m_rulesTable->item(row, 0);
		const bool enabled = enabledItem != nullptr && enabledItem->checkState() == Qt::Checked;

		QString tooltip;
		QColor normalColor = palette().base().color();
		QColor warningColor(255, 244, 204);
		QColor errorColor(255, 220, 220);

		if (regexItem != nullptr) {
			regexItem->setBackground(normalColor);
			const QString regexText = regexItem->text().trimmed();
			if (enabled && regexText.isEmpty()) {
				regexItem->setBackground(errorColor);
				tooltip += "Empty regex. ";
			}
			else if (enabled) {
				const QRegularExpression regex(regexText);
				if (!regex.isValid()) {
					regexItem->setBackground(errorColor);
					tooltip += QString("Invalid regex: %1. ").arg(regex.errorString());
				}
				else if (groupItem != nullptr && groupItem->text().toInt() > regex.captureCount()) {
					groupItem->setBackground(errorColor);
					tooltip += QString("Key group exceeds regex capture count (%1). ").arg(regex.captureCount());
				}
				else if (groupItem != nullptr) {
					groupItem->setBackground(normalColor);
				}
			}
			else if (groupItem != nullptr) {
				groupItem->setBackground(normalColor);
			}
		}

		if (labelItem != nullptr) {
			labelItem->setBackground(normalColor);
			const QString key = labelItem->text().trimmed().toLower();
			if (enabled && !key.isEmpty()) {
				if (labels.contains(key)) {
					labelItem->setBackground(warningColor);
					tooltip += "Duplicate rule label. ";
				}
				labels.insert(key);
			}
			labelItem->setToolTip(tooltip.trimmed());
		}
		if (regexItem != nullptr)
			regexItem->setToolTip(tooltip.trimmed());
		if (groupItem != nullptr)
			groupItem->setToolTip(tooltip.trimmed());
	}
}

QString DatasetAssemblerWidget::ruleDisplayName(const DatasetRule& _rule, int _index) const
{
	return _rule.label.trimmed().isEmpty() ? QString::number(_index + 1) : _rule.label.trimmed();
}

QStringList DatasetAssemblerWidget::splitPathSegments(const QString& _path) const
{
	return QDir::fromNativeSeparators(_path).split('/', Qt::SkipEmptyParts);
}

bool DatasetAssemblerWidget::rulesUseRelativeFolders(const std::vector<DatasetRule>& _rules) const
{
	for (const DatasetRule& rule : _rules)
		if (rule.enabled && !rule.relativeFolder.trimmed().isEmpty())
			return true;
	return false;
}

bool DatasetAssemblerWidget::folderContainsDatasetContent(const QString& _folderPath, const std::vector<DatasetRule>& _rules) const
{
	for (const DatasetRule& rule : _rules) {
		if (!rule.enabled)
			continue;

		QDir baseDir(_folderPath);
		if (!rule.relativeFolder.trimmed().isEmpty())
			baseDir = QDir(baseDir.filePath(rule.relativeFolder.trimmed()));
		if (!baseDir.exists())
			continue;

		const QRegularExpression regex(rule.regex);
		if (!regex.isValid())
			continue;

		QDirIterator it(baseDir.absolutePath(), QDir::Files | QDir::NoDotAndDotDot, QDirIterator::NoIteratorFlags);
		while (it.hasNext()) {
			const QString absPath = it.next();
			const QFileInfo info(absPath);
			if (regex.match(info.fileName()).hasMatch())
				return true;
		}
	}
	return false;
}

QStringList DatasetAssemblerWidget::discoverDatasetFolders(const QString& _rootFolder, const std::vector<DatasetRule>& _rules) const
{
	QStringList datasetFolders;
	const bool stopAboveRelativeFolders = rulesUseRelativeFolders(_rules);
	QDirIterator it(_rootFolder, QDir::Dirs | QDir::NoDotAndDotDot, QDirIterator::Subdirectories);

	QStringList allDirs;
	allDirs << QFileInfo(_rootFolder).absoluteFilePath();
	while (it.hasNext())
		allDirs << QFileInfo(it.next()).absoluteFilePath();
	std::sort(allDirs.begin(), allDirs.end(), [](const QString& a, const QString& b) { return a.count('/') < b.count('/'); });

	for (const QString& dirPath : allDirs) {
		QDir dir(dirPath);
		const QFileInfoList childDirs = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
		if (stopAboveRelativeFolders) {
			if (folderContainsDatasetContent(dirPath, _rules))
				datasetFolders << dirPath;
		}
		else if (childDirs.isEmpty() && folderContainsDatasetContent(dirPath, _rules)) {
			datasetFolders << dirPath;
		}
	}

	datasetFolders.removeDuplicates();
	return datasetFolders;
}

QStringList DatasetAssemblerWidget::hierarchySegmentsForDatasetFolder(const QString& _rootFolder, const QString& _datasetFolder) const
{
	QString relativeDir = QDir(QFileInfo(_rootFolder).absoluteFilePath()).relativeFilePath(QFileInfo(_datasetFolder).absoluteFilePath());
	QStringList segments = splitPathSegments(relativeDir);
	if (!segments.isEmpty() && segments.front() == ".")
		segments.removeFirst();
	return segments;
}

QTreeWidgetItem* DatasetAssemblerWidget::ensurePreviewNode(QTreeWidgetItem* _parent, const QString& _label, const QString& _type)
{
	const int childCount = _parent == nullptr ? m_previewTree->topLevelItemCount() : _parent->childCount();
	for (int index = 0; index < childCount; ++index) {
		QTreeWidgetItem* item = _parent == nullptr ? m_previewTree->topLevelItem(index) : _parent->child(index);
		if (item != nullptr && item->text(0) == _label && (_type.isEmpty() || item->text(1) == _type))
			return item;
	}

	QTreeWidgetItem* item = new QTreeWidgetItem(QStringList() << _label << _type);
	if (_parent == nullptr)
		m_previewTree->addTopLevelItem(item);
	else
		_parent->addChild(item);
	return item;
}

void DatasetAssemblerWidget::populatePreviewTree(const QStringList& _roots, const std::vector<DatasetRule>& _rules)
{
	m_previewTree->clear();
	for (const QString& rootFolder : _roots) {
		const QString rootName = QFileInfo(rootFolder).fileName().isEmpty() ? rootFolder : QFileInfo(rootFolder).fileName();
		QTreeWidgetItem* rootItem = ensurePreviewNode(nullptr, QString("%1 [%2]").arg(rootName, rootFolder), "Root");
		const ScanResult scan = scanRootFolder(rootFolder, _rules);
		for (auto it = scan.datasets.begin(); it != scan.datasets.end(); ++it) {
			const QString datasetKey = it.key();
			const DatasetEntry& entry = it.value();
			const QString datasetFolder = entry.datasetFolder.isEmpty() ? rootFolder : entry.datasetFolder;
			const QStringList segments = hierarchySegmentsForDatasetFolder(rootFolder, datasetFolder);

			QTreeWidgetItem* parentItem = rootItem;
			for (const QString& segment : segments)
				parentItem = ensurePreviewNode(parentItem, segment, "Level");

			bool missingRequired = false;
			for (int ruleIndex = 0; ruleIndex < (int)_rules.size(); ++ruleIndex) {
				const DatasetRule& rule = _rules[ruleIndex];
				if (rule.enabled && rule.required && !entry.filesByRule.contains(ruleIndex)) {
					missingRequired = true;
					break;
				}
			}
			QTreeWidgetItem* datasetItem = ensurePreviewNode(parentItem, datasetKey, missingRequired ? "Dataset (incomplete)" : "Dataset");
			for (auto fileIt = entry.filesByRule.begin(); fileIt != entry.filesByRule.end(); ++fileIt) {
				const DatasetRule& rule = _rules[fileIt.key()];
				ensurePreviewNode(datasetItem, QFileInfo(fileIt.value()).fileName(), ruleDisplayName(rule, fileIt.key()));
			}
		}
	}
	m_previewTree->expandAll();
}

DatasetAssemblerWidget::ImageMemoryEstimate DatasetAssemblerWidget::estimateImageMemoryForScan(const ScanResult& _scan) const
{
	ImageMemoryEstimate estimate;
	QSet<QString> seenFiles;
	for (auto it = _scan.datasets.begin(); it != _scan.datasets.end(); ++it) {
		const DatasetEntry& entry = it.value();
		for (auto fileIt = entry.filesByRule.begin(); fileIt != entry.filesByRule.end(); ++fileIt) {
			const QString filename = QFileInfo(fileIt.value()).absoluteFilePath();
			if (seenFiles.contains(filename) || !isTiffFilename(filename))
				continue;
			seenFiles.insert(filename);

			uint64_t bytes = 0;
			if (estimateTiffImageBytes(filename, bytes)) {
				estimate.cpuBytes += bytes;
				estimate.gpuBytes += bytes;
				++estimate.imageFiles;
			}
			else {
				++estimate.unreadableImageFiles;
				estimate.messages << QString("Could not estimate TIFF memory: %1").arg(filename);
			}
		}
	}
	return estimate;
}

bool DatasetAssemblerWidget::estimateTiffImageBytes(const QString& _filename, uint64_t& _bytes) const
{
	_bytes = 0;
	TinyTIFFReaderFile* tiffr = TinyTIFFReader_open(_filename.toStdString().c_str());
	if (!tiffr)
		return false;

	if (TinyTIFFReader_wasError(tiffr)) {
		TinyTIFFReader_close(tiffr);
		return false;
	}

	const uint32_t width = TinyTIFFReader_getWidth(tiffr);
	const uint32_t height = TinyTIFFReader_getHeight(tiffr);
	const uint16_t bitsPerSample = TinyTIFFReader_getBitsPerSample(tiffr, 0);
	const uint16_t samplesPerPixel = TinyTIFFReader_getSamplesPerPixel(tiffr);
	uint32_t depth = TinyTIFFReader_countFrames(tiffr);
	if (depth == 0)
		depth = 1;
	TinyTIFFReader_close(tiffr);

	if (width == 0 || height == 0 || bitsPerSample == 0 || samplesPerPixel == 0)
		return false;

	const uint64_t bytesPerSample = std::max<uint64_t>(1, static_cast<uint64_t>(bitsPerSample + 7) / 8);
	_bytes = safeMul(safeMul(safeMul(safeMul(width, height), depth), samplesPerPixel), bytesPerSample);
	return _bytes > 0;
}

DatasetAssemblerWidget::RuntimeMemoryStatus DatasetAssemblerWidget::queryRuntimeMemoryStatus() const
{
	RuntimeMemoryStatus status;

#if defined(_WIN32)
	MEMORYSTATUSEX mem;
	mem.dwLength = sizeof(mem);
	if (GlobalMemoryStatusEx(&mem)) {
		status.availableCpuBytes = static_cast<uint64_t>(mem.ullAvailPhys);
		status.hasCpuBytes = true;
	}
#elif defined(__APPLE__)
	mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
	vm_statistics64_data_t vmstat;
	if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vmstat), &count) == KERN_SUCCESS) {
		uint64_t pageSize = 0;
		size_t pageSizeLen = sizeof(pageSize);
		if (sysctlbyname("hw.pagesize", &pageSize, &pageSizeLen, nullptr, 0) == 0) {
			status.availableCpuBytes = static_cast<uint64_t>(vmstat.free_count + vmstat.inactive_count) * pageSize;
			status.hasCpuBytes = true;
		}
	}
#else
	struct sysinfo info;
	if (sysinfo(&info) == 0) {
		status.availableCpuBytes = static_cast<uint64_t>(info.freeram) * static_cast<uint64_t>(info.mem_unit);
		status.hasCpuBytes = true;
	}
#endif

	QOpenGLContext* context = QOpenGLContext::currentContext();
	if (context != nullptr) {
		QOpenGLFunctions* functions = context->functions();
		const QSet<QByteArray> extensions = context->extensions();
		GLint value = 0;
		if (extensions.contains("GL_NVX_gpu_memory_info")) {
			functions->glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &value);
			if (value > 0) {
				status.availableGpuBytes = static_cast<uint64_t>(value) * 1024ull;
				status.hasGpuBytes = true;
			}
		}
		else if (extensions.contains("GL_ATI_meminfo")) {
			GLint values[4] = { 0, 0, 0, 0 };
			functions->glGetIntegerv(GL_TEXTURE_FREE_MEMORY_ATI, values);
			if (values[0] > 0) {
				status.availableGpuBytes = static_cast<uint64_t>(values[0]) * 1024ull;
				status.hasGpuBytes = true;
			}
		}
	}

	return status;
}

QString DatasetAssemblerWidget::formatBytes(const uint64_t _bytes) const
{
	const double gib = static_cast<double>(_bytes) / (1024.0 * 1024.0 * 1024.0);
	const double mib = static_cast<double>(_bytes) / (1024.0 * 1024.0);
	if (gib >= 1.0)
		return QString::number(gib, 'f', 2) + " GiB";
	return QString::number(mib, 'f', 1) + " MiB";
}

bool DatasetAssemblerWidget::confirmImageMemoryPolicy(const ImageMemoryEstimate& _estimate, bool& _outOfCore, bool& _pyramidalRendering) const
{
	_outOfCore = false;
	_pyramidalRendering = false;
	if (_estimate.imageFiles == 0)
		return true;

	const RuntimeMemoryStatus status = queryRuntimeMemoryStatus();
	const double safety = 0.75;
	const uint64_t estimatedProcessBytesWithoutPyramid = _estimate.cpuBytes + _estimate.gpuBytes;
	const bool cpuExceeded = status.hasCpuBytes && static_cast<double>(_estimate.cpuBytes) > static_cast<double>(status.availableCpuBytes) * safety;
	const bool processExceeded = status.hasCpuBytes && static_cast<double>(estimatedProcessBytesWithoutPyramid) > static_cast<double>(status.availableCpuBytes) * safety;
	const bool gpuExceeded = status.hasGpuBytes && static_cast<double>(_estimate.gpuBytes) > static_cast<double>(status.availableGpuBytes) * safety;

	_outOfCore = cpuExceeded;
	_pyramidalRendering = gpuExceeded || processExceeded;

	if (!cpuExceeded && !gpuExceeded && !processExceeded)
		return true;

	QDialog dialog(const_cast<DatasetAssemblerWidget*>(this));
	dialog.setWindowTitle(tr("Image memory sanity check"));
	QVBoxLayout* layout = new QVBoxLayout(&dialog);

	QString message = tr("The assembler found %1 TIFF image(s). Estimated uncompressed image memory:").arg(_estimate.imageFiles);
	layout->addWidget(new QLabel(message, &dialog));
	layout->addWidget(new QLabel(tr("CPU RAM retained by full image pixels: %1").arg(formatBytes(_estimate.cpuBytes)), &dialog));
	layout->addWidget(new QLabel(tr("Full-resolution GPU texture memory if pyramidal rendering is disabled: %1").arg(formatBytes(_estimate.gpuBytes)), &dialog));
	layout->addWidget(new QLabel(tr("Possible Windows process memory impact without pyramidal rendering: %1").arg(formatBytes(estimatedProcessBytesWithoutPyramid)), &dialog));
	layout->addWidget(new QLabel(status.hasCpuBytes ? tr("Available CPU RAM: %1").arg(formatBytes(status.availableCpuBytes)) : tr("Available CPU RAM: unknown"), &dialog));
	layout->addWidget(new QLabel(status.hasGpuBytes ? tr("Available GPU memory: %1").arg(formatBytes(status.availableGpuBytes)) : tr("Available GPU memory: unknown"), &dialog));
	if (_estimate.unreadableImageFiles > 0)
		layout->addWidget(new QLabel(tr("Warning: %1 TIFF image(s) could not be estimated.").arg(_estimate.unreadableImageFiles), &dialog));

	QCheckBox* outOfCoreBox = new QCheckBox(tr("Use out-of-core image storage"), &dialog);
	outOfCoreBox->setChecked(_outOfCore);
	QCheckBox* pyramidBox = new QCheckBox(tr("Use pyramidal rendering"), &dialog);
	pyramidBox->setChecked(_pyramidalRendering);
	layout->addWidget(outOfCoreBox);
	layout->addWidget(pyramidBox);

	QDialogButtonBox* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
	layout->addWidget(buttons);
	connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
	connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);

	if (dialog.exec() != QDialog::Accepted)
		return false;

	_outOfCore = outOfCoreBox->isChecked();
	_pyramidalRendering = pyramidBox->isChecked();
	return true;
}

void DatasetAssemblerWidget::populateHierarchy(MyMultipleObject* _multipleObject, const std::vector<AssembledDatasetInfo>& _assembledInfos) const
{
	if (_multipleObject == nullptr)
		return;

	_multipleObject->clearHierarchy();
	if (_assembledInfos.empty())
		return;

	std::map<std::string, size_t> nodeByKey;
	for (const AssembledDatasetInfo& info : _assembledInfos) {
		if (info.object == nullptr)
			continue;

		size_t objectIndex = 0;
		bool foundObject = false;
		for (; objectIndex < _multipleObject->nbColors(); ++objectIndex) {
			if (_multipleObject->getObject(objectIndex) == info.object) {
				foundObject = true;
				break;
			}
		}
		if (!foundObject)
			continue;

		const QString rootLabel = QFileInfo(info.rootFolder).fileName().isEmpty() ? info.rootFolder : QFileInfo(info.rootFolder).fileName();
		QStringList hierarchyPath;
		hierarchyPath << rootLabel;
		for (const QString& segment : info.hierarchySegments)
			hierarchyPath << segment;

		std::string pathKey;
		int parentIndex = -1;
		for (int segmentIndex = 0; segmentIndex < hierarchyPath.size(); ++segmentIndex) {
			const QString segment = hierarchyPath[segmentIndex];
			if (!pathKey.empty())
				pathKey += "/";
			pathKey += segment.toStdString();
			auto nodeIt = nodeByKey.find(pathKey);
			if (nodeIt == nodeByKey.end()) {
				const size_t newIndex = _multipleObject->addHierarchyNode(segment.toStdString(), QString("Level %1").arg(segmentIndex + 1).toStdString(), parentIndex);
				nodeByKey[pathKey] = newIndex;
				parentIndex = (int)newIndex;
			}
			else {
				parentIndex = (int)nodeIt->second;
			}
		}

		_multipleObject->attachObjectToHierarchyNode(parentIndex, objectIndex);
	}
}
