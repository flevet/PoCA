/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ButtonLayer.cpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*/

#include "ButtonLayer.hpp"

#include <algorithm>
#include <map>
#include <string>

#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QColorDialog>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QMenu>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QSizePolicy>
#include <QtGui/QIcon>
#include <QtGui/QPainter>
#include <QtGui/QPixmap>

#include <General/Palette.hpp>
#include <Plot/Icons.hpp>
#include <Plot/Misc.h>

namespace poca::qt {

	ButtonLayer::ButtonLayer(const nlohmann::json& _configuration, QWidget* _parent) : QWidget(_parent), m_configuration(_configuration)
	{
		m_iconSize = intValue(m_configuration, "iconSize", 20);
		buildFromConfiguration();
	}

	ButtonLayer* ButtonLayer::create(const nlohmann::json& _configuration, QWidget* _parent)
	{
		return new ButtonLayer(_configuration, _parent);
	}

	ButtonLayer* generateButtonLayer(const nlohmann::json& _configuration, QWidget* _parent)
	{
		return ButtonLayer::create(_configuration, _parent);
	}

	QPushButton* ButtonLayer::button(const QString& _identifier) const
	{
		for (const CreatedButton& b : m_buttons)
			if (b.identifier == _identifier)
				return b.button;
		return nullptr;
	}

	QPushButton* ButtonLayer::paletteButton(const QString& _paletteName) const
	{
		for (const CreatedButton& b : m_paletteButtons)
			if (b.identifier == _paletteName)
				return b.button;
		return nullptr;
	}

	nlohmann::json ButtonLayer::saveConfiguration() const
	{
		nlohmann::json config = m_configuration;
		if (!config.contains("palettes") || !config["palettes"].is_array())
			return config;

		for (auto& item : config["palettes"]) {
			QString name = stringValue(item, "name");
			QPushButton* b = paletteButton(name);
			if (b != nullptr)
				item["name"] = b->property("poca_palette").toString().toStdString();
		}
		return config;
	}

	void ButtonLayer::buildFromConfiguration()
	{
		QHBoxLayout* line1 = new QHBoxLayout;
		QHBoxLayout* line2 = nullptr;
		const bool allowSecondLine = boolValue(m_configuration, "allowSecondLine", true);
		if (allowSecondLine)
			line2 = new QHBoxLayout;

		const int maxPalettesOnFirstLine = intValue(m_configuration, "maxPalettesOnFirstLine", 9);
		int paletteIndex = 0;
		if (m_configuration.contains("palettes") && m_configuration["palettes"].is_array())
			for (const auto& item : m_configuration["palettes"]) {
				addPaletteButton(item, line1, line2, maxPalettesOnFirstLine, paletteIndex);
				++paletteIndex;
			}

		QWidget* spacer = new QWidget;
		spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
		line1->addWidget(spacer);

		if (m_configuration.contains("actions") && m_configuration["actions"].is_array())
			for (const auto& item : m_configuration["actions"])
				addActionButton(item, line1, m_iconSize);

		QVBoxLayout* mainLayout = new QVBoxLayout;
		mainLayout->setContentsMargins(0, 0, 0, 0);
		mainLayout->setSpacing(0);
		mainLayout->addLayout(line1);
		if (line2 != nullptr && line2->count() > 0) {
			QWidget* spacer2 = new QWidget;
			spacer2->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
			line2->addWidget(spacer2);
			mainLayout->addLayout(line2);
		}
		setLayout(mainLayout);
	}

	void ButtonLayer::addPaletteButton(const nlohmann::json& _item, QHBoxLayout* _line1, QHBoxLayout* _line2, const int _maxPalettesOnFirstLine, const int _paletteIndex)
	{
		QString paletteName = stringValue(_item, "name");
		if (paletteName.isEmpty() || !isKnownPalette(paletteName.toStdString()))
			return;

		QPushButton* button = new QPushButton;
		button->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
		button->setMaximumSize(QSize(m_iconSize, m_iconSize));
		button->setProperty("poca_palette", paletteName);
		button->setToolTip(stringValue(_item, "tooltip", paletteName));
		button->setContextMenuPolicy(Qt::CustomContextMenu);
		updatePaletteButtonIcon(button, paletteName);

		QHBoxLayout* targetLine = (_line2 != nullptr && _paletteIndex >= _maxPalettesOnFirstLine) ? _line2 : _line1;
		targetLine->addWidget(button, 0, Qt::AlignLeft);
		connect(button, SIGNAL(pressed()), this, SLOT(processPaletteButton()));
		connect(button, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(processPaletteContextMenu(const QPoint&)));

		m_paletteButtons.push_back({ button, paletteName, "palette" });
	}

	void ButtonLayer::addActionButton(const nlohmann::json& _item, QHBoxLayout* _layout, const int _maxSize)
	{
		QString identifier = stringValue(_item, "identifier");
		if (identifier.isEmpty())
			identifier = stringValue(_item, "id");
		if (identifier.isEmpty())
			return;

		QPushButton* button = new QPushButton;
		button->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
		button->setMaximumSize(QSize(_maxSize, _maxSize));
		button->setProperty("poca_action", identifier);
		button->setToolTip(stringValue(_item, "tooltip", identifier));
		button->setIcon(iconFromIdentifier(stringValue(_item, "icon", identifier)));
		if (boolValue(_item, "checkable", false)) {
			button->setCheckable(true);
			button->setChecked(boolValue(_item, "checked", false));
			connect(button, SIGNAL(toggled(bool)), this, SLOT(processActionButton(bool)));
		}
		else
			connect(button, SIGNAL(pressed()), this, SLOT(processActionButton()));

		if (boolValue(_item, "exclusive", false)) {
			if (m_exclusiveActions == nullptr)
				m_exclusiveActions = new QButtonGroup(this);
			m_exclusiveActions->addButton(button);
		}

		_layout->addWidget(button, 0, Qt::AlignRight);
		m_buttons.push_back({ button, identifier, "action" });
	}

	void ButtonLayer::processPaletteButton()
	{
		QPushButton* button = qobject_cast<QPushButton*>(sender());
		if (button == nullptr)
			return;
		emit palettePressed(button->property("poca_palette").toString());
	}

	void ButtonLayer::processPaletteContextMenu(const QPoint& _pos)
	{
		QPushButton* button = qobject_cast<QPushButton*>(sender());
		if (button == nullptr)
			return;

		QMenu* menu = buildPaletteMenu(button);
		QAction* selected = menu->exec(button->mapToGlobal(_pos));
		if (selected == nullptr)
			return;

		QString previousPalette = button->property("poca_palette").toString();
		if (selected->data().toString() == "__custom_color__") {
			QColor color = QColorDialog::getColor(Qt::white, this, tr("Choose palette color"));
			if (!color.isValid())
				return;
			QString newName = QString("Custom:%1,%2,%3").arg(color.red()).arg(color.green()).arg(color.blue());
			button->setProperty("poca_palette", newName);
			updateMonochromePaletteButtonIcon(button, color);
			emit paletteChanged(previousPalette, newName);
			return;
		}

		QString newPalette = selected->data().toString();
		if (newPalette.isEmpty() || newPalette == previousPalette)
			return;
		button->setProperty("poca_palette", newPalette);
		button->setToolTip(newPalette);
		updatePaletteButtonIcon(button, newPalette);
		emit paletteChanged(previousPalette, newPalette);
	}

	void ButtonLayer::processActionButton()
	{
		QPushButton* button = qobject_cast<QPushButton*>(sender());
		if (button == nullptr)
			return;
		emit actionPressed(button->property("poca_action").toString());
	}

	void ButtonLayer::processActionButton(const bool _checked)
	{
		QPushButton* button = qobject_cast<QPushButton*>(sender());
		if (button == nullptr)
			return;
		emit actionToggled(button->property("poca_action").toString(), _checked);
	}

	QMenu* ButtonLayer::buildPaletteMenu(QPushButton* _button) const
	{
		QMenu* menu = new QMenu(_button);
		const QString current = _button->property("poca_palette").toString();
		for (const QString& name : knownPalettes()) {
			QAction* action = menu->addAction(name);
			action->setData(name);
			action->setCheckable(true);
			action->setChecked(name == current);
		}
		menu->addSeparator();
		QAction* custom = menu->addAction(tr("Custom color..."));
		custom->setData("__custom_color__");
		return menu;
	}

	void ButtonLayer::updatePaletteButtonIcon(QPushButton* _button, const QString& _paletteName) const
	{
		poca::core::Palette palette = poca::core::Palette::getStaticLut(_paletteName.toStdString());
		QImage image = poca::core::generateImage(m_iconSize, m_iconSize, &palette);
		_button->setIcon(QIcon(QPixmap::fromImage(image)));
	}

	void ButtonLayer::updateMonochromePaletteButtonIcon(QPushButton* _button, const QColor& _color) const
	{
		QPixmap pixmap(m_iconSize, m_iconSize);
		pixmap.fill(Qt::transparent);
		QPainter painter(&pixmap);
		painter.fillRect(0, 0, m_iconSize, m_iconSize, _color);
		painter.setPen(Qt::black);
		painter.drawRect(0, 0, m_iconSize - 1, m_iconSize - 1);
		_button->setIcon(QIcon(pixmap));
	}

	bool ButtonLayer::isKnownPalette(const std::string& _name)
	{
		const std::vector<std::string>& names = poca::core::Palette::getStaticLutNames();
		return std::find(names.begin(), names.end(), _name) != names.end();
	}

	QStringList ButtonLayer::knownPalettes()
	{
		QStringList result;
		const std::vector<std::string>& names = poca::core::Palette::getStaticLutNames();
		for (const std::string& name : names)
			result << QString::fromStdString(name);
		return result;
	}

	QIcon ButtonLayer::iconFromIdentifier(const QString& _identifier)
	{
		const std::string id = _identifier.toStdString();
		if (id == "brush" || id == "display") return QIcon(QPixmap(poca::plot::brushIcon));
		if (id == "fill") return QIcon(QPixmap(poca::plot::fillIcon));
		if (id == "save") return QIcon(QPixmap(poca::plot::saveIcon));
		if (id == "export") return QIcon(QPixmap(poca::plot::exportIcon));
		if (id == "object" || id == "createObjects") return QIcon(QPixmap(poca::plot::objectIcon));
		if (id == "pointRendering" || id == "points") return QIcon(QPixmap(poca::plot::pointRenderingIcon));
		if (id == "outlinePointRendering" || id == "outlines") return QIcon(QPixmap(poca::plot::outlinePointRenderingIcon));
		if (id == "polytopeRendering" || id == "polytope") return QIcon(QPixmap(poca::plot::polytopeRenderingIcon));
		if (id == "bbox" || id == "boundingBox") return QIcon(QPixmap(poca::plot::bboxIcon));
		if (id == "heatmap") return QIcon(QPixmap(poca::plot::heatmapIcon));
		if (id == "selection") return QIcon(QPixmap(poca::plot::selectionIcon));
		if (id == "invert") return QIcon(QPixmap(poca::plot::invertIcon));
		if (id == "parameters") return QIcon(QPixmap(poca::plot::parametersIcon));
		if (id == "delete") return QIcon(QPixmap(poca::plot::deleteIcon));
		if (id == "gaussian" || id == "gauss3D") return QIcon(QPixmap(poca::plot::gauss3DIcon));
		if (id == "world") return QIcon(QPixmap(poca::plot::worldIcon));
		if (id == "screen") return QIcon(QPixmap(poca::plot::screenIcon));
		if (id == "random") return QIcon(QPixmap(poca::plot::randomIcon));
		if (id == "clear") return QIcon(QPixmap(poca::plot::clearIcon));
		if (id == "reset") return QIcon(QPixmap(poca::plot::resetIcon));
		if (id == "color") return QIcon(QPixmap(poca::plot::colorIcon));
		if (id == "apply") return QIcon(QPixmap(poca::plot::applyIcon));
		if (id == "delaunay") return QIcon(QPixmap(poca::plot::delaunayIcon));
		if (id == "voronoi") return QIcon(QPixmap(poca::plot::voronoiIcon));
		if (id == "screenshot") return QIcon(QPixmap(poca::plot::screenShotIcon));
		if (id == "play") return QIcon(QPixmap(poca::plot::playIcon));
		if (id == "plus") return QIcon(QPixmap(poca::plot::plusIcon));
		return QIcon();
	}

	QString ButtonLayer::stringValue(const nlohmann::json& _json, const char* _key, const QString& _default)
	{
		if (!_json.contains(_key))
			return _default;
		try {
			if (_json[_key].is_string())
				return QString::fromStdString(_json[_key].get<std::string>());
		}
		catch (nlohmann::json::exception&) {}
		return _default;
	}

	bool ButtonLayer::boolValue(const nlohmann::json& _json, const char* _key, const bool _default)
	{
		if (!_json.contains(_key))
			return _default;
		try {
			if (_json[_key].is_boolean())
				return _json[_key].get<bool>();
		}
		catch (nlohmann::json::exception&) {}
		return _default;
	}

	int ButtonLayer::intValue(const nlohmann::json& _json, const char* _key, const int _default)
	{
		if (!_json.contains(_key))
			return _default;
		try {
			if (_json[_key].is_number_integer())
				return _json[_key].get<int>();
		}
		catch (nlohmann::json::exception&) {}
		return _default;
	}
}
