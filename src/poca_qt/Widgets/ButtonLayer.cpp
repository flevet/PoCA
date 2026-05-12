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
#include <utility>

#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDialog>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QSizePolicy>
#include <QtGui/QIcon>
#include <QtGui/QPainter>
#include <QtGui/QPixmap>

#include <General/Palette.hpp>
#include <General/Engine.hpp>
#include "CustomColorDialog.hpp"
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

	QSpinBox* ButtonLayer::spinBox(const QString& _identifier) const
	{
		for (const auto& s : m_spinBoxes)
			if (s.second == _identifier)
				return s.first;
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
		const int paletteColumns = intValue(m_configuration.contains("paletteLayout") ? m_configuration["paletteLayout"] : m_configuration, "columns", intValue(m_configuration, "maxPalettesOnFirstLine", 9));
		const int requestedPaletteLines = intValue(m_configuration.contains("paletteLayout") ? m_configuration["paletteLayout"] : m_configuration, "lines", 1);
		const int actionColumns = intValue(m_configuration.contains("actionLayout") ? m_configuration["actionLayout"] : m_configuration, "columns", 7);
		const int requestedActionLines = intValue(m_configuration.contains("actionLayout") ? m_configuration["actionLayout"] : m_configuration, "lines", 1);

		const int nbPalettes = (m_configuration.contains("palettes") && m_configuration["palettes"].is_array()) ? (int)m_configuration["palettes"].size() : 0;
		const int nbActions = (m_configuration.contains("actions") && m_configuration["actions"].is_array()) ? (int)m_configuration["actions"].size() : 0;
		const int paletteLines = std::max(requestedPaletteLines, rowIndexForItem(std::max(0, nbPalettes - 1), paletteColumns, requestedPaletteLines) + 1);
		const int actionLines = std::max(requestedActionLines, rowIndexForItem(std::max(0, nbActions - 1), actionColumns, requestedActionLines) + 1);
		const int nbLines = std::max(1, std::max(paletteLines, actionLines));

		std::vector<QHBoxLayout*> lines;
		for (int n = 0; n < nbLines; ++n)
			lines.push_back(new QHBoxLayout);

		int paletteIndex = 0;
		if (m_configuration.contains("palettes") && m_configuration["palettes"].is_array())
			for (const auto& item : m_configuration["palettes"]) {
				const int row = rowIndexForItem(paletteIndex, paletteColumns, requestedPaletteLines);
				if (row >= 0 && row < (int)lines.size())
					addPaletteButton(item, lines[row]);
				++paletteIndex;
			}

		for (QHBoxLayout* line : lines) {
			QWidget* spacer = new QWidget;
			spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
			line->addWidget(spacer);
		}

		int actionIndex = 0;
		if (m_configuration.contains("actions") && m_configuration["actions"].is_array())
			for (const auto& item : m_configuration["actions"]) {
				const int row = rowIndexForItem(actionIndex, actionColumns, requestedActionLines);
				if (row >= 0 && row < (int)lines.size()) {
					const QString type = stringValue(item, "type", "button");
					if (type == "spinbox" || type == "spinBox")
						addSpinBox(item, lines[row], m_iconSize);
					else
						addActionButton(item, lines[row], m_iconSize);
				}
				++actionIndex;
			}

		QVBoxLayout* mainLayout = new QVBoxLayout;
		mainLayout->setContentsMargins(0, 0, 0, 0);
		mainLayout->setSpacing(0);
		for (QHBoxLayout* line : lines)
			if (line->count() > 1)
				mainLayout->addLayout(line);
		setLayout(mainLayout);
	}


	void ButtonLayer::addPaletteButton(const nlohmann::json& _item, QHBoxLayout* _layout)
	{
		QString paletteName = stringValue(_item, "name");
		if (paletteName.isEmpty() || (!isKnownPalette(paletteName.toStdString()) && paletteName != "RandomOneColor" && paletteName != "Random"))
			return;

		QPushButton* button = new QPushButton;
		button->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
		button->setMaximumSize(QSize(m_iconSize, m_iconSize));
		button->setProperty("poca_palette", paletteName);
		button->setToolTip(stringValue(_item, "tooltip", paletteName));
		button->setContextMenuPolicy(Qt::CustomContextMenu);
		updatePaletteButtonIcon(button, paletteName);

		_layout->addWidget(button, 0, Qt::AlignLeft);
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

	void ButtonLayer::addSpinBox(const nlohmann::json& _item, QHBoxLayout* _layout, const int _maxSize)
	{
		QString identifier = stringValue(_item, "identifier");
		if (identifier.isEmpty())
			identifier = stringValue(_item, "id");
		if (identifier.isEmpty())
			return;

		const QString iconId = stringValue(_item, "icon");
		if (!iconId.isEmpty()) {
			QLabel* label = new QLabel;
			label->setMaximumSize(QSize(_maxSize, _maxSize));
			label->setPixmap(iconFromIdentifier(iconId).pixmap(_maxSize, _maxSize));
			label->setToolTip(stringValue(_item, "tooltip", identifier));
			_layout->addWidget(label, 0, Qt::AlignRight);
		}

		QSpinBox* spin = new QSpinBox;
		spin->setRange(intValue(_item, "minimum", intValue(_item, "min", 0)), intValue(_item, "maximum", intValue(_item, "max", 100)));
		spin->setValue(intValue(_item, "value", intValue(_item, "default", spin->minimum())));
		spin->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
		spin->setProperty("poca_action", identifier);
		spin->setToolTip(stringValue(_item, "tooltip", identifier));
		_layout->addWidget(spin, 0, Qt::AlignRight);
		connect(spin, SIGNAL(valueChanged(int)), this, SLOT(processSpinBox(int)));
		m_spinBoxes.push_back(std::make_pair(spin, identifier));
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

		QString previousPalette = button->property("poca_palette").toString();
		CustomColorDialog dialog(this);
		dialog.setSelectedPalette(previousPalette.toStdString());
		if (dialog.exec() != QDialog::Accepted)
			return;

		QString newPalette = QString::fromStdString(dialog.selectedPaletteName());
		if (newPalette.isEmpty() || newPalette == previousPalette)
			return;
		if (!isKnownPalette(newPalette.toStdString()))
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

	void ButtonLayer::processSpinBox(const int _value)
	{
		QSpinBox* spin = qobject_cast<QSpinBox*>(sender());
		if (spin == nullptr)
			return;
		emit spinBoxValueChanged(spin->property("poca_action").toString(), _value);
	}

	int ButtonLayer::rowIndexForItem(const int _index, const int _columns, const int)
	{
		if (_index < 0)
			return 0;
		return _index / std::max(1, _columns);
	}


	void ButtonLayer::updatePaletteButtonIcon(QPushButton* _button, const QString& _paletteName) const
	{
		if (_paletteName == "RandomOneColor") {
			_button->setIcon(QIcon(QPixmap(poca::plot::randomIcon)));
			return;
		}
		poca::core::Palette* enginePalette = poca::core::Engine::instance()->palette(_paletteName.toStdString());
		poca::core::Palette palette = enginePalette != nullptr ? *enginePalette : poca::core::Palette::getStaticLut(_paletteName.toStdString());
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
		if (_name == "RandomOneColor" || _name == "Random")
			return true;
		return poca::core::Engine::instance()->palette(_name) != nullptr;
	}

	QStringList ButtonLayer::knownPalettes()
	{
		QStringList result;
		const auto& palettes = poca::core::Engine::instance()->palettes();
		for (const auto& item : palettes)
			result << QString::fromStdString(item.first);
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
		if (id == "pointSize") return QIcon(QPixmap(poca::plot::pointSizeIcon));
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
		//if (id == "random") return QIcon(QPixmap(poca::plot::randomIcon));
		if (id == "clear") return QIcon(QPixmap(poca::plot::clearIcon));
		if (id == "reset") return QIcon(QPixmap(poca::plot::resetIcon));
		if (id == "color") return QIcon(QPixmap(poca::plot::colorIcon));
		if (id == "apply") return QIcon(QPixmap(poca::plot::applyIcon));
		if (id == "delaunay") return QIcon(QPixmap(poca::plot::delaunayIcon));
		if (id == "voronoi") return QIcon(QPixmap(poca::plot::voronoiIcon));
		if (id == "screenshot") return QIcon(QPixmap(poca::plot::screenShotIcon));
		if (id == "play") return QIcon(QPixmap(poca::plot::playIcon));
		if (id == "plus") return QIcon(QPixmap(poca::plot::plusIcon));
		if (id == "RandomOneColor") return QIcon(QPixmap(poca::plot::randomIcon));
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
