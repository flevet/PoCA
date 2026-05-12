/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ButtonLayer.hpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*/

#ifndef ButtonLayer_h__
#define ButtonLayer_h__

#include <QtWidgets/QWidget>
#include <QtCore/QString>
#include <QtCore/QStringList>
#include <QtCore/QPoint>

#include <vector>
#include <utility>

#include <General/json.hpp>

class QPushButton;
class QSpinBox;
class QButtonGroup;
class QHBoxLayout;
class QColor;
class QIcon;

namespace poca::qt {

	class ButtonLayer : public QWidget {
		Q_OBJECT

	public:
		struct CreatedButton {
			QPushButton* button{ nullptr };
			QString identifier;
			QString kind;
		};

		ButtonLayer(const nlohmann::json&, QWidget* = nullptr);
		~ButtonLayer() override = default;

		QPushButton* button(const QString&) const;
		QPushButton* paletteButton(const QString&) const;
		QSpinBox* spinBox(const QString&) const;
		const nlohmann::json& configuration() const { return m_configuration; }
		nlohmann::json saveConfiguration() const;

		static ButtonLayer* create(const nlohmann::json&, QWidget* = nullptr);
		static bool isKnownPalette(const std::string&);
		static QStringList knownPalettes();

	signals:
		void palettePressed(const QString& _paletteName);
		void paletteChanged(const QString& _previousPaletteName, const QString& _newPaletteName);
		void actionPressed(const QString& _identifier);
		void actionToggled(const QString& _identifier, const bool _checked);
		void spinBoxValueChanged(const QString& _identifier, const int _value);

	private slots:
		void processPaletteButton();
		void processPaletteContextMenu(const QPoint&);
		void processActionButton();
		void processActionButton(const bool);
		void processSpinBox(const int);

	private:
		void buildFromConfiguration();
		void addPaletteButton(const nlohmann::json&, QHBoxLayout*);
		void addActionButton(const nlohmann::json&, QHBoxLayout*, const int);
		void addSpinBox(const nlohmann::json&, QHBoxLayout*, const int);
		static int rowIndexForItem(const int, const int, const int);
		void updatePaletteButtonIcon(QPushButton*, const QString&) const;
		void updateMonochromePaletteButtonIcon(QPushButton*, const QColor&) const;
		static QIcon iconFromIdentifier(const QString&);
		static QString stringValue(const nlohmann::json&, const char*, const QString& = QString());
		static bool boolValue(const nlohmann::json&, const char*, const bool = false);
		static int intValue(const nlohmann::json&, const char*, const int);

	private:
		nlohmann::json m_configuration;
		std::vector<CreatedButton> m_buttons;
		std::vector<CreatedButton> m_paletteButtons;
		std::vector<std::pair<QSpinBox*, QString>> m_spinBoxes;
		QButtonGroup* m_exclusiveActions{ nullptr };
		int m_iconSize{ 20 };
	};

	ButtonLayer* generateButtonLayer(const nlohmann::json&, QWidget* = nullptr);
}

#endif
