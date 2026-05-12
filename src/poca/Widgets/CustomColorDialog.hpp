#ifndef CustomColorDialog_h__
#define CustomColorDialog_h__

#include <QtWidgets/QColorDialog>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QScrollArea>

#include <General/Palette.hpp>

class CustomColorDialog;

class ColorBarDialog : public QWidget
{
	Q_OBJECT

public:
	ColorBarDialog(poca::core::Palette*, Qt::Orientation = Qt::Horizontal, CustomColorDialog* = nullptr, QWidget* = nullptr);

	void setPalette(poca::core::Palette*);
	void addEllipse(QPainterPath&, qreal, qreal = 1.);
	void changePositionOfSelectedColor(const double);

protected:
	void mousePressEvent(QMouseEvent*) override;
	void mouseMoveEvent(QMouseEvent*) override;
	void mouseReleaseEvent(QMouseEvent*) override;
	void paintEvent(QPaintEvent*) override;

public:
	void setColorOfSelectedColor(const QColor&);
	bool hasSelectedColor() const { return m_selected != -1; }

private:
	CustomColorDialog* m_parent{ nullptr };
	Qt::Orientation m_orientation{ Qt::Horizontal };
	poca::core::Palette* m_palette{ nullptr };
	int m_moveIndex{ -1 }, m_selected{ -1 };
};

class CustomColorDialog : public QColorDialog
{
	Q_OBJECT
public:
	CustomColorDialog(QWidget* = nullptr);
	~CustomColorDialog();

public slots:
	void positionChanged();
	void newPalette();
	void paletteButtonClicked();
	void savePalette();
	void deletePalette();

protected:
	void paintEvent(QPaintEvent*) override;
	bool eventFilter(QObject*, QEvent*) override;

private:
	void layoutTreeParcours(QLayout*);
	void rebuildPaletteButtons();
	void loadPalette(const std::string&, const poca::core::Palette&, bool);
	void installRightClickColorFilters(QObject*);
	void applyColorToSelectedColor(const QColor&);
	QIcon iconForPalette(const poca::core::Palette&) const;

	ColorBarDialog* m_colorBar{ nullptr };
	QLineEdit* m_position{ nullptr };
	QLineEdit* m_nameEdit{ nullptr };
	QPushButton* m_saveButton{ nullptr };
	QPushButton* m_deleteButton{ nullptr };
	QWidget* m_paletteButtonsWidget{ nullptr };
	QGridLayout* m_paletteButtonsLayout{ nullptr };
	poca::core::Palette* m_editedPalette{ nullptr };
	std::string m_currentName;
	bool m_newPalette{ false };

	friend class ColorBarDialog;
};

#endif
