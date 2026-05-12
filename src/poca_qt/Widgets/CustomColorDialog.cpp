#include <algorithm>

#include <QtWidgets/QLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QDialogButtonBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QInputDialog>
#include <QtWidgets/QMessageBox>
#include <QtGui/QPainter>
#include <QtGui/QPainterPath>
#include <QtGui/QMouseEvent>
#include <QtGui/QPixmap>
#include <QtCore/QEvent>

#include <General/Engine.hpp>
#include <Plot/Misc.h>

#include "CustomColorDialog.hpp"

#define CIRCLE_RADIUS 15
#define BORDERS_SIZE 0
#define LUT_ICON_SIZE 20

ColorBarDialog::ColorBarDialog(poca::core::Palette* _palette, Qt::Orientation _o, CustomColorDialog* _dialog, QWidget* _parent) : QWidget(_parent), m_parent(_dialog), m_orientation(_o), m_palette(_palette)
{
#ifndef QT_NO_CURSOR
	setCursor(Qt::PointingHandCursor);
#endif
	setMinimumHeight(2 * (CIRCLE_RADIUS + BORDERS_SIZE + 1));
	setObjectName("ColorBarDialog");
}

void ColorBarDialog::mousePressEvent(QMouseEvent* _event)
{
	if (m_palette == nullptr || m_parent == nullptr) return;
	int index = -1;
	for (unsigned int i = 0; i < m_palette->size(); ++i) {
		QPainterPath painterPath;
		addEllipse(painterPath, m_palette->colorPosition(i));
		if (painterPath.contains(_event->pos())) index = (int)i;
	}
	if (_event->button() == Qt::LeftButton) {
		if (index == -1) {
			QColor color = m_parent->currentColor();
			if (color.isValid()) {
				qreal position = (_event->x() - BORDERS_SIZE) / (qreal)(width() - 2 * BORDERS_SIZE);
				position = std::max<qreal>(0., std::min<qreal>(1., position));
				m_palette->setColor((float)position, poca::core::Color4uc(color.red(), color.green(), color.blue(), color.alpha()));
				m_parent->m_position->setText(QString::number(position));
			}
		}
		else {
			m_selected = index;
			if (index != 0 && index != (int)m_palette->size() - 1) {
				m_moveIndex = index;
				m_parent->m_position->setText(QString::number(m_palette->colorPosition(m_moveIndex)));
			}
		}
	}
	else if (_event->button() == Qt::RightButton && index != -1) {
		QColor color = m_parent->currentColor();
		if (color.isValid())
			m_palette->setColorAt(index, poca::core::Color4uc(color.red(), color.green(), color.blue(), color.alpha()));
	}
	else if (_event->button() == Qt::MiddleButton && index > 0 && index != (int)m_palette->size() - 1) {
		m_palette->removeColorAt(index);
		m_moveIndex = m_selected = -1;
		m_parent->m_position->clear();
	}
	update();
}

void ColorBarDialog::mouseMoveEvent(QMouseEvent* _event)
{
	if (m_palette == nullptr || m_moveIndex == -1) return;
	qreal nextPosition = (_event->x() - BORDERS_SIZE) / (qreal)(width() - 2 * BORDERS_SIZE);
	if (nextPosition <= m_palette->colorPosition(m_moveIndex - 1))
		nextPosition = (m_palette->colorPosition(m_moveIndex - 1) * (width() - 2 * BORDERS_SIZE) + 1) / (qreal)(width() - 2 * BORDERS_SIZE);
	else if (nextPosition >= m_palette->colorPosition(m_moveIndex + 1))
		nextPosition = (m_palette->colorPosition(m_moveIndex + 1) * (width() - 2 * BORDERS_SIZE) - 1) / (qreal)(width() - 2 * BORDERS_SIZE);
	m_palette->setColorPosition(m_moveIndex, (float)nextPosition);
	m_parent->m_position->setText(QString::number(nextPosition));
	update();
}

void ColorBarDialog::mouseReleaseEvent(QMouseEvent*) { m_moveIndex = -1; }

void ColorBarDialog::paintEvent(QPaintEvent* _event)
{
	QPainter painter(this);
	if (m_palette != nullptr) {
		std::vector<float> pos;
		std::vector<poca::core::Color4uc> colors;
		m_palette->getGradientInfos(pos, colors);
		QLinearGradient gradient;
		for (size_t n = 0; n < pos.size(); n++)
			gradient.setColorAt(pos[n], QColor(colors[n][0], colors[n][1], colors[n][2], colors[n][3]));
		gradient.setStart(BORDERS_SIZE, BORDERS_SIZE);
		gradient.setFinalStop((qreal)(width() - BORDERS_SIZE), 0.0);
		painter.fillRect(BORDERS_SIZE, BORDERS_SIZE, width() - 2 * BORDERS_SIZE, height() - 2 * BORDERS_SIZE, gradient);
		QPainterPath painterPath, painterPathSelected;
		for (unsigned int i = 0; i < m_palette->size(); ++i) {
			if (m_moveIndex == (int)i || m_selected == (int)i) addEllipse(painterPathSelected, m_palette->colorPosition(i), .5);
			else addEllipse(painterPath, m_palette->colorPosition(i), .5);
		}
		painter.setRenderHint(QPainter::Antialiasing);
		painter.setPen(Qt::black);
		painter.fillPath(painterPath, Qt::white);
		painter.drawPath(painterPath);
		painter.setPen(Qt::red);
		painter.fillPath(painterPathSelected, Qt::white);
		painter.drawPath(painterPathSelected);
	}
	QWidget::paintEvent(_event);
}

void ColorBarDialog::setPalette(poca::core::Palette* _palette)
{
	m_palette = _palette;
	m_moveIndex = m_selected = -1;
	update();
}

void ColorBarDialog::addEllipse(QPainterPath& _painterPath, qreal _position, qreal _percentage)
{
	_painterPath.addEllipse(QPointF(_position * (width() - 2 * BORDERS_SIZE) + BORDERS_SIZE, height() * 0.5), CIRCLE_RADIUS * _percentage, CIRCLE_RADIUS * _percentage);
}

void ColorBarDialog::changePositionOfSelectedColor(const double _position)
{
	if (m_palette == nullptr) return;
	int index = m_moveIndex != -1 ? m_moveIndex : m_selected;
	if (index <= 0 || index == (int)m_palette->size() - 1) return;
	poca::core::Color4uc color = m_palette->colorAt(index);
	m_palette->removeColorAt(index);
	m_palette->setColor((float)_position, color);
	m_moveIndex = -1;
	m_selected = -1;
	for (unsigned int i = 0; i < m_palette->size() && m_selected == -1; ++i)
		if (m_palette->colorPosition(i) == _position) m_selected = (int)i;
	update();
}

void ColorBarDialog::setColorOfSelectedColor(const QColor& _color)
{
	if (m_palette == nullptr || m_selected == -1 || !_color.isValid()) return;
	m_palette->setColorAt(m_selected, poca::core::Color4uc(_color.red(), _color.green(), _color.blue(), _color.alpha()));
	update();
}

CustomColorDialog::CustomColorDialog(QWidget* _parent) : QColorDialog(_parent)
{
	setOption(QColorDialog::DontUseNativeDialog, true);
	m_editedPalette = new poca::core::Palette(poca::core::Color4uc(0, 0, 0, 255), poca::core::Color4uc(255, 255, 255, 255), "");
	QLayout* layout = this->layout();
	layoutTreeParcours(layout);

	m_colorBar = new ColorBarDialog(m_editedPalette, Qt::Horizontal, this, nullptr);
	m_colorBar->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

	QHBoxLayout* editLayout = new QHBoxLayout();
	editLayout->addWidget(new QLabel(tr("Name:")));
	m_nameEdit = new QLineEdit();
	editLayout->addWidget(m_nameEdit);
	editLayout->addWidget(new QLabel(tr("Position:")));
	m_position = new QLineEdit();
	editLayout->addWidget(m_position);
	QObject::connect(m_position, SIGNAL(returnPressed()), this, SLOT(positionChanged()));

	QScrollArea* scroll = new QScrollArea();
	scroll->setWidgetResizable(true);
	m_paletteButtonsWidget = new QWidget();
	m_paletteButtonsLayout = new QGridLayout(m_paletteButtonsWidget);
	m_paletteButtonsLayout->setContentsMargins(0, 0, 0, 0);
	m_paletteButtonsLayout->setSpacing(2);
	scroll->setWidget(m_paletteButtonsWidget);
	scroll->setMinimumHeight(60);
	rebuildPaletteButtons();

	QDialogButtonBox* buttons = new QDialogButtonBox();
	m_saveButton = buttons->addButton(tr("Save changes"), QDialogButtonBox::AcceptRole);
	QObject::connect(m_saveButton, SIGNAL(clicked()), this, SLOT(savePalette()));
	m_deleteButton = buttons->addButton(tr("Delete"), QDialogButtonBox::DestructiveRole);
	QObject::connect(m_deleteButton, SIGNAL(clicked()), this, SLOT(deletePalette()));
	QPushButton* select = buttons->addButton(tr("Select"), QDialogButtonBox::AcceptRole);
	QObject::connect(select, SIGNAL(clicked()), this, SLOT(selectPalette()));
	QPushButton* close = buttons->addButton(QDialogButtonBox::Close);
	QObject::connect(close, SIGNAL(clicked()), this, SLOT(reject()));

	QVBoxLayout* mainLay = dynamic_cast<QVBoxLayout*>(layout);
	if (mainLay) {
		mainLay->addWidget(new QLabel(tr("Palettes:")));
		mainLay->addWidget(scroll);
		mainLay->addWidget(m_colorBar);
		mainLay->addLayout(editLayout);
		mainLay->addWidget(buttons);
	}
	setWindowTitle("Palettes");
	setObjectName("CustomColorDialog");
	installRightClickColorFilters(this);
	newPalette();
}

CustomColorDialog::~CustomColorDialog()
{
	delete m_editedPalette;
}

void CustomColorDialog::layoutTreeParcours(QLayout* _layout)
{
	std::vector<QWidget*> toDelete;
	for (int i = 0; i < _layout->count(); i++) {
		QLayoutItem* layoutItem = _layout->itemAt(i);
		QLayout* layout2 = layoutItem->layout();
		if (layout2) layoutTreeParcours(layout2);
		else {
			QWidget* widg = layoutItem->widget();
			if (dynamic_cast<QDialogButtonBox*>(widg)) toDelete.push_back(widg);
		}
	}
	for (QWidget* widget : toDelete) {
		_layout->removeWidget(widget);
		delete widget;
	}
}

void CustomColorDialog::rebuildPaletteButtons()
{
	while (QLayoutItem* item = m_paletteButtonsLayout->takeAt(0)) {
		delete item->widget();
		delete item;
	}
	QPushButton* add = new QPushButton("+");
	add->setMaximumSize(QSize(LUT_ICON_SIZE, LUT_ICON_SIZE));
	add->setMinimumSize(QSize(LUT_ICON_SIZE, LUT_ICON_SIZE));
	add->setToolTip(tr("Add a palette"));
	m_paletteButtonsLayout->addWidget(add, 0, 0);
	QObject::connect(add, SIGNAL(clicked()), this, SLOT(newPalette()));

	int index = 1;
	const auto& palettes = poca::core::Engine::instance()->palettes();
	for (const auto& item : palettes) {
		QPushButton* button = new QPushButton();
		button->setMaximumSize(QSize(LUT_ICON_SIZE, LUT_ICON_SIZE));
		button->setMinimumSize(QSize(LUT_ICON_SIZE, LUT_ICON_SIZE));
		button->setIcon(iconForPalette(item.second));
		button->setProperty("paletteName", QString::fromStdString(item.first));
		button->setToolTip(QString::fromStdString(item.first));
		m_paletteButtonsLayout->addWidget(button, index / 12, index % 12);
		QObject::connect(button, SIGNAL(clicked()), this, SLOT(paletteButtonClicked()));
		index++;
	}
}

QIcon CustomColorDialog::iconForPalette(const poca::core::Palette& _palette) const
{
	poca::core::Palette palette(_palette);
	QImage im = poca::core::generateImage(LUT_ICON_SIZE, LUT_ICON_SIZE, &palette);
	return QIcon(QPixmap::fromImage(im));
}

void CustomColorDialog::loadPalette(const std::string& _name, const poca::core::Palette& _palette, bool _new)
{
	m_currentName = _name;
	m_newPalette = _new;
	m_editedPalette->setPalette(_palette);
	m_editedPalette->setName(_name);
	m_nameEdit->setText(QString::fromStdString(_name));
	m_nameEdit->setReadOnly(false);
	m_saveButton->setText(_new ? tr("Add palette") : tr("Save changes"));
	if (m_deleteButton != nullptr) m_deleteButton->setEnabled(!_new);
	m_colorBar->setPalette(m_editedPalette);
}

void CustomColorDialog::newPalette()
{
	float r = (float)rand() / (float)RAND_MAX;
	float g = (float)rand() / (float)RAND_MAX;
	float b = (float)rand() / (float)RAND_MAX;
	float a = 1.f;
	poca::core::Color4D color(r * 255.f, g * 255.f, b * 255.f, a * 255.f);
	poca::core::Palette palette(color, color, "New palette");
	loadPalette("New palette", palette, true);
}

void CustomColorDialog::paletteButtonClicked()
{
	QPushButton* button = qobject_cast<QPushButton*>(sender());
	if (button == nullptr) return;
	std::string name = button->property("paletteName").toString().toStdString();
	setSelectedPalette(name);
}

bool CustomColorDialog::setSelectedPalette(const std::string& _name)
{
	poca::core::Palette* palette = poca::core::Engine::instance()->palette(_name);
	if (palette == nullptr) return false;
	loadPalette(_name, *palette, false);
	return true;
}

void CustomColorDialog::savePalette()
{
	QString name = m_nameEdit->text().trimmed();
	if (name.isEmpty()) {
		QMessageBox::warning(this, tr("Palette"), tr("Please enter a palette name."));
		return;
	}
	std::string newName = name.toStdString();
	if (!m_newPalette && !m_currentName.empty() && newName != m_currentName)
		poca::core::Engine::instance()->removePalette(m_currentName);
	poca::core::Engine::instance()->addOrReplacePalette(newName, *m_editedPalette);
	m_currentName = newName;
	m_newPalette = false;
	m_saveButton->setText(tr("Save changes"));
	if (m_deleteButton != nullptr) m_deleteButton->setEnabled(true);
	rebuildPaletteButtons();
}


void CustomColorDialog::selectPalette()
{
	if (m_newPalette)
		savePalette();
	if (m_currentName.empty()) {
		QMessageBox::warning(this, tr("Palette"), tr("Please select a palette."));
		return;
	}
	accept();
}

void CustomColorDialog::deletePalette()
{
	if (m_newPalette || m_currentName.empty()) return;
	if (QMessageBox::question(this, tr("Palette"), tr("Delete palette '%1'?").arg(QString::fromStdString(m_currentName))) != QMessageBox::Yes)
		return;
	poca::core::Engine::instance()->removePalette(m_currentName);
	rebuildPaletteButtons();
	newPalette();
}

void CustomColorDialog::installRightClickColorFilters(QObject* _object)
{
	if (_object == nullptr) return;
	_object->installEventFilter(this);
	const QObjectList children = _object->children();
	for (QObject* child : children)
		installRightClickColorFilters(child);
}

void CustomColorDialog::applyColorToSelectedColor(const QColor& _color)
{
	if (m_colorBar == nullptr || !m_colorBar->hasSelectedColor()) return;
	m_colorBar->setColorOfSelectedColor(_color);
}

bool CustomColorDialog::eventFilter(QObject* _object, QEvent* _event)
{
	if (_event->type() == QEvent::MouseButtonPress) {
		QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(_event);
		if (mouseEvent->button() == Qt::RightButton) {
			QWidget* widget = qobject_cast<QWidget*>(_object);
			if (widget != nullptr && widget != m_colorBar && qobject_cast<QPushButton*>(widget) == nullptr && qobject_cast<QLineEdit*>(widget) == nullptr) {
				QColor color;
				QPixmap pixmap = widget->grab();
				QPoint pos = mouseEvent->pos();
				if (!pixmap.isNull() && pos.x() >= 0 && pos.y() >= 0 && pos.x() < pixmap.width() && pos.y() < pixmap.height())
					color = pixmap.toImage().pixelColor(pos);
				if (!color.isValid())
					color = currentColor();
				applyColorToSelectedColor(color);
			}
		}
	}
	return QColorDialog::eventFilter(_object, _event);
}

void CustomColorDialog::paintEvent(QPaintEvent* _event)
{
	QColorDialog::paintEvent(_event);
}

void CustomColorDialog::positionChanged()
{
	bool ok;
	double position = m_position->text().toDouble(&ok);
	if (ok) m_colorBar->changePositionOfSelectedColor(position);
}
