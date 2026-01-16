/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ColorButtonGridWidget.cpp
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

#include <iostream>
#include <QAbstractButton>
#include <QHBoxLayout>
#include <QLayout>
#include <QVBoxLayout>
#include "ColorButtonGridWidget.hpp"

ColorButtonGridWidget::ColorButtonGridWidget(QWidget* parent)
    : QWidget(parent)
{
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

    m_rootHBox = new QHBoxLayout;
    m_rootHBox->setContentsMargins(0, 0, 0, 0);
    m_rootHBox->setSpacing(0);

    m_rowsContainer = new QWidget(this);
    m_rowsContainer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);

    m_rowsVBox = new QVBoxLayout;
    m_rowsVBox->setContentsMargins(0, 0, 0, 0);
    m_rowsVBox->setSpacing(0);
    m_rowsContainer->setLayout(m_rowsVBox);

    // --- scroll area wraps rowsContainer ---
    m_scrollArea = new QScrollArea(this);
    m_scrollArea->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    m_scrollArea->setFrameShape(QFrame::NoFrame);
    m_scrollArea->setWidgetResizable(true);
    m_scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_scrollArea->setWidget(m_rowsContainer);

    // --- right column: right button on first line, toggle button below ---
    m_rightColumn = new QWidget(this);
    m_rightVBox = new QVBoxLayout(m_rightColumn);
    m_rightVBox->setContentsMargins(0, 0, 0, 0);
    m_rightVBox->setSpacing(0);

    m_rightButton = new QPushButton(this);
    m_rightButton->setVisible(false);
    m_rightButton->setCheckable(true);
    m_rightButton->setChecked(false);
    m_rightButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
    m_rightButton->setMaximumSize(QSize(m_buttonMaxSize, m_buttonMaxSize));

    m_toggleButton = new QToolButton(this);
    m_toggleButton->setCheckable(true);
    m_toggleButton->setChecked(false);
    m_toggleButton->setAutoRaise(true); // makes it look compact
    m_toggleButton->setText("v");       // collapsed indicator; you can change
    m_toggleButton->setMaximumSize(QSize(m_buttonMaxSize, m_buttonMaxSize));
    m_toggleButton->setToolTip("Collapse/expand");
    connect(m_toggleButton, &QToolButton::toggled,
        this, &ColorButtonGridWidget::onToggleCollapsed);

    m_rightVBox->addWidget(m_rightButton, 0, Qt::AlignRight | Qt::AlignTop);
    m_rightVBox->addWidget(m_toggleButton, 0, Qt::AlignRight | Qt::AlignTop);
    m_rightVBox->addStretch(1); // keep both at the top

    m_group = new QButtonGroup(this);
    m_group->setExclusive(true);

    connect(m_group, SIGNAL(buttonClicked(QAbstractButton*)),
        this, SLOT(onAnyButtonClicked(QAbstractButton*)));
    connect(m_rightButton, &QPushButton::released,
        this, &ColorButtonGridWidget::rightButtonClicked);

    m_rootHBox->addWidget(m_scrollArea, 1);      // left: scrollable area
    m_rootHBox->addWidget(m_rightColumn, 0);    // right: 2 stacked buttons
    setLayout(m_rootHBox);

    setCollapsed(false);
}

void ColorButtonGridWidget::setMaxPerRow(int v)
{
    m_maxPerRow = (v <= 0) ? 1 : v;
    rebuild();
}

void ColorButtonGridWidget::setButtonMaxSize(int px)
{
    m_buttonMaxSize = (px <= 0) ? 1 : px;
    ensureButtonSizing();

    if (m_rightButton)
        m_rightButton->setMaximumSize(QSize(m_buttonMaxSize, m_buttonMaxSize));
    if (m_toggleButton)
        m_toggleButton->setMaximumSize(QSize(m_buttonMaxSize, m_buttonMaxSize));

    rebuild();

    // if currently collapsed, recompute the two-row height
    if (m_collapsed)
        setCollapsed(true);
}

void ColorButtonGridWidget::setCount(int count)
{
    if (count < 0) count = 0;
    m_count = count;
    ensureButtons();
    rebuild();
}

int ColorButtonGridWidget::count() const
{
    return m_count;
}

void ColorButtonGridWidget::setCurrentIndex(int idx)
{
    m_current = idx;
    if (QAbstractButton* b = m_group->button(idx))
        b->setChecked(true);
}

int ColorButtonGridWidget::currentIndex() const
{
    return m_current;
}

void ColorButtonGridWidget::setRightButtonText(const QString& text)
{
    m_rightButton->setText(text);
    rebuild();
}

QPushButton* ColorButtonGridWidget::rightButton() const
{
    return m_rightButton;
}

void ColorButtonGridWidget::onAnyButtonClicked(QAbstractButton* b)
{
    const int id = m_group->id(b);
    m_current = id;
    emit indexClicked(id);
}

void ColorButtonGridWidget::ensureButtons()
{
    while (m_buttons.size() < m_count) {
        const int idx = m_buttons.size();
        auto* b = new QPushButton(QString::number(idx + 1), this);
        b->setCheckable(true);
        b->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
        b->setMaximumSize(QSize(m_buttonMaxSize, m_buttonMaxSize));

        m_buttons.push_back(b);
        m_group->addButton(b, idx);
    }

    for (int i = 0; i < m_buttons.size(); ++i) {
        m_buttons[i]->setText(QString::number(i + 1));
        m_buttons[i]->setVisible(i < m_count);
    }
}

void ColorButtonGridWidget::ensureButtonSizing()
{
    for (auto* b : m_buttons)
        b->setMaximumSize(QSize(m_buttonMaxSize, m_buttonMaxSize));
}

void ColorButtonGridWidget::rebuild()
{
    clearLayout(m_rowsVBox);

    if (m_count <= 0) {
        updateAuxVisibility(0);
        return;
    }

    const int rows = (m_count + m_maxPerRow - 1) / m_maxPerRow;

    int idx = 0;
    for (int r = 0; r < rows; ++r) {
        auto* rowW = new QWidget(m_rowsContainer);
        auto* rowH = new QHBoxLayout(rowW);
        rowH->setContentsMargins(0, 0, 0, 0);
        rowH->setSpacing(0);

        rowH->addStretch(1);
        for (int c = 0; c < m_maxPerRow && idx < m_count; ++c, ++idx)
            rowH->addWidget(m_buttons[idx], 0, Qt::AlignCenter);
        rowH->addStretch(1);

        m_rowsVBox->addWidget(rowW);
    }

    // Restore checked state
    if (m_current >= 0 && m_current < m_count) {
        if (QAbstractButton* b = m_group->button(m_current))
            b->setChecked(true);
    }

    updateAuxVisibility(rows);

    // Apply the correct height based on collapse/expand
    setCollapsed(m_collapsed);

    updateGeometry();
}


void ColorButtonGridWidget::clearLayout(QLayout* layout)
{
    while (QLayoutItem* item = layout->takeAt(0)) {
        if (QWidget* w = item->widget())
            w->setParent(nullptr);
        delete item;
    }
}

bool ColorButtonGridWidget::isCollapsed() const
{
    return m_collapsed;
}

void ColorButtonGridWidget::setCollapsed(bool on)
{
    if (m_count <= 0 || !m_scrollArea) {
        m_collapsed = false;
        if (m_scrollArea) m_scrollArea->setFixedHeight(0);
        return;
    }

    const int rows = (m_count + m_maxPerRow - 1) / m_maxPerRow;
    if (rows <= 1) {
        m_collapsed = false;
        m_scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        m_scrollArea->setFixedHeight(contentHeightPx());
        return;
    }

    m_collapsed = on;

    if (!m_collapsed) {
        m_scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        m_scrollArea->setFixedHeight(contentHeightPx());
        if (m_toggleButton) m_toggleButton->setText("v");
    }
    else {
        m_scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        m_scrollArea->setFixedHeight(computeTwoRowHeightPx());
        if (m_toggleButton) m_toggleButton->setText(">");
    }
}


void ColorButtonGridWidget::onToggleCollapsed(bool checked)
{
    setCollapsed(checked);
}

int ColorButtonGridWidget::computeTwoRowHeightPx() const
{
    // Each "row" is a QWidget with a QHBoxLayout; its height is basically the button size.
    // Add a small cushion for layout margins/spacing.
    const int rowH = m_buttonMaxSize;
    const int spacing = (m_rowsVBox ? m_rowsVBox->spacing() : 0);
    const int topBottom = 2; // small cushion; can tune

    // Two rows: row + spacing + row + cushion
    return (2 * rowH) + spacing + topBottom;
}

int ColorButtonGridWidget::contentHeightPx() const
{
    if (!m_rowsContainer || !m_scrollArea)
        return 0;

    const int h = m_rowsContainer->sizeHint().height();
    const int extra = m_scrollArea->contentsMargins().top()
        + m_scrollArea->contentsMargins().bottom();
    return h + extra;
}

void ColorButtonGridWidget::updateAuxVisibility(int rows)
{
    const bool hasAny = (m_count > 0);
    const bool hasMany = (m_count > 1);
    const bool multiRow = (rows > 1);

    // Whole widget hidden when empty
    setVisible(hasAny);

    // Right button only when > 1 item
    m_rightButton->setVisible(hasMany && !m_rightButton->text().isEmpty());

    // Toggle only when > 1 row
    m_toggleButton->setVisible(multiRow);

    // If it isn't multi-row, force expanded and uncheck toggle
    if (!multiRow) {
        m_collapsed = false;
        m_toggleButton->blockSignals(true);
        m_toggleButton->setChecked(false);
        m_toggleButton->blockSignals(false);
    }

    // If empty, also clamp heights so it doesn't reserve space
    if (!hasAny && m_scrollArea) {
        m_scrollArea->setFixedHeight(0);
        setFixedHeight(0);
        return;
    }

    // Let the widget height be driven by scroll area + small margins
    setMinimumHeight(0);
    setMaximumHeight(QWIDGETSIZE_MAX);

    if (multiRow && m_collapsed == false) setCollapsed(true);
}
