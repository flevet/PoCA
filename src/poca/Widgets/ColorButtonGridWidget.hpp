/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ColorButtonGridWidget.hpp
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

// A standalone widget that shows N checkable buttons in rows (maxPerRow per row),
// each row centered, plus an independent button pinned to the right.
//
// Usage:
//   auto* w = new ColorButtonGridWidget(this);
//   w->setMaxPerRow(20);
//   w->setCount(nbFiles);
//   w->setCurrentIndex(currentId);
//   w->setRightButtonText("...");
//   connect(w, &ColorButtonGridWidget::indexClicked, this, &YourClass::changeColorObjectIndex);
//   connect(w, &ColorButtonGridWidget::rightButtonClicked, this, &YourClass::onRightButton);
//
// Notes:
// - Buttons are 0-based index in signals, label shows 1..N.

#ifndef ColorButtonGridWidget_h__
#define ColorButtonGridWidget_h__

#include <QButtonGroup>
#include <QPushButton>
#include <QToolButton>
#include <QVector>
#include <QWidget>
#include <QScrollArea>
#include <QDialog>

class QHBoxLayout;
class QVBoxLayout;
class QDialogButtonBox;

class ParametersDialog : public QDialog
{
    Q_OBJECT
public:
    explicit ParametersDialog(QWidget* parent = nullptr);

signals:
    void gridReleased();
    void toggleGridCentered(bool);
    void exportAllObjects();

private:
    QPushButton* m_gridBtn = nullptr;
    QPushButton* m_centeredBtn = nullptr;
    QPushButton* m_exportAllObjectsBtn = nullptr;
    QDialogButtonBox* m_buttons = nullptr;
};

class ColorButtonGridWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ColorButtonGridWidget(QWidget* parent = nullptr);

    void setCount(int count);
    int  count() const;

    void setCurrentIndex(int idx);
    int  currentIndex() const;

    void setMaxPerRow(int v);
    void setButtonMaxSize(int px);

    void setRightButtonText(const QString& text);
    QPushButton* rightButton() const;

    void setCollapsed(bool on);
    bool isCollapsed() const;

    QToolButton* parametersButton() { return m_parametersButton; }

signals:
    void indexClicked(int idx);
    void rightButtonClicked();
    void parametersButtonClicked();

private slots:
    void onAnyButtonClicked(QAbstractButton* b);
    void onToggleCollapsed(bool checked);

private:
    void ensureButtons();
    void ensureButtonSizing();
    void rebuild();

    void clearLayout(QLayout*);
    int computeTwoRowHeightPx() const;
    int contentHeightPxForRows(int rows) const;

    void updateAuxVisibility(int);

    int rowHeightPx() const;

private:
    // Layouts
    QHBoxLayout* m_rootHBox = nullptr;
    QWidget* m_rowsContainer = nullptr;
    QVBoxLayout* m_rowsVBox = nullptr;

    // Buttons
    QVector<QPushButton*> m_buttons;
    QPushButton* m_rightButton = nullptr;
    QToolButton* m_parametersButton = nullptr;
    QButtonGroup* m_group = nullptr;

    QScrollArea* m_scrollArea = nullptr;
    QWidget* m_rightColumn = nullptr;
    QVBoxLayout* m_rightVBox = nullptr;
    QToolButton* m_toggleButton = nullptr;

    QWidget* m_rightTopRow = nullptr;
    QHBoxLayout* m_rightTopHBox = nullptr;

    bool m_collapsed = false;

    // State
    int m_count = 0;
    int m_current = -1;
    int m_maxPerRow = 20;
    int m_buttonMaxSize = 20;
};

#endif // MergeDatasetsChoiceDialog_h__

