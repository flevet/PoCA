#pragma once

#include <QtWidgets/QTableWidgetItem>

class SortableFloatItem : public QTableWidgetItem
{
public:
    SortableFloatItem(const QTableWidgetItem& other) : QTableWidgetItem(other) {}
    SortableFloatItem(const QIcon& icon, const QString& text, int type = Type) : QTableWidgetItem(icon, text, type) {}
    SortableFloatItem(const QString& text, int type = Type) : QTableWidgetItem(text, type) {}
    SortableFloatItem(int type = Type) : QTableWidgetItem(type) {}

    bool operator<(const QTableWidgetItem& other) const override
    {
        return text().toFloat() < other.text().toFloat();
    }
};
