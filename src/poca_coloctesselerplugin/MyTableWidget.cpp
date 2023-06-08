/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MyTableWidget.cpp
*
* Copyright: Florian Levet (2020-2022)
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

#include <QtGui/QKeyEvent>
#include <QtGui/QClipboard>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QApplication>
#include <fstream>
#include <iostream>

#include "MyTableWidget.hpp"

MyTableWidget::MyTableWidget(QWidget * _parent) :QTableWidget(_parent)
{

}

MyTableWidget::~MyTableWidget()
{

}

void MyTableWidget::keyPressEvent(QKeyEvent * _event)
{
	if (_event->key() == Qt::Key_C &&  _event->modifiers() == Qt::ControlModifier)
		copyToClipboardWithRowSelection();
	else
		QTableWidget::keyPressEvent(_event);

}

void MyTableWidget::copyToClipboardWithRowSelection()
{
	QItemSelectionModel * selection = this->selectionModel();
	QModelIndexList indexes = selection->selectedRows();
	if (indexes.size() < 1)
		return;//No row selected

	QString text;
	for (int col = 0; col < this->columnCount(); col++){
		QTableWidgetItem * header = this->horizontalHeaderItem(col);
		if (header)
			text.append(header->text());
		text.append('\t');
	}
	text.append('\n');
	foreach(QModelIndex idx, indexes){
		int row = idx.row();
		for (int col = 0; col < this->columnCount(); col++){
			QTableWidgetItem * item = this->item(row, col);
			if (item)
				text.append(item->text());
			text.append('\t');
		}
		text.append('\n');
	}
	QApplication::clipboard()->setText(text);
}

void MyTableWidget::copyToClipboard()
{
	QString selected_text = exportContent();
	QClipboard *clipboard = QApplication::clipboard();
	clipboard->setText(selected_text);
}

void MyTableWidget::exportResults(const QString & _dir, const bool _askUser)
{
	QString tmp(_dir), name;
	if (_askUser){
		if (!tmp.endsWith("/")) tmp.append("/");
		tmp.append("colocalization_coeffcicients.txt");
		name = QFileDialog::getSaveFileName(this, QObject::tr("Save coefficients..."), tmp, QObject::tr("Text files (*.txt)"), 0, QFileDialog::DontUseNativeDialog);
	}
	else
		name = tmp;
	std::cout << name.toLatin1().data() << std::endl;
	QFileInfo info(name);
	std::ofstream fs(info.absoluteFilePath().toStdString());
	if (!fs){
		std::cout << "Failed to open " << name.toLatin1().data() << " to save values" << std::endl;
		return;
	}
	QString selected_text = exportContent();
	fs << selected_text.toLatin1().data();
	fs.close();
}

const QString MyTableWidget::exportContent()
{
	QString selected_text;
	for (int column = 0; column < this->columnCount(); column++){
		QTableWidgetItem * header = this->horizontalHeaderItem(column);
		if (header)
			selected_text.append(header->text());
		if (column < this->columnCount() - 1)
			selected_text.append('\t');
	}
	for (int row = 0; row < this->rowCount(); row++){
		selected_text.append("\n");
		for (int column = 0; column < this->columnCount(); column++){
			QTableWidgetItem * item = this->item(row, column);
			if (item)
				selected_text.append(item->text());
			if (column < this->columnCount() - 1)
				selected_text.append("\t");
		}
	}
	return selected_text;
}