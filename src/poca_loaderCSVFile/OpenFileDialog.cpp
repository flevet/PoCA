/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      OpenFileDialog.cpp
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

#include <QtWidgets/QPushButton>
#include <QtWidgets/QBoxLayout>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <QtCore/QString>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QCheckBox>

#include <iostream>
#include <sstream>

#include <General/Misc.h>
#include <General/Command.hpp>

#include "OpenFileDialog.hpp"

OpenFileDialog::OpenFileDialog(std::ifstream& _fs, char _separator, QWidget * _parent, Qt::WindowFlags _f) :QDialog(_parent, _f), m_separator(_separator)
{
	QGroupBox* groupColumns = new QGroupBox("Column names and preview");
	groupColumns->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	//Read header line
	std::string s;
	std::getline(_fs, s);
	if (m_separator == 0) {
		std::string::size_type loc = s.find(",", 0);
		if (loc != std::string::npos)
			m_separator = ',';
		else {
			loc = s.find(";", 0);
			if (loc != std::string::npos)
				m_separator = ';';
			else {
				loc = s.find("\t", 0);
				if (loc != std::string::npos)
					m_separator = '\t';
				else {
					loc = s.find(" ", 0);
					if (loc != std::string::npos)
						m_separator = ' ';
					else
						return;
				}
			}
		}
	}
	m_knownHeaders = { std::make_pair("x", false), std::make_pair("y", false), std::make_pair("z", false), std::make_pair("intensity", false), std::make_pair("frame", false), 
		std::make_pair("sigmaXY", false), std::make_pair("sigmaZ", false), std::make_pair("nx", false), std::make_pair("ny", false), std::make_pair("nz", false) };
	
	std::vector < std::string > headers, line1, line2, line3;
	headers = poca::core::split(s, m_separator, headers);
	std::getline(_fs, s);
	line1 = poca::core::split(s, m_separator, line1);
	std::getline(_fs, s);
	line2 = poca::core::split(s, m_separator, line2);
	std::getline(_fs, s);
	line3 = poca::core::split(s, m_separator, line3);
	_fs.seekg(0, std::ios::beg);
	for (size_t n = 0; n < headers.size(); n++) {
		QCheckBox* cbox = new QCheckBox("select");
		cbox->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
		cbox->setChecked(false);
		m_useColumns.push_back(cbox);
		QComboBox* combo = new QComboBox;
		combo->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
		m_choiceColumns.push_back(combo);
		QTextEdit* tedit = new QTextEdit;
		tedit->setMaximumWidth(150);
		tedit->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
		tedit->setReadOnly(true);
		m_previewColumns.push_back(tedit);
	}
	for (size_t n = 0; n < headers.size(); n++) {
		m_useColumns[n]->setChecked(false);
		m_choiceColumns[n]->addItem(headers[n].c_str());
		for (const std::pair<std::string, bool>& header : m_knownHeaders)
			m_choiceColumns[n]->addItem(header.first.c_str());
		m_previewColumns[n]->append(line1[n].c_str());
		m_previewColumns[n]->append(line2[n].c_str());
		m_previewColumns[n]->append(line3[n].c_str());
	}

	QGridLayout* layourPreview = new QGridLayout;
	int lineCount = 0, columnCount = 0;
	for (QCheckBox* cbox : m_useColumns)
		layourPreview->addWidget(cbox, lineCount, columnCount++, 1, 1);
	lineCount++; columnCount = 0;
	for (QComboBox* combo : m_choiceColumns)
		layourPreview->addWidget(combo, lineCount, columnCount++, 1, 1);
	lineCount++; columnCount = 0;
	for (QTextEdit* tedit : m_previewColumns)
		layourPreview->addWidget(tedit, lineCount, columnCount++, 1, 1);
	groupColumns->setLayout(layourPreview);


	QLabel* lbl1 = new QLabel("Please assign variable names to each columns. The x and y coordinates have to be assigned to load the file");
	lbl1->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);

	QGroupBox * groupCalibration = new QGroupBox("Calibration");
	groupCalibration->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	QLabel* pixXLbl = new QLabel("Pixel X/Y:");
	pixXLbl->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_pixXYEdit = new QLineEdit("1");
	m_pixXYEdit->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	QLabel* pixZLbl = new QLabel("Pixel Z:");
	pixZLbl->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_pixZEdit = new QLineEdit("1");
	m_pixZEdit->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	QLabel* timeLbl = new QLabel("Time:");
	timeLbl->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	m_timeEdit = new QLineEdit("1");
	m_timeEdit->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	QLabel* dimensionUnitLbl = new QLabel("Dimension unit");
	dimensionUnitLbl->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	QStringList dimensionTypes;
	dimensionTypes << "pix" << "cm" << "mm" << "µm" << "nm";
	m_dimCombo = new QComboBox;
	m_dimCombo->addItems(dimensionTypes);
	m_dimCombo->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	QLabel* timeUnitLbl = new QLabel("Time unit");
	timeUnitLbl->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	QStringList timeTypes;
	timeTypes << "s" << "ms";
	m_timeCombo = new QComboBox;
	m_timeCombo->addItems(timeTypes);
	m_timeCombo->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	QGridLayout* layoutCalibration = new QGridLayout;
	lineCount = 0; columnCount = 0;
	layoutCalibration->addWidget(pixXLbl, lineCount, columnCount++, 1, 1);
	layoutCalibration->addWidget(m_pixXYEdit, lineCount, columnCount++, 1, 1);
	layoutCalibration->addWidget(dimensionUnitLbl, lineCount, columnCount++, 1, 1);
	layoutCalibration->addWidget(m_dimCombo, lineCount++, columnCount, 1, 1);
	columnCount = 0;
	layoutCalibration->addWidget(pixZLbl, lineCount, columnCount++, 1, 1);
	layoutCalibration->addWidget(m_pixZEdit, lineCount, columnCount++, 1, 1);
	layoutCalibration->addWidget(timeUnitLbl, lineCount, columnCount++, 1, 1);
	layoutCalibration->addWidget(m_timeCombo, lineCount++, columnCount, 1, 1);
	columnCount = 0;
	layoutCalibration->addWidget(timeLbl, lineCount, columnCount++, 1, 1);
	layoutCalibration->addWidget(m_timeEdit, lineCount, columnCount++, 1, 1);
	groupCalibration->setLayout(layoutCalibration);

	QGroupBox* groupRequired = new QGroupBox("Required columns");
	groupRequired->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_requiredColumns = new QTextEdit;
	m_requiredColumns->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_requiredColumns->setTextColor(Qt::red);
	QHBoxLayout* layoutRequired = new QHBoxLayout;
	layoutRequired->addWidget(m_requiredColumns);
	groupRequired->setLayout(layoutRequired);

	QGroupBox* groupOthers = new QGroupBox("Other usual columns");
	groupOthers->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	m_otherColumns = new QTextEdit;
	m_otherColumns->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
	QHBoxLayout* layoutOthers = new QHBoxLayout;
	layoutOthers->addWidget(m_otherColumns);
	groupOthers->setLayout(layoutOthers);

	QHBoxLayout* layoutMisc = new QHBoxLayout;
	layoutMisc->addWidget(groupCalibration);
	layoutMisc->addWidget(groupRequired);
	layoutMisc->addWidget(groupOthers);

	QPushButton * closeBtn = new QPushButton("Ok", this);
	QPushButton * cancelBtn = new QPushButton("Cancel", this);
	QHBoxLayout * layoutButton = new QHBoxLayout;
	layoutButton->addWidget(closeBtn);
	layoutButton->addWidget(cancelBtn);
	QWidget * widButton = new QWidget;
	widButton->setLayout(layoutButton);

	QVBoxLayout * layout = new QVBoxLayout;
	layout->addWidget(lbl1);
	layout->addWidget(groupColumns);
	layout->addLayout(layoutMisc);
	layout->addWidget(widButton);

	for (size_t i = 0; i < headers.size(); i++) {
		bool found = false;
		for (size_t j = 1; j < m_choiceColumns[i]->count() && !found; j++) {
			found = headers[i] == m_choiceColumns[i]->itemText(j).toStdString();
			if (found) {
				m_choiceColumns[i]->setCurrentIndex(j);
				m_knownHeaders[j - 1].second = true;
			}
		}
		m_useColumns[i]->setChecked(found);
	}

	for (size_t i = 0; i < m_knownHeaders.size(); i++) {
		if (m_knownHeaders[i].second == false) {
			if (i < 2)
				m_requiredColumns->append(QString(m_knownHeaders[i].first.c_str()) + " is not selected");
			else
				m_otherColumns->append(QString(m_knownHeaders[i].first.c_str()) + " is not selected");
		}
	}

	this->setLayout(layout);
	this->setWindowTitle("Open file");
	QPoint p = QCursor::pos();
	this->setGeometry(p.x() - sizeHint().width() / 2, p.y(), sizeHint().width(), sizeHint().height());

	QObject::connect(closeBtn, SIGNAL(clicked()), this, SLOT(accept()));
	QObject::connect(cancelBtn, SIGNAL(clicked()), this, SLOT(reject()));

	for (size_t n = 0; n < headers.size(); n++)
		QObject::connect(m_choiceColumns[n], SIGNAL(currentIndexChanged(int)), this, SLOT(actionNeeded(int)));
}

OpenFileDialog::~OpenFileDialog()
{

}

void OpenFileDialog::actionNeeded(int _index)
{
	QObject* sender = QObject::sender();
	bool update = false;
	for (size_t n = 0; n < m_choiceColumns.size() && !update; n++) {
		if (sender == m_choiceColumns[n]) {
			m_useColumns[n]->setChecked(true);
			update = true;
		}
	}
	if (update) {
		for (size_t i = 0; i < m_choiceColumns.size(); i++) {
			std::string header = m_choiceColumns[i]->currentText().toStdString();
			bool found = false;
			for (size_t j = 0; j < m_knownHeaders.size() && !found; j++) {
				found = m_knownHeaders[j].first == header;
				if (found)
					m_knownHeaders[j].second = true;
			}
		}
		m_requiredColumns->clear();
		m_otherColumns->clear();
		for (size_t i = 0; i < m_knownHeaders.size(); i++) {
			if (m_knownHeaders[i].second == false) {
				if (i < 2)
					m_requiredColumns->append(QString(m_knownHeaders[i].first.c_str()) + " is not selected");
				else
					m_otherColumns->append(QString(m_knownHeaders[i].first.c_str()) + " is not selected");
			}
		}
	}
}

const float OpenFileDialog::getXY() const
{
	bool ok;
	float val = m_pixXYEdit->text().toFloat(&ok);
	return ok ? val : 1.f;
}

const float OpenFileDialog::getZ() const
{
	bool ok;
	float val = m_pixZEdit->text().toFloat(&ok);
	return ok ? val : 1.f;
}

const float OpenFileDialog::getT() const
{
	bool ok;
	float val = m_timeEdit->text().toFloat(&ok);
	return ok ? val : 1.f;
}

const bool OpenFileDialog::areRequiredColumnsSelected() const
{
	return m_knownHeaders[0].second && m_knownHeaders[1].second;
}

void OpenFileDialog::getColumns(poca::core::CommandInfo* _com) const
{
	_com->addParameters("calibration_xy", getXY(), "calibration_z", getZ(), "calibration_t", getT(), "separator", m_separator);

	for (size_t n = 0; n < m_useColumns.size(); n++) {
		if (!m_useColumns[n]->isChecked()) continue;
		_com->addParameter(m_choiceColumns[n]->currentText().toStdString(), n);
	}
}

