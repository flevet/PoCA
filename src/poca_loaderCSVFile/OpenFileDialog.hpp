/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      OpenFileDialog.hpp
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

#ifndef OpenFileDialog_h__
#define OpenFileDialog_h__

#include <QtWidgets/QDialog>

#include <vector>
#include <fstream>

class QComboBox;
class QString;
class QLineEdit;
class QTextEdit;
class QCheckBox;

namespace poca::core {
	class CommandInfo;
}

class OpenFileDialog : public QDialog{
	Q_OBJECT

public:
	OpenFileDialog(std::ifstream&, char, QWidget * = 0, Qt::WindowFlags = 0);
	~OpenFileDialog();

	const float getXY() const;
	const float getZ() const;
	const float getT() const;

	const bool areRequiredColumnsSelected() const;
	void getColumns(poca::core::CommandInfo*) const;

	inline void setValues(float& _xy, float& _z, float& _t) const { _xy = getXY(); _z = getZ(); _t = getT(); }

protected slots:
	void actionNeeded(int);

protected:
	std::vector <QCheckBox*> m_useColumns;
	std::vector <QComboBox*> m_choiceColumns;
	std::vector <QTextEdit*> m_previewColumns;
	QTextEdit* m_requiredColumns, *m_otherColumns;

	std::vector <std::pair<std::string, bool>> m_knownHeaders;
	//Calibration
	QLineEdit * m_pixXYEdit, * m_pixZEdit, * m_timeEdit;
	QComboBox * m_dimCombo, * m_timeCombo;

	char m_separator;
};

#endif // ColocalizationChoiceDialog_h__

