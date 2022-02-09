/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      PythonParametersDialog.hpp
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

#ifndef PythonParametersDialog_h__
#define PythonParametersDialog_h__

#include <QtWidgets/QDialog>
#include <vector>

#include <General/json.hpp>

class QLabel;
class QPushButton;

class PythonParametersDialog : public QDialog{
	Q_OBJECT

public:
	PythonParametersDialog(const nlohmann::json&, QWidget * = 0, Qt::WindowFlags = 0);
	~PythonParametersDialog();

	const std::vector <std::string>& getNameParameters() const { return m_nameInJson; }
	const std::vector <std::string>& getPaths() const { return m_paths; }

protected slots:
	void actionNeeded();

protected:
	QLabel* m_lblPython, * m_lblPythonDLL, * m_lblPythonLib, * m_lblPythonSitePackage, * m_lblPythonScripts;
	QPushButton* m_btnPython, * m_btnPythonDLL, * m_btnPythonLib, * m_btnPythonSitePackage, * m_btnPythonScripts;
	std::vector <QLabel*> m_labels;
	std::vector <QPushButton*> m_buttons;
	std::vector <std::string> m_paths, m_nameInJson;
};

#endif // ColocalizationChoiceDialog_h__

