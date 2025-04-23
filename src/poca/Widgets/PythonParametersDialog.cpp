/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      PythonParametersDialog.cpp
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

#include <QtWidgets/QPushButton>
#include <QtWidgets/QBoxLayout>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QGroupBox>
#include <QtCore/QDir>
#include <QtWidgets/QFileDialog>

#include <Plot/Icons.hpp>

#include "PythonParametersDialog.hpp"

PythonParametersDialog::PythonParametersDialog(const nlohmann::json&_parameters, QWidget * _parent, Qt::WindowFlags _f) :QDialog(_parent, _f)
{
	std::vector <QString> labelGroupBoxes = { "Path to Python", "Path to Python DLL", 
		"Path to Python Lib", "Path to Python site packages", "Path to PoCA Python scripts" };
	m_labels.resize(labelGroupBoxes.size());
	m_buttons.resize(labelGroupBoxes.size());
	m_paths.resize(labelGroupBoxes.size());
	std::vector <QGroupBox*> groupBoxes(labelGroupBoxes.size());
	m_nameInJson = std::vector <std::string>{ "python_path", "python_dll_path", "python_lib_path", "python_packages_path", "python_scripts_path" };
	for (auto n = 0; n < m_nameInJson.size(); n++) {
		if (!_parameters.contains("PythonParameters")) continue;
		if (_parameters["PythonParameters"].contains(m_nameInJson[n]))
			m_paths[n] = _parameters["PythonParameters"][m_nameInJson[n]].get<std::string>();
	}

	int maxSize = 20;

	for (auto n = 0; n < labelGroupBoxes.size(); n++) {
		groupBoxes[n] = new QGroupBox(labelGroupBoxes[n]);
		groupBoxes[n]->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
		m_labels[n] = new QLabel(m_paths[n].c_str());
		m_labels[n]->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
		m_buttons[n] = new QPushButton();
		m_buttons[n]->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
		m_buttons[n]->setMaximumSize(QSize(maxSize, maxSize));
		m_buttons[n]->setIcon(QIcon(QPixmap(poca::plot::openDirIcon)));
		QObject::connect(m_buttons[n], SIGNAL(pressed()), this, SLOT(actionNeeded()));
		QHBoxLayout* layout = new QHBoxLayout;
		layout->addWidget(m_labels[n]);
		layout->addWidget(m_buttons[n]);
		groupBoxes[n]->setLayout(layout);
	}

	QPushButton* closeBtn = new QPushButton("Ok", this);
	QPushButton* cancelBtn = new QPushButton("Cancel", this);
	QVBoxLayout* layoutGroups = new QVBoxLayout;
	for(QGroupBox* group : groupBoxes)
		layoutGroups->addWidget(group);
	QHBoxLayout* layoutButton = new QHBoxLayout;
	layoutButton->addWidget(closeBtn);
	layoutButton->addWidget(cancelBtn);
	QVBoxLayout* layoutAll = new QVBoxLayout;
	layoutGroups->addLayout(layoutGroups);
	layoutGroups->addLayout(layoutButton);

	this->setLayout(layoutGroups);
	this->setWindowTitle("Parameters");
	QPoint p = QCursor::pos();
	this->setGeometry(p.x(), p.y(), this->sizeHint().width(), this->sizeHint().height());

	QObject::connect(closeBtn, SIGNAL(clicked()), this, SLOT(accept()));
	QObject::connect(cancelBtn, SIGNAL(clicked()), this, SLOT(reject()));
}

PythonParametersDialog::~PythonParametersDialog()
{

}

void PythonParametersDialog::actionNeeded()
{
	QObject* sender = QObject::sender();
	for (auto n = 0; n < m_buttons.size(); n++) {
		if (!(sender == m_buttons[n])) continue;
		std::string path = m_paths[n].empty() ? QDir::currentPath().toStdString() : m_paths[n];
		QString dirName = QFileDialog::getExistingDirectory(0,
			QObject::tr("Select directory"),
			path.c_str(),
			QFileDialog::DontUseNativeDialog | QFileDialog::DontResolveSymlinks);
		if (dirName.isEmpty()) return;
		m_labels[n]->setText(dirName);
		m_paths[n] = dirName.toStdString();
		if (n != 0) continue;
		std::vector <std::string> pythonFolders = { "\\DLLs", "\\Lib", "\\Lib\\site-packages" };
		for (auto i = 0; i < pythonFolders.size(); i++) {
			QDir pathDir((m_paths[n] + pythonFolders[i]).c_str());
			if (pathDir.exists()) {
				m_paths[i + 1] = pathDir.absolutePath().toStdString();
				m_labels[i + 1]->setText(pathDir.absolutePath());
			}
		}
	}
}