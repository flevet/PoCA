/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      PythonWidget.cpp
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

#ifndef NO_PYTHON

#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QColorDialog>
#include <QtGui/QRegExpValidator>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QOpenGLWidget>
#include <QtWidgets/QButtonGroup>
#include <fstream>

#include <General/Misc.h>
#include <Plot/Icons.hpp>
#include <Geometry/DetectionSet.hpp>
#include <General/PythonInterpreter.hpp>
#include <Objects/MyObject.hpp>
#include <Factory/ObjectListFactory.hpp>
#include <Geometry/ObjectList.hpp>
#include <General/MyData.hpp>

#include "../Widgets/PythonWidget.hpp"

PythonWidget::PythonWidget(poca::core::MediatorWObjectFWidget* _mediator, QWidget* _parent/*= 0*/, Qt::WindowFlags _f/*= 0 */) :QWidget(_parent, _f)
{
	m_parentTab = (QTabWidget*)_parent;
	m_mediator = _mediator;
	m_object = NULL;

	this->setObjectName("PythonWidget");
	this->addActionToObserve("LoadObjCharacteristicsAllWidgets");

	m_groupPreloadedPythonFiles = new QGroupBox(QObject::tr("Pre-loaded Python modules"));
	m_groupPreloadedPythonFiles->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
	int maxSize = 100;
	std::vector <std::string> namePythonFiles = { "nena", "ellipsoidFit", "CAML" };
	for (const std::string& name : namePythonFiles) {
		std::string filename = "./pythonScripts/" + name + ".py";

		if (!poca::core::file_exists(filename)) continue;
		m_buttonsPreloaded.push_back(std::make_pair(new QPushButton(), name));
		m_buttonsPreloaded.back().first->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
		m_buttonsPreloaded.back().first->setMaximumSize(QSize(maxSize, maxSize));
		m_buttonsPreloaded.back().first->setIconSize(QSize(maxSize, maxSize));;
		m_buttonsPreloaded.back().first->setIcon(QIcon(QPixmap(poca::plot::filePythonIcon)));
		QObject::connect(m_buttonsPreloaded.back().first, SIGNAL(pressed()), this, SLOT(actionNeeded()));
	}
	QGridLayout* layoutPredefined = new QGridLayout;
	int lineCount = 0, columeCount = 0;
	for (size_t n = 0; n < m_buttonsPreloaded.size(); n++) {
		QVBoxLayout* layout = new QVBoxLayout;
		layout->addWidget(m_buttonsPreloaded[n].first);
		layout->setAlignment(m_buttonsPreloaded[n].first, Qt::AlignHCenter);
		QLabel* lbl = new QLabel(m_buttonsPreloaded[n].second.c_str());
		lbl->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
		layout->addWidget(lbl);
		layout->setAlignment(lbl, Qt::AlignHCenter);
		layoutPredefined->addLayout(layout, lineCount, columeCount);
		if ((n + 1) % 3 == 0) {
			lineCount++;
			columeCount = 0;
		}
		else
			columeCount++;
	}
	m_groupPreloadedPythonFiles->setLayout(layoutPredefined);

	QWidget* emptyW = new QWidget;
	emptyW->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	QVBoxLayout* layout = new QVBoxLayout;
	layout->addWidget(m_groupPreloadedPythonFiles);
	layout->addWidget(emptyW);

	this->setLayout(layout);
}

PythonWidget::~PythonWidget()
{
}

void PythonWidget::actionNeeded()
{
	QObject* sender = QObject::sender();
	bool found = false;
	for (size_t n = 0; n < m_buttonsPreloaded.size() && !found; n++) {
		found = (m_buttonsPreloaded[n].first == sender);
		if (found) {
			if (m_buttonsPreloaded[n].second == "nena")
				executeNena();
			else if (m_buttonsPreloaded[n].second == "ellipsoidFit")
				executeEllipsoidFit();
			else if (m_buttonsPreloaded[n].second == "CAML")
				executeCAML();
		}
	}
}

void PythonWidget::actionNeeded(int _val)
{

}

void PythonWidget::actionNeeded(bool _val)
{

}

void PythonWidget::executeNena()
{
	poca::core::MyObjectInterface* obj = m_object->currentObject();
	poca::core::BasicComponent* bci = obj->getBasicComponent("DetectionSet");
	poca::geometry::DetectionSet* dset = dynamic_cast <poca::geometry::DetectionSet*>(bci);
	if (dset == NULL) return;

	if (!dset->hasData("frame")) return;

	const std::vector <float>& xs = dset->getData("x");
	const std::vector <float>& ys = dset->getData("y");
	const std::vector <float>& zs = dset->hasData("z") ? dset->getData("z") : std::vector <float>(xs.size(), 0.f);
	const std::vector <float>& times = dset->getData("frame");

	//Prepare data per frame
	std::vector <uint32_t> pointsPerFrame;
	float currentTime = times[0];
	uint32_t n = 0;
	pointsPerFrame.push_back(n++);
	for (; n < dset->nbPoints(); n++) {
		if (times[n] != currentTime) {
			currentTime = times[n];
			pointsPerFrame.push_back(n);
		}
	}
	pointsPerFrame.push_back(n - 1);

	QVector <double> distancesNena;

	double dmin = std::numeric_limits < double >::max(), d = 0.;
	for (size_t currentSlice = 0; currentSlice < pointsPerFrame.size() - 2; currentSlice++) {
		size_t nextSlice = currentSlice + 1;
		for (uint32_t i = pointsPerFrame[currentSlice]; i < pointsPerFrame[nextSlice]; i++) {
			dmin = std::numeric_limits < double >::max();
			bool found = false;
			for (uint32_t j = pointsPerFrame[nextSlice]; j < pointsPerFrame[nextSlice + 1]; j++) {
				d = poca::geometry::distance3DSqr(xs[i], ys[i], zs[i], xs[j], ys[j], zs[j]);
				if (d < dmin) {
					dmin = d;
					found = true;
				}
			}
			if (found) {
				distancesNena.push_back(sqrt(dmin));
			}
		}
	}
	poca::core::PythonInterpreter* py = poca::core::PythonInterpreter::instance();
	QVector <QVector <double>> distances;
	distances.push_back(distancesNena);
	QVector <double> coeffs;
	bool res = py->applyFunctionWithNArraysParameterAnd1ArrayReturned(coeffs, distances, "nena", "NeNA");
	if (res == EXIT_FAILURE)
		std::cout << "ERROR! NeNA was not run with error message: python script was not run." << std::endl;
	else
		std::cout << "The average lecalization accuracy by NeNA is at " << coeffs[0] << " nm." << std::endl;
}

void PythonWidget::executeEllipsoidFit()
{
	poca::core::MyObjectInterface* obj = m_object->currentObject();
	poca::core::BasicComponent* bci = obj->getBasicComponent("DetectionSet");
	poca::geometry::DetectionSet* dset = dynamic_cast <poca::geometry::DetectionSet*>(bci);

	if (dset == NULL) return;
	if (!dset->hasData("z")) return;

	const std::vector <float>& xs = dset->getData("x");
	const std::vector <float>& ys = dset->getData("y");
	const std::vector <float>& zs = dset->getData("z");

	QVector <QVector <double>> coordinates;
	coordinates.resize(3);
	for (size_t n = 0; n < 3; n++)
		coordinates[n].resize(xs.size());
	for (size_t n = 0; n < xs.size(); n++) {
		coordinates[0][n] = xs[n];
		coordinates[1][n] = ys[n];
		coordinates[2][n] = zs[n];
	}
	QVector <double> coeffs;
	poca::core::PythonInterpreter* py = poca::core::PythonInterpreter::instance();
	bool res = py->applyFunctionWithNArraysParameterAnd1ArrayReturned(coeffs, coordinates, "ellipsoidFit", "ls_ellipsoid");
	if (res == EXIT_FAILURE)
		std::cout << "ERROR! Command ellipsoid fit was not run with error message: python script was not run." << std::endl;
}

void PythonWidget::executeCAML()
{
	poca::core::MyObjectInterface* obj = m_object->currentObject();
	poca::core::BasicComponent* bci = obj->getBasicComponent("DetectionSet");
	poca::geometry::DetectionSet* dset = dynamic_cast <poca::geometry::DetectionSet*>(bci);
	if (dset == NULL) return;

	bool is3D = dset->dimension() == 3;

	const std::vector <float>& xs = dset->getData("x");
	const std::vector <float>& ys = dset->getData("y");

	QVector <QVector <double>> data(is3D ? 4 : 3);

	poca::core::BoundingBox bbox = dset->boundingBox();
	data[0].push_back(bbox[3] - bbox[0]);
	data[0].push_back(bbox[4] - bbox[1]);
	data[0].push_back(is3D ? (bbox[5] - bbox[2]) : 0);
	data[0].push_back(bbox[3]);
	data[0].push_back(bbox[0]);
	data[0].push_back(bbox[4]);
	data[0].push_back(bbox[1]);
	data[0].push_back(is3D ? bbox[5] : 0);
	data[0].push_back(is3D ? bbox[2] : 0);
	for (unsigned int n = 0; n < dset->nbPoints(); n++) {
		data[1].push_back(xs[n]);
		data[2].push_back(ys[n]);
	}
	if (is3D) {
		const std::vector <float>& zs = dset->getData("z");
		for (unsigned int n = 0; n < dset->nbPoints(); n++)
			data[3].push_back(zs[n]);
	}

	poca::core::PythonInterpreter* py = poca::core::PythonInterpreter::instance();
	QVector <double> res;
	bool res2 = py->applyFunctionWithNArraysParameterAnd1ArrayReturned(res, data, "CAML", "testCAML");
	if (res2 == EXIT_FAILURE)
		std::cout << "ERROR! Command CAML fit was not run with error message: python script was not run." << std::endl;
	else {
		std::vector <float> camlID(res.size(), 0.f);
		std::transform(res.begin(), res.end(), camlID.begin(), [](double x) { return (float)x; });
		dset->addFeature("camlID", new poca::core::MyData(camlID));

		std::vector<bool> selection(dset->nbPoints(), false);
		for (int n = 0; n < res.size(); n++) {
			selection[n] = res[n];
		}
		dset->setSelection(selection);
		dset->executeCommand(false, "updateFeature");

		poca::geometry::ObjectListFactory factory;
		poca::geometry::ObjectList* objects = factory.createObjectList(obj, selection, 40.f);
		objects->setBoundingBox(dset->boundingBox());
		obj->addBasicComponent(objects);
		obj->notify(poca::core::CommandInfo(false, "addCommandToSpecificComponent", "component", (poca::core::BasicComponent*)objects));

		m_object->notifyAll("updateGL");
	}
}

void PythonWidget::performAction(poca::core::MyObjectInterface* _obj, poca::core::CommandInfo* _ci)
{
}

void PythonWidget::update(poca::core::SubjectInterface* _subject, const poca::core::CommandInfo& _aspect)
{
	poca::core::MyObjectInterface* obj = dynamic_cast <poca::core::MyObjectInterface*> (_subject);
	poca::core::MyObjectInterface* objOneColor = obj->currentObject();
	if (objOneColor == NULL) {
		m_groupPreloadedPythonFiles->setVisible(false);
		return;
	}
	m_groupPreloadedPythonFiles->setVisible(true);

	m_object = obj;

	bool visible = (objOneColor != NULL && objOneColor->hasBasicComponent("DetectionSet"));
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
	m_parentTab->setTabVisible(m_parentTab->indexOf(this), visible);
#endif

	if (_aspect == "LoadObjCharacteristicsAllWidgets") {
	}
}

void PythonWidget::executeMacro(poca::core::MyObjectInterface* _wobj, poca::core::CommandInfo* _ci)
{
	this->performAction(_wobj, _ci);
}

#endif