/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      LoaderCSVFile.cpp
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

#include <fstream>
#include <iostream>
#include <sstream>
#include <QtCore\QElapsedTimer>

#include <General/Misc.h>
#include <Geometry/nanoflann.hpp>
#include <General/Command.hpp>

#include "LoaderCSVFile.hpp"
#include "OpenFileDialog.hpp"



const QString getTimeElapsed(QElapsedTimer _time)
{
	int elapsedTime = _time.elapsed();
	QString out = QString("%1:%2").arg(elapsedTime / 60000, 2, 10, QChar('0'))
		.arg((elapsedTime % 60000) / 1000, 2, 10, QChar('0'));
	return out;
}

void LoaderCSVFile::loadFile(const QString& _filename, std::map <std::string, std::vector <float>>& _data, poca::core::CommandInfo* _command)
{
	if (!_filename.endsWith(".csv")) return;

	std::ifstream fs(_filename.toLatin1().data());
	char separator = 0;
	float multXY = 1.f, multZ = 1.f, multT = 1.f;
	if (_command->nbParameters() == 1) {//Command has only the path parameter
		OpenFileDialog ofd(fs, separator);
		ofd.setModal(true);

		if (ofd.exec() == QDialog::Accepted) {
			if (!ofd.areRequiredColumnsSelected()) {
				std::cout << "Columns x and/or y were not selected, the file " << _filename.toStdString() << " will not be opened!" << std::endl;
				return;
			}
			ofd.getColumns(_command);
		}
		else
			return;
	}

	if(!_command->hasParameter("x") && !_command->hasParameter("y")) {
		std::cout << "Columns x and/or y were not selected, the file " << _filename.toStdString() << " will not be opened!" << std::endl;
		return;
	}

	std::vector<std::pair<size_t, std::string>> columns;
	const nlohmann::json& json = _command->json["open"];

	for (auto& [key, value] : json.items()) {
		if (key == "path") continue;
		if (key == "calibration_xy")
			multXY = value.get<float>();
		else if (key == "calibration_xy")
			multXY = value.get<float>();
		else if (key == "calibration_z")
			multZ = value.get<float>();
		else if (key == "calibration_t")
			multT = value.get<float>();
		else if (key == "separator")
			separator = value.get<char>();
		else
			columns.push_back(std::make_pair(value.get<size_t>(), key));
	}

	int nbPoints = 0, nbSlices = 0;

	QElapsedTimer time;
	time.start();

	std::string s;

	QString realName(_filename);
	realName = _filename.mid(_filename.lastIndexOf("/") + 1);
	int nbLines = std::count(std::istreambuf_iterator<char>(fs), std::istreambuf_iterator<char>(), '\n');
	fs.clear();
	fs.seekg(0, std::ios::beg);
	unsigned int currentLine = 0, nbForUpdate = nbLines / 100.;
	if (nbForUpdate == 0) nbForUpdate = 1;
	printf("Reading %s: %.2f %%", realName.toLatin1().data(), ((double)currentLine / nbLines * 100.));

	std::getline(fs, s);//read headers

	std::vector<std::vector<float>> chosenValues(columns.size());

	while (std::getline(fs, s)) {
		std::vector < std::string > values;
		values = poca::core::split(s, separator, values);

		for (size_t index = 0; index < columns.size(); index++)
			chosenValues[index].push_back(::atof(values[columns[index].first].c_str()));
		
		if (currentLine++ % nbForUpdate == 0) printf("\rReading %s: %.2f %%", realName.toLatin1().data(), ((double)currentLine / nbLines * 100.));
	}
	printf("\rReading %s: 100.00 %%\n", realName.toLatin1().data());

	uint32_t indexX, indexY, indexZ, indexFrame;
	bool hasZ = false, hasFrame = false;

	for (size_t index = 0; index < columns.size(); index++) {
		if (columns[index].second == "x")
			indexX = index;
		else if (columns[index].second == "y")
			indexY = index;
		else if (columns[index].second == "z") {
			indexZ = index;
			hasZ = true;
		}
		else if (columns[index].second == "frame") {
			indexFrame = index;
			hasFrame = true;
		}
	}

	if (multXY != 1.f) {
		for (size_t n = 0; n < chosenValues[indexX].size(); n++) {
			chosenValues[indexX][n] *= multXY;
			chosenValues[indexY][n] *= multXY;
		}
	}
	if (multZ != 1.f && hasZ)
		for (size_t n = 0; n < chosenValues[indexZ].size(); n++)
			chosenValues[indexZ][n] *= multZ;
	if (multT != 1.f && hasFrame)
		for (size_t n = 0; n < chosenValues[indexFrame].size(); n++)
			chosenValues[indexFrame][n] *= multT;

	//discard duplicate locs (too close to each others)
	QElapsedTimer time2;
	time2.start();
	KdPointCloud_3D_D cloud;
	cloud.m_pts.resize(chosenValues[indexX].size());
	for (int i = 0; i < chosenValues[indexX].size(); i++) {
		cloud.m_pts[i].m_x = chosenValues[indexX][i];
		cloud.m_pts[i].m_y = chosenValues[indexY][i];
		cloud.m_pts[i].m_z = hasZ ? chosenValues[indexZ][i] : 0.f;
	}
	std::vector <bool> treatedLocs(chosenValues[indexX].size(), false);
	std::vector <size_t> indexesKept(chosenValues[indexX].size(), 0);
	size_t cptKept = 0;
	const double DOUBLE_EPSILON = .00000000001;
	KdTree_3D_double kdtree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
	kdtree.buildIndex();
	const double search_radius = static_cast<double>(DOUBLE_EPSILON * DOUBLE_EPSILON);
	std::vector< std::pair< std::size_t, double > > ret_matches;
	nanoflann::SearchParams params;
	std::size_t nMatches;
	double nbPs = chosenValues[indexX].size();
	nbForUpdate = nbPs / 100.;
	if (nbForUpdate == 0) nbForUpdate = 1;
	printf("\rDetermination of identical localizations: %.2f %%", (0. / nbPs * 100.));
	for (int i = 0; i < chosenValues[indexX].size(); i++) {
		if (i % nbForUpdate == 0) printf("\rDetermination of identical localizations: %.2f %%", ((double)i / nbPs * 100.));
		if (treatedLocs[i]) continue;
		indexesKept[cptKept++] = i;

		const double queryPt[3] = { chosenValues[indexX][i], chosenValues[indexY][i], hasZ ? chosenValues[indexZ][i] : 0.f };
		nMatches = kdtree.radiusSearch(&queryPt[0], search_radius, ret_matches, params);
		for (size_t n = 0; n < nMatches; n++)
			treatedLocs[ret_matches[n].first] = true;
	}
	std::vector<std::vector<float>> correctedValues(columns.size());
	bool hasBeenCorrected = cptKept != chosenValues[indexX].size();
	if (hasBeenCorrected) {
		for(std::vector<float>& vec : correctedValues)
			vec.resize(cptKept);
		for (size_t i = 0; i < cptKept; i++) {
			size_t index = indexesKept[i];

			for (size_t n = 0; n < correctedValues.size(); n++)
				correctedValues[n][i] = chosenValues[n][index];
		}
	}
	printf("\rDetermination of identical localizations: 100.00 %%\n");
	std::cout << (chosenValues[indexX].size() - cptKept) << " localizations were identicals (computed in " << getTimeElapsed(time2).toLatin1().data() << ")" << std::endl;

	std::vector <float> ids(hasBeenCorrected ? correctedValues[0].size() : chosenValues[0].size());
	std::iota(std::begin(ids), std::end(ids), 1);
	_data["id"] = ids;

	for (size_t n = 0; n < chosenValues.size(); n++)
		_data[columns[n].second] = hasBeenCorrected ? correctedValues[n] : chosenValues[n];

	fs.close();
}