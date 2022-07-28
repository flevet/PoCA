/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      LoaderPDBFile.cpp
*
* Copyright: Florian Levet (2020-2021)
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
#include <gemmi/mmread.hpp>

#include "LoaderPDBFile.hpp"

const QString getTimeElapsed(QElapsedTimer _time)
{
	int elapsedTime = _time.elapsed();
	QString out = QString("%1:%2").arg(elapsedTime / 60000, 2, 10, QChar('0'))
		.arg((elapsedTime % 60000) / 1000, 2, 10, QChar('0'));
	return out;
}

void LoaderPDBFile::loadFile(const QString& _filename, std::map <std::string, std::vector <float>>& _data, poca::core::CommandInfo* _command)
{
	if (!_filename.endsWith(".pdb") && !_filename.endsWith(".cif")) return;

	/*pdb* P;
	P = initPDB();
	parsePDB(_filename.toStdString().data(), P, "");
	//printPDB(P);

	std::vector <float> xs, ys, zs;

	chain* C = NULL;
	atom* A = NULL;
	int chainId;
	for (chainId = 0; chainId < P->size; chainId++) {
		C = &P->chains[chainId];
		A = &C->residues[0].atoms[0];
		while (A != NULL) {
			xs.push_back(A->coor.x * 10.f);
			ys.push_back(A->coor.y * 10.f);
			zs.push_back(A->coor.z * 10.f);
			A = A->next;
		}
	}

	_data["x"] = xs;
	_data["y"] = ys;
	_data["z"] = zs;

	freePDB(P);*/

	std::vector <float> xs, ys, zs, idxChain, idxResidue, idxModel;
	gemmi::Structure st = gemmi::read_structure_file(_filename.toStdString());
	std::cout << "This file has " << st.models.size() << " models.\n";

	std::set <std::string> names;
	for (const auto& chain : st.children()[0].children())
		for (const auto& residue : chain.children())
			for (const auto& atom : residue.children()) {
				//std::cout << atom.padded_name() << std::endl;
				names.insert(atom.padded_name());
			}

	std::copy(names.begin(), names.end(), std::ostream_iterator<std::string>(std::cout, "\n"));

	uint32_t index = 1;
	std::map<std::string, uint32_t> namesWithIndex;
	for (const auto& name : names)
		namesWithIndex[name] = index++;

	float idModel = 1.f;
	for (const auto& model : st.children()) {
		float idchain = 1.f;
		//std::cout << "chain size = " << model.children().size() << std::endl;
		for (const auto& chain : model.children()) {
			float idresidue = 0.f;
			//std::cout << "residue size = " << chain.children().size() << std::endl;
			for (const auto& residue : chain.children()) {
				//std::cout << "atom size = " << residue.children().size() << std::endl;
				for (const auto& atom : residue.children()) {
					xs.push_back(atom.pos.x * 10.f);
					ys.push_back(atom.pos.y * 10.f);
					zs.push_back(atom.pos.z * 10.f);
					std::string tmp = atom.padded_name();
					auto it = namesWithIndex.find(atom.padded_name());
					idxChain.push_back(idchain);// it->second);
					idxResidue.push_back(idresidue);// it->second);
					idxModel.push_back(idModel);// it->second);
				}
				idresidue++;
			}
			idchain++;
		}
		idModel++;
	}
	if (xs.empty())
		std::cout << "No data loaded." << std::endl;
	else {
		_data["x"] = xs;
		_data["y"] = ys;
		_data["z"] = zs;
		_data["chain"] = idxChain;
		_data["residue"] = idxResidue;
		_data["model"] = idxModel;
	}
}