/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectIndicesFactory.cpp
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

#include <fstream>

#include <General/BasicComponent.hpp>
#include <Interfaces/HistogramInterface.hpp>
#include <Interfaces/MyObjectInterface.hpp>
#include <General/Misc.h>

#include "ObjectIndicesFactory.hpp"
#include "../Geometry/VoronoiDiagram.hpp"

namespace poca::geometry {
	ObjectIndicesFactoryInterface* createObjectIndicesFactory()
	{
		return new ObjectIndicesFactory();
	}

	ObjectIndicesFactory::ObjectIndicesFactory()
	{

	}

	ObjectIndicesFactory::~ObjectIndicesFactory()
	{

	}

	std::vector <uint32_t>  ObjectIndicesFactory::createObjects(poca::core::MyObjectInterface* _obj, const std::vector <bool>& _selection, const size_t _minNbLocs, const size_t _maxNbLocs)
	{
		poca::core::BasicComponentInterface* bci = _obj->getBasicComponent("VoronoiDiagram");
		VoronoiDiagram* voronoi = dynamic_cast <VoronoiDiagram*>(bci);
		if (!voronoi) return std::vector <uint32_t>();
		/*if (delaunay->dimension() == 2)
			return createObjects2D(delaunay, _selection, _minNbLocs, _maxNbLocs);
		else if (delaunay->dimension() == 3)
			return createObjects3D(delaunay, _selection, _minNbLocs, _maxNbLocs);*/

		std::vector <uint32_t> clustersIndices(_selection.size(), 0);

		const poca::core::MyArrayUInt32& neighbors = voronoi->getNeighbors();

		uint32_t currentIndex = 1;

		std::vector <bool> selection(_selection);
		for (auto n = 0; n < selection.size(); n++) {
			if (!selection[n]) continue;
			std::vector <uint32_t> indices;
			indices.push_back(n);
			selection[n] = false;
			size_t cur = 0, size = indices.size();
			while (cur < indices.size()) {
				//for (auto i = cur; i < size; i++) {
				size_t index = indices[cur];
				for (uint32_t neigh = 0; neigh < neighbors.nbElementsObject(index); neigh++) {
					uint32_t indexNeigh = neighbors.elementIObject(index, neigh);
					if (indexNeigh != std::numeric_limits<std::uint32_t>::max() && selection[indexNeigh]) {
						indices.push_back(indexNeigh);
						selection[indexNeigh] = false;
					}
				}
				cur++;
				//}
			}
			if (_minNbLocs <= indices.size() && indices.size() <= _maxNbLocs) {
				for (auto id : indices)
					clustersIndices[id] = currentIndex;
				currentIndex++;
			}
		}

		return clustersIndices;
	}

	std::vector <uint32_t>  ObjectIndicesFactory::createObjects2D(DelaunayTriangulationInterface* _delaunay, const std::vector <bool>& _selection, const size_t _minNbLocs, const size_t _maxNbLocs)
	{
		std::vector <uint32_t> clustersIndices(_selection.size(), 0);

		const poca::core::MyArrayUInt32& neighbors = _delaunay->getNeighbors();

		uint32_t currentIndex = 1;

		std::vector <bool> selection(_selection);
		for (auto n = 0; n < selection.size(); n++) {
			if (!selection[n]) continue;
			std::vector <uint32_t> indices;
			indices.push_back(n);
			selection[n] = false;
			size_t cur = 0, size = indices.size();
			while (cur < indices.size()) {
				//for (auto i = cur; i < size; i++) {
					size_t index = indices[cur];
					for (uint32_t neigh = 0; neigh < neighbors.nbElementsObject(index); neigh++) {
						uint32_t indexNeigh = neighbors.elementIObject(index, neigh);
						if (selection[indexNeigh]) {
							indices.push_back(indexNeigh);
							selection[indexNeigh] = false;
						}
					}
					cur++;
				//}
			}
			if (_minNbLocs <= indices.size() && indices.size() <= _maxNbLocs) {
				for (auto id : indices)
					clustersIndices[id] = currentIndex;
				currentIndex++;
			}
		}

		return clustersIndices;
	}

	std::vector <uint32_t>  ObjectIndicesFactory::createObjects3D(DelaunayTriangulationInterface* _delaunay, const std::vector <bool>& _selection, const size_t _minNbLocs, const size_t _maxNbLocs)
	{
		std::vector <uint32_t> clustersIndices(_selection.size(), 0);
		
		const float* xs = _delaunay->getXs();
		const float* ys = _delaunay->getYs();
		const float* zs = _delaunay->getZs();
		const poca::core::MyArrayUInt32& neighbors = _delaunay->getNeighbors();

		return clustersIndices;
	}
}