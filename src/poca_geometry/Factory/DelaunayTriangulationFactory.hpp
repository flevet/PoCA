/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DelaunayTriangulationFactory.hpp
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

#ifndef DelaunayTriangulationFactory_h__
#define DelaunayTriangulationFactory_h__

#include "../Interfaces/DelaunayTriangulationFactoryInterface.hpp"

namespace poca::geometry {
	class DelaunayTriangulationInterface;

	class DelaunayTriangulationFactory : public DelaunayTriangulationFactoryInterface {
	public:
		DelaunayTriangulationFactory();
		~DelaunayTriangulationFactory();

		DelaunayTriangulationInterface* createDelaunayTriangulation(poca::core::MyObjectInterface*, poca::core::PluginList*, const bool = true);
		DelaunayTriangulationInterface* createDelaunayTriangulationOnSphere(poca::core::MyObjectInterface*, poca::core::PluginList*, const poca::core::Vec3mf&, const float, const bool = true);

		DelaunayTriangulationInterface* createDelaunayTriangulation(const std::vector <float>&, const std::vector <float>&);
		DelaunayTriangulationInterface* createDelaunayTriangulation(const std::vector <float>&, const std::vector <float>&, const std::vector <float>&);
	
		DelaunayTriangulationInterface* createDelaunayTriangulationOnSphere(const std::vector <float>&, const std::vector <float>&, const std::vector <float>&, const poca::core::Vec3mf&, const float);
	};
}

#endif // DelaunayTriangulationFactory_h__

