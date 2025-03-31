/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ColocTesseler.hpp
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

#ifndef ColocTesseler_hpp__
#define ColocTesseler_hpp__

#include <General/BasicComponent.hpp>
#include <General/ArrayStatistics.hpp>
#include <General/Scatterplot.hpp>
#include <Geometry/VoronoiDiagram.hpp>
#include <General/Vec4.hpp>

class ColocTesseler: public poca::core::BasicComponent {
public:
	enum ClassColocTesseler{BACKGROUND = 1, HIGH_DENSITY = 2, COLOC = 3};

	ColocTesseler(poca::geometry::VoronoiDiagram*, poca::geometry::VoronoiDiagram*);
	~ColocTesseler();

	poca::core::BasicComponentInterface* copy();
	void executeCommand(poca::core::CommandInfo*);

	inline poca::geometry::VoronoiDiagram* voronoiAt(size_t _idx) const { return m_voronois[_idx]; }
	inline const std::vector <uint32_t>& indexTriOtherColorAt(size_t _idx) const { return m_indexTrianglePointsInOtherColor[_idx]; }
	inline const poca::core::Scatterplot& scattergramAt(size_t _idx) const { return m_scattergram[_idx]; }
	inline poca::core::Scatterplot* scattergramPtrAt(size_t _idx) { return &m_scattergram[_idx]; }
	inline poca::core::Scatterplot* scattergramLogPtrAt(size_t _idx) { return &m_scattergramLog[_idx]; }

	inline std::vector <unsigned char>& classesLocsAt(size_t _idx) { return m_classesLocs[_idx]; }
	inline const std::vector <unsigned char>& classesLocsAt(size_t _idx) const { return m_classesLocs[_idx]; }

	const std::array <float, 2>& getSpearmans() const { return m_spearmans; };
	const std::array <float, 2>& getManders() const { return m_manders; };
	const std::array<std::vector <float>, 2>& getSpearmans2() const { return m_spearmans2; };
	const std::array<std::vector <float>, 2>& getManders2() const { return m_manders2; };

	const uint32_t dimension() const { return m_voronois[0]->dimension(); }

protected:
	poca::geometry::VoronoiDiagram* m_voronois[2];
	std::vector <uint32_t> m_indexTrianglePointsInOtherColor[2];

	poca::core::Scatterplot m_scattergram[2], m_scattergramLog[2];

	std::vector <unsigned char> m_classesLocs[2];

	std::array <float, 2> m_spearmans, m_manders;
	std::array<std::vector <float>, 2> m_spearmans2, m_manders2;
};

#endif

