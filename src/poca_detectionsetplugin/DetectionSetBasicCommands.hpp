/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DetectionSetBasicCommands.hpp
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

#ifndef VoronoiDiagramBasicCommands_h__
#define VoronoiDiagramBasicCommands_h__

#include <General/Command.hpp>

namespace poca::geometry {
	class DetectionSet;
}

class DetectionSetBasicCommands : public poca::core::Command
{
public:
	DetectionSetBasicCommands(poca::geometry::DetectionSet*);
	DetectionSetBasicCommands(const DetectionSetBasicCommands&);
	~DetectionSetBasicCommands();

	void execute(poca::core::CommandInfo*);
	poca::core::Command* copy();
	const poca::core::CommandInfos saveParameters() const {
		return poca::core::CommandInfos();
	}
	poca::core::CommandInfo createCommand(const std::string&, const nlohmann::json&);

protected:
	void saveAsSVG(const QString&) const;
	void rotateLocsXY(const float);
	void rescaleFromGNN(const QString&);
	poca::geometry::DetectionSet* keepPercentageOfLocalizations(const float) const;

protected:
	poca::geometry::DetectionSet* m_dset;
};

#endif

