/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      CleanerCommand.hpp
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

#ifndef CleanerCommand_h__
#define CleanerCommand_h__

#include <General/Command.hpp>
#include <Geometry/DetectionSet.hpp>
#include <OpenGL/GLBuffer.hpp>

namespace poca::opengl {
	class Camera;
}

namespace poca::core {
	class EquationFit;
}

class CleanerCommand : public poca::core::Command {
public:
	CleanerCommand(poca::geometry::DetectionSet*);
	CleanerCommand(const CleanerCommand&);
	~CleanerCommand();

	void execute(poca::core::CommandInfo*);
	void freeGPUMemory();

	poca::core::Command* copy();
	const poca::core::CommandInfos saveParameters() const { return poca::core::CommandInfos(); }
	poca::core::CommandInfo createCommand(const std::string&, const nlohmann::json&);

	inline const uint32_t getNbEmissionBursts() const { return m_nbEmissionBursts; }
	inline const uint32_t getNbUncorrectedLocs() const { return m_nbUncorrectedLocs; }
	inline const uint32_t getNbCorrectedLocs() const { return m_nbCorrectedLocs; }
	inline const uint32_t getNbOriginalLocs() const { return m_nbOriginalLocs; }
	inline const uint32_t getNbSuppressedLocs() const { return m_nbSuppressedLocs; }

protected:
	void display(poca::opengl::Camera*, const bool);
	void createDisplay();
	poca::core::MyObjectInterface* cleanDetectionSet(const float, const uint32_t, const bool);

	size_t computeNbLocsCorrectedSpecifiedDT(const uint32_t, const float);
	uint32_t computeAnalysisParameters(const uint32_t, const float);

protected:
	poca::geometry::DetectionSet* m_dset;

	std::vector <uint32_t> m_pointsPerFrame, m_mergedPoints, m_firstsMerged, m_nbBlinks, m_nbSequencesOff, m_totalOffs;
	poca::core::EquationFit* m_eqnBlinks, * m_eqnTOns, * m_eqnTOffs;
	std::map <std::string, std::vector <float>> m_featuresCorrectedLocs;
	uint32_t m_nbEmissionBursts, m_nbUncorrectedLocs, m_nbCorrectedLocs, m_nbOriginalLocs, m_nbSuppressedLocs, m_darkTime;

	bool m_initializeDisplay;
	poca::opengl::PointGLBuffer <poca::core::Vec3mf> m_pointBuffer;
	poca::opengl::LineGLBuffer <poca::core::Vec3mf> m_lineBuffer;
};

#endif

