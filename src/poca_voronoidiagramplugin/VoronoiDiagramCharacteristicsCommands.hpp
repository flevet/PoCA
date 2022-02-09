/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      VoronoiDiagramCharacteristicsCommands.hpp
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

#ifndef VoronoiDiagramCharacteristicsCommands_h__
#define VoronoiDiagramCharacteristicsCommands_h__

#include <General/Command.hpp>
#include <Geometry/VoronoiDiagram.hpp>
#include <General/Histogram.hpp>

class VoronoiDiagramCharacteristicsCommands : public poca::core::Command
{
public:
	VoronoiDiagramCharacteristicsCommands(poca::geometry::VoronoiDiagram*);
	VoronoiDiagramCharacteristicsCommands(const VoronoiDiagramCharacteristicsCommands&);
	~VoronoiDiagramCharacteristicsCommands();

	void execute(poca::core::CommandInfo*);
	poca::core::Command* copy();
	const poca::core::CommandInfos saveParameters() const {
		return poca::core::CommandInfos();
	}
	poca::core::CommandInfo createCommand(const std::string&, const nlohmann::json&);

	const std::vector <float>& getExperimentalCurvePDF(float&);
	const std::vector <float>& getAnalyticCurvePDF(float&);
	const std::vector <float>& getExperimentalCurveCDF();
	const std::vector <float>& getAnalyticCurveCDF();

protected:
	void computeCharacteristics(const float, const bool, const uint32_t, const uint32_t);
	void computeAnalyticCurve();
	void computeAnalyticCurve2D();
	void computeAnalyticCurve3D();
	const float computeThreshold();

protected:
	poca::geometry::VoronoiDiagram* m_voronoi;

	bool m_is3D;
	unsigned int m_nbFaces;
	poca::core::BoundingBoxD m_bbox;

	unsigned int m_nbInitialLocs, m_nbBins, m_nbIterationsMonteCarlo, m_nbDegreePolynome;
	double m_maxHExp, m_maxHAna, m_maxHPolynome, m_r2, m_env;
	float m_normalization;
	double* m_monteCarloBinsMeans, * m_monteCarloBinsStdDev;

	poca::core::Histogram m_hist;
	std::vector <float> m_xs, m_ysPdfExp, m_ysCdfExp, m_ysPdfAna, m_ysCdfAna;
	std::vector <float> m_coeffsPolynome, m_ysPdfPolynome, m_ysCdfPolynome;

	float m_threshold;
};

#endif

