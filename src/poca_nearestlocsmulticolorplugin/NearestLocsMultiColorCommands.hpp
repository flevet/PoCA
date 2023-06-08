/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      NearestLocsMultiColorCommands.hpp
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

#ifndef NearestLocsMultiColorCommands_h__
#define NearestLocsMultiColorCommands_h__

#include <General/Command.hpp>
#include <OpenGL/Camera.hpp>

class NearestLocsMultiColorCommands : public poca::core::Command
{
public:
	NearestLocsMultiColorCommands(poca::core::MyObject*);
	NearestLocsMultiColorCommands(const NearestLocsMultiColorCommands&);
	~NearestLocsMultiColorCommands();

	void execute(poca::core::CommandInfo*);
	poca::core::Command* copy();
	const poca::core::CommandInfos saveParameters() const {
		return poca::core::CommandInfos();
	}
	poca::core::CommandInfo createCommand(const std::string&, const nlohmann::json&);
	//void saveCommands(nlohmann::json&);
	void freeGPUMemory();

	const std::vector <uint32_t>& getIndexObjects() const { return m_idxObjects; }
	const std::vector <float>& getDistanceToCentroids() const { return m_distancesToCentroids; }
	const std::vector <float>& getDistanceToOutlines() const { return m_distancesToOutlines; }

	virtual poca::core::HistogramInterface* getHistogramCentroids() { return m_histogramCentroids; }
	virtual poca::core::HistogramInterface* getHistogramOutlines() { return m_histogramOutlines; }
	virtual poca::core::PaletteInterface* getPalette() { return m_palette; }

protected:
	void display(poca::opengl::Camera*, const bool);
	void drawElements(poca::opengl::Camera*);
	void createDisplay(const std::vector <poca::core::Vec3mf>&, const std::vector <poca::core::Vec3mf>&);

	void computeNearestLocMulticolor(const bool, const uint32_t, const float);
	void transferObjects() const;
	void saveDistances(const std::string&) const;

protected:
	poca::core::MyObject* m_obj;
	uint32_t m_referenceId;
	bool m_displayToCentroids, m_displayToOutlines;
	std::vector <bool> m_selectedObjects;
	std::vector <uint32_t> m_idxObjects;
	std::vector <float> m_distancesToCentroids, m_distancesToOutlines;
	poca::core::HistogramInterface* m_histogramCentroids, * m_histogramOutlines;
	poca::core::PaletteInterface* m_palette;

	poca::opengl::LineSingleGLBuffer <poca::core::Vec3mf> m_toCentroidsBuffer, m_toOutlinesBuffer;
};

#endif

