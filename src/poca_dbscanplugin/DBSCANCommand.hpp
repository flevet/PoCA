/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DBSCANCommand.hpp
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

#include <General/Command.hpp>
#include <General/Histogram.hpp>
#include <OpenGL/GLBuffer.hpp>
#include <Geometry/dbscan.h>

namespace poca::geometry {
	class DetectionSet;
}

namespace poca::opengl {
	class Camera;
}

class DBSCANCommand: public poca::core::Command {
public:
	DBSCANCommand(poca::geometry::DetectionSet*);
	DBSCANCommand(const DBSCANCommand&);
	~DBSCANCommand();

	void execute(poca::core::CommandInfo*);

	poca::core::Command* copy();
	const poca::core::CommandInfos saveParameters() const { return poca::core::CommandInfos(); }
	poca::core::CommandInfo createCommand(const std::string&, const nlohmann::json&);

	inline const std::vector <float>& getSizeClusters() const { return m_sizeClusters; }
	inline const std::vector <float>& getMajorAxisClusters() const { return m_majorAxisClusters; }
	inline const std::vector <float>& getMinorAxisClusters() const { return m_minorAxisClusters; }
	inline const std::vector <uint32_t>& getNbLocsClusters() const { return m_nbLocsClusters; }
	inline const size_t nbClusters() const { return m_dbscan.Clusters.size(); }
	inline poca::core::HistogramInterface* getHistogram() { return m_histSizes; }

protected:
	void computeDBSCAN(const float, const uint32_t, const uint32_t);
	void display(poca::opengl::Camera*, const bool);
	void createDisplay();
	void updateColorBuffer();

protected:
	poca::geometry::DetectionSet* m_dset;

	DBSCAN <dbvec3f, float> m_dbscan;
	std::vector <float> m_sizeClusters, m_majorAxisClusters, m_minorAxisClusters;
	std::vector <uint32_t> m_nbLocsClusters;
	poca::core::Histogram* m_histSizes;

	poca::opengl::PointGLBuffer <poca::core::Vec3mf> m_pointBuffer;
	poca::opengl::PointGLBuffer <poca::core::Color4D> m_colorBuffer;
	bool m_updateColorBuffer;
};

#endif

