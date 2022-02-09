/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DetectionSet.hpp
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

#ifndef DetectionSet_h__
#define DetectionSet_h__

#include <vector>
#include <array>
#include <map>

#include <General/BasicComponent.hpp>
#include <General/Vec3.hpp>

#include "nanoflann.hpp"

namespace poca::core {
	template <typename T>
	struct Vec3Cloud
	{
		std::vector<Vec3<T>>  m_pts;

		Vec3<T>* data() { return m_pts.data(); }
		const Vec3<T>* data() const { return m_pts.data(); }
		void resize(const size_t _nb) { m_pts.resize(_nb); }
		void copy(const std::vector<Vec3<T>>& _o) { std::copy(_o.begin(), _o.end(), std::back_inserter(m_pts)); }

		// Must return the number of data points
		inline size_t kdtree_get_point_count() const { return m_pts.size(); }

		// Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
		inline T kdtree_distance(const T* p1, const size_t idx_p2, size_t /*size*/) const
		{
			const T d0 = p1[0] - m_pts[idx_p2][0];
			const T d1 = p1[1] - m_pts[idx_p2][1];
			const T d2 = p1[2] - m_pts[idx_p2][2];
			return d0 * d0 + d1 * d1 + d2 * d2;
		}

		// Returns the dim'th component of the idx'th point in the class:
		// Since this is inlined and the "dim" argument is typically an immediate value, the
		//  "if/else's" are actually solved at compile time.
		inline T kdtree_get_pt(const size_t idx, int dim) const
		{
			if (dim == 0) return m_pts[idx][0];
			else if (dim == 1) return m_pts[idx][1];
			else return m_pts[idx][2];
		}

		// Optional bounding-box computation: return false to default to a standard bbox computation loop.
		//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
		//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
		template <class BBOX>
		bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }
	};

	typedef Vec3Cloud < double >  DetectionPointCloud;
}

namespace poca::geometry {

	typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, poca::core::DetectionPointCloud >, poca::core::DetectionPointCloud, 3 /* dim */> KdTree_DetectionPoint;

	class DetectionSet : public poca::core::BasicComponent {
	public:
		DetectionSet();
		DetectionSet(const DetectionSet&);
		DetectionSet(const std::vector < DetectionSet* >&);
		DetectionSet(const std::map <std::string, std::vector <float>>&);
		~DetectionSet();

		inline const size_t nbSlices() const { return m_nbSlices; }
		inline const size_t nbPoints() const { return m_nbPoints; }
		const unsigned int memorySize() const;

		BasicComponent* copy();
		void setData(const std::map <std::string, std::vector <float>>&);

		size_t dimension() const { return hasData("z") ? 3 : 2; }

		const float averageDensity() const;
		DetectionSet* duplicateSelection() const;

		void saveDetections(std::ofstream&);

		inline KdTree_DetectionPoint* getKdTree() { return m_kdTree; }

	protected:
		size_t m_nbPoints, m_nbSlices;
		poca::core::DetectionPointCloud m_pointCloud;
		KdTree_DetectionPoint* m_kdTree;
	};
}

#endif // DetectionSet_h__

