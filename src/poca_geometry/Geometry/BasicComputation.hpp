/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      BasicComputation.hpp
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

#ifndef BasicComputation_h__
#define BasicComputation_h__

#include <General/Vec2.hpp>
#include <General/Vec3.hpp>

namespace poca::geometry {
	enum class WalkDirection { CW, CCW };

	float computeTriangleArea(const double, const double, const double, const double, const double, const double);
	float computePolygonArea(poca::core::Vec3md*, const unsigned int);
	float computePolygonArea2D(poca::core::Vec3md*, const unsigned int);
	float computePolygonArea(poca::core::Vec2md*, const unsigned int);
	float computePolygonArea(double*, size_t);
	float distance(const float, const float, const float, const float, const float, const float);
	float distanceSqr(const float, const float, const float, const float, const float, const float);

	//Computed using the Heron's formula
	//Variables are the three side lengthes of the triangle
	template <typename T>
	static T computeAreaTriangle(const T _a, const T _b, const T _c) {
		T semiPerimeter = (_a + _b + _c) / 2.;
		return (T)sqrt(fabs(semiPerimeter * (semiPerimeter - _a) * (semiPerimeter - _b) * (semiPerimeter - _c)));
	}

	template <class T>
	std::unordered_map<T, std::vector<T>> build_boundary_graph(const std::vector<std::pair<T, T>>& boundary_edges) {
		std::unordered_map<T, std::vector<T>> adjacency;

		for (const auto& e : boundary_edges) {
			adjacency[e.first].push_back(e.second);
			adjacency[e.second].push_back(e.first);
		}

		return adjacency;
	}

	template <class T>
	std::pair<std::vector<T>, std::vector<std::vector<T>>> remove_duplicate_loops(std::vector<T> loop) {
		std::unordered_map<T, T> vertex_to_index;
		std::vector<std::vector<T>> removed_loops;

		T i = 0;
		while (i < loop.size()) {
			auto [it, inserted] = vertex_to_index.emplace(loop[i], i);
			if (!inserted) {
				T first_occurrence = it->second;

				if (i > first_occurrence + 1) {
					std::cout << "Duplicate at vertex " << loop[i] << " positions " << first_occurrence << " and " << i << ", loop size: " << loop.size() << std::endl;
					// Extract the duplicate subloop
					std::vector<T> removed(loop.begin() + first_occurrence + 1, loop.begin() + i + 1);
					std::reverse(removed.begin(), removed.end());
					removed_loops.push_back(removed);

					// Remove the duplicate segment, leaving one occurrence
					loop.erase(loop.begin() + first_occurrence + 1, loop.begin() + i + 1);
				}
				else {
					// Direct duplicate, remove the second occurrence only
					loop.erase(loop.begin() + i);
				}

				// Start over with a fresh map
				vertex_to_index.clear();
				/*for (size_t j = 0; j < loop.size(); ++j) {
					vertex_to_index[loop[j]] = j;
				}*/

				i = 0; // Fully restart the scan after a modification
			}
			else {
				++i;
			}
		}

		return { loop, removed_loops };
	}

	template <class T>
	bool have_same_elements(const std::vector<T>& a, const std::vector<T>& b) {
		if (a.size() != b.size()) return false;

		std::unordered_map<T, T> count;

		for (auto x : a) ++count[x];
		for (auto x : b) {
			if (!count.count(x)) return false;
			if (--count[x] == 0) count.erase(x);
		}

		return count.empty();
	}

	void smoothOutline(std::vector<poca::core::Vec3mf>&, std::vector<poca::core::Vec3mf>&, uint32_t, uint32_t, float, bool = true);

	class BasicComputation {
	public:
		static double distance(const double, const double, const double, const double);
		static double distanceSqr(const double, const double, const double, const double);

		static double getTriangleArea(const poca::core::Vec2md&, const poca::core::Vec2md&, const poca::core::Vec2md&);
		static double getTriangleArea(const double, const double, const double, const double, const double, const double);

		static bool isRecInsideCircle(const double, const double, const double, const double, const double, const double, const double);
		static bool isInsideRec(const double, const double, const double, const double, const double, const double);
		static bool isLineIntersectCircle(const double, const double, const double, const double, const double, const double, const double);
		static void closestPointOnLine(const double, const double, const double, const double, const double, const double, double&, double&);
		static bool circleLineIntersect(const double, const double, const double, const double, const double, const double, const double);

		static void circleLineIntersect(const double, const double, const double, const double, const double, const double, const double, std::vector < poca::core::Vec2md >&);
		static double computeAreaCircularSegment(const double, const double, const double, const poca::core::Vec2md&, const poca::core::Vec2md&);

	};
}
#endif

