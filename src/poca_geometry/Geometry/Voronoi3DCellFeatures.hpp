/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Voronoi3DCellFeatures.hpp
*
* Copyright: Florian Levet (2020-2026)
*
* License:   LGPL v3
*/

#ifndef Voronoi3DCellFeatures_hpp__
#define Voronoi3DCellFeatures_hpp__

#include <map>
#include <string>
#include <vector>

#include <General/Vec3.hpp>
#include <General/MyArray.hpp>
#include <General/Vec6.hpp>

#include "CGAL_includes.hpp"

namespace poca::geometry {
	class Voronoi3DCellFeatures {
	public:
		struct FeatureSet {
			std::map<std::string, std::vector<float>> values;
		};

		static FeatureSet compute(
			const std::vector<Polyhedron_3_inexact>&,
			const std::vector<float>&,
			const float*, const float*, const float*,
			const poca::core::MyArrayUInt32&,
			const std::vector<bool>&);

		static FeatureSet compute(
			const std::vector<Surface_mesh_3_double>&,
			const std::vector<poca::core::Vec3mf>&,
			const poca::core::MyArrayUInt32&,
			const std::vector<bool>&,
			const poca::core::BoundingBox* = nullptr);

		static FeatureSet compute(
			const std::vector<Surface_mesh_3_double>&,
			const std::vector<poca::core::Vec3mf>&,
			const poca::core::MyArrayUInt32&,
			const std::vector<uint32_t>&,
			const poca::core::BoundingBox* = nullptr);

	private:
		Voronoi3DCellFeatures() = delete;
	};
}

#endif
