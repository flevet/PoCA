/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MultiPrimitiveData.hpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*/

#ifndef MultiPrimitiveData_h__
#define MultiPrimitiveData_h__

#include <cstdint>
#include <vector>

#include <General/Vec4.hpp>

namespace poca::opengl {

	struct PickMappingEntry {
		uint32_t objectIndex = 0;
		uint32_t localIndex = 0;
	};

	template<typename VertexT>
	struct MultiPointData {
		std::vector<VertexT> vertices;
		std::vector<float> ids;
		std::vector<float> features;
		std::vector<PickMappingEntry> pickMap;

		inline void clear() {
			vertices.clear();
			ids.clear();
			features.clear();
			pickMap.clear();
		}
	};

	template<typename VertexT>
	struct MultiLineData {
		std::vector<VertexT> vertices;
		std::vector<float> features;
		std::vector<VertexT> normals;
		std::vector<poca::core::Color4D> colors;

		inline void clear() {
			vertices.clear();
			features.clear();
			normals.clear();
			colors.clear();
		}
	};

	template<typename VertexT>
	struct MultiTriangleData {
		std::vector<VertexT> vertices;
		std::vector<float> ids;
		std::vector<float> features;
		std::vector<VertexT> normals;
		std::vector<poca::opengl::PickMappingEntry> pickMap;

		inline void clear() {
			vertices.clear();
			ids.clear();
			features.clear();
			normals.clear();
			pickMap.clear();
		}
	};
}

#endif
