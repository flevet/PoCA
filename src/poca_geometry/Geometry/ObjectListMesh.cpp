/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectListMesh.cpp
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

#include <algorithm>
#include <execution>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/poisson_surface_reconstruction.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <CGAL/mesh_segmentation.h>
#include <CGAL/squared_distance_2.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/clip.h>
#include <CGAL/enum.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>

#include <General/MyData.hpp>
#include <General/BasicComponent.hpp>
#include <Interfaces/HistogramInterface.hpp>
#include <General/Misc.h>

#include "ObjectListMesh.hpp"
#include "BasicComputation.hpp"
#include "DelaunayTriangulation.hpp"
#include "../Interfaces/ObjectFeaturesFactoryInterface.hpp"

namespace PMP = CGAL::Polygon_mesh_processing;
typedef CGAL::Mean_curvature_flow_skeletonization<Surface_mesh_3_double> Skeletonization;
typedef Skeletonization::Skeleton Skeleton;
typedef Skeleton::vertex_descriptor Skeleton_vertex;
typedef Skeleton::edge_descriptor Skeleton_edge;
typedef CGAL::Face_filtered_graph<Surface_mesh_3_double> Filtered_graph;

typedef Surface_mesh_3_double::Property_map<vertex_descriptor, Kernel::Vector_3> Vertex_vector_3_map;
typedef Surface_mesh_3_double::Property_map<face_descriptor, Kernel::Vector_3> Facet_vector_3_map;

namespace poca::geometry {
	ObjectListMesh::ObjectListMesh(std::vector <std::vector <poca::core::Vec3mf>>& _allVertices, std::vector < std::vector <std::vector <std::size_t>>>& _allTriangles, const bool _applyRemeshing, const double _target, const uint32_t _it)
		:ObjectListInterface("ObjectListMesh"), m_applyRemeshing(_applyRemeshing), m_targetLength(_target), m_iterations(_it)
	{
		std::vector <poca::core::Vec3mf> triPoCA, edges, links;
		std::vector <uint32_t> nbTriPoCA = { 0 }, nbEdges = { 0 }, nbLinks = { 0 };
		std::vector <float> volumes;

		clock_t t1 = clock(), t2;
		std::cout << std::string(10, '-');
		for (auto n = 0; n < _allVertices.size(); n++) {
			std::vector < Point_3_double> points(_allVertices[n].size());
			std::transform(std::execution::par, _allVertices[n].begin(), _allVertices[n].end(), points.begin(), [](const auto& value) {return Point_3_double(value[0], value[1], value[2]);});
			int percent = floor((float)n / (float)_allVertices.size() * 10.f);
			std::cout << "\r" << std::string(percent, '*') << std::string(10 - percent, '-') << " ; generating CGAL mesh number " << (n + 1) << " composed of " << points.size()  << " vertices";
			addObjectMesh(points, _allTriangles[n], triPoCA, nbTriPoCA, edges, nbEdges, links, nbLinks, volumes);
		}
		t2 = clock();
		long elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC;
		std::cout << "\r" << std::string(10, '*') << " ; time elapsed for creating all CGAL meshes ->  " << elapsed << " seconds.                                       " << std::endl;
		m_edgesSkeleton.initialize(edges, nbEdges);
		m_linksSkeleton.initialize(links, nbLinks);
		m_triangles.initialize(triPoCA, nbTriPoCA);

		m_centroids.resize(m_meshes.size());

		m_bbox.set(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min());
		std::vector <uint32_t> points, nbPts{ 0 }; //_mesh.number_of_vertices()
		for (const auto& mesh : m_meshes) {
			for (const auto& point : mesh.points()) {
				auto x = point.x(), y = point.y(), z = point.z();
				m_xs.push_back(x);
				m_ys.push_back(y);
				m_zs.push_back(z);
				if (x < m_bbox[0]) m_bbox[0] = x;
				if (y < m_bbox[1]) m_bbox[1] = y;
				if (z < m_bbox[2]) m_bbox[2] = z;
				if (x > m_bbox[3]) m_bbox[3] = x;
				if (y > m_bbox[4]) m_bbox[4] = y;
				if (z > m_bbox[5]) m_bbox[5] = z;
			}
			nbPts.push_back(m_xs.size());
		}
		points.resize(nbPts.back());
		std::iota(std::begin(points), std::end(points), 0);
		m_locs.initialize(points, nbPts);
		m_outlineLocs = m_locs;

		//Create area feature
		std::vector <float> nbLocs(m_locs.nbElements(), 0.f);
		for (size_t i = 0; i < m_triangles.nbElements(); i++)
			nbLocs[i] = m_locs.nbElementsObject(i);
		const poca::core::MyArrayUInt32& localizations = m_locs;// m_outlineLocs;
		ObjectFeaturesFactoryInterface* factory = createObjectFeaturesFactory();
		std::vector <float> sizes(m_locs.nbElements()), resPCA(factory->nbFeaturesPCA(true));
		std::vector <float> major(m_locs.nbElements()), minor(m_locs.nbElements()), minor2(m_locs.nbElements()), minMin2(m_locs.nbElements());
		m_axis.resize(m_locs.nbElements());
		for (size_t n = 0; n < m_locs.nbElements(); n++) {
			float* ptr = &resPCA[0];
			factory->computePCA(m_locs, n, m_xs.data(), m_ys.data(), m_zs.data(), ptr);
			m_centroids[n].set(resPCA[0], resPCA[1], resPCA[2]);
			major[n] = resPCA[3];
			minor[n] = resPCA[4];
			minor2[n] = resPCA[5];
			minMin2[n] = minor[n] + minor2[n];
			sizes[n] = (resPCA[3] + resPCA[4] + resPCA[5]) / 3.f;
			m_axis[n] = { poca::core::Vec3mf(resPCA[6], resPCA[7], resPCA[8]),
				poca::core::Vec3mf(resPCA[9], resPCA[10], resPCA[11]) ,
				poca::core::Vec3mf(resPCA[12], resPCA[13], resPCA[14]) };
		}
		delete factory;

		std::vector <float> ids(m_locs.nbElements());
		std::iota(std::begin(ids), std::end(ids), 1);

		m_data["volume"] = poca::core::generateDataWithLog(volumes);
		m_data["nbLocs"] = poca::core::generateDataWithLog(nbLocs);
		m_data["size"] = poca::core::generateDataWithLog(sizes);
		m_data["id"] = poca::core::generateDataWithLog(ids);
		m_data["major"] = poca::core::generateDataWithLog(major);
		m_data["minor"] = poca::core::generateDataWithLog(minor);
		m_data["minor2"] = poca::core::generateDataWithLog(minor2);
		m_data["minMin2"] = poca::core::generateDataWithLog(minMin2);

		m_selection.resize(volumes.size());
		setCurrentHistogramType("volume");
		forceRegenerateSelection();
	}

	ObjectListMesh::ObjectListMesh(std::vector <std::vector <Point_3_double>>& _allVertices, std::vector < std::vector <std::vector <std::size_t>>>& _allTriangles, const bool _applyRemeshing, const double _target, const uint32_t _it)
		:ObjectListInterface("ObjectListMesh"), m_applyRemeshing(_applyRemeshing), m_targetLength(_target), m_iterations(_it)
	{
		std::vector <poca::core::Vec3mf> triPoCA, edges, links;
		std::vector <uint32_t> nbTriPoCA = { 0 }, nbEdges = { 0 }, nbLinks = { 0 };
		std::vector <float> volumes;

		clock_t t1 = clock(), t2;
		std::cout << std::string(10, '-');
		for (auto n = 0; n < _allVertices.size(); n++) {
			addObjectMesh(_allVertices[n], _allTriangles[n], triPoCA, nbTriPoCA, edges, nbEdges, links, nbLinks, volumes);
			int percent = floor((float)n / (float)_allVertices.size() * 10.f);
			std::cout << "\r" << std::string(percent, '*') << std::string(10 - percent, '-') << " ; generating CGAL mesh number " << (n + 1) << " composed of " << _allVertices[n].size() << " vertices";
		}
		t2 = clock();
		long elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC;
		m_edgesSkeleton.initialize(edges, nbEdges);
		m_linksSkeleton.initialize(links, nbLinks);
		m_triangles.initialize(triPoCA, nbTriPoCA);

		m_centroids.resize(m_meshes.size());

		m_bbox.set(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min());
		std::vector <uint32_t> points, nbPts{ 0 }; //_mesh.number_of_vertices()
		for (const auto& mesh : m_meshes) {
			for (const auto& point : mesh.points()) {
				auto x = point.x(), y = point.y(), z = point.z();
				m_xs.push_back(x);
				m_ys.push_back(y);
				m_zs.push_back(z);
				if (x < m_bbox[0]) m_bbox[0] = x;
				if (y < m_bbox[1]) m_bbox[1] = y;
				if (z < m_bbox[2]) m_bbox[2] = z;
				if (x > m_bbox[3]) m_bbox[3] = x;
				if (y > m_bbox[4]) m_bbox[4] = y;
				if (z > m_bbox[5]) m_bbox[5] = z;
			}
			nbPts.push_back(m_xs.size());
		}
		points.resize(nbPts.back());
		std::iota(std::begin(points), std::end(points), 0);
		m_locs.initialize(points, nbPts);
		m_outlineLocs = m_locs;

		//Create area feature
		std::vector <float> nbLocs(m_locs.nbElements(), 0.f);
		for (size_t i = 0; i < m_triangles.nbElements(); i++)
			nbLocs[i] = m_locs.nbElementsObject(i);
		const poca::core::MyArrayUInt32& localizations = m_locs;// m_outlineLocs;
		ObjectFeaturesFactoryInterface* factory = createObjectFeaturesFactory();
		std::vector <float> sizes(m_locs.nbElements()), resPCA(factory->nbFeaturesPCA(true));
		std::vector <float> major(m_locs.nbElements()), minor(m_locs.nbElements()), minor2(m_locs.nbElements()), minMin2(m_locs.nbElements());
		m_axis.resize(m_locs.nbElements());
		for (size_t n = 0; n < m_locs.nbElements(); n++) {
			float* ptr = &resPCA[0];
			factory->computePCA(m_locs, n, m_xs.data(), m_ys.data(), m_zs.data(), ptr);
			m_centroids[n].set(resPCA[0], resPCA[1], resPCA[2]);
			major[n] = resPCA[3];
			minor[n] = resPCA[4];
			minor2[n] = resPCA[5];
			minMin2[n] = minor[n] + minor2[n];
			sizes[n] = (resPCA[3] + resPCA[4] + resPCA[5]) / 3.f;
			m_axis[n] = { poca::core::Vec3mf(resPCA[6], resPCA[7], resPCA[8]),
				poca::core::Vec3mf(resPCA[9], resPCA[10], resPCA[11]) ,
				poca::core::Vec3mf(resPCA[12], resPCA[13], resPCA[14]) };
		}
		delete factory;

		std::vector <float> ids(m_locs.nbElements());
		std::iota(std::begin(ids), std::end(ids), 1);

		m_data["volume"] = poca::core::generateDataWithLog(volumes);
		m_data["nbLocs"] = poca::core::generateDataWithLog(nbLocs);
		m_data["size"] = poca::core::generateDataWithLog(sizes);
		m_data["id"] = poca::core::generateDataWithLog(ids);
		m_data["major"] = poca::core::generateDataWithLog(major);
		m_data["minor"] = poca::core::generateDataWithLog(minor);
		m_data["minor2"] = poca::core::generateDataWithLog(minor2);
		m_data["minMin2"] = poca::core::generateDataWithLog(minMin2);

		m_selection.resize(volumes.size());
		setCurrentHistogramType("volume");
		forceRegenerateSelection();
	}

	ObjectListMesh::~ObjectListMesh()
	{
	}

	poca::core::BasicComponentInterface* ObjectListMesh::copy()
	{
		return new ObjectListMesh(*this);
	}

	const bool ObjectListMesh::addObjectMesh(std::vector <Point_3_double>& _vertices, std::vector<std::vector<std::size_t> >& _triangles,
		std::vector <poca::core::Vec3mf>& _trianglesPoCA, std::vector <std::uint32_t>& _nbTriPoCA,
		std::vector <poca::core::Vec3mf>& _edgesSkeleton, std::vector <std::uint32_t>& _nbSkeletons,
		std::vector <poca::core::Vec3mf>& _linksSkeleton, std::vector <std::uint32_t>& _nbLinks,
		std::vector <float>& _volumes)
	{
		if (_vertices.size() < 5)
			return false;
		PMP::repair_polygon_soup(_vertices, _triangles, CGAL::parameters::erase_all_duplicates(true).require_same_orientation(true));
		if (!PMP::orient_polygon_soup(_vertices, _triangles))
		{
			std::cerr << "Some duplication happened during polygon soup orientation" << std::endl;
		}

		if (!PMP::is_polygon_soup_a_polygon_mesh(_triangles))
		{
			std::cerr << "Warning: polygon soup does not describe a polygon mesh" << std::endl;
			return false;
		}

		m_meshes.push_back(Surface_mesh_3_double());
		Surface_mesh_3_double& mesh = m_meshes.back();
		PMP::polygon_soup_to_polygon_mesh(_vertices, _triangles, mesh);

		PMP::keep_large_connected_components(mesh, 10);

#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(6, 0, 0)
		PMP::remove_almost_degenerate_faces(mesh);
#endif

		/*double target_edge_length = 4;
		unsigned int nb_iter = 1;
		std::cout << "Start smoothing (" << num_faces(mesh) << " faces)..." << std::endl;
		PMP::isotropic_remeshing(faces(mesh), target_edge_length, mesh, CGAL::parameters::number_of_iterations(nb_iter));
		std::cout << "End remeshing (" << num_faces(mesh) << " faces)..." << std::endl;
		//CGAL::Polygon_mesh_processing::reverse_face_orientations(mesh);*/


		bool res = processSurfaceMesh(mesh, _trianglesPoCA, _nbTriPoCA, _edgesSkeleton, _nbSkeletons, _linksSkeleton, _nbLinks, _volumes);
		if (!res)
			m_meshes.pop_back();

		return res;
	}

	const bool ObjectListMesh::processSurfaceMesh(Surface_mesh_3_double& _mesh,
		std::vector <poca::core::Vec3mf>& _trianglesPoCA, std::vector <std::uint32_t>& _nbTriPoCA,
		std::vector <poca::core::Vec3mf>& _edgesSkeleton, std::vector <std::uint32_t>& _nbSkeletons,
		std::vector <poca::core::Vec3mf>& _linksSkeleton, std::vector <std::uint32_t>& _nbLinks,
		std::vector <float>& _volumes)
	{
		if (!CGAL::is_triangle_mesh(_mesh)) {
			std::cout << "Not a triangle mesh" << std::endl;
			return false;
		}

		if (!CGAL::Polygon_mesh_processing::is_outward_oriented(_mesh))
			CGAL::Polygon_mesh_processing::reverse_face_orientations(_mesh);

		if (m_applyRemeshing) {
			double target_edge_length = 1.;
			unsigned int nb_iter = 4;
			//std::cout << "Start smoothing (" << num_faces(mesh) << " faces)..." << std::endl;
			PMP::isotropic_remeshing(faces(_mesh), m_targetLength, _mesh, CGAL::parameters::number_of_iterations(m_iterations));
			//std::cout << "End remeshing (" << num_faces(mesh) << " faces)..." << std::endl;
		}
		float volume = PMP::volume(_mesh);
		if (volume < 0.f) {
			return false;
		}

		//std::cout << "Orientation mesh - " << CGAL::Polygon_mesh_processing::is_outward_oriented(_mesh) << std::endl;

		//Create Mesh for rendering
		for (Surface_mesh_3_double::Face_index fd : _mesh.faces()) {
			int j = 0;
			CGAL::Vertex_around_face_iterator<Surface_mesh_3_double> vbegin, vend;
			for (boost::tie(vbegin, vend) = vertices_around_face(_mesh.halfedge(fd), _mesh);
				vbegin != vend;
				vbegin++) {
				j++;
				auto p = _mesh.point(*vbegin);
				_trianglesPoCA.push_back(poca::core::Vec3mf(p.x(), p.y(), p.z()));
			}
		}
		_nbTriPoCA.push_back(_trianglesPoCA.size());
		_volumes.push_back(volume);

		Kernel::Iso_cuboid_3 bbox = CGAL::bounding_box(_mesh.points().begin(), _mesh.points().end());
		m_bboxMeshes.push_back(poca::core::BoundingBox(bbox.xmin(), bbox.ymin(), bbox.zmin(), bbox.xmax(), bbox.ymax(), bbox.zmax()));

		Facet_vector_3_map facetNormals = _mesh.add_property_map<face_descriptor, Kernel::Vector_3>("f:norm").first;
		PMP::compute_face_normals(_mesh, facetNormals);

		//std::cout << "Done creating normal per vertex" << std::endl;

		Vertex_vector_3_map vertexNormals = _mesh.add_property_map<vertex_descriptor, Kernel::Vector_3>("v:norm").first;
		PMP::compute_vertex_normals(_mesh, vertexNormals);

		return true;
	}

	void ObjectListMesh::generateLocs(std::vector <poca::core::Vec3mf>& _locs)
	{
		/*_locs.clear();
		for (const auto& mesh : m_meshes)
			for (const auto& point : mesh.points())
				_locs.push_back(poca::core::Vec3mf(point.x(), point.y(), point.z()));*/
		_locs.resize(m_locs.nbData());
		const std::vector <uint32_t>& indices = m_locs.getData();
		for (size_t n = 0; n < indices.size(); n++) {
			size_t index = indices.at(n);
			_locs[n].set(m_xs[index], m_ys[index], m_zs[index]);
		}
	}

	void ObjectListMesh::generateNormalLocs(std::vector <poca::core::Vec3mf>& _norms)
	{
		_norms.clear();
		for (const auto& mesh : m_meshes) {
#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(6, 0, 0)
			Vertex_vector_3_map normal_map = mesh.property_map<vertex_descriptor, Kernel::Vector_3>("v:norm").value();
#else
			Vertex_vector_3_map normal_map = mesh.property_map<vertex_descriptor, Kernel::Vector_3>("v:norm").first;
#endif
			Surface_mesh_3_double::Vertex_range r = mesh.vertices();
			Surface_mesh_3_double::Vertex_range::iterator  vb = r.begin(), ve = r.end();
			for (boost::tie(vb, ve) = mesh.vertices(); vb != ve; ++vb) {
				// Print vertex index and vertex coordinates
				const auto& normal = normal_map[*vb];
				_norms.push_back(poca::core::Vec3mf(normal.x(), normal.y(), normal.z()));
			}
		}
	}

	void ObjectListMesh::getLocsFeatureInSelection(std::vector <float>& _features, const std::vector <float>& _values, const std::vector <bool>& _selection, const float _notSelectedValue) const
	{
		/*_features.clear();
		int i = 0;
		for (const auto& mesh : m_meshes) {
			for (const auto& point : mesh.points())
				_features.push_back(_selection[i] ? _values[i] : _notSelectedValue);
			i++;
		}*/
		_features.resize(m_locs.nbData());

		size_t cpt = 0;
		for (size_t i = 0; i < m_locs.nbElements(); i++) {
			for (size_t j = 0; j < m_locs.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _values[i] : _notSelectedValue;
			}

		}
	}

	void ObjectListMesh::getLocsFeatureInSelectionHiLow(std::vector <float>& _features, const std::vector <bool>& _selection, const float _selectedValue, const float _notSelectedValue) const
	{
		/*_features.clear();
		int i = 0;
		for (const auto& mesh : m_meshes) {
			for (const auto& point : mesh.points())
				_features.push_back(_selection[i] ? _selectedValue : _notSelectedValue);
			i++;
		}*/
		_features.resize(m_locs.nbData());

		size_t cpt = 0;
		for (size_t i = 0; i < m_locs.nbElements(); i++) {
			for (size_t j = 0; j < m_locs.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _selectedValue : _notSelectedValue;
			}

		}
	}

	void ObjectListMesh::getOutlinesFeatureInSelection(std::vector <float>& _features, const std::vector <float>& _values, const std::vector <bool>& _selection, const float _notSelectedValue) const
	{
		/*_features.clear();
		int i = 0;
		for (const auto& mesh : m_meshes) {
			for (const auto& point : mesh.points())
				_features.push_back(_selection[i] ? _values[i] : _notSelectedValue);
			i++;
		}*/
		_features.resize(m_locs.nbData());

		size_t cpt = 0;
		for (size_t i = 0; i < m_locs.nbElements(); i++) {
			for (size_t j = 0; j < m_locs.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _values[i] : _notSelectedValue;
			}

		}
	}

	void ObjectListMesh::getOutlinesFeatureInSelectionHiLow(std::vector <float>& _features, const std::vector <bool>& _selection, const float _selectedValue, const float _notSelectedValue) const
	{
		/*_features.clear();
		int i = 0;
		for (const auto& mesh : m_meshes) {
			for (const auto& point : mesh.points())
				_features.push_back(_selection[i] ? _selectedValue : _notSelectedValue);
			i++;
		}*/
		_features.resize(m_locs.nbData());

		size_t cpt = 0;
		for (size_t i = 0; i < m_locs.nbElements(); i++) {
			for (size_t j = 0; j < m_locs.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _selectedValue : _notSelectedValue;
			}

		}
	}

	void ObjectListMesh::generateLocsPickingIndices(std::vector <float>& _ids) const
	{
		/*_ids.clear();
		int i = 0;
		for (const auto& mesh : m_meshes) {
			for (const auto& point : mesh.points())
				_ids.push_back(i + 1);
			i++;
		}*/
		_ids.resize(m_locs.nbData());

		size_t cpt = 0;
		for (size_t i = 0; i < m_locs.nbElements(); i++) {
			for (size_t j = 0; j < m_locs.nbElementsObject(i); j++) {
				_ids[cpt++] = i + 1;
			}

		}
	}

	void ObjectListMesh::generateTriangles(std::vector <poca::core::Vec3mf>& _triangles)
	{
		std::copy(m_triangles.getData().begin(), m_triangles.getData().end(), std::back_inserter(_triangles));
	}

	void ObjectListMesh::generateOutlines(std::vector <poca::core::Vec3mf>& _outlines)
	{
		std::copy(m_outlines.getData().begin(), m_outlines.getData().end(), std::back_inserter(_outlines));
	}

	void ObjectListMesh::generateNormals(std::vector <poca::core::Vec3mf>& _normals)
	{
		_normals.clear();
		for (const auto& mesh : m_meshes) {
#if CGAL_VERSION_NR >= CGAL_VERSION_NUMBER(6, 0, 0)
			Vertex_vector_3_map normal_map = mesh.property_map<vertex_descriptor, Kernel::Vector_3>("v:norm").value();
#else
			Vertex_vector_3_map normal_map = mesh.property_map<vertex_descriptor, Kernel::Vector_3>("v:norm").first;
#endif
			for (Surface_mesh_3_double::Face_index fd : mesh.faces()) {
				CGAL::Vertex_around_face_iterator<Surface_mesh_3_double> vbegin, vend;
				for (boost::tie(vbegin, vend) = vertices_around_face(mesh.halfedge(fd), mesh); vbegin != vend; vbegin++) {
					const auto& normal = normal_map[*vbegin];
					_normals.push_back(poca::core::Vec3mf(normal.x(), normal.y(), normal.z()));
				}
			}
		}
	}

	void ObjectListMesh::generatePickingIndices(std::vector <float>& _ids) const
	{
		const std::vector <poca::core::Vec3mf> triangles = m_triangles.getData();
		_ids.resize(triangles.size());

		size_t cpt = 0;
		for (size_t i = 0; i < m_triangles.nbElements(); i++) {
			for (size_t j = 0; j < m_triangles.nbElementsObject(i); j++) {
				_ids[cpt++] = i + 1;
			}

		}
	}

	void ObjectListMesh::getFeatureInSelection(std::vector <float>& _features, const std::vector <float>& _values, const std::vector <bool>& _selection, const float _notSelectedValue) const
	{
		const std::vector <poca::core::Vec3mf> triangles = m_triangles.getData();
		_features.resize(triangles.size());

		size_t cpt = 0;
		for (size_t i = 0; i < m_triangles.nbElements(); i++) {
			for (size_t j = 0; j < m_triangles.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _values[i] : _notSelectedValue;
			}

		}
	}

	void ObjectListMesh::getFeatureInSelectionHiLow(std::vector <float>& _features, const std::vector <bool>& _selection, const float _selectedValue, const float _notSelectedValue) const
	{
		const std::vector <poca::core::Vec3mf> triangles = m_triangles.getData();
		_features.resize(triangles.size());

		size_t cpt = 0;
		for (size_t i = 0; i < m_triangles.nbElements(); i++) {
			for (size_t j = 0; j < m_triangles.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _selectedValue : _notSelectedValue;
			}

		}
	}

	poca::core::BoundingBox ObjectListMesh::computeBoundingBoxElement(const int _idx) const
	{
		return m_bboxMeshes[_idx];
	}

	poca::core::Vec3mf ObjectListMesh::computeBarycenterElement(const int _idx) const
	{
		return m_centroids[_idx];
	}

	void ObjectListMesh::generateOutlineLocs(std::vector <poca::core::Vec3mf>& _locs)
	{
		/*_locs.clear();
		for (const auto& mesh : m_meshes)
			for (const auto& point : mesh.points())
				_locs.push_back(poca::core::Vec3mf(point.x(), point.y(), point.z()));*/
		_locs.resize(m_locs.nbData());
		const std::vector <uint32_t>& indices = m_locs.getData();
		for (size_t n = 0; n < indices.size(); n++) {
			size_t index = indices.at(n);
			_locs[n].set(m_xs[index], m_ys[index], m_zs[index]);
		}
	}

	void ObjectListMesh::getOutlineLocsFeatureInSelection(std::vector <float>& _features, const std::vector <float>& _values, const std::vector <bool>& _selection, const float _notSelectedValue) const
	{
		/*_features.clear();
		int i = 0;
		for (const auto& mesh : m_meshes) {
			for (const auto& point : mesh.points())
				_features.push_back(_selection[i] ? _values[i] : _notSelectedValue);
			i++;
		}*/
		_features.resize(m_locs.nbData());

		size_t cpt = 0;
		for (size_t i = 0; i < m_locs.nbElements(); i++) {
			for (size_t j = 0; j < m_locs.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _values[i] : _notSelectedValue;
			}

		}
	}

	void ObjectListMesh::getOutlineLocsFeatureInSelectionHiLow(std::vector <float>& _features, const std::vector <bool>& _selection, const float _selectedValue, const float _notSelectedValue) const
	{
		/*_features.clear();
		int i = 0;
		for (const auto& mesh : m_meshes) {
			for (const auto& point : mesh.points())
				_features.push_back(_selection[i] ? _selectedValue : _notSelectedValue);
			i++;
		}*/
		_features.resize(m_locs.nbData());

		size_t cpt = 0;
		for (size_t i = 0; i < m_locs.nbElements(); i++) {
			for (size_t j = 0; j < m_locs.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _selectedValue : _notSelectedValue;
			}

		}
	}

	void ObjectListMesh::computeSkeletons()
	{
		std::vector <poca::core::Vec3mf> skeletons, links;
		std::vector <uint32_t> nbSkeletons, nbLinks;
		int cpt = 0;
		clock_t t1 = clock(), t2;
		std::cout << std::string(10, '-');
		for (auto& mesh : m_meshes) {
			int percent = floor((float)cpt / (float)m_meshes.size() * 10.f);
			double target_edge_length = 2;
			unsigned int nb_iter = 5;

			std::cout << "\r" << std::string(percent, '*') << std::string(10 - percent, '-') << " ; computing skeleton for mesh " << (cpt + 1) << " composed of " << num_faces(mesh) << " triangles";
			//std::cout << "Start smoothing (" << num_faces(mesh) << " faces)..." << std::endl;
			//std::cout << "Start remeshing (" << num_faces(mesh) << " faces)..." << std::endl;
			//PMP::isotropic_remeshing(faces(mesh), target_edge_length, mesh, CGAL::parameters::number_of_iterations(nb_iter));

			//std::cout << "Start skeletonization (" << num_faces(mesh) << " faces)..." << std::endl;
			Skeleton skeleton;
			CGAL::extract_mean_curvature_flow_skeleton(mesh, skeleton);

			//std::cout << "Number of vertices of the skeleton: " << boost::num_vertices(skeleton) << "\n";
			//std::cout << "Number of edges of the skeleton: " << boost::num_edges(skeleton) << "\n";

			for (const Skeleton_edge& e : CGAL::make_range(edges(skeleton)))
			{
				const Point_3_double& s = skeleton[source(e, skeleton)].point;
				const Point_3_double& t = skeleton[target(e, skeleton)].point;
				skeletons.push_back(poca::core::Vec3mf(s.x(), s.y(), s.z()));
				skeletons.push_back(poca::core::Vec3mf(t.x(), t.y(), t.z()));
			}
			nbSkeletons.push_back(skeletons.size());

			for (Skeleton_vertex v : CGAL::make_range(vertices(skeleton)))
				for (vertex_descriptor vd : skeleton[v].vertices) {
					links.push_back(poca::core::Vec3mf(skeleton[v].point.x(), skeleton[v].point.y(), skeleton[v].point.z()));
					auto other = get(CGAL::vertex_point, mesh, vd);
					links.push_back(poca::core::Vec3mf(other.x(), other.y(), other.z()));
				}
			nbLinks.push_back(links.size());

			//std::cout << "Done creating skeleton and links \n";
			cpt++;
		}
		m_edgesSkeleton.initialize(skeletons, nbSkeletons);
		m_linksSkeleton.initialize(links, nbLinks);

		t2 = clock();
		long elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC;
		std::cout << "\r" << std::string(10, '*') << " ; time elapsed for creating all skeletons -> " << elapsed << " seconds.                                       " << std::endl;
	}

	void ObjectListMesh::saveAsOBJ(const std::string& _filename) const
	{
		std::ofstream fs(_filename);

		uint32_t curMesh = 0, countPoints = 0;
		std::vector <uint32_t> points, nbPts{ 0 }; //_mesh.number_of_vertices()
		for (const auto& mesh : m_meshes) {
			for (const auto& point : mesh.points()) {
				countPoints++;
				fs << "v " << point.x() << " " << point.y() << " " << point.z() << " " << curMesh << std::endl;
			}
			nbPts.push_back(countPoints);
			curMesh++;
		}

		curMesh = 0;
		for (const auto& mesh : m_meshes) {
			for (Surface_mesh_3_double::Face_index fd : mesh.faces()) {
				CGAL::Vertex_around_face_iterator<Surface_mesh_3_double> vbegin, vend;
				fs << "f";
				for (boost::tie(vbegin, vend) = vertices_around_face(mesh.halfedge(fd), mesh); vbegin != vend; vbegin++) {
					fs << " " << (/*nbPts[curMesh]*/ + (*vbegin) + 1);
				}
				fs << " " << curMesh << std::endl;
			}
			curMesh++;
		}
	}
}
