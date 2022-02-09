/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectList.cpp
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

#include <algorithm>

#include <General/MyData.hpp>
#include <General/BasicComponent.hpp>
#include <Interfaces/HistogramInterface.hpp>

#include "ObjectList.hpp"
#include "BasicComputation.hpp"
#include "DelaunayTriangulation.hpp"
#include "../Interfaces/ObjectFeaturesFactoryInterface.hpp"

namespace poca::geometry {
	ObjectList::ObjectList(const float* _xs, const float* _ys, const float* _zs, 
		const std::vector <uint32_t>& _locsAllObjects, const std::vector <uint32_t>& _firstsLocs,
		const std::vector <poca::core::Vec3mf>& _trianglesAllObjects, const std::vector <uint32_t>& _firstsTriangles,
		const std::vector <poca::core::Vec3mf>& _outlinesAllObjects, const std::vector <uint32_t>& _firstsOutlines, 
		const std::vector <uint32_t>& _linkTriangulationFacesToObjects)
		:BasicComponent("ObjectList"), m_xs(_xs), m_ys(_ys), m_zs(_zs), 
		m_locs(_locsAllObjects, _firstsLocs), m_triangles(_trianglesAllObjects, _firstsTriangles), m_outlines(_outlinesAllObjects, _firstsOutlines), m_linkTriangulationFacesToObjects(_linkTriangulationFacesToObjects)
	{
		//Create area feature
		std::vector <float> areas(m_triangles.nbElements(), 0.f), nbLocs(m_locs.nbElements(), 0.f);
		for (size_t i = 0; i < m_triangles.nbElements(); i++) {
			for (size_t j = 0; j < m_triangles.nbElementsObject(i); j+=3) {
				const poca::core::Vec3mf& v1 = m_triangles.elementIObject(i, j);
				const poca::core::Vec3mf& v2 = m_triangles.elementIObject(i, j + 1);
				const poca::core::Vec3mf& v3 = m_triangles.elementIObject(i, j + 2);
				areas[i] += poca::geometry::computeTriangleArea(v1.x(), v1.y(), v2.x(), v2.y(), v3.x(), v3.y());
			}
			nbLocs[i] = m_locs.nbElementsObject(i);

		}
		ObjectFeaturesFactoryInterface* factory = createObjectFeaturesFactory();
		std::vector <float> sizes(m_locs.nbElements()), resPCA(factory->nbFeaturesPCA(m_zs != NULL));
		for (size_t n = 0; n < m_locs.nbElements(); n++) {
			float* ptr = &resPCA[0];
			factory->computePCA(m_locs, n, m_xs, m_ys, m_zs, ptr);
			sizes[n] = (resPCA[8] + resPCA[9]) / 2.f;
		}
		delete factory;

		std::vector <float> ids(m_locs.nbElements());
		std::iota(std::begin(ids), std::end(ids), 1);

		m_data["area"] = new poca::core::MyData(areas);
		m_data["nbLocs"] = new poca::core::MyData(nbLocs);
		m_data["size"] = new poca::core::MyData(sizes);
		m_data["id"] = new poca::core::MyData(ids);
		m_selection.resize(areas.size());
		setCurrentHistogramType("area");
		forceRegenerateSelection();
	}

	ObjectList::ObjectList(const float* _xs, const float* _ys, const float* _zs,
		const std::vector <uint32_t>& _locsAllObjects, const std::vector <uint32_t>& _firstsLocs,
		const std::vector <poca::core::Vec3mf>& _trianglesAllObjects, const std::vector <uint32_t>& _firstsTriangles,
		const std::vector <float>& _volumes, 
		const std::vector <uint32_t>& _linkTriangulationFacesToObjects, 
		const std::vector <uint32_t>& _locsOutlineAllObject, const std::vector <uint32_t>& _firstsOutlineLocs, const std::vector <poca::core::Vec3mf>& _normalOutlineLocs)
		:BasicComponent("ObjectList"), m_xs(_xs), m_ys(_ys), m_zs(_zs),
		m_locs(_locsAllObjects, _firstsLocs), m_triangles(_trianglesAllObjects, _firstsTriangles), m_linkTriangulationFacesToObjects(_linkTriangulationFacesToObjects), m_outlineLocs(_locsOutlineAllObject, _firstsOutlineLocs), m_normalOutlineLocs(_normalOutlineLocs)
	{
		//Create area feature
		std::vector <float> areas(m_triangles.nbElements(), 0.f), nbLocs(m_locs.nbElements(), 0.f);
		for (size_t i = 0; i < m_triangles.nbElements(); i++) {
			for (size_t j = 0; j < m_triangles.nbElementsObject(i); j += 3) {
				const poca::core::Vec3mf& v1 = m_triangles.elementIObject(i, j);
				const poca::core::Vec3mf& v2 = m_triangles.elementIObject(i, j + 1);
				const poca::core::Vec3mf& v3 = m_triangles.elementIObject(i, j + 2);
				areas[i] += poca::geometry::computeTriangleArea(v1.x(), v1.y(), v2.x(), v2.y(), v3.x(), v3.y());
			}
			nbLocs[i] = m_locs.nbElementsObject(i);

		}
		const poca::core::MyArrayUInt32& localizations = m_outlineLocs;
		ObjectFeaturesFactoryInterface* factory = createObjectFeaturesFactory();
		std::vector <float> sizes(localizations.nbElements()), resPCA(factory->nbFeaturesPCA(m_zs != NULL));
		std::vector <float> major(localizations.nbElements()), minor(localizations.nbElements()), minor2(localizations.nbElements());
		m_axis.resize(localizations.nbElements());
		for (size_t n = 0; n < localizations.nbElements(); n++) {
			float* ptr = &resPCA[0];
			factory->computePCA(localizations, n, m_xs, m_ys, m_zs, ptr);
			major[n] = resPCA[3];
			minor[n] = resPCA[4];
			minor2[n] = resPCA[5];
			sizes[n] = (resPCA[3] + resPCA[4] + resPCA[5]) / 3.f;
			m_axis[n] = { poca::core::Vec3mf(resPCA[6], resPCA[7], resPCA[8]), 
				poca::core::Vec3mf(resPCA[9], resPCA[10], resPCA[11]) , 
				poca::core::Vec3mf(resPCA[12], resPCA[13], resPCA[14]) };
		}
		delete factory;

		std::vector <float> ids(m_locs.nbElements());
		std::iota(std::begin(ids), std::end(ids), 1);

		m_data["volume"] = new poca::core::MyData(_volumes);
		m_data["surface"] = new poca::core::MyData(areas);
		m_data["nbLocs"] = new poca::core::MyData(nbLocs);
		m_data["size"] = new poca::core::MyData(sizes);
		m_data["major"] = new poca::core::MyData(major);
		m_data["minor"] = new poca::core::MyData(minor);
		m_data["minor2"] = new poca::core::MyData(minor2);
		m_data["id"] = new poca::core::MyData(ids);
		m_selection.resize(areas.size());
		setCurrentHistogramType("volume");
		forceRegenerateSelection();
	}

	ObjectList::~ObjectList()
	{
	}

	poca::core::BasicComponent* ObjectList::copy()
	{
		return new ObjectList(*this);
	}

	void ObjectList::generateLocs(std::vector <poca::core::Vec3mf>& _locs)
	{
		_locs.resize(m_locs.nbData());
		const std::vector <uint32_t>& indices = m_locs.getData();
		if (m_zs != NULL) {
			for (size_t n = 0; n < indices.size(); n++) {
				size_t index = indices.at(n);
				_locs[n].set(m_xs[index], m_ys[index], m_zs[index]);
			}
		}
		else {
			for (size_t n = 0; n < indices.size(); n++) {
				size_t index = indices.at(n);
				_locs[n].set(m_xs[index], m_ys[index], 0.f);
			}
		}
	}

	void ObjectList::generateNormalLocs(std::vector <poca::core::Vec3mf>& _norms)
	{
		_norms.resize(m_locs.nbData());
		const std::vector <uint32_t>& indices = m_locs.getData(), & objects = m_locs.getFirstElements();
		if (m_zs != NULL) {
			for (uint32_t n = 0; n < m_locs.nbElements(); n++) {
				uint32_t nbLocs = objects[n + 1] - objects[n];
				float nbD = nbLocs;
				poca::core::Vec3mf centroid(0.f, 0.f, 0.f);
				for (uint32_t idx = objects[n]; idx < objects[n + 1]; idx++) {
					uint32_t index = indices[idx];
					centroid += poca::core::Vec3mf(m_xs[index], m_ys[index], m_zs[index]) / nbD;
				}
				for (uint32_t idx = objects[n]; idx < objects[n + 1]; idx++) {
					uint32_t index = indices[idx];
					poca::core::Vec3mf normal = poca::core::Vec3mf(m_xs[index], m_ys[index], m_zs[index]) - centroid;
					normal.normalize();
					_norms[idx] = normal;
				}
			}
		}
		else {
			for (uint32_t n = 0; n < m_locs.nbElements(); n++) {
				uint32_t nbLocs = objects[n + 1] - objects[n];
				float nbD = nbLocs;
				poca::core::Vec3mf centroid(0.f, 0.f, 0.f);
				for (uint32_t idx = objects[n]; idx < objects[n + 1]; idx++) {
					uint32_t index = indices[idx];
					centroid += poca::core::Vec3mf(m_xs[index], m_ys[index], 0.f) / nbD;
				}
				for (uint32_t idx = objects[n]; idx < objects[n + 1]; idx++) {
					uint32_t index = indices[idx];
					poca::core::Vec3mf normal = poca::core::Vec3mf(m_xs[index], m_ys[index], 0.f) - centroid;
					normal.normalize();
					_norms[idx] = normal;
				}
			}
		}
	}

	void ObjectList::getLocsFeatureInSelection(std::vector <float>& _features, const std::vector <float>& _values, const std::vector <bool>& _selection, const float _notSelectedValue) const
	{
		_features.resize(m_locs.nbData());

		size_t cpt = 0;
		for (size_t i = 0; i < m_locs.nbElements(); i++) {
			for (size_t j = 0; j < m_locs.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _values[i] : _notSelectedValue;
			}

		}

	}

	void ObjectList::getLocsFeatureInSelectionHiLow(std::vector <float>& _features, const std::vector <bool>& _selection, const float _selectedValue, const float _notSelectedValue) const
	{
		_features.resize(m_locs.nbData());

		size_t cpt = 0;
		for (size_t i = 0; i < m_locs.nbElements(); i++) {
			for (size_t j = 0; j < m_locs.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _selectedValue : _notSelectedValue;
			}

		}
	}

	void ObjectList::getOutlinesFeatureInSelection(std::vector <float>& _features, const std::vector <float>& _values, const std::vector <bool>& _selection, const float _notSelectedValue) const
	{
		_features.resize(m_outlines.nbData());// *2);

		size_t cpt = 0;
		for (size_t i = 0; i < m_outlines.nbElements(); i++) {
			for (size_t j = 0; j < m_outlines.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _values[i] : _notSelectedValue;
			}
		}
	}

	void ObjectList::getOutlinesFeatureInSelectionHiLow(std::vector <float>& _features, const std::vector <bool>& _selection, const float _selectedValue, const float _notSelectedValue) const
	{
		_features.resize(m_outlines.nbData());// *2);

		size_t cpt = 0;
		for (size_t i = 0; i < m_outlines.nbElements(); i++) {
			for (size_t j = 0; j < m_outlines.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _selectedValue : _notSelectedValue;
			}

		}
	}

	void ObjectList::generateLocsPickingIndices(std::vector <float>& _ids) const
	{
		_ids.resize(m_locs.nbData());

		size_t cpt = 0;
		for (size_t i = 0; i < m_locs.nbElements(); i++) {
			for (size_t j = 0; j < m_locs.nbElementsObject(i); j++) {
				_ids[cpt++] = i + 1;
			}

		}
	}

	void ObjectList::generateTriangles(std::vector <poca::core::Vec3mf>& _triangles)
	{
		std::copy(m_triangles.getData().begin(), m_triangles.getData().end(), std::back_inserter(_triangles));
	}

	void ObjectList::generateOutlines(std::vector <poca::core::Vec3mf>& _outlines)
	{
		std::copy(m_outlines.getData().begin(), m_outlines.getData().end(), std::back_inserter(_outlines));
	}

	void ObjectList::generateNormals(std::vector <poca::core::Vec3mf>& _normals)
	{
		_normals.resize(m_triangles.getData().size());
		if (m_zs == NULL)
			std::fill(_normals.begin(), _normals.end(), poca::core::Vec3mf(0.f, 0.f, 1.f));
		else {
			const std::vector <poca::core::Vec3mf>& triangles = m_triangles.getData();
			const std::vector <uint32_t>& firsts = m_triangles.getFirstElements();
			for (size_t n = 0; n < m_triangles.nbElements(); n++) {
				poca::core::Vec3mf centroid;
				float nb = m_locs.nbElementsObject(n);
				for (size_t i = 0; i < m_locs.nbElementsObject(n); i++) {
					uint32_t idLoc = m_locs.elementIObject(n, i);
					centroid += poca::core::Vec3mf(m_xs[idLoc], m_ys[idLoc], m_zs[idLoc]) / nb;
				}

				for (uint32_t idTri = firsts[n]; idTri < firsts[n + 1]; idTri += 3) {
					const poca::core::Vec3mf& v1 = triangles.at(idTri), v2 = triangles.at(idTri + 1), v3 = triangles.at(idTri + 2);
					poca::core::Vec3mf e1 = v2 - v1, e2 = v3 - v1, normal = -e1.cross(e2);
					normal.normalize();
					_normals[idTri] = normal;
					_normals[idTri + 1] = normal;
					_normals[idTri + 2] = normal;
				}
			}
		}
	}

	void ObjectList::getFeatureInSelection(std::vector <float>& _features, const std::vector <float>& _values, const std::vector <bool>& _selection, const float _notSelectedValue) const
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

	void ObjectList::getFeatureInSelectionHiLow(std::vector <float>& _features, const std::vector <bool>& _selection, const float _selectedValue, const float _notSelectedValue) const
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

	void ObjectList::generatePickingIndices(std::vector <float>& _ids) const
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

	poca::core::BoundingBox ObjectList::computeBoundingBoxElement(const int _idx) const
	{
		poca::core::BoundingBox bbox;
		if(m_zs == NULL)
			bbox.set(FLT_MAX, FLT_MAX, 0.f, -FLT_MAX, -FLT_MAX, 0.f);
		else
			bbox.set(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
		const std::vector <poca::core::Vec3mf> triangles = m_triangles.getData();
		for (size_t j = 0; j < m_triangles.nbElementsObject(_idx); j++) {
			poca::core::Vec3mf vertex = m_triangles.elementIObject(_idx, j);
			bbox[0] = vertex.x() < bbox[0] ? vertex.x() : bbox[0];
			bbox[1] = vertex.y() < bbox[1] ? vertex.y() : bbox[1];

			bbox[3] = vertex.x() > bbox[3] ? vertex.x() : bbox[3];
			bbox[4] = vertex.y() > bbox[4] ? vertex.y() : bbox[4];

			if (m_zs != NULL) {
				bbox[2] = vertex.z() < bbox[2] ? vertex.z() : bbox[2];
				bbox[5] = vertex.z() > bbox[5] ? vertex.z() : bbox[5];
			}
		}
		return bbox;
	}

	poca::core::Vec3mf ObjectList::computeBarycenterElement(const int _idx) const
	{
		poca::core::Vec3mf centroid(0.f, 0.f, 0.f);
		float nbs = m_locs.nbElementsObject(_idx);
		for (size_t j = 0; j < m_locs.nbElementsObject(_idx); j++) {
			uint32_t index = m_locs.elementIObject(_idx, j);
			centroid += poca::core::Vec3mf(m_xs[index], m_ys[index], m_zs != NULL ? m_zs[index] : 0.f) / nbs;
		}
		return centroid;
	}

	void ObjectList::generateOutlineLocs(std::vector <poca::core::Vec3mf>& _locs)
	{
		_locs.resize(m_outlineLocs.nbData());
		const std::vector <uint32_t>& indices = m_outlineLocs.getData();
		if (m_zs != NULL) {
			for (size_t n = 0; n < indices.size(); n++) {
				size_t index = indices.at(n);
				_locs[n].set(m_xs[index], m_ys[index], m_zs[index]);
			}
		}
		else {
			for (size_t n = 0; n < indices.size(); n++) {
				size_t index = indices.at(n);
				_locs[n].set(m_xs[index], m_ys[index], 0.f);
			}
		}
	}

	void ObjectList::getOutlineLocsFeatureInSelection(std::vector <float>& _features, const std::vector <float>& _values, const std::vector <bool>& _selection, const float _notSelectedValue) const
	{
		_features.resize(m_outlineLocs.nbData());

		size_t cpt = 0;
		for (size_t i = 0; i < m_outlineLocs.nbElements(); i++) {
			for (size_t j = 0; j < m_outlineLocs.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _values[i] : _notSelectedValue;
			}

		}

	}

	void ObjectList::getOutlineLocsFeatureInSelectionHiLow(std::vector <float>& _features, const std::vector <bool>& _selection, const float _selectedValue, const float _notSelectedValue) const
	{
		_features.resize(m_outlineLocs.nbData());

		size_t cpt = 0;
		for (size_t i = 0; i < m_outlineLocs.nbElements(); i++) {
			for (size_t j = 0; j < m_outlineLocs.nbElementsObject(i); j++) {
				_features[cpt++] = _selection[i] ? _selectedValue : _notSelectedValue;
			}

		}
	}
}

