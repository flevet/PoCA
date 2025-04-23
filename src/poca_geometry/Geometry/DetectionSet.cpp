/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DetectionSet.cpp
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

#include <float.h>
#include <set>
#include <fstream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <vector>

#include <General/Palette.hpp>
#include <General/MyData.hpp>
#include <General/Histogram.hpp>
#include <General/Misc.h>
#include <General/ranker.h>

#include "DetectionSet.hpp"


#define DOUBLE_EPSILON .00000000001

namespace poca::geometry {

	struct classcompSet {
		bool operator()(const std::pair< poca::core::DetectionPoint, int >& _p1, const std::pair< poca::core::DetectionPoint, int >& _p2) const
		{
			double dy = _p1.first.y() - _p2.first.y();
			if (fabs(dy) > DOUBLE_EPSILON) return(_p1.first.y() < _p2.first.y());
			double dx = _p1.first.x() - _p2.first.x();
			if (fabs(dx) > DOUBLE_EPSILON) return(_p1.first.x() < _p2.first.x());
			return false;
		}
	};

	struct classcompSort {
		bool operator()(const std::pair< poca::core::DetectionPoint, int >& _p1, const std::pair< poca::core::DetectionPoint, int >& _p2) const
		{
			return _p1.second < _p2.second;
		}
	} myComparator;

	DetectionSet::DetectionSet() :poca::core::BasicComponent("DetectionSet"), m_kdTree(NULL)
	{
	}

	DetectionSet::DetectionSet(const DetectionSet& _o) : poca::core::BasicComponent(_o), m_nbPoints(_o.m_nbPoints), m_nbSlices(_o.m_nbSlices), m_kdTree(NULL)
	{
		const std::vector <float>& xs = m_data["x"]->getOriginalData<float>(), & ys = m_data["y"]->getOriginalData<float>();
		m_pointCloud.resize(xs.size());
		if (hasData("z")) {
			const std::vector <float>& zs = m_data["z"]->getOriginalData<float>();
			for (size_t n = 0; n < xs.size(); n++)
				m_pointCloud.m_pts[n].set(xs[n], ys[n], zs[n]);
		}
		else 
			for (size_t n = 0; n < xs.size(); n++)
				m_pointCloud.m_pts[n].set(xs[n], ys[n], 0.);
		m_kdTree = new KdTree_DetectionPoint(3, m_pointCloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
		m_kdTree->buildIndex();
	}

	DetectionSet::DetectionSet(const std::vector < DetectionSet* >& _vect) : poca::core::BasicComponent("DetectionSet")
	{
		m_nbPoints = 0;
		m_nbSlices = 0;
		float xmax = -FLT_MAX, ymax = -FLT_MAX, zmin = FLT_MAX, zmax = -FLT_MAX;
		for (std::vector < DetectionSet* >::const_iterator it2 = _vect.begin(); it2 != _vect.end(); it2++) {
			DetectionSet* o = *it2;
			m_nbPoints += o->m_nbPoints;
			m_nbSlices += o->m_nbSlices;
			if (o->boundingBox()[3] > xmax)
				xmax = o->boundingBox()[3];
			if (o->boundingBox()[4] > ymax)
				ymax = o->boundingBox()[4];
			if (o->boundingBox()[2] < zmin)
				zmin = o->boundingBox()[2];
			if (o->boundingBox()[5] > zmax)
				zmax = o->boundingBox()[5];
		}

		std::map <std::string, std::vector <float>> data;
		float addingForStart = 0;

		for (std::map<std::string, poca::core::MyData*>::iterator it = _vect[0]->m_data.begin(); it != _vect[0]->m_data.end(); ++it)
			data[it->first] = std::vector <float>();

		for (std::vector < DetectionSet* >::const_iterator it2 = _vect.begin(); it2 != _vect.end(); it2++) {
			DetectionSet* o = *it2;
			for (std::map<std::string, poca::core::MyData*>::iterator it = o->m_data.begin(); it != o->m_data.end(); ++it) {
				const std::vector <float>& tmp = it->second->getOriginalData<float>();
				if (it->first == "time") {
					std::transform(tmp.begin(), tmp.end(), std::back_inserter(data[it->first]), [&addingForStart](auto i) { return  i + addingForStart; });
					addingForStart = data[it->first][data[it->first].size() - 1] + 1;
				}
				else
					std::copy(tmp.begin(), tmp.end(), std::back_inserter(data[it->first]));
			}
		}

		setData(data);
		setCurrentHistogramType("x");
		forceRegenerateSelection();
		this->setBoundingBox(0, 0, zmin < 0. ? floor(zmin) : 0, ceil(xmax), ceil(ymax), zmax < 0. ? floor(zmax) : ceil(zmax));
		std::cout << "Total number of detections -> " << m_nbPoints << std::endl;

		const std::vector <float>& xs = m_data["x"]->getOriginalData<float>(), & ys = m_data["y"]->getOriginalData<float>();
		m_pointCloud.resize(xs.size());
		if (hasData("z")) {
			const std::vector <float>& zs = m_data["z"]->getOriginalData<float>();
			for (size_t n = 0; n < xs.size(); n++)
				m_pointCloud.m_pts[n].set(xs[n], ys[n], zs[n]);
		}
		else
			for (size_t n = 0; n < xs.size(); n++)
				m_pointCloud.m_pts[n].set(xs[n], ys[n], 0.);
		m_kdTree = new KdTree_DetectionPoint(3, m_pointCloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
		m_kdTree->buildIndex();
	}

	DetectionSet::DetectionSet(const std::map <std::string, std::vector <float>>& _data) :poca::core::BasicComponent("DetectionSet"), m_nbPoints(0), m_nbSlices(0)
	{
		m_palette = poca::core::Palette::getStaticLutPtr("HotCold2");
		setData(_data);
		m_nbPoints = m_data["x"]->getOriginalData<float>().size();
		const std::vector <float>& xs = m_data["x"]->getOriginalData<float>(), & ys = m_data["y"]->getOriginalData<float>();
		const std::vector <float>& zs = hasData("z") ? m_data["z"]->getOriginalData<float>() : std::vector <float>(xs.size(), 0.f);
		float xmin = FLT_MAX, xmax = -FLT_MAX, ymin = FLT_MAX, ymax = -FLT_MAX, zmin = FLT_MAX, zmax = -FLT_MAX;
		m_pointCloud.resize(xs.size());
		for (size_t n = 0; n < xs.size(); n++) {
			if (xs[n] < xmin)
				xmin = xs[n];
			if (xs[n] > xmax)
				xmax = xs[n];
			if (ys[n] < ymin)
				ymin = ys[n];
			if (ys[n] > ymax)
				ymax = ys[n];
			if (zs[n] < zmin)
				zmin = zs[n];
			if (zs[n] > zmax)
				zmax = zs[n];
			m_pointCloud.m_pts[n].set(xs[n], ys[n], zs[n]);
		}

		this->setBoundingBox(floor(xmin), floor(ymin), floor(zmin), ceil(xmax), ceil(ymax), ceil(zmax));

		m_nbSlices = hasData("frame") ? (size_t)(m_data["frame"]->getOriginalData<float>().back() + 1) : 1;
		setCurrentHistogramType("x");
		forceRegenerateSelection();
		m_kdTree = new KdTree_DetectionPoint(3, m_pointCloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
		m_kdTree->buildIndex();
	}

	DetectionSet::~DetectionSet()
	{
	}

	const unsigned int DetectionSet::memorySize() const
	{
		size_t memoryS = 0;
		memoryS += 2 * sizeof(float);
		memoryS += 2 * sizeof(int);
		memoryS += sizeof(poca::core::DetectionPoint*);
		memoryS += sizeof(double*);
		memoryS += sizeof(unsigned int*);
		memoryS += sizeof(unsigned short*);
		if (m_nbPoints > 0) {
			memoryS += m_nbPoints * sizeof(poca::core::DetectionPoint);
			memoryS += m_nbPoints * sizeof(double);
			memoryS += m_nbPoints * sizeof(unsigned int);
			memoryS += m_nbPoints * sizeof(unsigned short);
		}
		return (unsigned int)memoryS;
	}

	poca::core::BasicComponentInterface* DetectionSet::copy()
	{
		return new DetectionSet(*this);
	}


	void DetectionSet::setData(const std::map <std::string, std::vector <float>>& _data)
	{
		poca::core::BasicComponent::setData(_data);
		for (std::map <std::string, std::vector <float>>::const_iterator it = _data.begin(); it != _data.end(); it++) {
			m_data[it->first] = poca::core::generateDataWithLog(it->second);
		}
	}

	const float DetectionSet::averageDensity() const
	{
		float nbs = nbElements(), averageD = 0.f;
		float w = m_bbox[3] - m_bbox[0], h = m_bbox[4] - m_bbox[1], t = m_bbox[5] - m_bbox[2];
		averageD = (dimension() == 3) ? nbs / (w * h * t) : nbs / (w * h);
		return averageD;
	}

	void DetectionSet::computeBBoxFromPoints()
	{
		m_bbox.set(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
		std::vector <std::string> headers = { "x", "y", "z" };
		for (auto n = 0; n < headers.size(); n++) {
			if (hasData(headers[n])) {
				std::vector <float>& values = m_data[headers[n]]->getOriginalData<float>();
				std::vector<float>::iterator min_v = std::min_element(values.begin(), values.end());
				std::vector<float>::iterator max_v = std::max_element(values.begin(), values.end());
				m_bbox[n] = *min_v;
				m_bbox[n + 3] = *max_v;
			}
		}
	}

	DetectionSet* DetectionSet::duplicateSelection() const
	{
		std::map <std::string, std::vector <float>> data;
		std::vector <float> feature(m_nbSelection);
		for (std::map <std::string, poca::core::MyData*>::const_iterator it = m_data.begin(); it != m_data.end(); it++) {
			size_t cpt = 0;
			const std::vector <float>& values = it->second->getOriginalData<float>();;// static_cast<poca::core::Histogram<float>*>(it->second->getOriginalHistogram())->getValues();
			for (size_t n = 0; n < values.size(); n++)
				if (m_selection[n])
					feature[cpt++] = values[n];
			data[it->first] = feature;
		}
		return new DetectionSet(data);
	}

	void DetectionSet::saveDetections(std::ofstream& _fs)
	{
		poca::core::stringList nameFeatures = getNameData();
		for (const string& name : nameFeatures)
			_fs << name.c_str() << ",";
		_fs << std::endl;
		std::vector <std::vector<float>*> data;
		for (string name : nameFeatures) {
			std::vector <float>& values = m_data[name]->getOriginalData<float>();
			data.push_back(&values);
		}
		for (size_t j = 0; j < m_nbPoints; j++) {
			for (size_t i = 0; i < nameFeatures.size(); i++)
				_fs << data[i]->at(j) << ",";
			_fs << std::endl;
		}
	}

	void DetectionSet::getFeaturesOfSelection(const std::vector <uint32_t>& _indexes, std::map <std::string, std::vector <float>>& _data)
	{
		poca::core::stringList features = getNameData();
		for (auto feature : features)
			_data[feature] = std::vector <float>();

		for (auto idx : _indexes) {
			for (auto feature : features)
				_data[feature].push_back(m_data[feature]->getOriginalData<float>()[idx]);
		}
	}

	DetectionSet* DetectionSet::copySelection(const std::vector <uint32_t>& _indexes)
	{
		std::map <std::string, std::vector <float>> data;
		this->getFeaturesOfSelection(_indexes, data);

		DetectionSet* dset = new DetectionSet(data);
		return dset;
	}
}

