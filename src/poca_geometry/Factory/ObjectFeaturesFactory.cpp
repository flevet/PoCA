/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectFeaturesFactory.cpp
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

#include <CGAL/Simple_cartesian.h>
#include <CGAL/linear_least_squares_fitting_2.h>
#include <CGAL/linear_least_squares_fitting_3.h>
#include <CGAL/Cartesian_d.h>
#include <CGAL/MP_Float.h>
#include <CGAL/Approximate_min_ellipsoid_d.h>
#include <CGAL/Approximate_min_ellipsoid_d_traits_d.h>

#include <General/ArrayStatistics.hpp>

#include "ObjectFeaturesFactory.hpp"

#ifndef M_PI
#define M_PI       3.14159265358979323846
#endif

namespace poca::geometry {
	typedef CGAL::Simple_cartesian< double >  Kernel;
	typedef Kernel::Line_2 KernelLine;
	typedef Kernel::Point_2 KernelPoint;
	typedef Kernel::Line_3 KernelLine3D;
	typedef Kernel::Point_3 KernelPoint3D;
	typedef Kernel::Plane_3 KernelPlane3D;
	typedef Kernel::Vector_3 KernelVector3D;

	typedef CGAL::Cartesian_d<double>                              KernelEllipse;
	typedef CGAL::MP_Float                                         ETEllipse;
	typedef CGAL::Approximate_min_ellipsoid_d_traits_d<KernelEllipse, ETEllipse> TraitsEllipse;
	typedef TraitsEllipse::Point                                         PointEllipse;
	typedef std::vector<PointEllipse>                                     Point_listEllipse;
	typedef CGAL::Approximate_min_ellipsoid_d<TraitsEllipse>              AME;

	ObjectFeaturesFactoryInterface* createObjectFeaturesFactory() {
		return new ObjectFeaturesFactory();
	}

	ObjectFeaturesFactory::ObjectFeaturesFactory()
	{

	}

	ObjectFeaturesFactory::~ObjectFeaturesFactory()
	{

	}

	void ObjectFeaturesFactory::computePCA(const poca::core::MyArrayUInt32& _locs, const size_t _index, const float* _xs, const float* _ys, const float* _zs, float* _res)
	{
		computePCA(_locs.allElements() + _locs.getFirstElements()[_index], _locs.nbElementsObject(_index), _xs, _ys, _zs, _res);
	}

	void ObjectFeaturesFactory::computePCA(const uint32_t* _locIds, const size_t _nbLocs, const float* _xs, const float* _ys, const float* _zs, float* _res)
	{
		if (_zs == NULL)
			computePCA2D(_locIds, _nbLocs, _xs, _ys, _res);
		else
			computePCA3D(_locIds, _nbLocs, _xs, _ys, _zs, _res);
	}

	void ObjectFeaturesFactory::computePCA2D(const uint32_t* _locIds, const size_t _nbLocs, const float* _xs, const float* _ys, float* _res)
	{
		float xc, yc, angle, axisX, axisY, diameter = 0., circ = 0., slope;
		poca::core::Vec2md p0, p1;
		std::vector < KernelPoint > points;
		for (size_t n = 0; n < _nbLocs; n++) {
			uint32_t index = _locIds[n];
			points.push_back(KernelPoint(_xs[index], _ys[index]));
		}
		KernelPoint centroid = CGAL::centroid(points.begin(), points.end(), CGAL::Dimension_tag < 0 >());
		KernelLine majorAxis;
		CGAL::linear_least_squares_fitting_2(points.begin(), points.end(), majorAxis, CGAL::Dimension_tag < 0 >());
		KernelLine minorAxis = majorAxis.perpendicular(centroid);
		float* distanceToMajorAxis = new float[points.size()];
		float* distanceToMinorAxis = new float[points.size()];
		float meanDMajor = 0., meanDMinor = 0., nb = points.size();
		p0.set(centroid.x(), centroid.y());
		p1.set(centroid.x(), centroid.y());
		float d0 = 0., d1 = 0., dpositive = 0., dnegative = 0.;
		for (int n = 0; n < points.size(); n++) {
			KernelPoint proj = majorAxis.projection(points[n]);
			distanceToMajorAxis[n] = sqrt(CGAL::squared_distance(proj, points[n]));
			if (majorAxis.has_on_positive_side(points[n])) {
				if (distanceToMajorAxis[n] > dpositive)
					dpositive = distanceToMajorAxis[n];
			}
			else {
				if (distanceToMajorAxis[n] > dnegative)
					dnegative = distanceToMajorAxis[n];
			}
			meanDMajor += (distanceToMajorAxis[n] / nb);
			if (minorAxis.has_on_positive_side(proj)) {
				float d = sqrt(CGAL::squared_distance(proj, centroid));
				if (d > d0) {
					p0.set(proj.x(), proj.y());
					d0 = d;
				}
			}
			else {
				float d = sqrt(CGAL::squared_distance(proj, centroid));
				if (d > d1) {
					p1.set(proj.x(), proj.y());
					d1 = d;
				}
			}
			proj = minorAxis.projection(points[n]);
			distanceToMinorAxis[n] = sqrt(CGAL::squared_distance(proj, points[n]));
			meanDMinor += (distanceToMinorAxis[n] / nb);
		}
		poca::core::ArrayStatistics statsMajor = poca::core::ArrayStatistics::generateArrayStatistics(distanceToMajorAxis, points.size());
		poca::core::ArrayStatistics statsMinor = poca::core::ArrayStatistics::generateArrayStatistics(distanceToMinorAxis, points.size());
		poca::core::Vec2mf v1 = p1 - p0, v2(1.f, 0.f);
		v1.normalize();
		float dot = v1.dot(v2);
		angle = acos(dot);
		angle = (angle * (180 / M_PI));
		KernelLine axis(KernelPoint(0., 0.), KernelPoint(1., 0.));
		if (axis.has_on_negative_side(KernelPoint(v1.x(), v1.y())))
			angle = 180. + 180 - angle;

		xc = centroid.x();
		yc = centroid.y();

		float dForRapport = (dpositive > dnegative) ? dnegative : dpositive;
		float rapport = 2.35; 
		axisY = rapport * statsMajor.getData(poca::core::ArrayStatistics::StdDev);
		axisX = rapport * statsMinor.getData(poca::core::ArrayStatistics::StdDev); 
		float major, minor;

		if (axisX > axisY) {
			major = axisX;
			minor = axisY;
		}
		else {
			major = axisY;
			minor = axisX;
		}
		circ = minor / major;
		diameter = ((minor + major) / 2.) * 2.;

		size_t index = 0;
		_res[index++] = majorAxis.a();
		_res[index++] = majorAxis.b();
		_res[index++] = xc;
		_res[index++] = yc;
		_res[index++] = angle;
		_res[index++] = axisX;
		_res[index++] = axisY;
		_res[index++] = circ;
		_res[index++] = major * 2.;
		_res[index++] = minor * 2.;

		delete[] distanceToMinorAxis;
		delete[] distanceToMajorAxis;
	}

	void ObjectFeaturesFactory::computePCA3D(const uint32_t* _locIds, const size_t _nbLocs, const float* _xs, const float* _ys, const float* _zs, float* _res)
	{
		double xc, yc, zc, angle, axisX, axisY, diameter = 0., circ = 0., slope;
		std::vector < KernelPoint3D > points;
		for (size_t n = 0; n < _nbLocs; n++) {
			uint32_t index = _locIds[n];
			points.push_back(KernelPoint3D(_xs[index], _ys[index], _zs[index]));
		}
		KernelPoint3D centroid = CGAL::centroid(points.begin(), points.end(), CGAL::Dimension_tag < 0 >());
		KernelLine3D majorAxisLine;
		KernelPlane3D majorAxisPlane;
		CGAL::linear_least_squares_fitting_3(points.begin(), points.end(), majorAxisLine, CGAL::Dimension_tag < 0 >());
		CGAL::linear_least_squares_fitting_3(points.begin(), points.end(), majorAxisPlane, CGAL::Dimension_tag < 0 >());
		KernelLine3D perpendicularLine = majorAxisPlane.perpendicular_line(centroid);
		KernelLine3D thirdLine(centroid, CGAL::cross_product(majorAxisLine.to_vector(), perpendicularLine.to_vector()));

		KernelLine3D lines[3] = { majorAxisLine , perpendicularLine , thirdLine };

		KernelPlane3D planes[3] = { KernelPlane3D(centroid, perpendicularLine.point(10), thirdLine.point(10)),
			KernelPlane3D(centroid, thirdLine.point(10), majorAxisLine.point(10)),
			KernelPlane3D(centroid, perpendicularLine.point(10), majorAxisLine.point(10))
		};
		std::vector <float> distanceToPlanes[3] = { std::vector <float>(points.size()),
			std::vector <float>(points.size()),
			std::vector <float>(points.size())
		};
		for (int n = 0; n < points.size(); n++) {
			for (unsigned int i = 0; i < 3; i++) {
				KernelPoint3D proj = planes[i].projection(points[n]);
				distanceToPlanes[i][n] = sqrt(CGAL::squared_distance(proj, points[n]));
			}
		}

		double rapport = 2.35; 
		double sizes[3];
		poca::core::ArrayStatistics stats[3];
		for (unsigned int i = 0; i < 3; i++) {
			stats[i] = poca::core::ArrayStatistics::generateArrayStatistics(distanceToPlanes[i].data(), points.size());
			sizes[i] = rapport * stats[i].getData(poca::core::ArrayStatistics::StdDev);
		}

		unsigned int idx[3] = { 0, 1, 2 };

		int cpt = 0;
		_res[cpt++] = (float)centroid.x();
		_res[cpt++] = (float)centroid.y();
		_res[cpt++] = (float)centroid.z();
		for (unsigned int i = 0; i < 3; i++)
			_res[cpt++] = (float)(sizes[idx[i]] * 2.);
		for (unsigned int i = 0; i < 3; i++) {
			KernelVector3D direction = lines[idx[i]].to_vector();
			poca::core::Vec3mf dirTmp(direction.x(), direction.y(), direction.z());
			dirTmp.normalize();
			_res[cpt++] = dirTmp.x();
			_res[cpt++] = dirTmp.y();
			_res[cpt++] = dirTmp.z();
		}
	}

	void ObjectFeaturesFactory::computeBoundingEllipse(const poca::core::MyArrayUInt32& _locs, const size_t _index, const float* _xs, const float* _ys, const float* _zs, float* _res)
	{
		computePCA(_locs.allElements() + _locs.getFirstElements()[_index], _locs.nbElementsObject(_index), _xs, _ys, _zs, _res);
	}

	void ObjectFeaturesFactory::computeBoundingEllipse(const uint32_t* _locIds, const size_t _nbLocs, const float* _xs, const float* _ys, const float* _zs, float* _res)
	{
		int dim = _zs == NULL ? 2 : 3;
		const double eps = 0.01;                // approximation ratio is (1+eps)

		Point_listEllipse P;
		if (dim == 2) {
			for (size_t n = 0; n < _nbLocs; n++) {
				uint32_t index = _locIds[n];
				P.push_back(PointEllipse((double)_xs[index], (double)_ys[index], 1.));
			}
		}
		else {
			for (size_t n = 0; n < _nbLocs; n++) {
				uint32_t index = _locIds[n];
				P.push_back(PointEllipse((double)_xs[index], (double)_ys[index], (double)_zs[index], 1.));
			}
		}
		// compute approximation:
		TraitsEllipse traits;
		AME ame(eps, P.begin(), P.end(), traits);

		uint32_t cpt = 0;
		// output center coordinates:
		std::cout << "Cartesian center coordinates: ";
		for (AME::Center_coordinate_iterator c_it = ame.center_cartesian_begin(); c_it != ame.center_cartesian_end(); ++c_it) {
			std::cout << *c_it << ' ';
			_res[cpt++] = *c_it;
		}
		std::cout << ".\n";
		// output lengthes axes:
		std::cout << "Length semiaxis: ";
		for (AME::Axes_lengths_iterator l_it = ame.axes_lengths_begin(); l_it != ame.axes_lengths_end(); ++l_it) {
			std::cout << *l_it << ' ';
			_res[cpt++] = *l_it;
		}
		std::cout << ".\n";
		// output  axes:
		AME::Axes_lengths_iterator axes = ame.axes_lengths_begin();
		for (int i = 0; i < dim; ++i) {
			std::cout << "Semiaxis " << i << " has length " << *axes++ << "\n"	<< "and Cartesian coordinates ";
			for (AME::Axes_direction_coordinate_iterator d_it = ame.axis_direction_cartesian_begin(i); d_it != ame.axis_direction_cartesian_end(i); ++d_it) {
				std::cout << *d_it << ' ';
				_res[cpt++] = *d_it;
			}
			std::cout << ".\n";
		}
	}
}

