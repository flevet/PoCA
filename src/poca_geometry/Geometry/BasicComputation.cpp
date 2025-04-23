/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      BasicComputation.cpp
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

#include <General/Misc.h>

#include "BasicComputation.hpp"

namespace poca::geometry {
	float computeTriangleArea(const double _x1, const double _y1, const double _x2, const double _y2, const double _x3, const double _y3)
	{
		double area = ((_x2 * _y1) - (_x1 * _y2)) + ((_x3 * _y2) - (_x2 * _y3)) + ((_x1 * _y3) - (_x3 * _y1));
		return fabs(area / 2.);
	}

	float computePolygonArea(poca::core::Vec3md* _points, const unsigned int _nbPoints)
	{
		double nbd = _nbPoints;
		poca::core::Vec3md centroid;
		for (unsigned int i = 0; i < _nbPoints; i++)
			centroid += _points[i] / nbd;
		double totalArea = 0;
		for (unsigned int i = 0; i < _nbPoints; i++) {
			uint32_t idnext = (i + 1) % _nbPoints;
			double sideA = distance3D<double>(centroid.x(), centroid.y(), centroid.z(), _points[i].x(), _points[i].y(), _points[i].z()),
				sideB = distance3D<double>(centroid.x(), centroid.y(), centroid.z(), _points[idnext].x(), _points[idnext].y(), _points[idnext].z()),
				sideC = distance3D<double>(_points[idnext].x(), _points[idnext].y(), _points[idnext].z(), _points[i].x(), _points[i].y(), _points[i].z());
			totalArea += computeAreaTriangle<double>(sideA, sideB, sideC);
		}
		return (float)totalArea;
	}

	float computePolygonArea2D(poca::core::Vec3md* _points, const unsigned int _nbPoints)
	{
		float totalArea = 0;
		for (unsigned int i = 0; i < _nbPoints - 1; i++)
			totalArea += ((_points[i].x() - _points[i + 1].x()) * (_points[i + 1].y() + (_points[i].y() - _points[i + 1].y()) / 2));
		// need to do points[point.length-1] and points[0].
		totalArea += ((_points[_nbPoints - 1].x() - _points[0].x()) * (_points[0].y() + (_points[_nbPoints - 1].y() - _points[0].y()) / 2));
		return (float)abs(totalArea);
	}

	float computePolygonArea(poca::core::Vec2md* _points, const unsigned int _nbPoints)
	{
		float totalArea = 0;
		for (unsigned int i = 0; i < _nbPoints - 1; i++)
			totalArea += ((_points[i].x() - _points[i + 1].x()) * (_points[i + 1].y() + (_points[i].y() - _points[i + 1].y()) / 2));
		// need to do points[point.length-1] and points[0].
		totalArea += ((_points[_nbPoints - 1].x() - _points[0].x()) * (_points[0].y() + (_points[_nbPoints - 1].y() - _points[0].y()) / 2));
		return (float)abs(totalArea);
	}

	float computePolygonArea(double* _points, const size_t _nbPoints)
	{
		float totalArea = 0;
		for (unsigned int i = 0; i < _nbPoints - 1; i++) {
			size_t cur = i * 2, next = (i + 1) * 2;
			totalArea += ((_points[cur] - _points[next]) * (_points[next+1] + (_points[cur+1] - _points[next+1]) / 2));
		}
		// need to do points[point.length-1] and points[0].
		size_t cur = (_nbPoints - 1) * 2, next = 0;
		totalArea += ((_points[cur] - _points[next]) * (_points[next + 1] + (_points[cur + 1] - _points[next + 1]) / 2));
		return (float)abs(totalArea);
	}

	float distance(const float _x0, const float _y0, const float _z0, const float _x1, const float _y1, const float _z1)
	{
		return sqrt(distanceSqr(_x0, _y0, _z0, _x1, _y1, _z1));
	}

	float distanceSqr(const float _x0, const float _y0, const float _z0, const float _x1, const float _y1, const float _z1)
	{
		double x = _x1 - _x0, y = _y1 - _y0, z = _z1 - _z0;
		return x * x + y * y + z * z;
	}

	double BasicComputation::distance(const double _x0, const double _y0, const double _x1, const double _y1)
	{
		return sqrt(distanceSqr(_x0, _y0, _x1, _y1));
	}

	double BasicComputation::distanceSqr(const double _x0, const double _y0, const double _x1, const double _y1)
	{
		double x = _x1 - _x0, y = _y1 - _y0;
		return x * x + y * y;
	}

	double BasicComputation::getTriangleArea(const poca::core::Vec2md& _v1, const poca::core::Vec2md& _v2, const poca::core::Vec2md& _v3)
	{
		double x1 = _v1.x(), y1 = _v1.y(), x2 = _v2.x(), y2 = _v2.y(), x3 = _v3.x(), y3 = _v3.y();
		double area = ((x2 * y1) - (x1 * y2)) + ((x3 * y2) - (x2 * y3)) + ((x1 * y3) - (x3 * y1));
		return fabs(area / 2.);
	}

	double BasicComputation::getTriangleArea(const double _x1, const double _y1, const double _x2, const double _y2, const double _x3, const double _y3)
	{
		double area = ((_x2 * _y1) - (_x1 * _y2)) + ((_x3 * _y2) - (_x2 * _y3)) + ((_x1 * _y3) - (_x3 * _y1));
		return fabs(area / 2.);
	}

	bool BasicComputation::isRecInsideCircle(const double _x0, const double _y0, const double _x1, const double _y1, const double _cX, const double _cY, const double _cr)
	{
		double l1 = distance(_x0, _y0, _cX, _cY);
		double l2 = distance(_x1, _y0, _cX, _cY);
		double l3 = distance(_x1, _y1, _cX, _cY);
		double l4 = distance(_x0, _y1, _cX, _cY);
		return (l1 <= _cr && l2 <= _cr && l3 <= _cr && l4 <= _cr);
	}

	bool BasicComputation::isInsideRec(const double _origX, const double _origY, const double _boundsX, const double _boundsY, const double _centerX, const double _centerY)
	{
		double x0 = _origX, y0 = _origY, x1 = _origX + _boundsX, y1 = _origY + _boundsY;
		return (x0 <= _centerX && _centerX <= x1 && y0 <= _centerY && _centerY <= y1);
	}

	bool BasicComputation::isLineIntersectCircle(const double _lx1, const double _ly1, const double _lx2, const double _ly2, const double _cX, const double _cY, const double _cr)
	{
		double pointX = 0., pointY = 0.;
		closestPointOnLine(_lx1, _ly1, _lx2, _ly2, _cX, _cY, pointX, pointY);
		double vecX = pointX - _cX, vecY = pointY - _cY;
		double length = sqrt(vecX * vecX + vecY * vecY);
		return (length <= _cr && _lx1 <= pointX && pointX <= _lx2 && _ly1 <= pointY && pointY <= _ly2);
	}

	void BasicComputation::closestPointOnLine(const double _lx1, const double _ly1, const double _lx2, const double _ly2, const double _cX, const double _cY, double& _px, double& _py)
	{
		double A1 = _ly2 - _ly1;
		double B1 = _lx1 - _lx2;
		double C1 = (_ly2 - _ly1) * _lx1 + (_lx1 - _lx2) * _ly1;
		double C2 = -B1 * _cX + A1 * _cY;
		double det = A1 * A1 - -B1 * B1;
		if (det != 0) {
			_px = ((A1 * C1 - B1 * C2) / det);
			_py = ((A1 * C2 - -B1 * C1) / det);
		}
		else {
			_px = _cX;
			_py = _cY;
		}
	}

	bool BasicComputation::circleLineIntersect(const double _x1, const double _y1, const double _x2, const double _y2, const double _cx, const double _cy, const double _cr)
	{
		double dx = _x2 - _x1;
		double dy = _y2 - _y1;
		double a = dx * dx + dy * dy;
		double b = 2 * (dx * (_x1 - _cx) + dy * (_y1 - _cy));
		double c = _cx * _cx + _cy * _cy;
		c += _x1 * _x1 + _y1 * _y1;
		c -= 2 * (_cx * _x1 + _cy * _y1);
		c -= _cr * _cr;
		double bb4ac = b * b - 4 * a * c;

		if (bb4ac < 0) {  // Not intersecting
			return false;
		}
		else {

			double mu = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
			double ix1 = _x1 + mu * (dx);
			double iy1 = _y1 + mu * (dy);
			mu = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
			double ix2 = _x1 + mu * (dx);
			double iy2 = _y1 + mu * (dy);

			double testX;
			double testY;
			// Figure out which point is closer to the circle
			if (distance(_x1, _y1, _cx, _cy) < distance(_x2, _y2, _cx, _cy)) {
				testX = _x2;
				testY = _y2;
			}
			else {
				testX = _x1;
				testY = _y1;
			}

			if (distance(testX, testY, ix1, iy1) < distance(_x1, _y1, _x2, _y2) || distance(testX, testY, ix2, iy2) < distance(_x1, _y1, _x2, _y2)) {
				return true;
			}
			else {
				return false;
			}
		}
	}

	void BasicComputation::circleLineIntersect(const double _x1, const double _y1, const double _x2, const double _y2, const double _cx, const double _cy, const double _cr, std::vector < poca::core::Vec2md >& _points)
	{
		double dx = _x2 - _x1;
		double dy = _y2 - _y1;
		double a = dx * dx + dy * dy;
		double b = 2 * (dx * (_x1 - _cx) + dy * (_y1 - _cy));
		double c = _cx * _cx + _cy * _cy;
		c += _x1 * _x1 + _y1 * _y1;
		c -= 2 * (_cx * _x1 + _cy * _y1);
		c -= _cr * _cr;
		double bb4ac = b * b - 4 * a * c;


		if (bb4ac < 0) {  // Not intersecting
			return;
		}
		else {

			double mu = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
			double ix1 = _x1 + mu * (dx);
			double iy1 = _y1 + mu * (dy);
			mu = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
			double ix2 = _x1 + mu * (dx);
			double iy2 = _y1 + mu * (dy);
			if (_x1 <= ix1 && ix1 <= _x2 && _y1 <= iy1 && iy1 <= _y2)
				_points.push_back(poca::core::Vec2md(ix1, iy1));
			if (_x1 <= ix2 && ix2 <= _x2 && _y1 <= iy2 && iy2 <= _y2)
				_points.push_back(poca::core::Vec2md(ix2, iy2));
		}
	}

	double BasicComputation::computeAreaCircularSegment(const double _cx, const double _cy, const double _r, const poca::core::Vec2md& _p1, const poca::core::Vec2md& _p2) {
		double x = (_p1.x() + _p2.x()) / 2., y = (_p1.y() + _p2.y()) / 2.;
		double smallR = BasicComputation::distance(x, y, _cx, _cy);
		double h = _r - smallR;

		double area = (_r * _r) * acos((_r - h) / _r) - ((_r - h) * sqrt((2. * _r * h) - (h * h)));
		return area;
	}
}

