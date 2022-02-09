/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Roi.cpp
*
* Copyright: Florian Levet (2020-2021)
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

#include <Windows.h>
#include <gl/glew.h>
#include <gl/GL.h>
#include <QtCore/qmath.h>

#include <General/Vec3.hpp>

#include "Roi.hpp"

namespace poca::core {
	ROI::ROI(const std::string& _type) :m_type(_type), m_changed(false)
	{
	}

	ROI::ROI(const ROI& _o) : m_name(_o.m_name), m_type(_o.m_type), m_changed(_o.m_changed)
	{
	}

	ROI::~ROI()
	{
	}

	LineROI::LineROI(const std::string& _type) :ROI(_type)
	{
	}

	LineROI::LineROI(const LineROI& _o) : ROI(_o)
	{
		std::copy(&_o.m_pts[0], &_o.m_pts[0] + 1, &m_pts[0]);
	}

	LineROI::~LineROI()
	{
		m_lineBuffer.freeGPUMemory();
	}

	void LineROI::updateDisplay()
	{
		if(m_lineBuffer.empty())
			m_lineBuffer.generateBuffer(2, 512 * 512, 3, GL_FLOAT);
		std::vector <poca::core::Vec3mf> pts = { poca::core::Vec3mf(m_pts[0].x(), m_pts[0].y(), 0.f),
					poca::core::Vec3mf(m_pts[1].x(), m_pts[1].y(), 0.f) };
		m_lineBuffer.updateBuffer(pts);
	}

	void LineROI::draw(poca::opengl::Camera*)
	{
		if (m_changed)
			updateDisplay();
		/*if( !ROI::MODIFY_ROI ){
			if( m_p0 == NULL ) return;
			if( m_p1 != NULL ){
				glBegin( GL_LINES );
				glVertex2dv( m_p0->getValues() );
				glVertex2dv( m_p1->getValues() );
				glEnd();
			}
			if( m_tmpPoint != NULL ){
				glLineStipple( 4, 0xAAAA );
				glEnable( GL_LINE_STIPPLE );
				glBegin( GL_LINES );
				glVertex2dv( m_p0->getValues() );
				glVertex2dv( m_tmpPoint->getValues() );
				glEnd();
				glDisable( GL_LINE_STIPPLE );
			}
		}
		else{
			if( m_p0 == NULL || m_p1 == NULL ) return;
			glBegin( GL_LINES );
			glVertex2dv( m_p0->getValues() );
			glVertex2dv( m_p1->getValues() );
			glEnd();
			glPushMatrix();
			glColor3d( 1.f, 0.f, 0.f );
			drawCircle( *m_p0, ROI::DISTANCE_MODIFY );
			drawCircle( *m_p1, ROI::DISTANCE_MODIFY );
			glPopMatrix();
		}*/
	}

	bool LineROI::inside(const float _x, const float _y, const float _z) const
	{
		return false;
	}

	void LineROI::onClick(const float _x, const float _y, const bool _modify)
	{
		m_pts[0].set(_x, _y);
		m_pts[1].set(_x, _y);
	}

	void LineROI::onMove(const float _x, const float _y, const bool _modify)
	{
		m_pts[1].set(_x, _y);
	}

	void LineROI::finalize(const float _x, const float _y, const bool _modify)
	{
		m_pts[1].set(_x, _y);
	}

	float LineROI::getFeature(const std::string& _typeFeature) const
	{
		if (_typeFeature == "perimeter")
			return m_pts[0].distance(m_pts[1]);
		return std::numeric_limits <float>::max();
	}

	void LineROI::save(std::ofstream& _fs) const
	{
		_fs << "LineROI" << std::endl;
		_fs << m_pts[0].x() << " " << m_pts[0].y() << " " << m_pts[1].x() << " " << m_pts[1].y() << std::endl;
	}

	void LineROI::load(std::ifstream& _fs)
	{
		std::string s;
		std::getline(_fs, s);
		std::istringstream is2(s);
		float x0, y0, x1, y1;
		is2 >> x0 >> y0 >> x1 >> y1;
		this->onClick(x0, y0);
		this->finalize(x1, y1);
	}

	const std::string LineROI::toStdString() const
	{
		std::string text("LineROI, " + m_name + "\n" + 
			"[" + std::to_string(m_pts[0].x()) + ", " + std::to_string(m_pts[0].y()) + "]" +
			"[" + std::to_string(m_pts[1].x()) + ", " + std::to_string(m_pts[2].y()) + "]");
		return text;
	}

	ROIInterface* LineROI::copy() const
	{
		return (ROIInterface*)(new LineROI(*this));
	}

	CircleROI::CircleROI(const std::string& _type) :ROI(_type), m_radius(0.f)
	{
	}

	CircleROI::CircleROI(const CircleROI& _o) : ROI(_o), m_radius(_o.m_radius)
	{
	}

	CircleROI::~CircleROI()
	{
	}

	void CircleROI::updateDisplay()
	{
		if (m_centerBuffer.empty())
			m_centerBuffer.generateBuffer(1, 512 * 512, 3, GL_FLOAT);
		std::vector <poca::core::Vec3mf> pt = { poca::core::Vec3mf(m_center.x(),m_center.y(), 0.f) };
		m_centerBuffer.updateBuffer(pt);
	}

	void CircleROI::draw(poca::opengl::Camera*)
	{
		if (m_centerBuffer.empty())
			updateDisplay();
	}

	bool CircleROI::inside(const float _x, const float _y, const float _z) const
	{
		return (m_center.distance(Vec2md(_x, _y)) <= m_radius);
	}

	void CircleROI::onClick(const float _x, const float _y, const bool _modify)
	{
		m_center.set(_x, _y);
	}

	void CircleROI::onMove(const float _x, const float _y, const bool _modify)
	{
		m_radius = m_center.distance(Vec2md(_x, _y));
	}

	void CircleROI::finalize(const float _x, const float _y, const bool _modify)
	{
		m_radius = m_center.distance(Vec2md(_x, _y));
	}

	float CircleROI::getFeature(const std::string& _typeFeature) const
	{
		if(_typeFeature == "perimeter")
			return 2. * M_PI * m_radius;
		else if (_typeFeature == "area")
			return M_PI * m_radius * m_radius;
		return std::numeric_limits <float>::max();
	}

	void CircleROI::save(std::ofstream& _fs) const
	{
		_fs << "CircleROI" << std::endl;
		_fs << m_center.x() << " " << m_center.y() << " " << m_radius << std::endl;
	}

	void CircleROI::load(std::ifstream& _fs)
	{
		std::string s;
		std::getline(_fs, s);
		std::istringstream is2(s);
		float x0, y0, radius;
		is2 >> x0 >> y0 >> radius;
		this->onClick(x0, y0);
		this->finalize(x0 + radius, y0);
	}

	ROIInterface* CircleROI::copy() const
	{
		return (ROIInterface*)(new CircleROI(*this));
	}

	const std::string CircleROI::toStdString() const
	{
		std::string text("CircleROI, " + m_name + "\n" +
			"[" + std::to_string(m_center.x()) + ", " + std::to_string(m_center.y()) + "]" +
			", radius = " + std::to_string(m_radius));
		return text;
	}

	/*PolylineROI::PolylineROI(const std::string& _type) :ROI(_type), m_finalized(false)
	{

	}

	PolylineROI::PolylineROI(const PolylineROI& _o) : ROI(_o), m_finalized(_o.m_finalized), m_centerSelected(_o.m_centerSelected)
	{
		m_points.resize(_o.m_points.size());
		std::copy(_o.m_points.begin(), _o.m_points.end(), m_points.begin());
	}

	PolylineROI::~PolylineROI()
	{
		m_points.clear();
	}

	void PolylineROI::draw() const
	{
		if (m_points.empty()) return;
		if (!ROI::MODIFY_ROI) {
			glPushMatrix();
			glBegin(GL_LINE_STRIP);
			for (std::vector < Vec2md >::const_iterator it = m_points.begin(); it != m_points.end(); it++)
				glVertex2dv(it->getValues());
			if (m_finalized)
				glVertex2dv(m_points.front().getValues());
			glEnd();
			if (m_tmpPoint != NULL) {
				glLineStipple(4, 0xAAAA);
				glEnable(GL_LINE_STIPPLE);
				glBegin(GL_LINE_STRIP);
				glVertex2dv(m_points.back().getValues());
				glVertex2dv(m_tmpPoint->getValues());
				glVertex2dv(m_points.front().getValues());
				glEnd();
				glDisable(GL_LINE_STIPPLE);
			}
			glPopMatrix();
		}
		else {
			glPushMatrix();
			glColor3d(1.f, 0.f, 0.f);
			glBegin(GL_LINE_STRIP);
			for (std::vector < Vec2md >::const_iterator it = m_points.begin(); it != m_points.end(); it++)
				glVertex2dv(it->getValues());
			glVertex2dv(m_points.front().getValues());
			glEnd();
			for (std::vector < Vec2md >::const_iterator it = m_points.begin(); it != m_points.end(); it++)
				drawCircle(*it, ROI::DISTANCE_MODIFY);
			drawCircle(*m_center, ROI::DISTANCE_MODIFY);
			glPopMatrix();
		}
	}

	bool PolylineROI::inside(const float _x, const float _y) const
	{
		float epsilon = 0.00001;
		for (std::vector < Vec2md >::const_iterator it = m_points.begin(); it != m_points.end(); it++) {
			float d = opensmlm::geometry::BasicComputation::distance(_x, _y, it->x(), it->y());
			if (d < epsilon)
				return true;
		}

		int cn = 0;// the  crossing number counter

		std::vector < Vec2md >::const_iterator prec = m_points.end();
		prec--;
		// loop through all edges of the polygon
		for (std::vector < Vec2md >::const_iterator current = m_points.begin(); current != m_points.end(); current++) {
			if (((prec->y() <= _y) && (current->y() > _y))     // an upward crossing
				|| ((prec->y() > _y) && (current->y() <= _y))) { // a downward crossing
					// compute  the actual edge-ray intersect x-coordinate
				float vt = (float)(_y - prec->y()) / (current->y() - prec->y());
				if (_x < prec->x() + vt * (current->x() - prec->x())) // P.x < intersect
					++cn;   // a valid crossing of y=P.y right of P.x
			}
			prec = current;
		}
		return (cn & 1) == 1;    // 0 if even (out), and 1 if  odd (in)
	}

	ROI* PolylineROI::insideModify(const float _x, const float _y)
	{
		m_currentChange = -1;
		m_centerSelected = false;
		float d = opensmlm::geometry::BasicComputation::distance(m_center->x(), m_center->y(), _x, _y);
		if (d < ROI::DISTANCE_MODIFY)
			m_centerSelected = true;
		if (m_centerSelected) return this;
		int cpt = 0;
		for (std::vector < Vec2md >::const_iterator it = m_points.begin(); it != m_points.end(); it++, cpt++) {
			float d = opensmlm::geometry::BasicComputation::distance(it->x(), it->y(), _x, _y);
			if (d < ROI::DISTANCE_MODIFY)
				m_currentChange = cpt;
		}
		if (m_currentChange != -1) return this;
		return NULL;
	}

	void PolylineROI::onClick(const float _x, const float _y, const bool _modify)
	{
		if (m_points.empty())
			m_tmpPoint = new Vec2md(_x, _y);
		m_points.push_back(Vec2md(_x, _y));
	}

	void PolylineROI::onMove(const float _x, const float _y, const bool _modify)
	{
		if (!_modify) {
			if (m_tmpPoint != NULL) m_tmpPoint->set(_x, _y);
		}
		else {
			if (m_centerSelected) {
				float xd = _x - m_center->x(), yd = _y - m_center->y();
				m_center->set(_x, _y);
				for (std::vector < Vec2md >::iterator it = m_points.begin(); it != m_points.end(); it++)
					it->set(it->x() + xd, it->y() + yd);
			}
			else
				m_points[m_currentChange].set(_x, _y);
		}
	}

	void PolylineROI::finalize(const float _x, const float _y, const bool _modify)
	{
		if (!_modify) {
			if (m_tmpPoint != NULL)
				delete m_tmpPoint;
			m_tmpPoint = NULL;
			m_finalized = true;
			float xc = 0., yc = 0., nb = m_points.size();
			for (std::vector < Vec2md >::const_iterator current = m_points.begin(); current != m_points.end(); current++) {
				xc += current->x() / nb;
				yc += current->y() / nb;
			}
			m_center = new Vec2md(xc, yc);
		}
		else {
			if (m_centerSelected) {
				float xd = _x - m_center->x(), yd = _y - m_center->y();
				m_center->set(_x, _y);
				for (std::vector < Vec2md >::iterator it = m_points.begin(); it != m_points.end(); it++)
					it->set(it->x() + xd, it->y() + yd);
			}
			else {
				m_points[m_currentChange].set(_x, _y);
				float xc = 0., yc = 0., nb = m_points.size();
				for (std::vector < Vec2md >::const_iterator current = m_points.begin(); current != m_points.end(); current++) {
					xc += current->x() / nb;
					yc += current->y() / nb;
				}
				m_center = new Vec2md(xc, yc);
			}
		}
	}

	float PolylineROI::getPerimeter() const
	{
		float perimeter = 0.;
		std::vector < Vec2md >::const_iterator prec = m_points.end();
		prec--;
		// loop through all edges of the polygon
		for (std::vector < Vec2md >::const_iterator current = m_points.begin(); current != m_points.end(); current++) {
			perimeter += prec->distance(*current);
			prec = current;
		}
		return perimeter;
	}

	float PolylineROI::getArea() const
	{
		float totalArea = 0;
		std::vector < Vec2md >::const_iterator prec = m_points.end();
		prec--;
		// loop through all edges of the polygon
		for (std::vector < Vec2md >::const_iterator current = m_points.begin(); current != m_points.end(); current++) {
			totalArea += ((prec->x() - current->x()) * (current->y() + (prec->y() - current->y()) / 2));
			prec = current;
		}
		return abs(totalArea);
	}

	const std::string PolylineROI::statusBarMessage(Calibration* _cal) const
	{
		if (m_points.empty() || m_tmpPoint == NULL) return std::string();
		float d = opensmlm::geometry::BasicComputation::distance(m_points.back().x(), m_points.back().y(), m_tmpPoint->x(), m_tmpPoint->y());
		std::string s("# points: " + std::string::number(m_points.size() + 1) + ", current distance: " + std::string::number(d * _cal->getPixelXY()) + " " + _cal->getDimensionUnit().c_str());
		return s;
	}

	void PolylineROI::save(std::ofstream& _fs) const
	{
		_fs << "PolygonROI" << std::endl;
		_fs << this->nbPoints() << std::endl;
		const std::vector < Vec2md >& points = this->getPoints();
		for (std::vector < Vec2md >::const_iterator it = points.begin(); it != points.end(); it++)
			_fs << it->x() << " " << it->y() << std::endl;
	}

	void PolylineROI::load(std::ifstream& _fs)
	{
		std::string s;
		std::getline(_fs, s);
		std::istringstream is2(s);
		float x, y;
		unsigned int nbPoints;
		is2 >> nbPoints;
		for (std::size_t n2 = 0; n2 < nbPoints - 1; n2++) {
			std::getline(_fs, s);
			std::istringstream is3(s);
			is3 >> x >> y;
			this->onClick(x, y);
		}
		std::getline(_fs, s);
		std::istringstream is3(s);
		is3 >> x >> y;
		this->onClick(x, y);
		this->finalize(x, y);
	}

	void PolylineROI::load(std::ifstream& _fs, const unsigned int _nb)
	{
		std::string s;
		float x, y;
		unsigned int nbPoints = _nb;
		for (std::size_t n2 = 0; n2 < nbPoints - 1; n2++) {
			std::getline(_fs, s);
			std::istringstream is3(s);
			is3 >> x >> y;
			this->onClick(x, y);
		}
		std::getline(_fs, s);
		std::istringstream is3(s);
		is3 >> x >> y;
		this->onClick(x, y);
		this->finalize(x, y);
	}

	ROI* PolylineROI::copy() const
	{
		return new PolylineROI(*this);
	}

	SquareROI::SquareROI(const std::string& _type) :ROI(_type)
	{
		m_p0 = m_p1 = NULL;
	}

	SquareROI::SquareROI(const SquareROI& _o) : ROI(_o)
	{
		if (_o.m_p0 == NULL)
			m_p0 = NULL;
		else
			m_p0 = new Vec2md(*_o.m_p0);
		if (_o.m_p1 == NULL)
			m_p1 = NULL;
		else
			m_p1 = new Vec2md(*_o.m_p1);
	}

	SquareROI::~SquareROI()
	{
		if (m_p0 != NULL)
			delete m_p0;
		if (m_p1 != NULL)
			delete m_p1;
	}

	void SquareROI::draw() const
	{
		if (!ROI::MODIFY_ROI) {
			if (m_p0 == NULL) return;
			if (m_p1 != NULL) {
				glBegin(GL_LINE_STRIP);
				glVertex2d(m_p0->x(), m_p0->y());
				glVertex2d(m_p0->x(), m_p1->y());
				glVertex2d(m_p1->x(), m_p1->y());
				glVertex2d(m_p1->x(), m_p0->y());
				glVertex2d(m_p0->x(), m_p0->y());
				glEnd();
			}
			if (m_tmpPoint != NULL) {
				glLineStipple(4, 0xAAAA);
				glEnable(GL_LINE_STIPPLE);
				glBegin(GL_LINE_STRIP);
				glVertex2d(m_p0->x(), m_p0->y());
				glVertex2d(m_p0->x(), m_tmpPoint->y());
				glVertex2d(m_tmpPoint->x(), m_tmpPoint->y());
				glVertex2d(m_tmpPoint->x(), m_p0->y());
				glVertex2d(m_p0->x(), m_p0->y());
				glEnd();
				glDisable(GL_LINE_STIPPLE);
			}
		}
		else {
			if (m_p0 == NULL || m_p1 == NULL) return;
			glBegin(GL_LINE_STRIP);
			glVertex2d(m_p0->x(), m_p0->y());
			glVertex2d(m_p0->x(), m_p1->y());
			glVertex2d(m_p1->x(), m_p1->y());
			glVertex2d(m_p1->x(), m_p0->y());
			glVertex2d(m_p0->x(), m_p0->y());
			glEnd();
			glPushMatrix();
			glColor3d(1.f, 0.f, 0.f);
			drawCircle(*m_p0, ROI::DISTANCE_MODIFY);
			drawCircle(*m_p1, ROI::DISTANCE_MODIFY);
			glPopMatrix();
		}
	}

	bool SquareROI::inside(const float _x, const float _y) const
	{
		float xmin = (m_p0->x() < m_p1->x()) ? m_p0->x() : m_p1->x();
		float xmax = (m_p0->x() > m_p1->x()) ? m_p0->x() : m_p1->x();
		float ymin = (m_p0->y() < m_p1->y()) ? m_p0->y() : m_p1->y();
		float ymax = (m_p0->y() > m_p1->y()) ? m_p0->y() : m_p1->y();

		return (xmin <= _x && _x <= xmax && ymin <= _y && _y <= ymax);
	}

	ROI* SquareROI::insideModify(const float _x, const float _y)
	{
		m_currentChange = -1;
		float d = m_p0->distance(Vec2md(_x, _y));
		if (d < ROI::DISTANCE_MODIFY) m_currentChange = 0;
		d = m_p1->distance(Vec2md(_x, _y));
		if (d < ROI::DISTANCE_MODIFY) m_currentChange = 1;
		if (m_currentChange != -1) return this;
		return NULL;
	}

	void SquareROI::onClick(const float _x, const float _y, const bool _modify)
	{
		if (!_modify) {
			if (m_p0 == NULL) {
				m_p0 = new Vec2md(_x, _y);
				m_tmpPoint = new Vec2md(_x, _y);
			}
			else if (m_p1 == NULL)
				m_p1 = new Vec2md(_x, _y);
			else
				m_p1->set(_x, _y);
		}
		else {
			m_currentChange = -1;
			float d = m_p0->distance(Vec2md(_x, _y));
			if (d < ROI::DISTANCE_MODIFY) m_currentChange = 0;
			d = m_p1->distance(Vec2md(_x, _y));
			if (d < ROI::DISTANCE_MODIFY) m_currentChange = 1;
		}
	}

	void SquareROI::onMove(const float _x, const float _y, const bool _modify)
	{
		if (!_modify) {
			if (m_tmpPoint != NULL) m_tmpPoint->set(_x, _y);
		}
		else {
			if (m_currentChange == 0) m_p0->set(_x, _y);
			if (m_currentChange == 1) m_p1->set(_x, _y);
		}
	}

	void SquareROI::finalize(const float _x, const float _y, const bool _modify)
	{
		if (!_modify) {
			if (m_p1 == NULL)
				m_p1 = new Vec2md(_x, _y);
			else
				m_p1->set(_x, _y);
			delete m_tmpPoint;
			m_tmpPoint = NULL;
			m_center = new Vec2md((m_p0->x() + m_p1->x()) / 2., (m_p0->y() + m_p1->y()) / 2.);
		}
		else {
			if (m_currentChange == 0) m_p0->set(_x, _y);
			if (m_currentChange == 1) m_p1->set(_x, _y);
			m_center = new Vec2md((m_p0->x() + m_p1->x()) / 2., (m_p0->y() + m_p1->y()) / 2.);
		}
	}

	float SquareROI::getPerimeter() const
	{
		float d = 0;
		d += 2. * opensmlm::geometry::BasicComputation::distance(m_p0->x(), m_p0->y(), m_p0->x(), m_p1->y());
		d += 2. * opensmlm::geometry::BasicComputation::distance(m_p0->x(), m_p1->y(), m_p1->x(), m_p1->y());
		return d;
	}

	float SquareROI::getArea() const
	{
		return opensmlm::geometry::BasicComputation::distance(m_p0->x(), m_p0->y(), m_p0->x(), m_p1->y()) * opensmlm::geometry::BasicComputation::distance(m_p0->x(), m_p1->y(), m_p1->x(), m_p1->y());
	}

	const std::string SquareROI::statusBarMessage(Calibration* _cal) const
	{
		if (m_p0 == NULL || m_tmpPoint == NULL) return std::string();
		float w = opensmlm::geometry::BasicComputation::distance(m_p0->x(), 0, m_tmpPoint->x(), 0), h = opensmlm::geometry::BasicComputation::distance(0, m_p0->y(), 0, m_tmpPoint->y());
		std::string s("Rectangle: [" + std::string::number(w * _cal->getPixelXY()) + " " + _cal->getDimensionUnit().c_str() + ", " + std::string::number(h * _cal->getPixelXY()) + " " + _cal->getDimensionUnit().c_str() + "]");
		return s;
	}

	void SquareROI::save(std::ofstream& _fs) const
	{
		_fs << "SquareROI" << std::endl;
		_fs << m_p0->x() << " " << m_p0->y() << " " << m_p1->x() << " " << m_p1->y() << std::endl;
	}

	void SquareROI::load(std::ifstream& _fs)
	{
		std::string s;
		std::getline(_fs, s);
		std::istringstream is2(s);
		float x0, y0, x1, y1;
		is2 >> x0 >> y0 >> x1 >> y1;
		this->onClick(x0, y0);
		this->finalize(x1, y1);
	}

	ROI* SquareROI::copy() const
	{
		return new SquareROI(*this);
	}


	TriangleROI::TriangleROI(const std::string& _type) :ROI(_type), m_changing(false)
	{
		m_p0 = m_p1 = m_p2 = NULL;
	}

	TriangleROI::TriangleROI(const TriangleROI& _o) : ROI(_o), m_changing(_o.m_changing)
	{
		if (_o.m_p0 == NULL)
			m_p0 = NULL;
		else
			m_p0 = new Vec2md(*_o.m_p0);
		if (_o.m_p1 == NULL)
			m_p1 = NULL;
		else
			m_p1 = new Vec2md(*_o.m_p1);
		if (_o.m_p2 == NULL)
			m_p2 = NULL;
		else
			m_p2 = new Vec2md(*_o.m_p2);
	}

	TriangleROI::~TriangleROI()
	{
		if (m_p0 != NULL)
			delete m_p0;
		if (m_p1 != NULL)
			delete m_p1;
		if (m_p2 != NULL)
			delete m_p2;
	}

	void TriangleROI::draw() const
	{
		if (!ROI::MODIFY_ROI) {
			if (m_changing) {
				glLineStipple(4, 0xAAAA);
				glEnable(GL_LINE_STIPPLE);
				glBegin(GL_LINE_STRIP);
				glVertex2d(m_p0->x(), m_p0->y());
				glVertex2d(m_p1->x(), m_p1->y());
				glVertex2d(m_p2->x(), m_p2->y());
				glVertex2d(m_p0->x(), m_p0->y());
				glEnd();
				glDisable(GL_LINE_STIPPLE);
			}
			else {
				glBegin(GL_LINE_STRIP);
				glVertex2d(m_p0->x(), m_p0->y());
				glVertex2d(m_p1->x(), m_p1->y());
				glVertex2d(m_p2->x(), m_p2->y());
				glVertex2d(m_p0->x(), m_p0->y());
				glEnd();
			}
		}
		else {
			if (m_p0 == NULL || m_p1 == NULL || m_p2 == NULL) return;
			glBegin(GL_LINE_STRIP);
			glVertex2d(m_p0->x(), m_p0->y());
			glVertex2d(m_p1->x(), m_p1->y());
			glVertex2d(m_p2->x(), m_p2->y());
			glVertex2d(m_p0->x(), m_p0->y());
			glEnd();
			glPushMatrix();
			glColor3d(1.f, 0.f, 0.f);
			drawCircle(*m_p0, ROI::DISTANCE_MODIFY);
			drawCircle(*m_tmpPoint, ROI::DISTANCE_MODIFY);
			glPopMatrix();
		}
	}

	bool TriangleROI::inside(const float _x, const float _y) const
	{
		float epsilon = 0.00001;
		float d = opensmlm::geometry::BasicComputation::distance(_x, _y, m_p0->x(), m_p0->y());
		if (d < epsilon) return true;
		d = opensmlm::geometry::BasicComputation::distance(_x, _y, m_p1->x(), m_p1->y());
		if (d < epsilon) return true;
		d = opensmlm::geometry::BasicComputation::distance(_x, _y, m_p2->x(), m_p2->y());
		if (d < epsilon) return true;

		int cn = 0;// the  crossing number counter

		// loop through all edges of the triangles
		Vec2md* prec, * current;
		for (unsigned int n = 0; n < 3; n++) {
			if (n == 0) { prec = m_p2; current = m_p0; }
			if (n == 1) { prec = m_p0; current = m_p1; }
			if (n == 2) { prec = m_p1; current = m_p2; }
			if (((prec->y() <= _y) && (current->y() > _y)) || ((prec->y() > _y) && (current->y() <= _y))) { // first upward test crossing then downward test crossing
					// compute  the actual edge-ray intersect x-coordinate
				float vt = (float)(_y - prec->y()) / (current->y() - prec->y());
				if (_x < prec->x() + vt * (current->x() - prec->x())) // P.x < intersect
					++cn;   // a valid crossing of y=P.y right of P.x
			}
		}
		return (cn & 1) == 1;    // 0 if even (out), and 1 if  odd (in)
	}

	ROI* TriangleROI::insideModify(const float _x, const float _y)
	{
		m_currentChange = -1;
		float d = m_p0->distance(Vec2md(_x, _y));
		if (d < ROI::DISTANCE_MODIFY) m_currentChange = 0;
		d = m_tmpPoint->distance(Vec2md(_x, _y));
		if (d < ROI::DISTANCE_MODIFY) m_currentChange = 1;
		if (m_currentChange != -1) return this;
		return NULL;
	}

	void TriangleROI::onClick(const float _x, const float _y, const bool _modify)
	{
		if (!_modify) {
			if (m_p0 == NULL) {
				m_p0 = new Vec2md(_x, _y);
				m_p1 = new Vec2md(_x, _y);
				m_p2 = new Vec2md(_x, _y);
				m_tmpPoint = new Vec2md(_x, _y);
			}
			else
				m_tmpPoint = new Vec2md(_x, _y);
			m_changing = true;
		}
		else {
			m_currentChange = -1;
			float d = m_p0->distance(Vec2md(_x, _y));
			if (d < ROI::DISTANCE_MODIFY) m_currentChange = 0;
			d = m_tmpPoint->distance(Vec2md(_x, _y));
			if (d < ROI::DISTANCE_MODIFY) m_currentChange = 1;
			if (m_currentChange != -1) m_changing = true;
		}
	}

	void TriangleROI::onMove(const float _x, const float _y, const bool _modify)
	{
		if (!_modify) {
			if (m_tmpPoint != NULL) {
				m_tmpPoint->set(_x, _y);
				opensmlm::geometry::StraightLine line(m_p0->x(), m_p0->y(), m_tmpPoint->x(), m_tmpPoint->y(), opensmlm::geometry::StraightLine::PARALLELE_LINE);
				float d = opensmlm::geometry::BasicComputation::distance(m_p0->x(), m_p0->y(), m_tmpPoint->x(), m_tmpPoint->y());
				d /= (sqrt(3.) / 2.);
				d /= 2.;
				Vec2md tmp = line.findOrthoPoint(m_tmpPoint->x(), m_tmpPoint->y(), d);
				m_p1->set(tmp.x(), tmp.y());
				tmp = line.findOrthoPoint(m_tmpPoint->x(), m_tmpPoint->y(), -d);
				m_p2->set(tmp.x(), tmp.y());
			}
		}
		else {
			if (m_currentChange == 0) m_p0->set(_x, _y);
			if (m_currentChange == 1) m_tmpPoint->set(_x, _y);
			opensmlm::geometry::StraightLine line(m_p0->x(), m_p0->y(), m_tmpPoint->x(), m_tmpPoint->y(), opensmlm::geometry::StraightLine::PARALLELE_LINE);
			float d = opensmlm::geometry::BasicComputation::distance(m_p0->x(), m_p0->y(), m_tmpPoint->x(), m_tmpPoint->y());
			d /= (sqrt(3.) / 2.);
			d /= 2.;
			Vec2md tmp = line.findOrthoPoint(m_tmpPoint->x(), m_tmpPoint->y(), d);
			m_p1->set(tmp.x(), tmp.y());
			tmp = line.findOrthoPoint(m_tmpPoint->x(), m_tmpPoint->y(), -d);
			m_p2->set(tmp.x(), tmp.y());
		}
	}

	void TriangleROI::finalize(const float _x, const float _y, const bool _modify)
	{
		m_changing = false;
		if (!_modify) {
			m_tmpPoint->set(_x, _y);
			opensmlm::geometry::StraightLine line(m_p0->x(), m_p0->y(), m_tmpPoint->x(), m_tmpPoint->y(), opensmlm::geometry::StraightLine::PARALLELE_LINE);
			float d = opensmlm::geometry::BasicComputation::distance(m_p0->x(), m_p0->y(), m_tmpPoint->x(), m_tmpPoint->y());
			d /= (sqrt(3.) / 2.);
			d /= 2.;
			Vec2md tmp = line.findOrthoPoint(m_tmpPoint->x(), m_tmpPoint->y(), d);
			m_p1->set(tmp.x(), tmp.y());
			tmp = line.findOrthoPoint(m_tmpPoint->x(), m_tmpPoint->y(), -d);
			m_p2->set(tmp.x(), tmp.y());
			m_center = new Vec2md((m_p0->x() + m_p1->x() + m_p2->x()) / 3., (m_p0->y() + m_p1->y() + m_p2->y()) / 3.);
		}
		else {
			if (m_currentChange == 0) m_p0->set(_x, _y);
			if (m_currentChange == 1) m_tmpPoint->set(_x, _y);
			opensmlm::geometry::StraightLine line(m_p0->x(), m_p0->y(), m_tmpPoint->x(), m_tmpPoint->y(), opensmlm::geometry::StraightLine::PARALLELE_LINE);
			float d = opensmlm::geometry::BasicComputation::distance(m_p0->x(), m_p0->y(), m_tmpPoint->x(), m_tmpPoint->y());
			d /= (sqrt(3.) / 2.);
			d /= 2.;
			Vec2md tmp = line.findOrthoPoint(m_tmpPoint->x(), m_tmpPoint->y(), d);
			m_p1->set(tmp.x(), tmp.y());
			tmp = line.findOrthoPoint(m_tmpPoint->x(), m_tmpPoint->y(), -d);
			m_p2->set(tmp.x(), tmp.y());
			m_center = new Vec2md((m_p0->x() + m_p1->x() + m_p2->x()) / 3., (m_p0->y() + m_p1->y() + m_p2->y()) / 3.);
		}
	}

	float TriangleROI::getPerimeter() const
	{
		float d = 0;
		d += 2. * opensmlm::geometry::BasicComputation::distance(m_p0->x(), m_p0->y(), m_p1->x(), m_p1->y());
		d += 2. * opensmlm::geometry::BasicComputation::distance(m_p1->x(), m_p1->y(), m_p2->x(), m_p2->y());
		d += 2. * opensmlm::geometry::BasicComputation::distance(m_p2->x(), m_p2->y(), m_p0->x(), m_p0->y());
		return d;
	}

	float TriangleROI::getArea() const
	{
		float totalArea = 0;
		totalArea += ((m_p0->x() - m_p1->x()) * (m_p1->y() + (m_p0->y() - m_p1->y()) / 2));
		totalArea += ((m_p1->x() - m_p2->x()) * (m_p2->y() + (m_p1->y() - m_p2->y()) / 2));
		totalArea += ((m_p2->x() - m_p0->x()) * (m_p0->y() + (m_p2->y() - m_p0->y()) / 2));
		return abs(totalArea);
	}

	const std::string TriangleROI::statusBarMessage(Calibration* _cal) const
	{
		if (m_p0 == NULL) return std::string();
		//	float w = Geometry::distance( m_p0->x(), 0, m_tmpPoint->x(), 0 ), h = Geometry::distance( 0, m_p0->y(), 0, m_tmpPoint->y() );
		//	std::string s( "Rectangle: [" + std::string::number( w * _cal->getPixelXY() ) + " " + _cal->getDimensionUnit() + ", " + std::string::number( h * _cal->getPixelXY() ) + " " + _cal->getDimensionUnit() + "]" );
		std::string s("Triangle");
		return s;
	}

	void TriangleROI::save(std::ofstream& _fs) const
	{
		_fs << "TriangleROI" << std::endl;
		_fs << m_p0->x() << " " << m_p0->y() << " " << m_p1->x() << " " << m_p1->y() << " " << m_p2->x() << " " << m_p2->y() << std::endl;
	}

	void TriangleROI::load(std::ifstream& _fs)
	{
		std::string s;
		std::getline(_fs, s);
		std::istringstream is2(s);
		float x0, y0, x1, y1, x2, y2;
		is2 >> x0 >> y0 >> x1 >> y1 >> x2 >> y2;
		m_p0->set(x0, y0);
		m_p1->set(x1, y1);
		m_p2->set(x2, y2);
		m_center = new Vec2md((m_p0->x() + m_p1->x() + m_p2->x()) / 3., (m_p0->y() + m_p1->y() + m_p2->y()) / 3.);
	}

	ROI* TriangleROI::copy() const
	{
		return new TriangleROI(*this);
	}*/
}

