/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Roi.cpp
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

#include <Windows.h>
#include <gl/glew.h>
#include <gl/GL.h>
#include <QtCore/qmath.h>
#include <glm/gtx/string_cast.hpp>

#include <General/Vec3.hpp>
#include <General/Misc.h>

#include "Roi.hpp"
#include "../OpenGL/Camera.hpp"
#include "../OpenGL/Shader.hpp"

namespace poca::core {
	ROI::ROI(const std::string& _type) :m_type(_type), m_changed(false), m_selected(false)
	{
	}

	ROI::ROI(const ROI& _o) : m_name(_o.m_name), m_type(_o.m_type), m_changed(_o.m_changed)
	{
	}

	ROI::~ROI()
	{
	}

	ROIInterface* getROIFromType(const int _type)
	{
		if (_type == poca::opengl::Camera::Line2DRoiDefinition)
			return new LineROI("LineROI");
		else if (_type == poca::opengl::Camera::Circle2DRoiDefinition)
			return new CircleROI("CircleROI");
		else if (_type == poca::opengl::Camera::Square2DRoiDefinition)
			return new SquareROI("SquareROI");
		else if (_type == poca::opengl::Camera::Triangle2DRoiDefinition)
			return new TriangleROI("TriangleROI");
		else if (_type == poca::opengl::Camera::Polyline2DRoiDefinition)
			return new PolylineROI("PolylineROI");
		else if (_type == poca::opengl::Camera::Sphere3DRoiDefinition)
			return new SphereROI("SphereROI");
		return NULL;
	}

	ROIInterface* getROIFromType(const std::string& _type)
	{
		if (_type == "LineROI")
			return new LineROI("LineROI");
		else if (_type == "CircleROI")
			return new CircleROI("CircleROI");
		else if (_type == "PolygonROI")
			return new PolylineROI("PolylineROI");
		else if (_type == "SquareROI")
			return new SquareROI("SquareROI");
		else if (_type == "TriangleROI")
			return new TriangleROI("TriangleROI");
		return NULL;
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
		m_changed = false;
	}

	void LineROI::draw(poca::opengl::Camera* _cam, const std::array <float, 4>& _color, const float _thickness, const float _antialias)
	{
		if (m_changed)
			updateDisplay();

		glCullFace(GL_BACK);
		glDisable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		poca::opengl::Shader* shader = _cam->getShader("line2DShader");
		const glm::mat4& proj = _cam->getProjectionMatrix(), & view = _cam->getViewMatrix(), & model = _cam->getModelMatrix();
		shader->use();
		shader->setMat4("MVP", proj * view * model);
		shader->setUVec4("viewport", _cam->getViewport());
		shader->setVec2("resolution", _cam->width(), _cam->height());
		shader->setFloat("thickness", _thickness * 2.f);
		shader->setFloat("antialias", _antialias);
		shader->setVec4("singleColor", _color[0], _color[1], _color[2], _color[3]);
		shader->setBool("useSpecialColors", false);
		glEnableVertexAttribArray(0);
		m_lineBuffer.bindBuffer(0, 0);
		glDrawArrays(m_lineBuffer.getMode(), 0, m_lineBuffer.getSizeBuffers()[0]);
		glDisableVertexAttribArray(0);
		shader->release();
	}

	bool LineROI::inside(const float _x, const float _y, const float _z) const
	{
		return false;
	}

	void LineROI::onClick(const float _x, const float _y, const float _z, const bool _modify)
	{
		m_changed = true;
		m_pts[0].set(_x, _y);
		m_pts[1].set(_x, _y);
	}

	void LineROI::onMove(const float _x, const float _y, const float _z, const bool _modify)
	{
		m_changed = true;
		m_pts[1].set(_x, _y);
	}

	void LineROI::finalize(const float _x, const float _y, const float _z, const bool _modify)
	{
		m_changed = true;
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
		return new LineROI(*this);
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
		m_changed = false;
	}

	void CircleROI::draw(poca::opengl::Camera* _cam, const std::array <float, 4>& _color, const float _thickness, const float _antialias)
	{
		if (m_radius <= 0.f) return;

		if (m_changed)
			updateDisplay();

		glDisable(GL_DEPTH_TEST);
		glEnable(GL_POINT_SPRITE);
		glEnable(GL_PROGRAM_POINT_SIZE);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		poca::opengl::Shader* shader = _cam->getShader("circle2DShader");
		const glm::mat4& proj = _cam->getProjectionMatrix(), & view = _cam->getViewMatrix(), & model = _cam->getModelMatrix();
		shader->use();
		shader->setMat4("MVP", proj * view * model);
		shader->setUVec4("viewport", _cam->getViewport());
		shader->setFloat("radius", m_radius);
		shader->setFloat("thickness", _thickness);
		shader->setFloat("antialias", _antialias);
		shader->setBool("activatedAntialias", _cam->isAntialiasActivated());
		shader->setVec4("singleColor", _color[0], _color[1], _color[2], _color[3]);
		glEnableVertexAttribArray(0);
		m_centerBuffer.bindBuffer(0, 0);
		glDrawArrays(m_centerBuffer.getMode(), 0, m_centerBuffer.getSizeBuffers()[0]);
		glDisableVertexAttribArray(0);
		shader->release();
	}

	bool CircleROI::inside(const float _x, const float _y, const float _z) const
	{
		return (m_center.distance(Vec2mf(_x, _y)) <= m_radius);
	}

	void CircleROI::onClick(const float _x, const float _y, const float _z, const bool _modify)
	{
		m_changed = true;
		m_center.set(_x, _y);
	}

	void CircleROI::onMove(const float _x, const float _y, const float _z, const bool _modify)
	{
		m_changed = true;
		m_radius = m_center.distance(Vec2mf(_x, _y));
	}

	void CircleROI::finalize(const float _x, const float _y, const float _z, const bool _modify)
	{
		m_changed = true;
		m_radius = m_center.distance(Vec2mf(_x, _y));
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
		return new CircleROI(*this);
	}

	const std::string CircleROI::toStdString() const
	{
		std::string text("CircleROI, " + m_name + "\n" +
			"[" + std::to_string(m_center.x()) + ", " + std::to_string(m_center.y()) + "]" +
			", radius = " + std::to_string(m_radius));
		return text;
	}

	PolylineROI::PolylineROI(const std::string& _type) :ROI(_type)
	{

	}

	PolylineROI::PolylineROI(const PolylineROI& _o) : ROI(_o)
	{
		m_pts.resize(_o.m_pts.size());
		std::copy(_o.m_pts.begin(), _o.m_pts.end(), m_pts.begin());
	}

	PolylineROI::~PolylineROI()
	{
		m_pts.clear();
	}

	void PolylineROI::updateDisplay()
	{
		if (m_buffer.empty())
			m_buffer.generateBuffer(6, 3, GL_FLOAT);
		std::vector <poca::core::Vec3mf> pts;
		for (Vec2mf tmp : m_pts)
			pts.push_back(poca::core::Vec3mf(tmp.x(), tmp.y(), 0.f));
		if(pts.size() > 2)
			pts.push_back(poca::core::Vec3mf(m_pts[0].x(), m_pts[0].y(), 0.f));

		std::vector <uint32_t> indices(pts.size());
		std::iota(std::begin(indices), std::end(indices), 0);
		indices.insert(indices.begin(), indices[indices.size() - 1]);
		indices.push_back(indices[0]);

		m_buffer.updateBuffer(pts);
		m_buffer.updateIndices(indices);

		m_changed = false;
	}

	void PolylineROI::draw(poca::opengl::Camera* _cam, const std::array <float, 4>& _color, const float _thickness, const float _antialias)
	{
		if (m_changed)
			updateDisplay();

		glCullFace(GL_BACK);
		glDisable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		poca::opengl::Shader* shader = _cam->getShader("polyline2DShader");
		const glm::mat4& proj = _cam->getProjectionMatrix(), & view = _cam->getViewMatrix(), & model = _cam->getModelMatrix();
		shader->use();
		shader->setMat4("MVP", proj * view * model);
		shader->setMat4("projection", proj);
		shader->setUVec4("viewport", _cam->getViewport());
		shader->setVec2("resolution", _cam->width(), _cam->height());
		shader->setFloat("thickness", _thickness);
		shader->setFloat("antialias", _antialias);
		shader->setFloat("miter_limit", 1.f);
		shader->setVec4("singleColor", _color[0], _color[1], _color[2], _color[3]);
		glEnableVertexAttribArray(0);
		m_buffer.bindBuffer(0);
		if (m_buffer.getBufferIndices() != 0) {
			glDrawElements(m_buffer.getMode(), m_buffer.getNbIndices(), GL_UNSIGNED_INT, (void*)0);
		}
		else
			glDrawArrays(m_buffer.getMode(), 0, m_buffer.getNbElements());
		glDisableVertexAttribArray(0);
		shader->release();
	}

	bool PolylineROI::inside(const float _x, const float _y, const float _z) const
	{
		float epsilon = 0.00001;
		for (std::vector < Vec2mf >::const_iterator it = m_pts.begin(); it != m_pts.end(); it++) {
			float d = poca::geometry::distance(_x, _y, it->x(), it->y());
			if (d < epsilon)
				return true;
		}

		int cn = 0;// the  crossing number counter

		std::vector < Vec2mf >::const_iterator prec = m_pts.end();
		prec--;
		// loop through all edges of the polygon
		for (std::vector < Vec2mf >::const_iterator current = m_pts.begin(); current != m_pts.end(); current++) {
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

	void PolylineROI::onClick(const float _x, const float _y, const float _z, const bool _modify)
	{
		m_changed = true;
		m_pts.push_back(Vec2md(_x, _y));
	}

	void PolylineROI::onMove(const float _x, const float _y, const float _z, const bool _modify)
	{
		if (m_pts.size() < 3) return;
		m_changed = true;
		m_pts.back().set(_x, _y);
	}

	void PolylineROI::finalize(const float _x, const float _y, const float _z, const bool _modify)
	{
		if (m_pts.size() < 3) return;
		m_changed = true;
		m_pts.back().set(_x, _y);
	}

	float PolylineROI::getFeature(const std::string& _typeFeature) const
	{
		if (_typeFeature == "perimeter") {
			float perimeter = 0.;
			std::vector < Vec2mf >::const_iterator prec = m_pts.end();
			prec--;
			// loop through all edges of the polygon
			for (std::vector < Vec2mf >::const_iterator current = m_pts.begin(); current != m_pts.end(); current++) {
				perimeter += prec->distance(*current);
				prec = current;
			}
			return perimeter;
		}
		else if (_typeFeature == "area") {
			float totalArea = 0;
			std::vector < Vec2mf >::const_iterator prec = m_pts.end();
			prec--;
			// loop through all edges of the polygon
			for (std::vector < Vec2mf >::const_iterator current = m_pts.begin(); current != m_pts.end(); current++) {
				totalArea += ((prec->x() - current->x()) * (current->y() + (prec->y() - current->y()) / 2));
				prec = current;
			}
			return abs(totalArea);
		}
		return std::numeric_limits <float>::max();
	}

	void PolylineROI::save(std::ofstream& _fs) const
	{
		_fs << "PolygonROI" << std::endl;
		_fs << this->nbPoints() << std::endl;
		const std::vector < Vec2mf >& points = this->getPoints();
		for (std::vector < Vec2mf >::const_iterator it = points.begin(); it != points.end(); it++)
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

	const std::string PolylineROI::toStdString() const
	{
		std::string text("PolylineROI, " + m_name + "\n# points = " + std::to_string(this->nbPoints()) + "\n");
		for (std::vector < Vec2mf >::const_iterator it = m_pts.begin(); it != m_pts.end(); it++)
			text.append("[" + std::to_string(it->x()) + ", " + std::to_string(it->y()) + "]\n");
		return text;
	}

	ROIInterface* PolylineROI::copy() const
	{
		return new PolylineROI(*this);
	}

	SquareROI::SquareROI(const std::string& _type) :ROI(_type)
	{
	}

	SquareROI::SquareROI(const SquareROI& _o) : ROI(_o)
	{
		std::copy(&_o.m_pts[0], &_o.m_pts[0] + 1, &m_pts[0]);
	}

	SquareROI::~SquareROI()
	{
		m_buffer.freeGPUMemory();
	}

	void SquareROI::updateDisplay()
	{
		if (m_buffer.empty())
			m_buffer.generateBuffer(6, 3, GL_FLOAT);
		std::vector <poca::core::Vec3mf> pts;
		pts.push_back(poca::core::Vec3mf(m_pts[0].x(), m_pts[0].y(), 0.f));
		pts.push_back(poca::core::Vec3mf(m_pts[0].x(), m_pts[1].y(), 0.f));
		pts.push_back(poca::core::Vec3mf(m_pts[1].x(), m_pts[1].y(), 0.f));
		pts.push_back(poca::core::Vec3mf(m_pts[1].x(), m_pts[0].y(), 0.f));
		pts.push_back(poca::core::Vec3mf(m_pts[0].x(), m_pts[0].y(), 0.f));

		std::vector <uint32_t> indices(pts.size());
		std::iota(std::begin(indices), std::end(indices), 0);
		indices.insert(indices.begin(), indices[0]);
		indices.push_back(indices[indices.size() - 1]);

		m_buffer.updateBuffer(pts);
		m_buffer.updateIndices(indices);

		m_changed = false;
	}

	void SquareROI::draw(poca::opengl::Camera* _cam, const std::array <float, 4>& _color, const float _thickness, const float _antialias)
	{
		if (m_changed)
			updateDisplay();

		glCullFace(GL_BACK); 
		glDisable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		poca::opengl::Shader* shader = _cam->getShader("polyline2DShader");
		const glm::mat4& proj = _cam->getProjectionMatrix(), & view = _cam->getViewMatrix(), & model = _cam->getModelMatrix();
		shader->use();
		shader->setMat4("MVP", proj * view * model);
		shader->setMat4("projection", proj);
		shader->setUVec4("viewport", _cam->getViewport());
		shader->setVec2("resolution", _cam->width(), _cam->height());
		shader->setFloat("thickness", _thickness);
		shader->setFloat("antialias", _antialias);
		shader->setFloat("miter_limit", 1.f);
		shader->setVec4("singleColor", _color[0], _color[1], _color[2], _color[3]);
		glEnableVertexAttribArray(0);
		m_buffer.bindBuffer(0);
		if (m_buffer.getBufferIndices() != 0) {
			glDrawElements(m_buffer.getMode(), m_buffer.getNbIndices(), GL_UNSIGNED_INT, (void*)0);
		}
		else
			glDrawArrays(m_buffer.getMode(), 0, m_buffer.getNbElements());
		glDisableVertexAttribArray(0);
		shader->release();
	}

	bool SquareROI::inside(const float _x, const float _y, const float _z) const
	{
		float xmin = (m_pts[0].x() < m_pts[1].x()) ? m_pts[0].x() : m_pts[1].x();
		float xmax = (m_pts[0].x() > m_pts[1].x()) ? m_pts[0].x() : m_pts[1].x();
		float ymin = (m_pts[0].y() < m_pts[1].y()) ? m_pts[0].y() : m_pts[1].y();
		float ymax = (m_pts[0].y() > m_pts[1].y()) ? m_pts[0].y() : m_pts[1].y();

		return (xmin <= _x && _x <= xmax && ymin <= _y && _y <= ymax);
	}

	void SquareROI::onClick(const float _x, const float _y, const float _z, const bool _modify)
	{
		m_changed = true;
		m_pts[0].set(_x, _y);
		m_pts[1].set(_x, _y);
	}

	void SquareROI::onMove(const float _x, const float _y, const float _z, const bool _modify)
	{
		m_changed = true;
		m_pts[1].set(_x, _y);
	}

	void SquareROI::finalize(const float _x, const float _y, const float _z, const bool _modify)
	{
		m_changed = true;
		m_pts[1].set(_x, _y);
	}

	float SquareROI::getFeature(const std::string& _typeFeature) const
	{
		if (_typeFeature == "perimeter") {
			float d = 0;
			d += 2. * poca::geometry::distance(m_pts[0].x(), m_pts[0].y(), m_pts[0].x(), m_pts[1].y());
			d += 2. * poca::geometry::distance(m_pts[0].x(), m_pts[1].y(), m_pts[1].x(), m_pts[1].y());
			return d;
		}
		else if (_typeFeature == "area")
			return poca::geometry::distance(m_pts[0].x(), m_pts[0].y(), m_pts[0].x(), m_pts[1].y()) * poca::geometry::distance(m_pts[0].x(), m_pts[1].y(), m_pts[1].x(), m_pts[1].y());
		return std::numeric_limits <float>::max();
	}

	void SquareROI::save(std::ofstream& _fs) const
	{
		_fs << "SquareROI" << std::endl;
		_fs << m_pts[0].x() << " " << m_pts[0].y() << " " << m_pts[1].x() << " " << m_pts[1].y() << std::endl;
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

	const std::string SquareROI::toStdString() const
	{
		std::string text("SquareROI, " + m_name + "\n" +
			"[" + std::to_string(m_pts[0].x()) + ", " + std::to_string(m_pts[0].y()) + "]" +
			"[" + std::to_string(m_pts[1].x()) + ", " + std::to_string(m_pts[1].y()) + "]");
		return text;
	}

	ROIInterface* SquareROI::copy() const
	{
		return new SquareROI(*this);
	}


	TriangleROI::TriangleROI(const std::string& _type) :ROI(_type)
	{
	}

	TriangleROI::TriangleROI(const TriangleROI& _o) : ROI(_o)
	{
		std::copy(&_o.m_pts[0], &_o.m_pts[0] + 2, &m_pts[0]);
	}

	TriangleROI::~TriangleROI()
	{
	}

	void TriangleROI::updateDisplay()
	{
		if (m_buffer.empty())
			m_buffer.generateBuffer(6, 3, GL_FLOAT);
		std::vector <poca::core::Vec3mf> pts;
		for(Vec2mf tmp : m_pts)
			pts.push_back(poca::core::Vec3mf(tmp.x(), tmp.y(), 0.f));
		pts.push_back(poca::core::Vec3mf(m_pts[0].x(), m_pts[0].y(), 0.f));

		std::vector <uint32_t> indices(pts.size());
		std::iota(std::begin(indices), std::end(indices), 0);
		indices.insert(indices.begin(), indices[0]);
		indices.push_back(indices[indices.size() - 1]);

		m_buffer.updateBuffer(pts);
		m_buffer.updateIndices(indices);

		m_changed = false;
	}

	void TriangleROI::draw(poca::opengl::Camera* _cam, const std::array <float, 4>& _color, const float _thickness, const float _antialias)
	{
		if (m_changed)
			updateDisplay();

		glCullFace(GL_BACK);
		glDisable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		poca::opengl::Shader* shader = _cam->getShader("polyline2DShader");
		const glm::mat4& proj = _cam->getProjectionMatrix(), & view = _cam->getViewMatrix(), & model = _cam->getModelMatrix();
		shader->use();
		shader->setMat4("MVP", proj * view * model);
		shader->setMat4("projection", proj);
		shader->setUVec4("viewport", _cam->getViewport());
		shader->setVec2("resolution", _cam->width(), _cam->height());
		shader->setFloat("thickness", _thickness);
		shader->setFloat("antialias", _antialias);
		shader->setFloat("miter_limit", 1.f);
		shader->setVec4("singleColor", _color[0], _color[1], _color[2], _color[3]);
		glEnableVertexAttribArray(0);
		m_buffer.bindBuffer(0);
		if (m_buffer.getBufferIndices() != 0) {
			glDrawElements(m_buffer.getMode(), m_buffer.getNbIndices(), GL_UNSIGNED_INT, (void*)0);
		}
		else
			glDrawArrays(m_buffer.getMode(), 0, m_buffer.getNbElements());
		glDisableVertexAttribArray(0);
		shader->release();
	}

	bool TriangleROI::inside(const float _x, const float _y, const float _z) const
	{
		float epsilon = 0.00001;
		float d = poca::geometry::distance(_x, _y, m_pts[0].x(), m_pts[0].y());
		if (d < epsilon) return true;
		d = poca::geometry::distance(_x, _y, m_pts[1].x(), m_pts[1].y());
		if (d < epsilon) return true;
		d = poca::geometry::distance(_x, _y, m_pts[2].x(), m_pts[2].y());
		if (d < epsilon) return true;

		int cn = 0;// the  crossing number counter

		// loop through all edges of the triangles
		Vec2mf prec, current;
		for (unsigned int n = 0; n < 3; n++) {
			if (n == 0) { prec = m_pts[2]; current = m_pts[0]; }
			if (n == 1) { prec = m_pts[0]; current = m_pts[1]; }
			if (n == 2) { prec = m_pts[1]; current = m_pts[2]; }
			if (((prec.y() <= _y) && (current.y() > _y)) || ((prec.y() > _y) && (current.y() <= _y))) { // first upward test crossing then downward test crossing
					// compute  the actual edge-ray intersect x-coordinate
				float vt = (float)(_y - prec.y()) / (current.y() - prec.y());
				if (_x < prec.x() + vt * (current.x() - prec.x())) // P.x < intersect
					++cn;   // a valid crossing of y=P.y right of P.x
			}
		}
		return (cn & 1) == 1;    // 0 if even (out), and 1 if  odd (in)
	}

	void TriangleROI::onClick(const float _x, const float _y, const float _z, const bool _modify)
	{
		m_changed = true;
		std::fill(&m_pts[0], &m_pts[0] + 3, Vec2md(_x, _y));
	}

	void TriangleROI::onMove(const float _x, const float _y, const float _z, const bool _modify)
	{
		m_changed = true;
		Vec2mf tmpPt(_x, _y), vect(tmpPt - m_pts[0]);
		float d = vect.length();
		vect.normalize();
		d /= (sqrt(3.f) / 2.f);
		d /= 2.f;
		m_pts[1].set(_x + d * -vect[1], _y + d * vect[0]);
		m_pts[2].set(_x - d * -vect[1], _y - d * vect[0]);
	}

	void TriangleROI::finalize(const float _x, const float _y, const float _z, const bool _modify)
	{
		m_changed = true;
		Vec2mf tmpPt(_x, _y), vect(tmpPt - m_pts[0]);
		float d = vect.length();
		vect.normalize();
		d /= (sqrt(3.f) / 2.f);
		d /= 2.f;
		m_pts[1].set(_x + d * -vect[1], _y + d * vect[0]);
		m_pts[2].set(_x - d * -vect[1], _y - d * vect[0]);
	}

	float TriangleROI::getFeature(const std::string& _typeFeature) const
	{
		if (_typeFeature == "perimeter") {
			float d = 0;
			d += 2. * poca::geometry::distance(m_pts[0].x(), m_pts[0].y(), m_pts[1].x(), m_pts[1].y());
			d += 2. * poca::geometry::distance(m_pts[1].x(), m_pts[1].y(), m_pts[2].x(), m_pts[2].y());
			d += 2. * poca::geometry::distance(m_pts[2].x(), m_pts[2].y(), m_pts[0].x(), m_pts[0].y());
			return d;
		}
		else if (_typeFeature == "area") {
			float totalArea = 0;
			totalArea += ((m_pts[0].x() - m_pts[1].x()) * (m_pts[1].y() + (m_pts[0].y() - m_pts[1].y()) / 2));
			totalArea += ((m_pts[1].x() - m_pts[2].x()) * (m_pts[2].y() + (m_pts[1].y() - m_pts[2].y()) / 2));
			totalArea += ((m_pts[2].x() - m_pts[0].x()) * (m_pts[0].y() + (m_pts[2].y() - m_pts[0].y()) / 2));
			return totalArea;
		}
		return std::numeric_limits <float>::max();
	}

	void TriangleROI::save(std::ofstream& _fs) const
	{
		_fs << "TriangleROI" << std::endl;
		_fs << m_pts[0].x() << " " << m_pts[0].y() << " " << m_pts[1].x() << " " << m_pts[1].y() << " " << m_pts[2].x() << " " << m_pts[2].y() << std::endl;
	}

	void TriangleROI::load(std::ifstream& _fs)
	{
		std::string s;
		std::getline(_fs, s);
		std::istringstream is2(s);
		float x0, y0, x1, y1, x2, y2;
		is2 >> x0 >> y0 >> x1 >> y1 >> x2 >> y2;
		m_pts[0].set(x0, y0);
		m_pts[1].set(x1, y1);
		m_pts[2].set(x2, y2);
	}

	const std::string TriangleROI::toStdString() const
	{
		std::string text("TriangleROI, " + m_name + "\n" +
			"[" + std::to_string(m_pts[0].x()) + ", " + std::to_string(m_pts[0].y()) + "]" +
			"[" + std::to_string(m_pts[1].x()) + ", " + std::to_string(m_pts[1].y()) + "]" +
			"[" + std::to_string(m_pts[2].x()) + ", " + std::to_string(m_pts[2].y()) + "]");
		return text;
	}

	ROIInterface* TriangleROI::copy() const
	{
		return new TriangleROI(*this);
	}

	SphereROI::SphereROI(const std::string& _type) :ROI(_type), m_radius(0.f)
	{
	}

	SphereROI::SphereROI(const SphereROI& _o) : ROI(_o), m_radius(_o.m_radius)
	{
	}

	SphereROI::~SphereROI()
	{
	}

	void SphereROI::updateDisplay()
	{
		if (m_centerBuffer.empty()) {
			m_centerBuffer.generateBuffer(1, 3, GL_FLOAT);
			m_radiusBuffer.generateBuffer(1, 1, GL_FLOAT);
		}
		std::vector <poca::core::Vec3mf> pt = { poca::core::Vec3mf(m_center.x(), m_center.y(), m_center.z()) };
		m_centerBuffer.updateBuffer(pt);
		std::vector <float> rad = { m_radius };
		m_radiusBuffer.updateBuffer(rad);
		m_changed = false;
	}

	void SphereROI::draw(poca::opengl::Camera* _cam, const std::array <float, 4>& _color, const float _thickness, const float _antialias)
	{
		if (m_radius <= 0.f) return;

		if (m_changed)
			updateDisplay();

		glEnable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		glDisable(GL_CULL_FACE);
		poca::opengl::Shader* shader = _cam->getShader("sphere3DShader");
		const poca::core::BoundingBox& crop = _cam->getCurrentCrop();
		const glm::mat4& proj = _cam->getProjectionMatrix(), & view = _cam->getViewMatrix(), & model = _cam->getModelMatrix();
		shader->use();
		shader->setMat4("MVP", proj * view * model);
		shader->setMat4("projection", proj);
		shader->setMat4("view", view);
		shader->setVec4("singleColor", _color[0], _color[1], _color[2], _color[3]);
		shader->setVec4("clipPlaneX", _cam->getClipPlaneX());
		shader->setVec4("clipPlaneY", _cam->getClipPlaneY());
		shader->setVec4("clipPlaneZ", _cam->getClipPlaneZ());
		shader->setVec4("clipPlaneW", _cam->getClipPlaneW());
		shader->setVec4("clipPlaneH", _cam->getClipPlaneH());
		shader->setVec4("clipPlaneT", _cam->getClipPlaneT());

		glm::vec3 lightColor(1., 1., 1.);
		const float linear = 0.09;
		const float quadratic = 0.032;
		shader->setVec3("light.Position", _cam->getCenter());
		shader->setVec3("light.Color", lightColor);
		shader->setFloat("light.Linear", linear);
		shader->setFloat("light.Quadratic", quadratic);
		shader->setVec3("bboxPt1", crop[0], crop[1], crop[2]);
		shader->setVec3("bboxPt2", crop[3], crop[4], crop[5]);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		m_centerBuffer.bindBuffer(0);
		m_radiusBuffer.bindBuffer(1);
		glDrawArrays(m_centerBuffer.getMode(), 0, m_centerBuffer.getNbElements());
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		shader->release();
	}

	bool SphereROI::inside(const float _x, const float _y, const float _z) const
	{
		return (m_center.distance(Vec3mf(_x, _y, _z)) <= m_radius);
	}

	void SphereROI::onClick(const float _x, const float _y, const float _z, const bool _modify)
	{
		m_changed = true;
		m_center.set(_x, _y, _z);
	}

	void SphereROI::onMove(const float _x, const float _y, const float _z, const bool _modify)
	{
		m_changed = true;
		m_radius = m_center.distance(Vec3mf(_x, _y, _z));
	}

	void SphereROI::finalize(const float _x, const float _y, const float _z, const bool _modify)
	{
		m_changed = true;
		m_radius = m_center.distance(Vec3mf(_x, _y, _z));
	}

	float SphereROI::getFeature(const std::string& _typeFeature) const
	{
		if (_typeFeature == "perimeter")
			return 2. * M_PI * m_radius;
		else if (_typeFeature == "area")
			return M_PI * m_radius * m_radius;
		return std::numeric_limits <float>::max();
	}

	void SphereROI::save(std::ofstream& _fs) const
	{
		_fs << "CircleROI" << std::endl;
		_fs << m_center.x() << " " << m_center.y() << " " << m_radius << std::endl;
	}

	void SphereROI::load(std::ifstream& _fs)
	{
		std::string s;
		std::getline(_fs, s);
		std::istringstream is2(s);
		float x0, y0, radius;
		is2 >> x0 >> y0 >> radius;
		this->onClick(x0, y0);
		this->finalize(x0 + radius, y0);
	}

	ROIInterface* SphereROI::copy() const
	{
		return new SphereROI(*this);
	}

	const std::string SphereROI::toStdString() const
	{
		std::string text("SphereROI, " + m_name + "\n" +
			"[" + std::to_string(m_center.x()) + ", " + std::to_string(m_center.y()) + ", " + std::to_string(m_center.z()) + "]" +
			", radius = " + std::to_string(m_radius));
		return text;
	}

}

