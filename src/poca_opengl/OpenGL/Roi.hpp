/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Roi.hpp      Roi.hpp
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

#ifndef ROI_h__
#define ROI_h__

#include <list>
#include <vector>

#include <General/Vec2.hpp>
#include <Interfaces/ROIInterface.hpp>

#include "GLBuffer.hpp"

namespace poca::core {
	class ROI: public ROIInterface {
	public:
		virtual ~ROI();

		virtual void draw(poca::opengl::Camera*) = 0;
		virtual bool inside(const float, const float, const float = 0.f) const = 0;
		virtual void onClick(const float, const float, const bool = false) = 0;
		virtual void onMove(const float, const float, const bool = false) = 0;
		virtual void finalize(const float, const float, const bool = false) = 0;
		virtual float getFeature(const std::string&) const = 0;

		virtual void save(std::ofstream&) const = 0;
		virtual void load(std::ifstream&) = 0;
		virtual const std::string toStdString() const = 0;

		virtual ROIInterface* copy() const = 0;

		inline void setName(const std::string& _name) { m_name = _name; }
		inline const std::string& getName() const { return m_name; }
		inline void setType(const std::string& _type) { m_type = _type; }
		inline const std::string& getType() const { return m_type; }

	protected:
		ROI(const std::string&);
		ROI(const ROI&);

	protected:
		std::string m_name, m_type;
		bool m_changed;
	};

	class LineROI : public ROI {
	public:
		LineROI(const std::string&);
		LineROI(const LineROI&);
		~LineROI();

		virtual void updateDisplay();
		virtual void draw(poca::opengl::Camera*);
		virtual bool inside(const float, const float, const float = 0.f) const;
		virtual void onClick(const float, const float, const bool = false);
		virtual void onMove(const float, const float, const bool = false);
		virtual void finalize(const float, const float, const bool = false);
		virtual float getFeature(const std::string&) const;
		virtual void save(std::ofstream&) const;
		virtual void load(std::ifstream&);
		virtual const std::string toStdString() const;
		virtual ROIInterface* copy() const;

	protected:
		Vec2mf m_pts[2];
		poca::opengl::LineGLBuffer <Vec3mf> m_lineBuffer;
	};

	class CircleROI : public ROI {
	public:
		CircleROI(const std::string&);
		CircleROI(const CircleROI&);
		~CircleROI();

		virtual void updateDisplay();
		virtual void draw(poca::opengl::Camera*);
		virtual bool inside(const float, const float, const float = 0.f) const;
		virtual void onClick(const float, const float, const bool = false);
		virtual void onMove(const float, const float, const bool = false);
		virtual void finalize(const float, const float, const bool = false);
		virtual float getFeature(const std::string&) const;
		virtual void save(std::ofstream&) const;
		virtual void load(std::ifstream&);
		virtual const std::string toStdString() const;
		virtual ROIInterface* copy() const;

	protected:
		Vec2mf m_center;
		float m_radius;
		poca::opengl::PointGLBuffer <Vec3mf> m_centerBuffer;
	};

	/*class PolylineROI : public ROI {
	public:
		PolylineROI(const std::string&);
		PolylineROI(const PolylineROI&);
		~PolylineROI();

		virtual void draw(poca::opengl::Camera*) const;
		virtual bool inside(const float, const float, const float = 0.f) const;
		virtual ROI* insideModify(const float, const float);
		virtual void onClick(const float, const float, const bool = false);
		virtual void onMove(const float, const float, const bool = false);
		virtual void finalize(const float, const float, const bool = false);
		virtual float getPerimeter() const;
		virtual float getArea() const;
		virtual void save(std::ofstream&) const;
		virtual void load(std::ifstream&);
		void load(std::ifstream&, const unsigned int);
		virtual ROI* copy() const;

		inline const std::vector < Vec2md >& getPoints() const { return m_points; }
		inline const std::size_t nbPoints() const { return m_points.size(); }

	protected:
		std::vector < Vec2md > m_points;
		bool m_finalized, m_centerSelected;
	};

	class SquareROI : public ROI {
	public:
		SquareROI(const std::string&);
		SquareROI(const SquareROI&);
		~SquareROI();

		virtual void draw(poca::opengl::Camera*) const;
		virtual bool inside(const float, const float, const float = 0.f) const;
		virtual ROI* insideModify(const float, const float);
		virtual void onClick(const float, const float, const bool = false);
		virtual void onMove(const float, const float, const bool = false);
		virtual void finalize(const float, const float, const bool = false);
		virtual float getPerimeter() const;
		virtual float getArea() const;
		virtual void save(std::ofstream&) const;
		virtual void load(std::ifstream&);
		virtual ROI* copy() const;

		inline const float getWidth() const { return m_p1->x() - m_p0->x(); }
		inline const float getHeight() const { return m_p1->y() - m_p0->y(); }

	protected:
		Vec2md* m_p0, * m_p1;
	};

	class TriangleROI : public ROI {
	public:
		TriangleROI(const std::string&);
		TriangleROI(const TriangleROI&);
		~TriangleROI();

		virtual void draw(poca::opengl::Camera*) const;
		virtual bool inside(const float, const float, const float = 0.f) const;
		virtual ROI* insideModify(const float, const float);
		virtual void onClick(const float, const float, const bool = false);
		virtual void onMove(const float, const float, const bool = false);
		virtual void finalize(const float, const float, const bool = false);
		virtual float getPerimeter() const;
		virtual float getArea() const;
		virtual void save(std::ofstream&) const;
		virtual void load(std::ifstream&);
		virtual ROI* copy() const;

	protected:
		Vec2md* m_p0, * m_p1, * m_p2;
		bool m_changing;
	};*/
}

#endif // ROI_h__

