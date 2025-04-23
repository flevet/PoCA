/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Roi.hpp      Roi.hpp
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

#ifndef ROI_h__
#define ROI_h__

#include <list>
#include <vector>

#include <General/Vec2.hpp>
#include <General/Vec3.hpp>
#include <General/Vec6.hpp>
#include <Interfaces/ROIInterface.hpp>

#include "../OpenGL/GLBuffer.hpp"

namespace poca::core {
	class ROI: public ROIInterface {
	public:
		virtual ~ROI();

		virtual void draw(poca::opengl::Camera*, const std::array <float, 4>&, const float = 5.f, const float = 1.f) = 0;
		virtual bool inside(const float, const float, const float = 0.f) const = 0;
		virtual void onClick(const float, const float, const float = 0.f, const bool = false) = 0;
		virtual void onMove(const float, const float, const float = 0.f, const bool = false) = 0;
		virtual void finalize(const float, const float, const float = 0.f, const bool = false) = 0;
		virtual float getFeature(const std::string&) const = 0;
		virtual const BoundingBox boundingBox() const = 0;

		virtual void save(std::ofstream&) const = 0;
		virtual void load(std::ifstream&) = 0;
		virtual const std::string toStdString() const = 0;

		virtual ROIInterface* copy() const = 0;

		void load(const std::vector<std::array<float, 2>>&);

		void applyCalibrationXY(const float = 1.f){}

		inline void setName(const std::string& _name) { m_name = _name; }
		inline const std::string& getName() const { return m_name; }
		inline void setType(const std::string& _type) { m_type = _type; }
		inline const std::string& getType() const { return m_type; }

		const bool selected() const { return m_selected; }
		void setSelected(const bool _selected) { m_selected = _selected; }

	protected:
		ROI(const std::string&);
		ROI(const ROI&);

	protected:
		std::string m_name, m_type;
		bool m_changed, m_selected;
	};

	class LineROI : public ROI {
	public:
		LineROI(const std::string&);
		LineROI(const LineROI&);
		~LineROI();

		virtual void updateDisplay();
		virtual void draw(poca::opengl::Camera*, const std::array <float, 4>&, const float = 5.f, const float = 1.f);
		virtual bool inside(const float, const float, const float = 0.f) const;
		virtual void onClick(const float, const float, const float = 0.f, const bool = false);
		virtual void onMove(const float, const float, const float = 0.f, const bool = false);
		virtual void finalize(const float, const float, const float = 0.f, const bool = false);
		virtual float getFeature(const std::string&) const;
		virtual void save(std::ofstream&) const;
		virtual void load(std::ifstream&);
		virtual const std::string toStdString() const;
		virtual ROIInterface* copy() const;
		virtual const BoundingBox boundingBox() const;

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
		virtual void draw(poca::opengl::Camera*, const std::array <float, 4>&, const float = 5.f, const float = 1.f);
		virtual bool inside(const float, const float, const float = 0.f) const;
		virtual void onClick(const float, const float, const float = 0.f, const bool = false);
		virtual void onMove(const float, const float, const float = 0.f, const bool = false);
		virtual void finalize(const float, const float, const float = 0.f, const bool = false);
		virtual float getFeature(const std::string&) const;
		virtual void save(std::ofstream&) const;
		virtual void load(std::ifstream&);
		virtual const std::string toStdString() const;
		virtual ROIInterface* copy() const;
		virtual const BoundingBox boundingBox() const;

		inline const Vec2mf& getCenter() const { return m_center; }
		inline const float& getRadius() const { return m_radius; }

	protected:
		Vec2mf m_center;
		float m_radius;
		poca::opengl::PointSingleGLBuffer <Vec3mf> m_centerBuffer;
	};

	class PolylineROI : public ROI {
	public:
		PolylineROI(const std::string&);
		PolylineROI(const PolylineROI&);
		~PolylineROI();

		virtual void updateDisplay();
		virtual void draw(poca::opengl::Camera*, const std::array <float, 4>&, const float = 5.f, const float = 1.f);
		virtual bool inside(const float, const float, const float = 0.f) const;
		virtual void onClick(const float, const float, const float = 0.f, const bool = false);
		virtual void onMove(const float, const float, const float = 0.f, const bool = false);
		virtual void finalize(const float, const float, const float = 0.f, const bool = false);
		virtual float getFeature(const std::string&) const;
		void applyCalibrationXY(const float = 1.f);
		virtual void save(std::ofstream&) const;
		virtual void load(std::ifstream&);
		void load(std::ifstream&, const unsigned int);
		virtual const std::string toStdString() const;
		virtual ROIInterface* copy() const;
		virtual const BoundingBox boundingBox() const;

		inline const std::vector < Vec2mf >& getPoints() const { return m_pts; }
		inline const std::size_t nbPoints() const { return m_pts.size(); }

	protected:
		std::vector < Vec2mf > m_pts;
		poca::opengl::LineStripAdjacencySingleGLBuffer <Vec3mf> m_buffer;
	};

	class SquareROI : public ROI {
	public:
		SquareROI(const std::string&);
		SquareROI(const SquareROI&);
		~SquareROI();

		virtual void updateDisplay();
		virtual void draw(poca::opengl::Camera*, const std::array <float, 4>&, const float = 5.f, const float = 1.f);
		virtual bool inside(const float, const float, const float = 0.f) const;
		virtual void onClick(const float, const float, const float = 0.f, const bool = false);
		virtual void onMove(const float, const float, const float = 0.f, const bool = false);
		virtual void finalize(const float, const float, const float = 0.f, const bool = false);
		virtual float getFeature(const std::string&) const;
		virtual void save(std::ofstream&) const;
		virtual void load(std::ifstream&);
		virtual const std::string toStdString() const;
		virtual ROIInterface* copy() const;
		virtual const BoundingBox boundingBox() const;

		inline const float getWidth() const { return m_pts[0].x() - m_pts[1].x(); }
		inline const float getHeight() const { return m_pts[0].y() - m_pts[1].y(); }

	protected:
		Vec2mf m_pts[2];
		poca::opengl::LineStripAdjacencySingleGLBuffer <Vec3mf> m_buffer;
	};

	class TriangleROI : public ROI {
	public:
		TriangleROI(const std::string&);
		TriangleROI(const TriangleROI&);
		~TriangleROI();

		virtual void updateDisplay();
		virtual void draw(poca::opengl::Camera*, const std::array <float, 4>&, const float = 5.f, const float = 1.f);
		virtual bool inside(const float, const float, const float = 0.f) const;
		virtual void onClick(const float, const float, const float = 0.f, const bool = false);
		virtual void onMove(const float, const float, const float = 0.f, const bool = false);
		virtual void finalize(const float, const float, const float = 0.f, const bool = false);
		virtual float getFeature(const std::string&) const;
		virtual void save(std::ofstream&) const;
		virtual void load(std::ifstream&);
		virtual const std::string toStdString() const;
		virtual ROIInterface* copy() const;
		virtual const BoundingBox boundingBox() const;

	protected:
		Vec2mf m_pts[3];
		poca::opengl::LineStripAdjacencySingleGLBuffer <Vec3mf> m_buffer;
	};

	class SphereROI : public ROI {
	public:
		SphereROI(const std::string&);
		SphereROI(const SphereROI&);
		~SphereROI();

		virtual void updateDisplay();
		virtual void draw(poca::opengl::Camera*, const std::array <float, 4>&, const float = 5.f, const float = 1.f);
		virtual bool inside(const float, const float, const float = 0.f) const;
		virtual void onClick(const float, const float, const float = 0.f, const bool = false);
		virtual void onMove(const float, const float, const float = 0.f, const bool = false);
		virtual void finalize(const float, const float, const float = 0.f, const bool = false);
		virtual float getFeature(const std::string&) const;
		virtual void save(std::ofstream&) const;
		virtual void load(std::ifstream&);
		virtual const std::string toStdString() const;
		virtual ROIInterface* copy() const;
		virtual const BoundingBox boundingBox() const;

		inline const Vec3mf& getCenter() const { return m_center; }
		inline const float& getRadius() const { return m_radius; }

	protected:
		Vec3mf m_center;
		float m_radius;
		poca::opengl::PointSingleGLBuffer <Vec3mf> m_centerBuffer;
		poca::opengl::FeatureSingleGLBuffer <float> m_radiusBuffer;
	};

	class PlaneROI : public ROI {
	public:
		PlaneROI(const std::string&);
		PlaneROI(const PlaneROI&);
		~PlaneROI();

		virtual void updateDisplay();
		virtual void draw(poca::opengl::Camera*, const std::array <float, 4>&, const float = 5.f, const float = 1.f);
		virtual bool inside(const float, const float, const float = 0.f) const;
		virtual void onClick(const float, const float, const float = 0.f, const bool = false);
		virtual void onMove(const float, const float, const float = 0.f, const bool = false);
		virtual void finalize(const float, const float, const float = 0.f, const bool = false);
		virtual float getFeature(const std::string&) const;
		virtual void save(std::ofstream&) const;
		virtual void load(std::ifstream&);
		virtual const std::string toStdString() const;
		virtual ROIInterface* copy() const;
		virtual const BoundingBox boundingBox() const;

		void addFinalPoint(const Vec3mf&);

		inline const Vec3mf& getPoint(const int _id) const { return m_pts[_id]; }
		inline const Vec3mf& getFinalPoint(const int _id) const { return m_finalPoints[_id]; }

	protected:
		Vec3mf m_pts[2];
		std::vector <Vec3mf> m_finalPoints;
		poca::opengl::LineSingleGLBuffer <Vec3mf> m_buffer;
		poca::opengl::LineGLBuffer <Vec3mf> m_lineBuffer;
		poca::opengl::TriangleSingleGLBuffer<Vec3mf> m_triangleBuffer;
	};

	class PolyplaneROI : public ROI {
	public:
		PolyplaneROI(const std::string&);
		PolyplaneROI(const PolyplaneROI&);
		~PolyplaneROI();

		virtual void updateDisplay();
		virtual void draw(poca::opengl::Camera*, const std::array <float, 4>&, const float = 5.f, const float = 1.f);
		virtual bool inside(const float, const float, const float = 0.f) const;
		virtual void onClick(const float, const float, const float = 0.f, const bool = false);
		virtual void onMove(const float, const float, const float = 0.f, const bool = false);
		virtual void finalize(const float, const float, const float = 0.f, const bool = false);
		virtual float getFeature(const std::string&) const;
		virtual void save(std::ofstream&) const;
		virtual void load(std::ifstream&);
		void load(std::ifstream&, const unsigned int);
		virtual const std::string toStdString() const;
		virtual ROIInterface* copy() const;
		virtual const BoundingBox boundingBox() const;

		void addFinalPoint(const Vec3mf&);

		inline const std::vector < Vec3mf >& getPoints() const { return m_pts; }
		inline const std::vector < Vec3mf >& getFinalPoints() const { return m_finalPoints; }
		inline const Vec3mf& getFinalPoint(const int _id) const { return m_finalPoints[_id]; }
		inline const std::size_t nbPoints() const { return m_pts.size(); }
		inline const std::size_t nbFinalPoints() const { return m_finalPoints.size(); }

	protected:
		std::vector < Vec3mf > m_pts;
		Vec3mf m_addedPointRendering;
		std::vector <Vec3mf> m_finalPoints;
		poca::opengl::LineStripAdjacencySingleGLBuffer <Vec3mf> m_buffer;
		poca::opengl::LineSingleGLBuffer <Vec3mf> m_outlineBuffer;
		poca::opengl::TriangleSingleGLBuffer<Vec3mf> m_triangleBuffer;
	};
}

#endif // ROI_h__

