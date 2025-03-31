/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ROIInterface.hpp
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

#ifndef ROIInterface_h__
#define ROIInterface_h__

#include <string>
#include <fstream>

namespace poca::opengl {
	class Camera;
}

namespace poca::core {
	class ROIInterface {
	public:
		virtual ~ROIInterface() = default;

		virtual void draw(poca::opengl::Camera*, 
			const std::array <float, 4>&,
			const float = 5.f,
			const float = 1.f) = 0;

		virtual bool inside(const float, const float, const float = 0.f) const = 0;
		virtual void onClick(const float, const float, const float = 0.f, const bool = false) = 0;
		virtual void onMove(const float, const float, const float = 0.f, const bool = false) = 0;
		virtual void finalize(const float, const float, const float = 0.f, const bool = false) = 0;
		virtual float getFeature(const std::string&) const = 0;
		virtual void applyCalibrationXY(const float = 1.f) = 0;

		virtual void setName(const std::string& _name) = 0;
		virtual const std::string& getName() const = 0;
		virtual void setType(const std::string& _type) = 0;
		virtual const std::string& getType() const = 0;

		virtual void load(const std::vector<std::array<float, 2>>&) = 0;

		virtual void save(std::ofstream&) const = 0;
		virtual void load(std::ifstream&) = 0;
		virtual const std::string toStdString() const = 0;

		virtual ROIInterface* copy() const = 0;

		virtual const bool selected() const = 0;
		virtual void setSelected(const bool) = 0;
	};

	ROIInterface* getROIFromType(const int);
	ROIInterface* getROIFromType(const std::string&);
}
#endif

