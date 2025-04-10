/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      HistogramInterface.hpp
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

#ifndef HistogramInterface_h__
#define HistogramInterface_h__

#include <vector>

namespace poca::core {

	class HistogramInterface {
	public:
		virtual ~HistogramInterface() = default;

		virtual const std::vector <float>& getBins() const = 0;
		virtual std::vector <float>& getBins() = 0;
		virtual const std::vector <float>& getTs() const = 0;
		virtual std::vector <float>& getTs() = 0;

		virtual const float getCurrentMin() const = 0;
		virtual void setCurrentMin(const float) = 0;
		virtual const float getCurrentMax() const = 0;
		virtual void setCurrentMax(const float) = 0;

		virtual const std::size_t getNbValues() const = 0;
		virtual const std::size_t getNbBins() const = 0;

		virtual const float getMean() const = 0;
		virtual const float getMedian() const = 0;
		virtual const float getStdDev() const = 0;
		virtual const float getMin() const = 0;
		virtual const float getMax() const = 0;
		virtual const float getStepX() const = 0;
		virtual const float getMaxY() const = 0;

		virtual void changeHistogramBounds(const float, const float) = 0;

		virtual void setHistogram(const bool, const int = 100, const bool = false, const float = 0., const bool = false, const float = 0.) = 0;
		virtual void setSelection(std::vector <bool>&) = 0;
		virtual void saveValues(std::ofstream&) const = 0;
		virtual HistogramInterface* computeLogHistogram() const = 0;
		virtual const size_t nbElements() const = 0;

		virtual void setInteraction(const bool) = 0;
		virtual bool hasInteraction() const = 0;
	};
}

#endif

