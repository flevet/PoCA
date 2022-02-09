/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectFeaturesFactory.hpp
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

#ifndef ObjectFeaturesFactory_h__
#define ObjectFeaturesFactory_h__

#include "../Interfaces/ObjectFeaturesFactoryInterface.hpp"

namespace poca::geometry {
	class ObjectFeaturesFactory :public ObjectFeaturesFactoryInterface {
	public:
		ObjectFeaturesFactory();
		~ObjectFeaturesFactory();

		virtual void computePCA(const poca::core::MyArrayUInt32&, const size_t, const float*, const float*, const float*, float *);
		virtual void computePCA(const uint32_t*, const size_t, const float*, const float*, const float*, float*);
		virtual size_t nbFeaturesPCA(const bool _is3D) const { return _is3D ? nbFeaturesPCAFor3D() : nbFeaturesPCAFor2D(); }

		virtual void computeBoundingEllipse(const poca::core::MyArrayUInt32&, const size_t, const float*, const float*, const float*, float*);
		virtual void computeBoundingEllipse(const uint32_t*, const size_t, const float*, const float*, const float*, float*);
		virtual size_t nbFeaturesBoundingEllipse(const bool _is3D) const { return _is3D ? nbFeaturesBoundingEllipseFor3D() : nbFeaturesBoundingEllipseFor2D(); }

	protected:	
		virtual void computePCA2D(const uint32_t*, const size_t, const float*, const float*, float*);
		virtual void computePCA3D(const uint32_t*, const size_t, const float*, const float*, const float*, float*);

		virtual size_t nbFeaturesPCAFor2D() const { return 10; }
		virtual size_t nbFeaturesPCAFor3D() const { return 15; }

		virtual size_t nbFeaturesBoundingEllipseFor2D() const { return 10; }
		virtual size_t nbFeaturesBoundingEllipseFor3D() const { return 15; }
	};
}

#endif

