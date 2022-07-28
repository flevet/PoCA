/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MyArray.hpp
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

#ifndef MyArray_h__
#define MyArray_h__

#include <string>
#include <vector>

#include "../General/Vec2.hpp"
#include "../General/Vec3.hpp"

namespace poca::core {

	template< class T >
	class MyArray {
	public:
		MyArray();
		MyArray(const std::vector <T>&, const std::vector <uint32_t>&);
		MyArray(const std::vector<std::vector <T>>&);
		MyArray(const MyArray < T >&);
		~MyArray();

		void initialize(const std::vector <T>&, const std::vector <uint32_t>&);
		void initialize(const std::vector<std::vector <T>>&);
		const std::string toString() const;

		const uint32_t memorySize() const;

		void clear();
		const bool empty() const;

		inline const size_t nbElements() const { return m_nbElements; }
		inline const size_t nbData() const { return m_nbData; }
		inline T* elementsObject(const size_t _index) { return &m_data[m_firstElementObjects[_index]]; }
		inline const T* elementsObject(const size_t _index) const { return &m_data[m_firstElementObjects[_index]]; }
		inline const T& elementIObject(const size_t _indexObject, const size_t _indexPoly) const { return m_data[m_firstElementObjects[_indexObject] + _indexPoly]; }
		inline const uint32_t nbElementsObject(const size_t _index) const { return m_firstElementObjects[_index + 1] - m_firstElementObjects[_index]; }

		inline T* allElements() { return m_data.data(); }
		inline const T* allElements() const { return m_data.data(); }
		inline uint32_t* allFirstElements() { return m_firstElementObjects.data(); }
		inline unsigned const int* allFirstElements() const { return m_firstElementObjects.data(); }

		inline const std::vector <T>& getData() const { return m_data; }
		inline const std::vector <uint32_t>& getFirstElements() const { return m_firstElementObjects; }

	protected:
		std::vector <T> m_data;
		std::vector <uint32_t> m_firstElementObjects;
		size_t m_nbElements, m_nbData;
	};

	template< class T >
	MyArray<T>::MyArray() :m_nbElements(0), m_nbData(0)
	{
	}

	template< class T >
	MyArray<T>::MyArray(const std::vector <T>& _data, const std::vector <uint32_t>& _firsts)
	{
		initialize(_data, _firsts);
	}

	template< class T >
	MyArray<T>::MyArray(const std::vector<std::vector <T>>& _data)
	{
		initialize(_data);
	}

	template< class T >
	MyArray<T>::MyArray(const MyArray < T >& _o) :m_data(_o.m_data), m_firstElementObjects(_o.m_firstElementObjects), m_nbElements(_o.m_nbElements), m_nbData(_o.m_nbData)
	{
	}

	template< class T >
	MyArray<T>::~MyArray()
	{
	}

	template< class T >
	void MyArray<T>::initialize(const std::vector <T>& _data, const std::vector <uint32_t>& _firsts)
	{
		m_nbElements = _firsts.size() - 1;
		m_nbData = _data.size();
		m_data.clear();
		std::copy(_data.begin(), _data.end(), std::back_inserter(m_data));
		m_firstElementObjects.clear();
		std::copy(_firsts.begin(), _firsts.end(), std::back_inserter(m_firstElementObjects));
	}

	template< class T >
	void MyArray<T>::initialize(const std::vector<std::vector <T>>& _data)
	{
		std::vector <T> data;
		std::vector <uint32_t> firsts;
		firsts.push_back(0);
		for (const std::vector<T>& elems : _data) {
			std::copy(elems.begin(), elems.end(), std::back_inserter(data));
			firsts.push_back((uint32_t)data.size());
		}
		initialize(data, firsts);
	}

	template< class T >
	const std::string MyArray<T>::toString() const
	{
		std::string s("# elements: " + QString::number(m_nbElements) + "\n");
		for (size_t n = 0; n < m_nbElements; n++) {
			s.append("Element " + std::to_string(n) + " is composed of " + std::to_string(m_firstElementObjects[n+1] - m_firstElementObjects[n]) + " objects\n");
		}
		return s;
	}

	template< class T >
	const uint32_t MyArray<T>::memorySize() const
	{
		uint32_t memoryS = 0;
		memoryS += m_nbData * sizeof(T);
		memoryS += m_nbElements * sizeof(size_t);
		memoryS += m_nbElements * sizeof(size_t);
		memoryS += 2 * sizeof(size_t);
		return memoryS;
	}

	template< class T >
	void MyArray<T>::clear()
	{
		m_data.clear();
		m_firstElementObjects.clear();
		m_nbElements = m_nbData = 0;
	}

	template< class T >
	const bool MyArray<T>::empty() const
	{
		return m_data.empty();
	}

	typedef MyArray < size_t > MyArraySizeT;
	typedef MyArray < uint32_t > MyArrayUInt32;
	typedef MyArray < unsigned int > MyArrayUInt;
	typedef MyArray < int > MyArrayInt;
	typedef MyArray < unsigned short > MyArrayUShort;
	typedef MyArray < float > MyArrayFloat;
	typedef MyArray < double > MyArrayDouble;
	typedef MyArray < Vec2md > MyArrayVec2md;
	typedef MyArray < Vec2mf > MyArrayVec2mf;
	typedef MyArray < Vec3md > MyArrayVec3md;
	typedef MyArray < Vec3mf > MyArrayVec3mf;
}

#endif // MyArray_h__

