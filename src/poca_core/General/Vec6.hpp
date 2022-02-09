/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Vec6.hpp
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

#ifndef Vec6_h__
#define Vec6_h__

#include <iostream>
#include <string>
#include <sstream>
#include <limits>

#include <General/json.hpp>

namespace poca::core {

	template< class T >
	class Vec6
	{

	public:

		/*----- methods -----*/

		inline static Vec6 zero();
		inline static Vec6 initBBox();

		/*----- methods -----*/

		template< class S > Vec6(const Vec6<S>& vec)
		{
			_e[0] = (T)vec(0);
			_e[1] = (T)vec(1);
			_e[2] = (T)vec(2);
			_e[3] = (T)vec(3);
			_e[4] = (T)vec(4);
			_e[5] = (T)vec(5);
		}

		Vec6() { _e[0] = _e[1] = _e[2] = _e[3] = _e[4] = _e[5] = 0; }
		Vec6(const Vec6<T>& vec);
		Vec6(const T& e0, const T& e1, const T& e2, const T& e3, const T& e4, const T& e5);

		~Vec6() {}

		T* ptr();
		const T* ptr() const;

		T* getArray();
		T* getValues();
		const T* getArray() const;
		const T* getValues() const;

		Vec6 operator+(const Vec6<T>& rhs) const;
		Vec6 operator-(const Vec6<T>& rhs) const;
		Vec6 operator-() const;
		Vec6 operator*(const T& rhs) const;
		Vec6 operator*(const Vec6<T>& rhs) const;
		Vec6 operator/(const T& rhs) const;
		Vec6 operator/(const Vec6<T>& rhs) const;

		Vec6& operator+=(const Vec6<T>& rhs);
		Vec6& operator-=(const Vec6<T>& rhs);
		Vec6& operator*=(const T& rhs);
		Vec6& operator*=(const Vec6<T>& rhs);
		Vec6& operator/=(const T& rhs);
		Vec6& operator/=(const Vec6<T>& rhs);
		Vec6& operator=(const Vec6<T>& rsh);

		bool operator==(const Vec6<T>& rhs) const;
		bool operator!=(const Vec6<T>& rhs) const;

		T& operator()(int idx);
		const T& operator()(int idx) const;

		T& operator[](int idx);
		const T& operator[](int idx) const;

		void set(T const x, T const y, T const z, T const w, T const a, T const b);

		void setX(const T& x);
		void setY(const T& y);
		void setZ(const T& z);
		void setWidth(const T& _width);
		void setHeight(const T& _height);
		void setThick(const T& _thick);

		T& x();
		T& y();
		T& z();
		T& width();
		T& height();
		T& thick();

		const T& x() const;
		const T& y() const;
		const T& z() const;
		const T& width() const;
		const T& height() const;
		const T& thick() const;

		const std::string print() const;

		void addPointBBox(const T, const T, const T);

		const T realWidth() const;
		const T realHeight() const;
		const T realThick() const;
		const T longestSide() const;
		const T smallestSide() const;

		const Vec6 intersect(const Vec6<T>& rhs) const;

	private:

		/*----- data members -----*/

		T _e[6];
	};

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline Vec6<T> Vec6<T>::zero()
	{
		return Vec6(0, 0, 0, 0, 0, 0);
	}

	template< class T >
	inline Vec6<T> Vec6<T>::initBBox()
	{
		return Vec6(std::numeric_limits<T>::max(), std::numeric_limits<T>::max(), std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest());
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline Vec6<T>::Vec6(const Vec6<T>& vec)
	{
		_e[0] = vec._e[0];
		_e[1] = vec._e[1];
		_e[2] = vec._e[2];
		_e[3] = vec._e[3];
		_e[4] = vec._e[4];
		_e[5] = vec._e[5];
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline Vec6<T>::Vec6(const T& e0, const T& e1, const T& e2, const T& e3, const T& e4, const T& e5)
	{
		_e[0] = e0;
		_e[1] = e1;
		_e[2] = e2;
		_e[3] = e3;
		_e[4] = e4;
		_e[5] = e5;
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline T* Vec6<T>::ptr()
	{
		return _e;
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline const T* Vec6<T>::ptr() const
	{
		return _e;
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline T* Vec6<T>::getArray()
	{
		return _e;
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline T* Vec6<T>::getValues()
	{
		return _e;
	}
	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline const T* Vec6<T>::getArray() const
	{
		return _e;
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline const T* Vec6<T>::getValues() const
	{
		return _e;
	}
	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline Vec6<T> Vec6<T>::operator+(const Vec6<T>& rhs) const
	{
		return Vec6<T>(
			_e[0] + rhs._e[0],
			_e[1] + rhs._e[1],
			_e[2] + rhs._e[2],
			_e[3] + rhs._e[3],
			_e[4] + rhs._e[4],
			_e[5] + rhs._e[5]
			);
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline Vec6<T> Vec6<T>::operator-(const Vec6<T>& rhs) const
	{
		return Vec6<T>(
			_e[0] - rhs._e[0],
			_e[1] - rhs._e[1],
			_e[2] - rhs._e[2],
			_e[3] - rhs._e[3],
			_e[4] - rhs._e[4],
			_e[5] - rhs._e[5]
			);
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline Vec6<T> Vec6<T>::operator-() const
	{
		return Vec6<T>(-_e[0], -_e[1], -_e[2], -_e[3], -_e[4], -_e[5]);
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline Vec6<T> Vec6<T>::operator*(const T& rhs) const
	{
		return Vec6<T>(_e[0] * rhs, _e[1] * rhs, _e[2] * rhs, _e[3] * rhs, _e[4] * rhs, _e[5] * rhs);
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline Vec6<T> Vec6<T>::operator*(const Vec6<T>& rhs) const
	{
		return Vec6<T>(_e[0] * rhs._e[0], _e[1] * rhs._e[1], _e[2] * rhs._e[2], _e[3] * rhs._e[3], _e[4] * rhs._e[4], _e[5] * rhs._e[5]);
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline Vec6<T> Vec6<T>::operator/(const T& rhs) const
	{
		return Vec6<T>(_e[0] / rhs, _e[1] / rhs, _e[2] / rhs, _e[3] / rhs, _e[4] / rhs, _e[5] / rhs);
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline Vec6<T> Vec6<T>::operator/(const Vec6<T>& rhs) const
	{
		return Vec6<T>(_e[0] / rhs._e[0], _e[1] / rhs._e[1], _e[2] / rhs._e[2], _e[3] / rhs._e[3], _e[4] / rhs._e[4], _e[5] / rhs._e[5]);
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline Vec6<T>& Vec6<T>::operator+=(const Vec6<T>& rhs)
	{
		_e[0] += rhs._e[0];
		_e[1] += rhs._e[1];
		_e[2] += rhs._e[2];
		_e[3] += rhs._e[3];
		_e[4] += rhs._e[4];
		_e[5] += rhs._e[5];
		return *this;
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline Vec6<T>& Vec6<T>::operator-=(const Vec6<T>& rhs)
	{
		_e[0] -= rhs._e[0];
		_e[1] -= rhs._e[1];
		_e[2] -= rhs._e[2];
		_e[3] -= rhs._e[3];
		_e[4] -= rhs._e[4];
		_e[5] -= rhs._e[5];
		return *this;
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline Vec6<T>& Vec6<T>::operator*=(const T& rhs)
	{
		_e[0] *= rhs;
		_e[1] *= rhs;
		_e[2] *= rhs;
		_e[3] *= rhs;
		_e[4] *= rhs;
		_e[5] *= rhs;
		return *this;
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline Vec6<T>& Vec6<T>::operator*=(const Vec6<T>& rhs)
	{
		_e[0] *= rhs._e[0];
		_e[1] *= rhs._e[1];
		_e[2] *= rhs._e[2];
		_e[3] *= rhs._e[3];
		_e[4] *= rhs._e[4];
		_e[5] *= rhs._e[5];
		return *this;
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline Vec6<T>& Vec6<T>::operator/=(const T& rhs)
	{
		_e[0] /= rhs;
		_e[1] /= rhs;
		_e[2] /= rhs;
		_e[3] /= rhs;
		_e[4] /= rhs;
		_e[5] /= rhs;
		return *this;
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline Vec6<T>& Vec6<T>::operator/=(const Vec6<T>& rhs)
	{
		_e[0] /= rhs._e[0];
		_e[1] /= rhs._e[1];
		_e[2] /= rhs._e[2];
		_e[3] /= rhs._e[3];
		_e[4] /= rhs._e[4];
		_e[5] /= rhs._e[5];
		return *this;
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline Vec6<T>& Vec6<T>::operator=(const Vec6<T>& rhs)
	{
		_e[0] = rhs._e[0];
		_e[1] = rhs._e[1];
		_e[2] = rhs._e[2];
		_e[3] = rhs._e[3];
		_e[4] = rhs._e[4];
		_e[5] = rhs._e[5];
		return *this;
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline bool Vec6<T>::operator==(const Vec6<T>& rhs) const
	{
		return _e[0] == rhs._e[0] && _e[1] == rhs._e[1] && _e[2] == rhs._e[2] && _e[3] == rhs._e[3] && _e[4] == rhs._e[4] && _e[5] == rhs._e[5];
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline bool Vec6<T>::operator!=(const Vec6<T>& rhs) const
	{
		return _e[0] != rhs._e[0] || _e[1] != rhs._e[1] || _e[2] != rhs._e[2] || _e[3] != rhs._e[3] || _e[4] != rhs._e[4] || _e[5] != rhs._e[5];
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline T& Vec6<T>::operator()(int idx)
	{
		return _e[idx];
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline const T& Vec6<T>::operator()
		(int idx) const
	{
		return _e[idx];
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline T& Vec6<T>::operator[](int idx)
	{
		return _e[idx];
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline const T& Vec6<T>::operator[](int idx) const
	{
		return _e[idx];
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	void
		Vec6<T>::set(T const x, T const y, T const z, T const w, T const a, T const b)
	{
		_e[0] = x;
		_e[1] = y;
		_e[2] = z;
		_e[3] = w;
		_e[4] = a;
		_e[5] = b;
	}


	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline T& Vec6<T>::x() {
		return _e[0];
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline T& Vec6<T>::y() {
		return _e[1];
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline T& Vec6<T>::z() {
		return _e[2];
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline T& Vec6<T>::width() {
		return _e[3];
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline T& Vec6<T>::height() {
		return _e[4];
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline T& Vec6<T>::thick() {
		return _e[5];
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline const T& Vec6<T>::x() const {
		return _e[0];
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline const T& Vec6<T>::y() const {
		return _e[1];
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline const T& Vec6<T>::z() const {
		return _e[2];
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline const T& Vec6<T>::width() const {
		return _e[3];
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline const T& Vec6<T>::height() const {
		return _e[4];
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline const T& Vec6<T>::thick() const {
		return _e[5];
	}

	template< class T >
	inline const T Vec6<T>::realWidth() const {
		return _e[3] - _e[0];
	}

	template< class T >
	inline const T Vec6<T>::realHeight() const {
		return _e[4] - _e[1];
	}

	template< class T >
	inline const T Vec6<T>::realThick() const {
		return _e[5] - _e[2];
	}

	template< class T >
	inline const T Vec6<T>::longestSide() const {
		const T w = realWidth(), h = realHeight(), t = realThick();
		T val = w > h ? w : h;
		val = val > t ? val : t;
		return val;
	}

	template< class T >
	inline const T Vec6<T>::smallestSide() const {
		const T w = realWidth(), h = realHeight(), t = realThick();
		T val = w < h ? w : h;
		val = val < t ? val : t;
		return val;
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline void Vec6<T>::setX(const T& x) {
		_e[0] = x;
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline void Vec6<T>::setY(const T& y) {
		_e[1] = y;
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline void Vec6<T>::setZ(const T& z) {
		_e[2] = z;
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline void Vec6<T>::setWidth(const T& _width) {
		_e[3] = _width;
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline void Vec6<T>::setHeight(const T& _height) {
		_e[4] = _height;
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline void Vec6<T>::setThick(const T& _thick) {
		_e[5] = _thick;
	}

	//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
//!
	template< class T >
	inline Vec6<T> operator*(const T& val, const Vec6<T>& vec)
	{
		return Vec6<T>(vec(0) * val, vec(1) * val, vec(2) * val, vec(3) * val, vec(4) * val, vec(5) * val);
	}


	template< class T >
	inline void Vec6<T>::addPointBBox(const T _x, const T _y, const T _z) {
		if (_x < _e[0]) _e[0] = _x;
		if (_x > _e[3]) _e[3] = _x;
		if (_y < _e[1]) _e[1] = _y;
		if (_y > _e[4]) _e[4] = _y;
		if (_z < _e[2]) _e[2] = _z;
		if (_z > _e[5]) _e[5] = _z;
	}

	template< class T >
	inline const Vec6<T> Vec6<T>::intersect(const Vec6<T>& rhs) const {
		Vec6<T> res;
		for (size_t n = 0; n < 3; n++)
			res[n] = _e[n] > rhs._e[n] ? _e[n] : rhs._e[n];
		for (size_t n = 3; n < 6; n++)
			res[n] = _e[n] < rhs._e[n] ? _e[n] : rhs._e[n];
		return res;
	}

	//------------------------------------------------------------------------------
	//!
	template< class T >
	inline const std::string Vec6<T>::print() const
	{
		std::ostringstream s;
		s << "[" << _e[0] << ", " << _e[1] << ", " << _e[2] << ", " << _e[3] << ", " << _e[4] << ", " << _e[5] << "]";
		return s.str();
	}

	/*==============================================================================
	  TYPEDEF
	  ==============================================================================*/

	typedef Vec6< int >    Vec6mi;
	typedef Vec6< float >  Vec6mf;
	typedef Vec6< double > Vec6md;
	typedef Vec6< bool > Vec6mb;
	typedef Vec6< float >  Vector6;
	typedef Vec6< std::size_t> Vec6ms;

	typedef Vec6 < float > BoundingBox;
	typedef Vec6 < float > BoundingBoxD;


	//External Operator
	template< class T >
	inline std::ostream&
		operator<<(std::ostream& os, Vec6<T> const& v)
	{
		return os << v.print().c_str();
	}

	template <typename T>
	void to_json(nlohmann::json& j, const Vec6<T>& P) {
		j = { { "x", P[0] }, { "y", P[1] }, { "z", P[2] }, { "w", P[3] }, { "h", P[4] }, { "t", P[5] } };
	};

	template <typename T>
	void from_json(const nlohmann::json& j, Vec6<T>& P) {
		P[0] = j.at("x").get<T>();
		P[1] = j.at("y").get<T>();
		P[2] = j.at("z").get<T>();
		P[3] = j.at("w").get<T>();
		P[4] = j.at("h").get<T>();
		P[5] = j.at("t").get<T>();
	}
}

#endif // Vec6_h__

