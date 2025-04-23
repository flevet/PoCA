/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Vec3.hpp
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

#ifndef Vec3_h__
#define Vec3_h__

#include <iostream>
#include <string>
#include <sstream>
#include <array>

#include <General/json.hpp>
       
/*==============================================================================
  CLASS Vec3
  ==============================================================================*/

//! 2D std::vector class.

namespace poca::core {

    template< class T >
    class Vec3
    {

    public:

        /*----- methods -----*/

        inline static Vec3 zero();

        /*----- methods -----*/

        template< class S > Vec3(const Vec3<S>& vec)
        {
            _e[0] = (T)vec(0);
            _e[1] = (T)vec(1);
            _e[2] = (T)vec(2);
        }

        Vec3() { _e[0] = _e[1] = _e[2] = 0; }
        Vec3(const Vec3<T>& vec);
        Vec3(const T& e0, const T& e1, const T& e2);

        ~Vec3() {}

        T* ptr();
        const T* ptr() const;

        T* getArray();
        T* getValues();
        const T* getArray() const;
        const T* getValues() const;

        T length() const;
        T lengthSquare() const;
        T sqrLength() const;
        T dot(const Vec3<T>& vec) const;
        Vec3<T> cross(const Vec3<T>& vec) const;
        T distance(const Vec3<T>& _vec)const;
        T distanceSqr(const Vec3<T>& _vec)const;

        Vec3  normal() const;
        Vec3& normalEq();
        Vec3& normalize();
        Vec3& normalEq(const T len);
        Vec3& negateEq();
        Vec3& clampToMaxEq(const T& max);

        Vec3 operator+(const Vec3<T>& rhs) const;
        Vec3 operator-(const Vec3<T>& rhs) const;
        Vec3 operator-() const;
        Vec3 operator*(const T& rhs) const;
        Vec3 operator*(const Vec3<T>& rhs) const;
        Vec3 operator/(const T& rhs) const;
        Vec3 operator/(const Vec3<T>& rhs) const;

        Vec3& operator+=(const Vec3<T>& rhs);
        Vec3& operator-=(const Vec3<T>& rhs);
        Vec3& operator*=(const T& rhs);
        Vec3& operator*=(const Vec3<T>& rhs);
        Vec3& operator/=(const T& rhs);
        Vec3& operator/=(const Vec3<T>& rhs);
        Vec3& operator=(const Vec3<T>& rsh);

        bool operator==(const Vec3<T>& rhs) const;
        bool operator!=(const Vec3<T>& rhs) const;

        bool operator<(const Vec3<T>& rhs) const;

        T& operator()(int idx);
        const T& operator()(int idx) const;

        T& operator[](int idx);
        const T& operator[](int idx) const;

        void set(T const x, T const y, T const z);
        void setX(const T& x);
        void setY(const T& y);
        void setZ(const T& z);


        T& x();
        T& y();
        T& z();

        const T& x() const;
        const T& y() const;
        const T& z() const;

        const std::string print() const;

    private:

        /*----- data members -----*/

        std::array<T, 3> _e;
    };

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T> Vec3<T>::zero()
    {
        return Vec3(0, 0, 0);
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T>::Vec3(const Vec3<T>& vec)
    {
        _e[0] = vec._e[0];
        _e[1] = vec._e[1];
        _e[2] = vec._e[2];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T>::Vec3(const T& e0, const T& e1, const T& e2)
    {
        _e[0] = e0;
        _e[1] = e1;
        _e[2] = e2;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T* Vec3<T>::ptr()
    {
        return _e;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline const T* Vec3<T>::ptr() const
    {
        return _e;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T* Vec3<T>::getArray()
    {
        return _e;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T* Vec3<T>::getValues()
    {
        return _e;
    }
    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline const T* Vec3<T>::getArray() const
    {
        return _e.data();
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline const T* Vec3<T>::getValues() const
    {
        return _e.data();
    }
    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T Vec3<T>::length() const
    {
        return (T)sqrt(_e[0] * _e[0] + _e[1] * _e[1] + _e[2] * _e[2]);
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T Vec3<T>::lengthSquare() const
    {
        return _e[0] * _e[0] + _e[1] * _e[1] + _e[2] * _e[2];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T Vec3<T>::sqrLength() const
    {
        return _e[0] * _e[0] + _e[1] * _e[1] + _e[2] * _e[2];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T Vec3<T>::dot(const Vec3<T>& vec) const
    {
        return _e[0] * vec._e[0] + _e[1] * vec._e[1] + _e[2] * vec._e[2];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T Vec3<T>::distance(const Vec3<T>& vec) const
    {
        T x = _e[0] - vec._e[0], y = _e[1] - vec._e[1], z = _e[2] - vec._e[2];
        return (T)sqrt(x * x + y * y + z * z);
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T Vec3<T>::distanceSqr(const Vec3<T>& vec) const
    {
        T x = _e[0] - vec._e[0], y = _e[1] - vec._e[1], z = _e[2] - vec._e[2];
        return (T)(x * x + y * y + z * z);
    }

    //------------------------------------------------------------------------------
    //!

    template< class T >
    inline Vec3<T> Vec3<T>::cross(const Vec3<T>& rhs) const
    {
        Vec3<T> res;
        res[0] = _e[1] * rhs[2] - _e[2] * rhs[1];
        res[1] = _e[2] * rhs[0] - _e[0] * rhs[2];
        res[2] = _e[0] * rhs[1] - _e[1] * rhs[0];

        return res;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T> Vec3<T>::normal() const
    {
        T tmp = (T)1 / length();
        return Vec3<T>(_e[0] * tmp, _e[1] * tmp, _e[2] * tmp);
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T>& Vec3<T>::normalEq()
    {
        T tmp = (T)1 / length();
        _e[0] *= tmp;
        _e[1] *= tmp;
        _e[2] *= tmp;
        return *this;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T>& Vec3<T>::normalize()
    {
        T tmp = (T)1 / length();
        _e[0] *= tmp;
        _e[1] *= tmp;
        _e[2] *= tmp;
        return *this;
    }
    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T>& Vec3<T>::normalEq(const T len)
    {
        T tmp = len / length();
        _e[0] *= tmp;
        _e[1] *= tmp;
        _e[2] *= tmp;
        return *this;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T>& Vec3<T>::negateEq()
    {
        _e[0] = -_e[0];
        _e[1] = -_e[1];
        _e[2] = -_e[2];
        return *this;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T>& Vec3<T>::clampToMaxEq(const T& max)
    {
        if (_e[0] > max)
        {
            _e[0] = max;
        }
        if (_e[1] > max)
        {
            _e[1] = max;
        }
        if (_e[2] > max)
        {
            _e[2] = max;
        }
        return *this;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T> Vec3<T>::operator+(const Vec3<T>& rhs) const
    {
        return Vec3<T>(
            _e[0] + rhs._e[0],
            _e[1] + rhs._e[1],
            _e[2] + rhs._e[2]
            );
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T> Vec3<T>::operator-(const Vec3<T>& rhs) const
    {
        return Vec3<T>(
            _e[0] - rhs._e[0],
            _e[1] - rhs._e[1],
            _e[2] - rhs._e[2]
            );
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T> Vec3<T>::operator-() const
    {
        return Vec3<T>(-_e[0], -_e[1], -_e[2]);
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T> Vec3<T>::operator*(const T& rhs) const
    {
        return Vec3<T>(_e[0] * rhs, _e[1] * rhs, _e[2] * rhs);
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T> Vec3<T>::operator*(const Vec3<T>& rhs) const
    {
        return Vec3<T>(_e[0] * rhs._e[0], _e[1] * rhs._e[1], _e[2] * rhs._e[2]);
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T> Vec3<T>::operator/(const T& rhs) const
    {
        return Vec3<T>(_e[0] / rhs, _e[1] / rhs, _e[2] / rhs);
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T> Vec3<T>::operator/(const Vec3<T>& rhs) const
    {
        return Vec3<T>(_e[0] / rhs._e[0], _e[1] / rhs._e[1], _e[2] / rhs._e[2]);
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T>& Vec3<T>::operator+=(const Vec3<T>& rhs)
    {
        _e[0] += rhs._e[0];
        _e[1] += rhs._e[1];
        _e[2] += rhs._e[2];
        return *this;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T>& Vec3<T>::operator-=(const Vec3<T>& rhs)
    {
        _e[0] -= rhs._e[0];
        _e[1] -= rhs._e[1];
        _e[2] -= rhs._e[2];
        return *this;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T>& Vec3<T>::operator*=(const T& rhs)
    {
        _e[0] *= rhs;
        _e[1] *= rhs;
        _e[2] *= rhs;
        return *this;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T>& Vec3<T>::operator*=(const Vec3<T>& rhs)
    {
        _e[0] *= rhs._e[0];
        _e[1] *= rhs._e[1];
        _e[2] *= rhs._e[2];
        return *this;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T>& Vec3<T>::operator/=(const T& rhs)
    {
        _e[0] /= rhs;
        _e[1] /= rhs;
        _e[2] /= rhs;
        return *this;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T>& Vec3<T>::operator/=(const Vec3<T>& rhs)
    {
        _e[0] /= rhs._e[0];
        _e[1] /= rhs._e[1];
        _e[2] /= rhs._e[2];
        return *this;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T>& Vec3<T>::operator=(const Vec3<T>& rhs)
    {
        _e[0] = rhs._e[0];
        _e[1] = rhs._e[1];
        _e[2] = rhs._e[2];
        return *this;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline bool Vec3<T>::operator==(const Vec3<T>& rhs) const
    {
        return _e[0] == rhs._e[0] && _e[1] == rhs._e[1] && _e[2] == rhs._e[2];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline bool Vec3<T>::operator!=(const Vec3<T>& rhs) const
    {
        return _e[0] != rhs._e[0] || _e[1] != rhs._e[1] || _e[2] != rhs._e[2];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline bool Vec3<T>::operator<(const Vec3<T>& rhs) const
    {
        if (_e[0] != rhs._e[0]) return _e[0] < rhs._e[0];
        if (_e[1] != rhs._e[1]) return _e[1] < rhs._e[1];
        return _e[2] < rhs._e[2];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T& Vec3<T>::operator()(int idx)
    {
        return _e[idx];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline const T& Vec3<T>::operator()
        (int idx) const
    {
        return _e[idx];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T& Vec3<T>::operator[](int idx)
    {
        return _e[idx];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline const T& Vec3<T>::operator[](int idx) const
    {
        return _e[idx];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    void
        Vec3<T>::set(T const x, T const y, T const z)
    {
        _e[0] = x;
        _e[1] = y;
        _e[2] = z;
    }



    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T& Vec3<T>::x() {
        return _e[0];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T& Vec3<T>::y() {
        return _e[1];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T& Vec3<T>::z() {
        return _e[2];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline const T& Vec3<T>::x() const {
        return _e[0];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline const T& Vec3<T>::y() const {
        return _e[1];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline const T& Vec3<T>::z() const {
        return _e[2];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline void Vec3<T>::setX(const T& x) {
        _e[0] = x;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline void Vec3<T>::setY(const T& y) {
        _e[1] = y;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline void Vec3<T>::setZ(const T& z) {
        _e[2] = z;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec3<T> operator*(const T& val, const Vec3<T>& vec)
    {
        return Vec3<T>(vec(0) * val, vec(1) * val, vec(2) * val);
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline const std::string Vec3<T>::print() const
    {
        std::ostringstream s;
        s << "[" << _e[0] << ", " << _e[1] << ", " << _e[2] << "]";
        return s.str();
    }

    /*==============================================================================
      TYPEDEF
      ==============================================================================*/

    typedef Vec3< int >    Vec3mi;
    typedef Vec3< unsigned int> Vec3mui;
    typedef Vec3< float >  Vec3mf;
    typedef Vec3< double > Vec3md;
    typedef Vec3< unsigned char > Vec3muc;
    typedef Vec3< unsigned char > Color3uc;

    typedef Vec3 < double >  DetectionPoint;

    typedef Vec3< float >  Vector3;

    //External Operator
    template< class T >
    inline std::ostream&
        operator<<(std::ostream& os, Vec3<T> const& v)
    {
        return os << v.print().c_str();
    }

    template <typename T>
    void to_json(nlohmann::json& j, const Vec3<T>& P) {
        j = { { "x", P[0] }, { "y", P[1] }, { "z", P[2] } };
    };

    template <typename T>
    void from_json(const nlohmann::json& j, Vec3<T>& P) {
        P[0] = j.at("x").get<T>();
        P[1] = j.at("y").get<T>();
        P[2] = j.at("z").get<T>();
    }
}
#endif // Vec3_h__

