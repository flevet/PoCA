/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Vec2.hpp
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

#ifndef CLASS_VEC2_HPP
#define CLASS_VEC2_HPP

#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <sstream>

#include <General/json.hpp>

namespace poca::core {
    /*==============================================================================
  CLASS Vec2
  ==============================================================================*/

  //! 2D vector class.

    template< class T >
    class Vec2
    {

    public:

        /*----- methods -----*/

        inline static Vec2 zero();

        /*----- methods -----*/

        template< class S > Vec2(const Vec2<S>& vec)
        {
            _e[0] = (T)vec(0);
            _e[1] = (T)vec(1);
        }

        Vec2() { _e[0] = _e[1] = 0; }
        Vec2(const Vec2<T>& vec);
        Vec2(const T& e0, const T& e1);

        ~Vec2() {}

        T* ptr();
        const T* ptr() const;

        T* getArray();
        T* getValues();
        const T* getArray() const;
        const T* getValues() const;

        T length() const;
        T sqrLength() const;
        T dot(const Vec2<T>& vec) const;
        T distance(const Vec2<T>& _vec)const;
        T distanceSqr(const Vec2<T>& _vec)const;
        T findSignedAngle(const Vec2<T>& _vec)const;
        T findSignedAngle(const T, const T)const;

        Vec2  normal() const;
        Vec2& normalEq();
        Vec2& normalize();
        Vec2& normalEq(const T len);
        Vec2& negateEq();
        Vec2& clampToMaxEq(const T& max);

        Vec2 operator+(const Vec2<T>& rhs) const;
        Vec2 operator-(const Vec2<T>& rhs) const;
        Vec2 operator-() const;
        Vec2 operator*(const T& rhs) const;
        Vec2 operator*(const Vec2<T>& rhs) const;
        Vec2 operator/(const T& rhs) const;
        Vec2 operator/(const Vec2<T>& rhs) const;

        Vec2& operator+=(const Vec2<T>& rhs);
        Vec2& operator-=(const Vec2<T>& rhs);
        Vec2& operator*=(const T& rhs);
        Vec2& operator*=(const Vec2<T>& rhs);
        Vec2& operator/=(const T& rhs);
        Vec2& operator/=(const Vec2<T>& rhs);
        Vec2& operator=(const Vec2<T>& rsh);

        bool operator==(const Vec2<T>& rhs) const;
        bool operator!=(const Vec2<T>& rhs) const;

        T& operator()(int idx);
        const T& operator()(int idx) const;

        T& operator[](int idx);
        const T& operator[](int idx) const;

        void set(T const x, T const y);
        void setX(const T& x);
        void setY(const T& y);


        T& x();
        T& y();

        const T& x() const;
        const T& y() const;

        const std::string print() const;

    private:

        /*----- data members -----*/

        T _e[2];
    };

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T> Vec2<T>::zero()
    {
        return Vec2(0, 0);
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T>::Vec2(const Vec2<T>& vec)
    {
        _e[0] = vec._e[0];
        _e[1] = vec._e[1];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T>::Vec2(const T& e0, const T& e1)
    {
        _e[0] = e0;
        _e[1] = e1;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T* Vec2<T>::ptr()
    {
        return _e;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline const T* Vec2<T>::ptr() const
    {
        return _e;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T* Vec2<T>::getArray()
    {
        return _e;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T* Vec2<T>::getValues()
    {
        return _e;
    }
    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline const T* Vec2<T>::getArray() const
    {
        return _e;
    }


    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline const T* Vec2<T>::getValues() const
    {
        return _e;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T Vec2<T>::length() const
    {
        return (T)sqrt(_e[0] * _e[0] + _e[1] * _e[1]);
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T Vec2<T>::sqrLength() const
    {
        return _e[0] * _e[0] + _e[1] * _e[1];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T Vec2<T>::dot(const Vec2<T>& vec) const
    {
        return _e[0] * vec._e[0] + _e[1] * vec._e[1];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T Vec2<T>::distance(const Vec2<T>& vec) const
    {
        T x = _e[0] - vec._e[0], y = _e[1] - vec._e[1];
        return (T)sqrt(x * x + y * y);
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T Vec2<T>::distanceSqr(const Vec2<T>& vec) const
    {
        T x = _e[0] - vec._e[0], y = _e[1] - vec._e[1];
        return (T)(x * x + y * y);
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T Vec2<T>::findSignedAngle(const Vec2<T>& vec) const
    {
        return findSignedAngle(vec._e[0], vec._e[1]);
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T Vec2<T>::findSignedAngle(const T _x, const T _y) const
    {
        T x = _x - _e[0];
        T y = _y - _e[1];
        T angle = atan2(y, x);
        //		double angle = (float)(FastMath.atan2_fast((float)y, (float)x));
        //		double angle = (float)(FastMath.atan2((float)y, (float)x));
        return angle;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T> Vec2<T>::normal() const
    {
        T tmp = (T)1 / length();
        return Vec2<T>(_e[0] * tmp, _e[1] * tmp);
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T>& Vec2<T>::normalEq()
    {
        T tmp = (T)1 / length();
        _e[0] *= tmp;
        _e[1] *= tmp;
        return *this;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T>& Vec2<T>::normalize()
    {
        T tmp = (T)1 / length();
        _e[0] *= tmp;
        _e[1] *= tmp;
        return *this;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T>& Vec2<T>::normalEq(const T len)
    {
        T tmp = len / length();
        _e[0] *= tmp;
        _e[1] *= tmp;
        return *this;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T>& Vec2<T>::negateEq()
    {
        _e[0] = -_e[0];
        _e[1] = -_e[1];
        return *this;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T>& Vec2<T>::clampToMaxEq(const T& max)
    {
        if (_e[0] > max)
        {
            _e[0] = max;
        }
        if (_e[1] > max)
        {
            _e[1] = max;
        }
        return *this;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T> Vec2<T>::operator+(const Vec2<T>& rhs) const
    {
        return Vec2<T>(
            _e[0] + rhs._e[0],
            _e[1] + rhs._e[1]
            );
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T> Vec2<T>::operator-(const Vec2<T>& rhs) const
    {
        return Vec2<T>(
            _e[0] - rhs._e[0],
            _e[1] - rhs._e[1]
            );
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T> Vec2<T>::operator-() const
    {
        return Vec2<T>(-_e[0], -_e[1]);
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T> Vec2<T>::operator*(const T& rhs) const
    {
        return Vec2<T>(_e[0] * rhs, _e[1] * rhs);
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T> Vec2<T>::operator*(const Vec2<T>& rhs) const
    {
        return Vec2<T>(_e[0] * rhs._e[0], _e[1] * rhs._e[1]);
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T> Vec2<T>::operator/(const T& rhs) const
    {
        return Vec2<T>(_e[0] / rhs, _e[1] / rhs);
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T> Vec2<T>::operator/(const Vec2<T>& rhs) const
    {
        return Vec2<T>(_e[0] / rhs._e[0], _e[1] / rhs._e[1]);
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T>& Vec2<T>::operator+=(const Vec2<T>& rhs)
    {
        _e[0] += rhs._e[0];
        _e[1] += rhs._e[1];
        return *this;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T>& Vec2<T>::operator-=(const Vec2<T>& rhs)
    {
        _e[0] -= rhs._e[0];
        _e[1] -= rhs._e[1];
        return *this;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T>& Vec2<T>::operator*=(const T& rhs)
    {
        _e[0] *= rhs;
        _e[1] *= rhs;
        return *this;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T>& Vec2<T>::operator*=(const Vec2<T>& rhs)
    {
        _e[0] *= rhs._e[0];
        _e[1] *= rhs._e[1];
        return *this;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T>& Vec2<T>::operator/=(const T& rhs)
    {
        _e[0] /= rhs;
        _e[1] /= rhs;
        return *this;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T>& Vec2<T>::operator/=(const Vec2<T>& rhs)
    {
        _e[0] /= rhs._e[0];
        _e[1] /= rhs._e[1];
        return *this;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T>& Vec2<T>::operator=(const Vec2<T>& rhs)
    {
        _e[0] = rhs._e[0];
        _e[1] = rhs._e[1];
        return *this;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline bool Vec2<T>::operator==(const Vec2<T>& rhs) const
    {
        return _e[0] == rhs._e[0] && _e[1] == rhs._e[1];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline bool Vec2<T>::operator!=(const Vec2<T>& rhs) const
    {
        return _e[0] != rhs._e[0] || _e[1] != rhs._e[1];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T& Vec2<T>::operator()(int idx)
    {
        return _e[idx];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline const T& Vec2<T>::operator()
        (int idx) const
    {
        return _e[idx];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T& Vec2<T>::operator[](int idx)
    {
        return _e[idx];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline const T& Vec2<T>::operator[](int idx) const
    {
        return _e[idx];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    void
        Vec2<T>::set(T const x, T const y)
    {
        _e[0] = x;
        _e[1] = y;
    }



    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline void Vec2<T>::setX(const T& x) {
        _e[0] = x;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline void Vec2<T>::setY(const T& y) {
        _e[1] = y;
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T& Vec2<T>::x() {
        return _e[0];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline T& Vec2<T>::y() {
        return _e[1];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline const T& Vec2<T>::x() const {
        return _e[0];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline const T& Vec2<T>::y() const {
        return _e[1];
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline Vec2<T> operator*(const T& val, const Vec2<T>& vec)
    {
        return Vec2<T>(vec(0) * val, vec(1) * val);
    }

    //------------------------------------------------------------------------------
    //!
    template< class T >
    inline const std::string Vec2<T>::print() const
    {
        std::ostringstream s;
        s << "[" << _e[0] << ", " << _e[1] << "]";
        return s.str();
    }

    /*==============================================================================
      TYPEDEF
      ==============================================================================*/

    typedef Vec2< int >    Vec2im;
    typedef Vec2< float >  Vec2fm;
    typedef Vec2< double > Vec2dm;

    typedef Vec2< int >    Vec2mi;
    typedef Vec2< uint32_t >    Vec2mui32;
    typedef Vec2< float >  Vec2mf;
    typedef Vec2< double > Vec2md;
    typedef Vec2< int > Vec2ms;

    typedef Vec2< float >  Vector2;

    //External Operator
    template< class T >
    inline std::ostream&
        operator<<(std::ostream& os, Vec2<T> const& v)
    {
        return os << v.print().c_str();
    }

    template <typename T>
    void to_json(nlohmann::json& j, const Vec2<T>& P) {
        j = { { "x", P[0] }, { "y", P[1] } };
    };

    template <typename T>
    void from_json(const nlohmann::json& j, Vec2<T>& P) {
        P[0] = j.at("x").get<T>();
        P[1] = j.at("y").get<T>();
    }
}
#endif

