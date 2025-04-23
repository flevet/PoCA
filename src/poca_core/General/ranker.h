/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ranker.h
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

#ifndef RANKER_H
#define RANKER_H

#include <vector>
#include <string>
#include <algorithm>

using std::vector;
using std::string;

#ifndef uint
typedef unsigned int uint;
#endif

template <class T>
class lt { public: static int compare(T a, T b) { return(a < b); } };
template <class T>
class gt { public: static int compare(T a, T b) { return(a > b); } };

template <class T, class C>
class ranker
{
 private:
  const T* p;
  uint sz;

 public:
  ranker(const vector<T>& v) : p(&v[0]), sz(v.size()) { }
  ranker(const T* tp, uint s) : p(tp), sz(s) { }

  int operator()(uint i1, uint i2) const { return(C::compare(p[i1],p[i2])); }

  template <class S>
  void get_orders(vector<S>& w) const {
    w.resize(sz);
    w.front() = 0;
    for (typename vector<S>::iterator i = w.begin(); i != w.end() - 1; ++i)
      *(i + 1) = *i + 1;
    std::sort(w.begin(), w.end(), *this);
  }

  template <class S>
  void get_partial_orders(vector<S>& w, uint num) const {
    if (num > sz) num = sz;
    w.resize(sz);
    w.front() = 0;
    for (typename vector<S>::iterator i = w.begin(); i != w.end() - 1; ++i)
      *(i + 1) = *i + 1;
    std::partial_sort(w.begin(), w.begin() + num, w.end(), *this);
    w.resize(num);
  }

  template <class S>
  void get_ranks(vector<S>& w, const string& method) const {
    w.resize(sz);
    vector<uint> tmp(w.size());
    get_orders(tmp);
    if (method == "average") {
      for (uint c = 0, reps; c < w.size(); c += reps) { reps = 1;
        while (c + reps < w.size() && p[tmp[c]] == p[tmp[c + reps]]) ++reps;
	for (uint k = 0; k < reps; ++k)
          w[tmp[c + k]] = S(2 * c + reps - 1) / 2 + 1;
      }
    } else if (method == "min") {
      for (uint c = 0, reps; c < w.size(); c += reps) { reps = 1;
        while (c + reps < w.size() && p[tmp[c]] == p[tmp[c + reps]]) ++reps;
	for (uint k = 0; k < reps; ++k) w[tmp[c + k]] = c + 1;
      }
    } else if (method == "max") {
      for (uint c = 0, reps; c < w.size(); c += reps) { reps = 1;
        while (c + reps < w.size() && p[tmp[c]] == p[tmp[c + reps]]) ++reps;
	for (uint k = 0; k < reps; ++k) w[tmp[c + k]] = c + reps;
      }
    } else // default
      for (uint c = 0; c < w.size(); ++c) w[tmp[c]] = c + 1;
  }

  template <class S>
  void get_partial_ranks(vector<S>& w, const string& method, size_t num) const {
    if (num > sz) num = sz;
    vector<uint> tmp(sz);
    get_partial_orders(tmp, num);
    w.resize(sz);
    fill(w.begin(), w.end(), 0);
    if (method == "average") {
      for (uint c = 0, reps; c < num; c += reps) { reps = 1;
        while (c + reps < num && p[tmp[c]] == p[tmp[c + reps]]) ++reps;
	for (uint k = 0; k < reps; ++k)
          w[tmp[c + k]] = S(2 * c + reps - 1) / 2 + 1;
      }
    } else if (method == "min") {
      for (uint c = 0, reps; c < num; c += reps) { reps = 1;
        while (c + reps < num && p[tmp[c]] == p[tmp[c + reps]]) ++reps;
	for (uint k = 0; k < reps; ++k) w[tmp[c + k]] = c + 1;
      }
    } else if (method == "max") {
      for (uint c = 0, reps; c < num; c += reps) { reps = 1;
        while (c + reps < num && p[tmp[c]] == p[tmp[c + reps]]) ++reps;
	for (uint k = 0; k < reps; ++k) w[tmp[c + k]] = c + reps;
      }
    } else // default
      for (uint c = 0; c < num; ++c) w[tmp[c]] = c + 1;
  }

};

template <class T, class S>
inline void rank(const vector<T>& v, vector<S>& w,
          const string& method = "average")
  { ranker<T, lt<T> > r(v); r.get_ranks(w, method); }

template <class T, class S>
inline void rank(const T* d, uint size, vector<S>& w,
          const string& method = "average")
  { ranker<T, lt<T> > r(d, size); r.get_ranks(w, method); }

template <class T, class S>
inline void partial_rank(const vector<T>& v, vector<S>& w, uint num,
          const string& method = "average")
  { ranker<T, lt<T> > r(v); r.get_partial_ranks(w, method, num); }

template <class T, class S>
inline void partial_rank(const T* d, uint size, vector<S>& w, uint num,
          const string& method = "average")
  { ranker<T, lt<T> > r(d, size); r.get_partial_ranks(w, method, num); }

template <class T, class S>
inline void order(const vector<T>& v, vector<S>& w)
  { ranker<T, lt<T> > r(v); r.get_orders(w); }

template <class T, class S>
inline void order(const T* d, uint size, vector<S>& w)
  { ranker<T, lt<T> > r(d, size); r.get_orders(w); }

template <class T, class S>
inline void partial_order(const vector<T>& v, vector<S>& w, uint num)
  { ranker<T, lt<T> > r(v); r.get_partial_orders(w, num); }

template <class T, class S>
inline void partial_order(const T* d, uint size, vector<S>& w, uint num)
  { ranker<T, lt<T> > r(d, size); r.get_partial_orders(w, num); }

template <class T, class S>
inline void rankhigh(const vector<T>& v, vector<S>& w,
          const string& method = "average")
  { ranker<T, gt<T> > r(v); r.get_ranks(w, method); }

template <class T, class S>
inline void rankhigh(const T* d, uint size, vector<S>& w,
          const string& method = "average")
  { ranker<T, gt<T> > r(d, size); r.get_ranks(w, method); }

template <class T, class S>
inline void partial_rankhigh(const vector<T>& v, vector<S>& w, uint num,
          const string& method = "average")
  { ranker<T, gt<T> > r(v); r.get_partial_ranks(w, method, num); }

template <class T, class S>
inline void partial_rankhigh(const T* d, uint size, vector<S>& w, uint num,
          const string& method = "average")
  { ranker<T, gt<T> > r(d, size); r.get_partial_ranks(w, method, num); }

template <class T, class S>
inline void orderhigh(const vector<T>& v, vector<S>& w)
  { ranker<T, gt<T> > r(v); r.get_orders(w); }

template <class T, class S>
inline void orderhigh(const T* d, uint size, vector<S>& w)
  { ranker<T, gt<T> > r(d, size); r.get_orders(w); }

template <class T, class S>
inline void partial_orderhigh(const vector<T>& v, vector<S>& w, uint num)
  { ranker<T, gt<T> > r(v); r.get_partial_orders(w, num); }

template <class T, class S>
inline void partial_orderhigh(const T* d, uint size, vector<S>& w, uint num)
  { ranker<T, gt<T> > r(d, size); r.get_partial_orders(w, num); }

template <class T>
inline T quantile(const T* d, const uint size, const double q)
{
  if (size == 0) return T(0);
  if (size == 1) return d[0];
  if (q <= 0) return *std::min_element(d, d + size);
  if (q >= 1) return *std::max_element(d, d + size);

  double pos = (size - 1) * q;
  uint ind = uint(pos);
  double delta = pos - ind;
  vector<T> w(size); std::copy(d, d + size, w.begin());
  std::nth_element(w.begin(), w.begin() + ind, w.end());
  T i1 = *(w.begin() + ind);
  T i2 = *std::min_element(w.begin() + ind + 1, w.end());
  return i1 * (1.0 - delta) + i2 * delta;
}

template <class T>
inline T quantile(const vector<T>& v, const double q)
  { return quantile(&v[0], v.size(), q); }

#endif


