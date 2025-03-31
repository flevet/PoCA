/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Misc.h
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

#ifndef Misc_h__
#define Misc_h__

#include <QtWidgets/QTabWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtCore/QString>

#include <vector>
#include <map>
#include <any>

#include "../General/Vec2.hpp"
#include "../General/Vec3.hpp"
#include "../General/Vec4.hpp"
#include "../General/Vec6.hpp"
#include "../General/ranker.h"
#include "../General/Histogram.hpp"
#include "../General/MyData.hpp"

namespace poca::core{
	enum PHOTOPHYSICS_PARAMETERS {NB_BLINKS = 0, TOTAL_ON = 1, TOTAL_OFF = 2, NB_ON = 3, NB_OFF = 4, LIFETIME = 5};
	enum ImageType { UINT8, UINT16, UINT32, INT32, FLOAT, RAW, LABEL, NONE };

	static size_t NbObjects = 1;
	static const std::size_t ENV_BUF_SIZE = 2048; // Enough for your PATH?

	Color4D contrastColor(const Color4D & _backC);
	int roundUpGeneric(int, int);
	int roundUpPowerOfTwo(int, int);
	float roundFromZero(float);

	float sRGBtoLin(const float);
	float luminance(const float, const float, const float);
	float perceivedLightness(const float);
	Color4uc randomColorB();
	Color4D randomColorD();
	Color4uc randomBrightColorB(const uint32_t = 128);
	Color4D randomBrightColorD(const float = 0.5f);
	Color4uc randomDarkColorB(const uint32_t = 128);
	Color4D randomDarkColorD(const float = 0.5f);


	bool file_exists(const std::string&);
	void getColorString(double, double, double, char [32]);
	void getColorStringUC(unsigned char, unsigned char, unsigned char, char[32]);

	void randomPointsOnUnitSphere(const uint32_t, std::vector<Vec3mf>&);

	double linear(double, const double*);
	double linearFixed(double, const double*);

	void computePhotophysicsParameters(const std::vector <float>&, std::vector <float>&);

	void PrintFullPath(const char*);

	template <class T>
	MyData* generateDataWithLog(const std::vector<T>& _data)
	{
		return new poca::core::MyData(new poca::core::Histogram<T>(_data, false), true);
	}

	template <class T>
	MyData* generateDataWithoutLog(const std::vector<T>& _data)
	{
		return new poca::core::MyData(new poca::core::Histogram<T>(_data, false), false);
	}

	template <class T>
	MyData* generateLogData(const std::vector<T>& _data)
	{
		return new poca::core::MyData(new poca::core::Histogram<T>(_data, true), false);
	}

	/*
	 * Generic function to find duplicates elements in vector.
	 * It adds the duplicate elements and their duplication count in given map countMap
	 */
	template <typename T>
	void findDuplicates(std::vector<T>& vecOfElements, std::map<T, int>& countMap)
	{
		// Iterate over the vector and store the frequency of each element in map
		for (auto& elem : vecOfElements)
		{
			auto result = countMap.insert(std::pair<T, int>(elem, 1));
			if (result.second == false)
				result.first->second++;
		}
		// Remove the elements from Map which has 1 frequency count
		for (auto it = countMap.begin(); it != countMap.end();)
		{
			if (it->second == 1)
				it = countMap.erase(it);
			else
				it++;
		}
	}

	template <typename T>
	vector<size_t> sort_indexes(const vector<T>& v) {

		// initialize original index locations
		vector<size_t> idx(v.size());
		iota(idx.begin(), idx.end(), 0);

		// sort indexes based on comparing values in v
		// using std::stable_sort instead of std::sort
		// to avoid unnecessary index re-orderings
		// when v contains elements of equal values 
		stable_sort(idx.begin(), idx.end(),
			[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });

		return idx;
	}
	
	template <typename T, typename M>
	void sort_indexes(const vector<T>& v, vector<M>& idx) {

		// initialize original index locations
		iota(idx.begin(), idx.end(), 0);

		// sort indexes based on comparing values in v
		// using std::stable_sort instead of std::sort
		// to avoid unnecessary index re-orderings
		// when v contains elements of equal values 
		stable_sort(idx.begin(), idx.end(),
			[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });
	}

	template <typename Container, typename T = typename std::decay<decltype(*std::begin(std::declval<Container>()))>::type>
	T stdDev(Container&& c)
	{
		auto b = std::begin(c), e = std::end(c);
		auto size = std::distance(b, e);
		auto sum = std::accumulate(b, e, T());
		auto mean = sum / size;
		T accum = T();
		for (const auto d : c)
			accum += (d - mean) * (d - mean);
		return std::sqrt(accum / (size - 1));
	}

	template <typename Container, typename T = typename std::decay<decltype(*std::begin(std::declval<Container>()))>::type>
	T variance(Container&& c)
	{
		auto b = std::begin(c), e = std::end(c);
		auto size = std::distance(b, e);
		auto sum = std::accumulate(b, e, T());
		auto mean = sum / size;
		T accum = T();
		for (const auto d : c)
			accum += (d - mean) * (d - mean);
		return accum / (size - 1);
	}

	template <typename Container, typename T = typename std::decay<decltype(*std::begin(std::declval<Container>()))>::type>
	T mean(Container&& c)
	{
		auto b = std::begin(c), e = std::end(c);
		auto size = std::distance(b, e);
		auto sum = std::accumulate(b, e, T());
		auto mean = sum / size;
		return mean;
	}

	template <typename Container, typename T = typename std::decay<decltype(*std::begin(std::declval<Container>()))>::type>
	T median(Container&& c)
	{
		auto b = std::begin(c), e = std::end(c);
		auto size = std::distance(b, e);
		if (size % 2 == 0) {
			auto med1 = size / 2 - 1, med2 = size / 2;
			std::nth_element(b, b + med1, e);
			std::nth_element(b, b + med2, e);
			return (*(b + med1) + *(b + med2)) / 2.;
		}
		else {
			auto med = size / 2;
			std::nth_element(b, b + med, e);
			return *(b + med);
		}
	}

	template <typename Container, typename T = typename std::decay<decltype(*std::begin(std::declval<Container>()))>::type>
	T quantile(Container&& c, float q)
	{
		auto b = std::begin(c), e = std::end(c);
		auto qcorrect = (q <= 1) ? q : q / 100.;
		const auto pos = qcorrect * std::distance(b, e);
		std::nth_element(b, b + pos, e);
		return *(b + pos);
	}

	template <typename T>
	void flipTriangles(std::vector<T>& _triangles) {
		assert(_triangles.size() % 3 == 0);
		for (auto n = 0; n < _triangles.size(); n += 3)
			std::swap(_triangles[n], _triangles[n + 2]);
	}

	std::vector < std::string >& split(const std::string&, char, std::vector < std::string >&);
}

namespace poca::geometry {
	void createCubeFromVector(std::vector <poca::core::Vec3mf>&, const poca::core::Vec6md&);
	template <typename T> int sgn(T);
	float sign(const poca::core::Vec3mf&, const poca::core::Vec3mf&, const poca::core::Vec3mf&);
	bool pointInTriangle(const poca::core::Vec3mf&, const poca::core::Vec3mf&, const poca::core::Vec3mf&, const poca::core::Vec3mf&);
	bool sameSide(const poca::core::Vec3mf&, const poca::core::Vec3mf&, const poca::core::Vec3mf&, const poca::core::Vec3mf&, const poca::core::Vec3mf&);
	bool pointInTetrahedron(const poca::core::Vec3mf&, const poca::core::Vec3mf&, const poca::core::Vec3mf&, const poca::core::Vec3mf&, const poca::core::Vec3mf&);

	bool onSegment(const double, const double, const double, const double, const double, const double);
	size_t orientation(const double, const double, const double, const double, const double, const double);
	bool doIntersect(const double, const double, const double, const double, const double, const double, const double, const double);
	
	template <class T>
	T spearmanRankCorrelationCoefficient(const std::vector <T>& X, const std::vector <T>& Y)
	{
		std::vector <T> XRanks(X.size()), YRanks(Y.size());
		rank(X, XRanks);
		rank(Y, YRanks);
		size_t n = XRanks.size();
		double sum_X = 0, sum_Y = 0, sum_XY = 0;
		double squareSum_X = 0, squareSum_Y = 0;

		for (int i = 0; i < n; i++)
		{
			// sum of elements of array X. 
			sum_X = sum_X + XRanks[i];

			// sum of elements of array Y. 
			sum_Y = sum_Y + YRanks[i];

			// sum of X[i] * Y[i]. 
			sum_XY = sum_XY + XRanks[i] * YRanks[i];

			// sum of square of array elements. 
			squareSum_X = squareSum_X + XRanks[i] * XRanks[i];
			squareSum_Y = squareSum_Y + YRanks[i] * YRanks[i];
		}

		// use formula for calculating 
		// correlation coefficient. 
		T corr = (T)(n * sum_XY - sum_X * sum_Y) / sqrt((n * squareSum_X - sum_X * sum_X) * (n * squareSum_Y - sum_Y * sum_Y));

		return corr;
	}
	//float spearmanRankCorrelationCoefficient2(const std::vector <float>&, const std::vector <float>&);

	void circleLineIntersect(const float, const float, const float, const float, const float, const float, const float, std::vector <poca::core::Vec2mf>&);
	float computeAreaCircularSegment(const float, const float, const float, const poca::core::Vec2mf&, const poca::core::Vec2mf&);
	//float computePolygonArea(poca::core::Vec3md*, const unsigned int);

	template <class T>
	T distance(const T _x0, const T _y0, const T _x1, const T _y1)
	{
		return sqrt(distanceSqr(_x0, _y0, _x1, _y1));
	}

	template <class T>
	T distanceSqr(const T _x0, const T _y0, const T _x1, const T _y1)
	{
		float x = _x1 - _x0, y = _y1 - _y0;
		return x * x + y * y;
	}

	template <class T>
	T distance3D(const T _x0, const T _y0, const T _z0, const T _x1, const T _y1, const T _z1)
	{
		return sqrt(distance3DSqr(_x0, _y0, _z0, _x1, _y1, _z1));
	}

	template <class T>
	T distance3DSqr(const T _x0, const T _y0, const T _z0, const T _x1, const T _y1, const T _z1)
	{
		float x = _x1 - _x0, y = _y1 - _y0, z = _z1 - _z0;
		return x * x + y * y + z * z;
	}

	template <class T>
	T sign(const T _p1x, const T _p1y, const T _p2x, const T _p2y, const T _p3x, const T _p3y)
	{
		return (_p1x - _p3x) * (_p2y - _p3y) - (_p2x - _p3x) * (_p1y - _p3y);
	}

	template <class T>
	bool pointInTriangle(const T _ptx, const T _pty, const T _v1x, const T _v1y, const T _v2x, const T _v2y, const T _v3x, const T _v3y)
	{
		T d1, d2, d3;
		bool has_neg, has_pos;

		d1 = sign<T>(_ptx, _pty, _v1x, _v1y, _v2x, _v2y);
		d2 = sign<T>(_ptx, _pty, _v2x, _v2y, _v3x, _v3y);
		d3 = sign<T>(_ptx, _pty, _v3x, _v3y, _v1x, _v1y);

		has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
		has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

		return !(has_neg && has_pos);
	}

	template <class T>
	void barycenter(T& _bx, T& _by, const T _p1x, const T _p1y, const T _p2x, const T _p2y, const T _p3x, const T _p3y)
	{
		_bx = (_p1x + _p2x + _p3x) / (T)3;
		_by = (_p1y + _p2y + _p3y) / (T)3;
	}
}

namespace poca::core::utils {
	QTabWidget* addSingleTabWidget(QTabWidget* _parent, const QString& _nameMainTab, const QString& _nameSubTab, QWidget* _widget);
	QTabWidget* addWidget(QTabWidget* _parent, const QString& _nameMainTab, const QString& _nameSubTab, QWidget* _widget, bool _first = true);
	const bool isFileExtensionInList(const QString&, const QStringList&);
	const bool isExtensionInList(const QString&, const QStringList&);
	void getJsonsFromString(const QString&, std::vector <nlohmann::json>&);
}

#endif

