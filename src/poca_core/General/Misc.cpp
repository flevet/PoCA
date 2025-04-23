/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Misc.cpp
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

#include <numeric>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <random>
#include <assert.h>
#include <sys/stat.h>

#include "Misc.h"

namespace poca::core {
	Color4D contrastColor(const Color4D& _backC)
	{
		float d = 0.f;

		// Counting the perceptive luminance - human eye favors green color... 
		double a = 1 - (0.299 * _backC[0] + 0.587 * _backC[1] + 0.114 * _backC[2]) / 255;

		if (a < 0.5)
			d = 0.F; // bright colors - black font
		else
			d = 1.f; // dark colors - white font

		return Color4D(d, d, d, 1.f);
	}

	int roundUpGeneric(int numToRound, int multiple)
	{
		assert(multiple);
		return ((numToRound + multiple - 1) / multiple) * multiple;
	}

	int roundUpPowerOfTwo(int numToRound, int multiple)
	{
		assert(multiple);
		int isPositive = (int)(numToRound >= 0);
		return ((numToRound + isPositive * (multiple - 1)) / multiple) * multiple;
	}

	float roundFromZero(float _x) {
		return _x < 0.f ? floor(_x) : ceil(_x);
	}

	bool file_exists(const std::string& _name) {
		struct stat buffer;
		return (stat(_name.c_str(), &buffer) == 0);
	}

	void getColorString(double _r, double _g, double _b, char _str[32])
	{
		int r = (int)(255. * _r);
		int g = (int)(255. * _g);
		int b = (int)(255. * _b);
		int rc = (r < 0) ? 0 : (r > 255) ? 255 : r;
		int gc = (g < 0) ? 0 : (g > 255) ? 255 : g;
		int bc = (b < 0) ? 0 : (b > 255) ? 255 : b;
		sprintf(_str, "#%2.2x%2.2x%2.2x", rc, gc, bc);
	}

	void getColorStringUC(unsigned char _r, unsigned char _g, unsigned char _b, char _str[32])
	{
		int r = _r;
		int g = _g;
		int b = _b;
		int rc = (r < 0) ? 0 : (r > 255) ? 255 : r;
		int gc = (g < 0) ? 0 : (g > 255) ? 255 : g;
		int bc = (b < 0) ? 0 : (b > 255) ? 255 : b;
		sprintf(_str, "#%2.2x%2.2x%2.2x", rc, gc, bc);
	}

	std::vector < std::string >& split(const std::string& s, char delim, std::vector < std::string >& elems) {
		std::stringstream ss(s);
		std::string item;
		int cpt = 0;
		while (std::getline(ss, item, delim)) {
			elems.push_back(item);
		}
		return elems;
	}

	float sRGBtoLin(const float _colorChannel)
	{
		// Send this function a decimal sRGB gamma encoded color value
		// between 0.0 and 1.0, and it returns a linearized value.

		if (_colorChannel <= 0.04045f) {
			return _colorChannel / 12.92f;
		}
		else {
			return pow(((_colorChannel + 0.055f) / 1.055f), 2.4f);
		}
	}

	float luminance(const float _r, const float _g, const float _b)
	{
		return 0.2126f * sRGBtoLin(_r) + 0.7152f * sRGBtoLin(_g) + 0.0722f * sRGBtoLin(_b);
	}

	float perceivedLightness(const float _luminance)
	{
		// Send this function a luminance value between 0.0 and 1.0,
		// and it returns L* which is "perceptual lightness"

		if (_luminance <= (216.f / 24389.f)) {       // The CIE standard states 0.008856 but 216/24389 is the intent for 0.008856451679036
			return _luminance * (24389.f / 27.f);  // The CIE standard states 903.3, but 24389/27 is the intent, making 903.296296296296296
		}
		else {
			return pow(_luminance, (1.f / 3.f)) * 116.f - 16.f;
		}
	}

	Color4uc randomColorB()
	{
		Color4D color = randomColorD();
		return Color4uc(color[0] * 255.f, color[1] * 255.f, color[2] * 255.f, color[3] * 255.f);
	}

	Color4D randomColorD()
	{
		float r = (float)rand() / (float)RAND_MAX;
		float g = (float)rand() / (float)RAND_MAX;
		float b = (float)rand() / (float)RAND_MAX;
		float a = 1.f;
		return Color4D(r, g, b, a);
	}

	Color4uc randomBrightColorB(const uint32_t _tresh)
	{
		Color4D color = randomBrightColorD((float)_tresh / 255.f);
		return Color4uc(color[0] * 255.f, color[1] * 255.f, color[2] * 255.f, color[3] * 255.f);
	}

	Color4D randomBrightColorD(const float _thesh)
	{
		float a;
		Color4D color;
		do {
			color = randomColorD();
			// Counting the perceptive luminance - human eye favors green color... 
			a = 1 - (0.299f * color[0] + 0.587f * color[1] + 0.114f * color[2]);
		} while (a > _thesh);
		return Color4D(color[0], color[1], color[2], color[3]);
	}

	Color4uc randomDarkColorB(const uint32_t _tresh)
	{
		Color4D color = randomBrightColorD((float)_tresh / 255.f);
		return Color4uc(color[0] * 255.f, color[1] * 255.f, color[2] * 255.f, color[3] * 255.f);
	}

	Color4D randomDarkColorD(const float _thesh)
	{
		float a;
		Color4D color;
		do {
			color = randomColorD();
			// Counting the perceptive luminance - human eye favors green color... 
			a = 1 - (0.299f * color[0] + 0.587f * color[1] + 0.114f * color[2]);
		} while (a > _thesh);
		return Color4D(color[0], color[1], color[2], color[3]);
	}

	void randomPointsOnUnitSphere(const uint32_t _nbPoints, std::vector<Vec3mf>& _points)
	{
		std::default_random_engine generator;
		std::normal_distribution<float> distribution(0.f, 1.f);

		_points.resize(_nbPoints);
		for (auto n = 0; n < _nbPoints; n++) {
			Vec3mf pt(distribution(generator), distribution(generator), distribution(generator)), norm(pt);
			pt.normalize();
			_points[n] = pt;
		}
	}

	double linear(double _x, const double* _p)
	{
		return _x * _p[0] + _p[1];
	}

	double linearFixed(double _x, const double* _p)
	{
		return _x * _p[0];
	}

	void computePhotophysicsParameters(const std::vector <float>& _frames, std::vector <float>& _photophysics)
	{
		bool isDown = false;
		assert(_photophysics.size() == 6);

		std::fill(_photophysics.begin(), _photophysics.begin() + 6, 0);
		_photophysics[poca::core::TOTAL_ON] = _frames.size();
		_photophysics[poca::core::NB_ON] = 1;

		if (_frames.size() == 1)
			_photophysics[poca::core::LIFETIME] = 1;
		else {
			for (auto n = 1; n < _frames.size(); n++) {
				auto dt = (_frames[n] - _frames[n - 1]) - 1;
				if (dt > 0) {
					_photophysics[poca::core::NB_ON] = _photophysics[poca::core::NB_ON] + 1;
					_photophysics[poca::core::NB_OFF] = _photophysics[poca::core::NB_OFF] + 1;
					_photophysics[poca::core::NB_BLINKS] = _photophysics[poca::core::NB_BLINKS] + 1;
					_photophysics[poca::core::TOTAL_OFF] = _photophysics[poca::core::TOTAL_OFF] + dt;
				}
			}
			_photophysics[poca::core::LIFETIME] = _photophysics[poca::core::TOTAL_ON] + _photophysics[poca::core::TOTAL_OFF];
		}
	}

	void PrintFullPath(const char* partialPath)
	{
		printf("Connecting to Python.\n");
		char full[_MAX_PATH];
		if (_fullpath(full, partialPath, _MAX_PATH) != NULL)
			printf("Full path is: %s\n", full);
		else
			printf("Invalid path\n");
	}
}

namespace poca::geometry {
	void createCubeFromVector(poca::core::Vec3mf* _cubeV, const poca::core::Vec6md& _vector)
	{
		poca::core::Vec3mf verticesBBox[8];
		//Cube -> 6 faces -> 12 lines -> 24 vertices

		double x = _vector[0], y = _vector[1], z = _vector[2], w = _vector[3], h = _vector[4], t = _vector[5];
		const int indexes[24] = { 0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 1, 5, 0, 4, 2, 6, 3, 7 };

		verticesBBox[0].set(x, y, z);
		verticesBBox[1].set(x, h, z);
		verticesBBox[2].set(w, h, z);
		verticesBBox[3].set(w, y, z);
		verticesBBox[4].set(x, y, t);
		verticesBBox[5].set(x, h, t);
		verticesBBox[6].set(w, h, t);
		verticesBBox[7].set(w, y, t);

		for (unsigned int n = 0; n < 24; n++)
			_cubeV[n] = verticesBBox[indexes[n]];
	}

	void createCubeFromVector(std::vector <poca::core::Vec3mf>& _cubeV, const poca::core::Vec6md& _vector)
	{
		poca::core::Vec3mf verticesBBox[8];
		//Cube -> 6 faces -> 12 lines -> 24 vertices
		_cubeV.resize(24);

		double x = _vector[0], y = _vector[1], z = _vector[2], w = _vector[3], h = _vector[4], t = _vector[5];
		const int indexes[24] = { 0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 1, 5, 0, 4, 2, 6, 3, 7 };

		verticesBBox[0].set(x, y, z);
		verticesBBox[1].set(x, h, z);
		verticesBBox[2].set(w, h, z);
		verticesBBox[3].set(w, y, z);
		verticesBBox[4].set(x, y, t);
		verticesBBox[5].set(x, h, t);
		verticesBBox[6].set(w, h, t);
		verticesBBox[7].set(w, y, t);

		for (unsigned int n = 0; n < 24; n++)
			_cubeV[n] = verticesBBox[indexes[n]];
	}

	float sign(const poca::core::Vec3mf& _p1, const poca::core::Vec3mf& _p2, const poca::core::Vec3mf& _p3)
	{
		return (_p1.x() - _p3.x()) * (_p2.y() - _p3.y()) - (_p2.x() - _p3.x()) * (_p1.y() - _p3.y());
	}

	template <typename T> int sgn(T val) {
		return (T(0) < val) - (val < T(0));
	}

	bool pointInTriangle(const poca::core::Vec3mf& _pt, const poca::core::Vec3mf& _v1, const poca::core::Vec3mf& _v2, const poca::core::Vec3mf& _v3)
	{
		float d1, d2, d3;
		bool has_neg, has_pos;

		d1 = sign(_pt, _v1, _v2);
		d2 = sign(_pt, _v2, _v3);
		d3 = sign(_pt, _v3, _v1);

		has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
		has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

		return !(has_neg && has_pos);
	}

	bool sameSide(const poca::core::Vec3mf& _v1, const poca::core::Vec3mf& _v2, const poca::core::Vec3mf& _v3, const poca::core::Vec3mf& _v4, const poca::core::Vec3mf& _p)
	{
		poca::core::Vec3mf normal = (_v2 - _v1).cross(_v3 - _v1);
		float dotV4 = normal.dot(_v4 - _v1);
		float dotP = normal.dot(_p - _v1);
		return sgn<float>(dotV4) == sgn<float>(dotP);
	}

	bool pointInTetrahedron(const poca::core::Vec3mf& _v1, const poca::core::Vec3mf& _v2, const poca::core::Vec3mf& _v3, const poca::core::Vec3mf& _v4, const poca::core::Vec3mf& _p)
	{
		return sameSide(_v1, _v2, _v3, _v4, _p) &&
			sameSide(_v2, _v3, _v4, _v1, _p) &&
			sameSide(_v3, _v4, _v1, _v2, _p) &&
			sameSide(_v4, _v1, _v2, _v3, _p);
	}

	// Given three colinear points p, q, r, the function checks if 
	// point q lies on line segment 'pr' 
	bool onSegment(const double _px, const double _py, const double _qx, const double _qy, const double _rx, const double _ry)
	{
		if (_qx <= std::max(_px, _rx) && _qx >= std::min(_px, _rx) &&
			_qy <= std::max(_py, _ry) && _qy >= std::min(_py, _ry))
			return true;

		return false;
	}

	// To find orientation of ordered triplet (p, q, r). 
	// The function returns following values 
	// 0 --> p, q and r are colinear 
	// 1 --> Clockwise 
	// 2 --> Counterclockwise
	size_t orientation(const double _px, const double _py, const double _qx, const double _qy, const double _rx, const double _ry)
	{
		// See https://www.geeksforgeeks.org/orientation-3-ordered-points/ 
		// for details of below formula. 
		int val = (_qy - _py) * (_rx - _qx) -
			(_qx - _px) * (_ry - _qy);

		if (val == 0) return 0;  // colinear 

		return (val > 0) ? 1 : 2; // clock or counterclock wise 
	}

	// The main function that returns true if line segment 'p1q1' 
	// and 'p2q2' intersect. 
	bool doIntersect(const double _p1x, const double _p1y, const double _q1x, const double _q1y, const double _p2x, const double _p2y, const double _q2x, const double _q2y)
	{
		// Find the four orientations needed for general and 
		// special cases 
		size_t o1 = orientation(_p1x, _p1y, _q1x, _q1y, _p2x, _p2y);
		size_t o2 = orientation(_p1x, _p1y, _q1x, _q1y, _q2x, _q2y);
		size_t o3 = orientation(_p2x, _p2y, _q2x, _q2y, _p1x, _p1y);
		size_t o4 = orientation(_p2x, _p2y, _q2x, _q2y, _q1x, _q1y);
		
		// General case 
		if (o1 != o2 && o3 != o4)
			return true;

		// Special Cases 
		// p1, q1 and p2 are colinear and p2 lies on segment p1q1 
		if (o1 == 0 && onSegment(_p1x, _p1y, _p2x, _p2y, _q1x, _q1y)) return true;

		// p1, q1 and q2 are colinear and q2 lies on segment p1q1 
		if (o2 == 0 && onSegment(_p1x, _p1y, _q2x, _q2y, _q1x, _q1y)) return true;

		// p2, q2 and p1 are colinear and p1 lies on segment p2q2 
		if (o3 == 0 && onSegment(_p2x, _p2y, _p1x, _p1y, _q2x, _q2y)) return true;

		// p2, q2 and q1 are colinear and q1 lies on segment p2q2 
		if (o4 == 0 && onSegment(_p2x, _p2y, _q1x, _q1y, _q2x, _q2y)) return true;

	}

	void circleLineIntersect(const float _x1, const float _y1, const float _x2, const float _y2, const float _cx, const float _cy, const float _cr, std::vector <poca::core::Vec2mf>& _points)
	{
		float dx = _x2 - _x1;
		float dy = _y2 - _y1;
		float a = dx * dx + dy * dy;
		float b = 2 * (dx * (_x1 - _cx) + dy * (_y1 - _cy));
		float c = _cx * _cx + _cy * _cy;
		c += _x1 * _x1 + _y1 * _y1;
		c -= 2 * (_cx * _x1 + _cy * _y1);
		c -= _cr * _cr;
		float bb4ac = b * b - 4 * a * c;

		if (bb4ac < 0) {  // Not intersecting
			return;
		}
		else {

			float mu = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
			float ix1 = _x1 + mu * (dx);
			float iy1 = _y1 + mu * (dy);
			mu = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
			float ix2 = _x1 + mu * (dx);
			float iy2 = _y1 + mu * (dy);
			if (_x1 <= ix1 && ix1 <= _x2 && _y1 <= iy1 && iy1 <= _y2)
				_points.push_back(poca::core::Vec2mf(ix1, iy1));
			if (_x1 <= ix2 && ix2 <= _x2 && _y1 <= iy2 && iy2 <= _y2)
				_points.push_back(poca::core::Vec2mf(ix2, iy2));
		}
	}

	float computeAreaCircularSegment(const float _cx, const float _cy, const float _r, const poca::core::Vec2mf& _p1, const poca::core::Vec2mf& _p2) {
		float x = (_p1.x() + _p2.x()) / 2., y = (_p1.y() + _p2.y()) / 2.;
		float smallR = distance(x, y, _cx, _cy);
		float h = _r - smallR;

		float area = (_r * _r) * acos((_r - h) / _r) - ((_r - h) * sqrt((2. * _r * h) - (h * h)));
		return area;
	}
}

namespace poca::core::utils {
	QTabWidget* addSingleTabWidget(QTabWidget* _parent, const QString& _nameMainTab, const QString& _nameSubTab, QWidget* _widget)
	{
		QTabWidget* tabW = NULL;
		int pos = -1;
		for (int n = 0; n < _parent->count(); n++)
			if (_parent->tabText(n) == _nameMainTab)
				pos = n;
		if (pos != -1) {
			tabW = static_cast <QTabWidget*>(_parent->widget(pos));
		}
		else {
			pos = _parent->addTab(new QTabWidget, _nameMainTab);
			tabW = static_cast <QTabWidget*>(_parent->widget(pos));
		}

		int index = tabW->addTab(_widget, _nameSubTab);
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
		tabW->setTabVisible(index, false);
#endif
		return tabW;
	}

	QTabWidget* addWidget(QTabWidget* _parent, const QString& _nameMainTab, const QString& _nameSubTab, QWidget* _widget, bool _first)
	{
		QTabWidget* parentTab = NULL;
		int pos = -1;
		for (int n = 0; n < _parent->count(); n++)
			if (_parent->tabText(n) == _nameMainTab)
				pos = n;
		if (pos != -1) {
			parentTab = static_cast <QTabWidget*>(_parent->widget(pos));
			pos = -1;
			for (int n = 0; n < parentTab->count(); n++)
				if (parentTab->tabText(n) == _nameSubTab)
					pos = n;
			QTabWidget* tabW2 = NULL;
			if (pos == -1) {
				tabW2 = new QTabWidget;
				QVBoxLayout* layout = new QVBoxLayout;
				/*QWidget* emptyW = new QWidget;
				emptyW->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
				layout->addWidget(emptyW);*/
				tabW2->setLayout(layout);
				parentTab->addTab(tabW2, _nameSubTab);
			}
			else
				tabW2 = static_cast <QTabWidget*>(parentTab->widget(pos));
			QLayout* layout = tabW2->layout();
			QVBoxLayout* vlayout = dynamic_cast <QVBoxLayout*>(layout);
			/*if (!vlayout)
				tabW2->addTab(w, QObject::tr("Nearest neighbor distribution"));
			else*/ {
				int size = vlayout->count();
				size = size > 0 ? size - 1 : size;
				vlayout->insertWidget(_first ? 0 : size, _widget);
				parentTab->update();
			}
		}
		else {
			parentTab = new QTabWidget;
			_parent->addTab(parentTab, _nameMainTab);
			QTabWidget* miscWidget = new QTabWidget;
			QVBoxLayout* layout = new QVBoxLayout;
			QWidget* emptyW = new QWidget;
			emptyW->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
			layout->addWidget(emptyW);
			miscWidget->setLayout(layout);
			parentTab->addTab(miscWidget, _nameSubTab);
			int size = layout->count();
			size = size > 0 ? size - 1 : size;
			layout->insertWidget(size, _widget);
			miscWidget->update();
		}
		return parentTab;
	}

	const bool isFileExtensionInList(const QString& _filename, const QStringList& _extensions)
	{
		int index = _filename.lastIndexOf(".");
		if (index == -1)
			return false;
		QString ext = _filename.right(_filename.size() - index);
		return isExtensionInList(ext, _extensions);
	}

	const bool isExtensionInList(const QString& _ext, const QStringList& _extensions)
	{
		for (const auto& extension : _extensions)
			if (extension == _ext)
				return true;
		return false;
	}

	void getJsonsFromString(const QString& _text, std::vector <nlohmann::json>& _jsons)
	{
		_jsons.clear();
		for (QString commandTxt : _text.split("\n")) {
			if (commandTxt.isEmpty()) continue;
			std::stringstream ss;
			ss.str(commandTxt.toStdString());
			nlohmann::json json;
			ss >> json;
			if (json.empty()) continue;
			_jsons.push_back(json);
		}
	}
}

