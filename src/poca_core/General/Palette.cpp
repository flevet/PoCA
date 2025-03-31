/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Palette.cpp
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

/*Copyright (c) 2010 Maxime Petitjean

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.*/

#include <assert.h>
#include <time.h>

#include "../General/Palette.hpp"
#include "../General/Misc.h"

namespace poca::core {

	uint32_t Palette::m_seedRand = 20;

	Palette::Palette()
	{

	}

	Palette::Palette(const Color4uc _color_begin, const Color4uc _color_end, const std::string& _name, const bool _hilow) : m_name(_name), m_hilow(_hilow)
	{
		m_gradient.insert(std::make_pair(0.f, _color_begin));
		m_gradient.insert(std::make_pair(1.f, _color_end));
		m_begin = 0.; m_end = 1.;
		m_from = 0; m_to = 256;
	}

	Palette::Palette(const Palette& _o) :m_gradient(_o.m_gradient), m_from(_o.m_from), m_to(_o.m_to), m_begin(_o.m_begin), m_end(_o.m_end), m_name(_o.m_name), m_autoscale(_o.m_autoscale), m_hilow(_o.m_hilow), m_filterMin(_o.m_filterMin), m_filterMax(_o.m_filterMax)
	{
	}

	void Palette::setPalette(PaletteInterface* _pal)
	{
		Palette* palette = dynamic_cast<Palette*>(_pal);
		if (palette == NULL)
			return;
		setPalette(*palette);
	}

	void Palette::setPalette(const Palette& _o)
	{
		m_gradient = _o.m_gradient;
		m_name = _o.m_name;
		m_begin = _o.m_begin;
		m_end = _o.m_end;
		m_from = _o.m_from;
		m_to = _o.m_to;
		m_autoscale = _o.m_autoscale;
		m_hilow = _o.m_hilow;
		m_filterMin = _o.m_filterMin;
		m_filterMax = _o.m_filterMax;
	}

	std::set <std::pair<float, Color4uc>>::iterator Palette::getElement(unsigned int _index)
	{
		assert(_index < (unsigned int)m_gradient.size());
		std::set <std::pair<float, Color4uc>>::iterator it = m_gradient.begin();
		for (unsigned int n = 0; n < _index; n++) it++;
		return it;
	}

	std::set <std::pair<float, Color4uc>>::const_iterator Palette::getElement(unsigned int _index) const
	{
		assert(_index < (unsigned int)m_gradient.size());
		std::set <std::pair<float, Color4uc>>::const_iterator it = m_gradient.begin();
		for (unsigned int n = 0; n < _index; n++) it++;
		return it;
	}

	void Palette::setColor(float _position, Color4uc _color)
	{
		assert(_position >= 0.0 && _position <= 1.0);
		m_gradient.insert(std::make_pair(_position, _color));
	}

	void Palette::removeColorAt(unsigned int _index)
	{
		std::set <std::pair<float, Color4uc>>::iterator it = getElement(_index);
		m_gradient.erase(it);
	}

	Color4uc Palette::colorAt(unsigned int _index) const
	{
		std::set <std::pair<float, Color4uc>>::const_iterator it = getElement(_index);
		return it->second;
	}


	void Palette::setColorAt(unsigned int _index, Color4uc _color)
	{
		std::set <std::pair<float, Color4uc>>::iterator it = getElement(_index);
		float pos = it->first;
		m_gradient.erase(it);
		m_gradient.insert(std::make_pair(pos, _color));
	}

	float Palette::colorPosition(unsigned int _index) const
	{
		std::set <std::pair<float, Color4uc>>::const_iterator it = getElement(_index);
		return it->first;
	}

	void Palette::setColorPosition(unsigned int _index, float _position)
	{
		std::set <std::pair<float, Color4uc>>::iterator it = getElement(_index);
		Color4uc color = it->second;
		m_gradient.erase(it);
		m_gradient.insert(std::pair(_position, color));
	}

	void Palette::getGradientInfos(std::vector <float>& _pos, std::vector <Color4uc>& _colors) const
	{
		for (std::set <std::pair<float, Color4uc>>::const_iterator it = m_gradient.begin(); it != m_gradient.end(); it++) {
			_pos.push_back(it->first);
			_colors.push_back(it->second);
		}
	}

	Palette Palette::getStaticLut(const std::string& _lut)
	{
		if (_lut == std::string("Gray")) {
			return Palette(Color4uc(0, 0, 0, 255), Color4uc(255, 255, 255, 255), _lut);
		}
		else if (_lut == std::string("Red")) {
			return Palette(Color4uc(0, 0, 0, 255), Color4uc(255, 0, 0, 255), _lut);
		}
		else if (_lut == std::string("Green")) {
			return Palette(Color4uc(0, 0, 0, 255), Color4uc(0, 255, 0, 255), _lut);
		}
		else if (_lut == std::string("Blue")) {
			return Palette(Color4uc(0, 0, 0, 255), Color4uc(0, 0, 255, 255), _lut);
		}
		else if (_lut == std::string("AllGray")) {
			return Palette(Color4uc(125, 125, 125, 255), Color4uc(125, 125, 125, 255), _lut);
		}
		else if (_lut == std::string("AllYellow")) {
			return Palette(Color4uc(255, 255, 0, 255), Color4uc(255, 255, 0, 255), _lut);
		}
		else if (_lut == std::string("AllRedColorBlind")) {
			return Palette(Color4uc(128, 0, 128, 255), Color4uc(128, 0, 128, 255), _lut);
		}
		else if (_lut == std::string("AllGreenColorBlind")) {
			return Palette(Color4uc(0, 200, 0, 255), Color4uc(0, 200, 0, 255), _lut);
		}
		else if (_lut == std::string("Fire")) {
			int r[] = { 0,0,1,25,49,73,98,122,146,162,173,184,195,207,217,229,240,252,255,255,255,255,255,255,255,255,255,255,255,255,255,255 };
			int g[] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,14,35,57,79,101,117,133,147,161,175,190,205,219,234,248,255,255,255,255 };
			int b[] = { 0,61,96,130,165,192,220,227,210,181,151,122,93,64,35,5,0,0,0,0,0,0,0,0,0,0,0,35,98,160,223,255 };
			int w = 32;
			std::vector < Color4uc > colors;
			for (int i = 1; i < w - 1; i++)
				colors.push_back(Color4uc(r[i], g[i], b[i], 255));
			Palette palette(Color4uc(0, 0, 0, 255), Color4uc(255, 255, 255, 255), _lut);
			float step = 1.f / 32.f;
			float cur = step;
			for (unsigned int i = 0; i < colors.size(); i++, cur += step)
				palette.setColor(cur, colors[i]);
			return palette;
		}
		else if (_lut == std::string("InvFire")) {
			int r[] = { 0,0,1,25,49,73,98,122,146,162,173,184,195,207,217,229,240,252,255,255,255,255,255,255,255,255,255,255,255,255,255,255 };
			int g[] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,14,35,57,79,101,117,133,147,161,175,190,205,219,234,248,255,255,255,255 };
			int b[] = { 0,61,96,130,165,192,220,227,210,181,151,122,93,64,35,5,0,0,0,0,0,0,0,0,0,0,0,35,98,160,223,255 };
			int w = 32;
			std::vector < Color4uc > colors;
			for (int i = w - 2; i >= 0; i--)
				colors.push_back(Color4uc(r[i], g[i], b[i], 255));
			Palette palette(Color4uc(255, 255, 255, 255), Color4uc(0, 0, 0, 255), _lut);
			float step = 1.f / 32.f;
			float cur = step;
			for (unsigned int i = 0; i < colors.size(); i++, cur += step)
				palette.setColor(cur, colors[i]);
			return palette;
		}
		else if (_lut == std::string("Ice")) {
			int r[] = { 0,0,0,0,0,0,19,29,50,48,79,112,134,158,186,201,217,229,242,250,250,250,250,251,250,250,250,250,251,251,243,230 };
			int g[] = { 156,165,176,184,190,196,193,184,171,162,146,125,107,93,81,87,92,97,95,93,93,90,85,69,64,54,47,35,19,0,4,0 };
			int b[] = { 140,147,158,166,170,176,209,220,234,225,236,246,250,251,250,250,245,230,230,222,202,180,163,142,123,114,106,94,84,64,26,27 };
			int w = 32;
			std::vector < Color4uc > colors;
			for (int i = 1; i < w - 1; i++)
				colors.push_back(Color4uc(r[i], g[i], b[i], 255));
			Palette palette(Color4uc(0, 0, 0, 255), Color4uc(255, 255, 255, 255), _lut);
			float step = 1.f / 32.f;
			float cur = step;
			for (unsigned int i = 0; i < colors.size(); i++, cur += step)
				palette.setColor(cur, colors[i]);
			return palette;
		}
		else if (_lut == std::string("AllBlue")) {
			return Palette(Color4uc(0, 85, 255, 255), Color4uc(0, 85, 255, 255), _lut);
		}
		else if (_lut == std::string("AllGreen")) {
			return Palette(Color4uc(0, 170, 127, 255), Color4uc(0, 170, 127, 255), _lut);
		}
		else if (_lut == std::string("AllRed")) {
			return Palette(Color4uc(255, 0, 0, 255), Color4uc(255, 0, 0, 255), _lut);
		}
		else if (_lut == std::string("AllOrange")) {
			return Palette(Color4uc(255, 165, 0, 255), Color4uc(255, 165, 0, 255), _lut);
		}
		else if (_lut == std::string("AllTomato")) {
			return Palette(Color4uc(255, 69, 0, 255), Color4uc(255, 69, 0, 255), _lut);
		}
		else if (_lut == std::string("AllCyan")) {
			return Palette(Color4uc(0, 255, 255, 255), Color4uc(0, 255, 255, 255), _lut);
		}
		else if (_lut == std::string("AllTurquoise")) {
			return Palette(Color4uc(64, 224, 208, 255), Color4uc(64, 224, 208, 255), _lut);
		}
		else if (_lut == std::string("AllWhite")) {
			return Palette(Color4uc(255, 255, 255, 255), Color4uc(255, 255, 255, 255), _lut);
		}
		else if (_lut == std::string("AllBlack")) {
			return Palette(Color4uc(0, 0, 0, 255), Color4uc(0, 0, 0, 255), _lut);
		}
		else if (_lut == std::string("HotCold")) {
			Palette palette(Color4uc(0, 0, 255, 255), Color4uc(170, 0, 255, 255), _lut);
			palette.setColor(0.1f, Color4uc(0, 170, 255, 255));
			palette.setColor(0.225f, Color4uc(103, 255, 139, 255));
			palette.setColor(0.35f, Color4uc(255, 255, 0, 255));
			palette.setColor(0.5f, Color4uc(255, 170, 0, 255));
			palette.setColor(0.7f, Color4uc(255, 0, 0, 255));
			return palette;
		}
		else if (_lut == std::string("HotCold2")) {
			Palette palette(Color4uc(0, 0, 255, 255), Color4uc(170, 0, 255, 255), _lut);
			palette.setColor(0.16f, Color4uc(0, 170, 255, 255));
			palette.setColor(0.33f, Color4uc(103, 255, 139, 255));
			palette.setColor(0.49f, Color4uc(255, 255, 0, 255));
			palette.setColor(0.66f, Color4uc(255, 170, 0, 255));
			palette.setColor(0.82f, Color4uc(255, 0, 0, 255));
			return palette;
		}
		else if (_lut == std::string("HiLow")) {
			Palette palette(Color4uc(128, 0, 128, 255), Color4uc(0, 200, 0, 255), _lut, true);
			palette.setColor(0.499, Color4uc(128, 0, 128, 255));
			palette.setColor(0.501, Color4uc(0, 200, 0, 255));
			return palette;
		}
		else if (_lut == std::string("Heatmap")) {
			Palette palette(Color4uc(33, 102, 172, 0), Color4uc(178, 24, 43, 255), _lut);
			palette.setColor(0.2f, Color4uc(103, 169, 207, 255));
			palette.setColor(0.4f, Color4uc(209, 229, 240, 255));
			palette.setColor(0.6f, Color4uc(253, 219, 199, 255));
			palette.setColor(0.8f, Color4uc(239, 138, 98, 255));
			return palette;
		}
		else if (_lut == std::string("HiLo")) {
			Palette palette(Color4uc(128, 0, 128, 255), Color4uc(0, 200, 0, 255), _lut);
			palette.setHiLow(true);
			return palette;
		}
		else if (_lut == std::string("Random")) {
			srand(m_seedRand);
			Palette palette(randomColorB(), randomColorB(), _lut);
			float step = 0.0001f;
			for (float cur = step; cur < 1.f; cur += step)
				palette.setColor(cur, randomColorB());
			m_seedRand = time(NULL);
			return palette;
		}
		else if (_lut == std::string("LightGrayscale")) {
			return Palette(Color4uc(0, 0, 0, 255), Color4uc(255, 255, 255, 255), _lut);
		}
		else if (_lut == std::string("DarkGrayscale")) {
			return Palette(Color4uc(255, 255, 255, 255), Color4uc(0, 0, 0, 255), _lut);
		}
		return Palette();
	}

	Palette* Palette::getStaticLutPtr(const std::string& _lut)
	{
		if (_lut == std::string("Gray")) {
			return new Palette(Color4uc(0, 0, 0, 255), Color4uc(255, 255, 255, 255), _lut);
		}
		else if (_lut == std::string("Red")) {
			return new Palette(Color4uc(0, 0, 0, 255), Color4uc(255, 0, 0, 255), _lut);
		}
		else if (_lut == std::string("Green")) {
			return new Palette(Color4uc(0, 0, 0, 255), Color4uc(0, 255, 0, 255), _lut);
		}
		else if (_lut == std::string("Blue")) {
			return new Palette(Color4uc(0, 0, 0, 255), Color4uc(0, 0, 255, 255), _lut);
		}
		else if (_lut == std::string("AllGray")) {
			return new Palette(Color4uc(125, 125, 125, 255), Color4uc(125, 125, 125, 255), _lut);
		}
		else if (_lut == std::string("AllYellow")) {
			return new Palette(Color4uc(255, 255, 0, 255), Color4uc(255, 255, 0, 255), _lut);
		}
		else if (_lut == std::string("AllRedColorBlind")) {
			return new Palette(Color4uc(128, 0, 128, 255), Color4uc(128, 0, 128, 255), _lut);
		}
		else if (_lut == std::string("AllGreenColorBlind")) {
			return new Palette(Color4uc(0, 200, 0, 255), Color4uc(0, 200, 0, 255), _lut);
		}
		else if (_lut == std::string("Fire")) {
			int r[] = { 0,0,1,25,49,73,98,122,146,162,173,184,195,207,217,229,240,252,255,255,255,255,255,255,255,255,255,255,255,255,255,255 };
			int g[] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,14,35,57,79,101,117,133,147,161,175,190,205,219,234,248,255,255,255,255 };
			int b[] = { 0,61,96,130,165,192,220,227,210,181,151,122,93,64,35,5,0,0,0,0,0,0,0,0,0,0,0,35,98,160,223,255 };
			int w = 32;
			std::vector < Color4uc > colors;
			for (int i = 1; i < w - 1; i++)
				colors.push_back(Color4uc(r[i], g[i], b[i], 255));
			Palette* palette = new Palette(Color4uc(0, 0, 0, 255), Color4uc(255, 255, 255, 255), _lut);
			float step = 1.f / 32.f;
			float cur = step;
			for (unsigned int i = 0; i < colors.size(); i++, cur += step)
				palette->setColor(cur, colors[i]);
			return palette;
		}
		else if (_lut == std::string("InvFire")) {
			int r[] = { 0,0,1,25,49,73,98,122,146,162,173,184,195,207,217,229,240,252,255,255,255,255,255,255,255,255,255,255,255,255,255,255 };
			int g[] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,14,35,57,79,101,117,133,147,161,175,190,205,219,234,248,255,255,255,255 };
			int b[] = { 0,61,96,130,165,192,220,227,210,181,151,122,93,64,35,5,0,0,0,0,0,0,0,0,0,0,0,35,98,160,223,255 };
			int w = 32;
			std::vector < Color4uc > colors;
			for (int i = w - 2; i >= 0; i--)
				colors.push_back(Color4uc(r[i], g[i], b[i], 255));
			Palette* palette = new Palette(Color4uc(255, 255, 255, 255), Color4uc(0, 0, 0, 255), _lut);
			float step = 1.f / 32.f;
			float cur = step;
			for (unsigned int i = 0; i < colors.size(); i++, cur += step)
				palette->setColor(cur, colors[i]);
			return palette;
		}
		else if (_lut == std::string("Ice")) {
			int r[] = { 0,0,0,0,0,0,19,29,50,48,79,112,134,158,186,201,217,229,242,250,250,250,250,251,250,250,250,250,251,251,243,230 };
			int g[] = { 156,165,176,184,190,196,193,184,171,162,146,125,107,93,81,87,92,97,95,93,93,90,85,69,64,54,47,35,19,0,4,0 };
			int b[] = { 140,147,158,166,170,176,209,220,234,225,236,246,250,251,250,250,245,230,230,222,202,180,163,142,123,114,106,94,84,64,26,27 };
			int w = 32;
			std::vector < Color4uc > colors;
			for (int i = 1; i < w - 1; i++)
				colors.push_back(Color4uc(r[i], g[i], b[i], 255));
			Palette* palette = new Palette(Color4uc(0, 0, 0, 255), Color4uc(255, 255, 255, 255), _lut);
			float step = 1.f / 32.f;
			float cur = step;
			for (unsigned int i = 0; i < colors.size(); i++, cur += step)
				palette->setColor(cur, colors[i]);
			return palette;
		}
		else if (_lut == std::string("AllBlue")) {
			return new Palette(Color4uc(0, 85, 255, 255), Color4uc(0, 85, 255, 255), _lut);
		}
		else if (_lut == std::string("AllGreen")) {
			return new Palette(Color4uc(0, 170, 127, 255), Color4uc(0, 170, 127, 255), _lut);
		}
		else if (_lut == std::string("AllRed")) {
			return new Palette(Color4uc(255, 0, 0, 255), Color4uc(255, 0, 0, 255), _lut);
		}
		else if (_lut == std::string("AllOrange")) {
			return new Palette(Color4uc(255, 165, 0, 255), Color4uc(255, 165, 0, 255), _lut);
		}
		else if (_lut == std::string("AllTomato")) {
			return new Palette(Color4uc(255, 69, 0, 255), Color4uc(255, 69, 0, 255), _lut);
		}
		else if (_lut == std::string("AllCyan")) {
			return new Palette(Color4uc(0, 255, 255, 255), Color4uc(0, 255, 255, 255), _lut);
		}
		else if (_lut == std::string("AllTurquoise")) {
			return new Palette(Color4uc(64, 224, 208, 255), Color4uc(64, 224, 208, 255), _lut);
		}
		else if (_lut == std::string("AllWhite")) {
			return new Palette(Color4uc(255, 255, 255, 255), Color4uc(255, 255, 255, 255), _lut);
		}
		else if (_lut == std::string("AllBlack")) {
			return new Palette(Color4uc(0, 0, 0, 255), Color4uc(0, 0, 0, 255), _lut);
		}
		else if (_lut == std::string("HotCold")) {
			Palette* palette = new Palette(Color4uc(0, 0, 255, 255), Color4uc(170, 0, 255, 255), _lut);
			palette->setColor(0.1f, Color4uc(0, 170, 255, 255));
			palette->setColor(0.225f, Color4uc(103, 255, 139, 255));
			palette->setColor(0.35f, Color4uc(255, 255, 0, 255));
			palette->setColor(0.5f, Color4uc(255, 170, 0, 255));
			palette->setColor(0.7f, Color4uc(255, 0, 0, 255));
			return palette;
		}
		else if (_lut == std::string("HotCold2")) {
			Palette* palette = new Palette(Color4uc(0, 0, 255, 255), Color4uc(170, 0, 255, 255), _lut);
			palette->setColor(0.16f, Color4uc(0, 170, 255, 255));
			palette->setColor(0.33f, Color4uc(103, 255, 139, 255));
			palette->setColor(0.49f, Color4uc(255, 255, 0, 255));
			palette->setColor(0.66f, Color4uc(255, 170, 0, 255));
			palette->setColor(0.82f, Color4uc(255, 0, 0, 255));
			return palette;
		}
		else if (_lut == std::string("HiLow")) {
			Palette* palette = new Palette(Color4uc(128, 0, 128, 255), Color4uc(0, 200, 0, 255), _lut, true);
			palette->setColor(0.499, Color4uc(128, 0, 128, 255));
			palette->setColor(0.501, Color4uc(0, 200, 0, 255));
			return palette;
		}
		else if (_lut == std::string("Heatmap")) {
			Palette* palette = new Palette(Color4uc(33, 102, 172, 0), Color4uc(178, 24, 43, 255), _lut);
			palette->setColor(0.2f, Color4uc(103, 169, 207, 255));
			palette->setColor(0.4f, Color4uc(209, 229, 240, 255));
			palette->setColor(0.6f, Color4uc(253, 219, 199, 255));
			palette->setColor(0.8f, Color4uc(239, 138, 98, 255));
			return palette;
		}
		else if (_lut == std::string("HiLo")) {
			Palette* palette = new Palette(Color4uc(128, 0, 128, 255), Color4uc(0, 200, 0, 255), _lut);
			palette->setHiLow(true);
			return palette;
		}
		else if (_lut == std::string("Random")) {
			Palette* palette = new Palette(randomColorB(), randomColorB(), _lut);
			float step = 0.0000001f;
			for(float cur = step; cur < 1.f; cur += step)
				palette->setColor(cur, randomColorB());
			return palette;
		}
		return NULL;
	}

	PaletteInterface* Palette::copy() const
	{
		Palette* palette = new Palette(*this);
		return palette;
	}

	void Palette::copy(Palette* _palette)
	{
		this->m_gradient = _palette->m_gradient;
		this->m_begin = _palette->m_begin;
		this->m_end = _palette->m_end;
		this->m_from = _palette->m_from;
		this->m_to = _palette->m_to;
	}

	const Color4uc Palette::getColorLUT(const float _pos) const
	{
		if (m_hilow) {
			Color4uc c1(128, 0, 128, 255), c2(0, 200, 0, 255);
			return _pos < .5f ? c2 : c1;
		}
		else {
			std::set <std::pair<float, Color4uc>>::iterator it = m_gradient.begin(), it_next = it;
			it_next++;
			for (; it_next != m_gradient.end(); it++, it_next++) {
				float pos1 = it->first, pos2 = it_next->first;
				if (pos1 == _pos)
					return it->second;
				if (pos2 == _pos)
					return it_next->second;
				if (pos1 < _pos && _pos < pos2) {
					Color4uc c2 = it->second, c1 = it_next->second;
					float segmentLength = pos2 - pos1;
					float pdist = _pos - pos1;
					float ratio = pdist / segmentLength;
					int red = (int)(ratio * c1[0] + (1 - ratio) * c2[0]);
					int green = (int)(ratio * c1[1] + (1 - ratio) * c2[1]);
					int blue = (int)(ratio * c1[2] + (1 - ratio) * c2[2]);
					int alpha = (int)(ratio * c1[3] + (1 - ratio) * c2[3]);
					return Color4uc(red, green, blue, alpha);
				}
			}
			return m_gradient.begin()->second;
		}
	}

	const Color4uc Palette::getColor(const float _pos) const
	{
		if (m_hilow) {
			Color4uc c1(128, 0, 128, 255), c2(0, 200, 0, 255);
			if (m_filterMin < _pos && _pos < m_filterMax)
				return c2;
			else
				return c1;
		}
		else {
			std::set <std::pair<float, Color4uc>>::iterator it = m_gradient.begin(), it_next = it;
			it_next++;
			for (; it_next != m_gradient.end(); it++, it_next++) {
				float pos1 = it->first, pos2 = it_next->first;
				if (pos1 == _pos)
					return it->second;
				if (pos2 == _pos)
					return it_next->second;
				if (pos1 < _pos && _pos < pos2) {
					Color4uc c2 = it->second, c1 = it_next->second;
					float segmentLength = pos2 - pos1;
					float pdist = _pos - pos1;
					float ratio = pdist / segmentLength;
					int red = (int)(ratio * c1[0] + (1 - ratio) * c2[0]);
					int green = (int)(ratio * c1[1] + (1 - ratio) * c2[1]);
					int blue = (int)(ratio * c1[2] + (1 - ratio) * c2[2]);
					int alpha = (int)(ratio * c1[3] + (1 - ratio) * c2[3]);
					return Color4uc(red, green, blue, alpha);
				}
			}
			return m_gradient.begin()->second;
		}
	}

	const Color4uc Palette::getColorNoInterpolation(const float _pos) const
	{
		if (m_hilow) {
			Color4uc c1(128, 0, 128, 255), c2(0, 200, 0, 255);
			if (m_filterMin < _pos && _pos < m_filterMax)
				return c2;
			else
				return c1;
		}
		else {
			std::set <std::pair<float, Color4uc>>::iterator it = m_gradient.begin(), it_next = it;
			it_next++;
			for (; it_next != m_gradient.end(); it++, it_next++) {
				float pos1 = it->first, pos2 = it_next->first;
				if (_pos < pos1 || _pos > pos2) continue;
				float inter = pos2 - pos1;
				if(_pos < pos1 + inter)
					return it->second;
				else
					return it_next->second;
			}
			return m_gradient.begin()->second;
		}
	}

	Palette* Palette::getMonochromePalette(const int _r, const int _g, const int _b)
	{
		return new Palette(Color4uc(_r, _g, _b, 255), Color4uc(_r, _g, _b, 255));
	}

	const unsigned int Palette::memorySize() const
	{
		unsigned int memoryS = 2 * sizeof(double);
		memoryS += 2 * sizeof(int);
		memoryS += 2 * sizeof(bool);

		return memoryS;
	}

	Color4uc Palette::getColor(const std::string& _name)
	{
		if (_name == "white")
			return Color4uc(255, 255, 255, 255);
		else if (_name == "black")
			return Color4uc(0, 0, 0, 255);
		else if (_name == "red")
			return Color4uc(255, 0, 0, 255);
		else if (_name == "blue")
			return Color4uc(255, 0, 255, 255);
		else if (_name == "green")
			return Color4uc(0, 255, 0, 255);
		else if (_name == "cyan")
			return Color4uc(0, 255, 255, 255);
		else if (_name == "magenta")
			return Color4uc(255, 0, 255, 255);
		else if (_name == "yellow")
			return Color4uc(255, 255, 0, 255);
		else if (_name == "gray")
			return Color4uc(125, 125, 125, 255);
		return Color4uc(255, 0, 0, 255);
	}
}

