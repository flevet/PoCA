/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Palette.hpp
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

#ifndef _PALETTE_H_
#define _PALETTE_H_

#include <set>

#include "../Interfaces/PaletteInterface.hpp"

namespace poca::core {

	struct Compare final
	{
		bool operator()(const std::pair<float, Color4uc>& lhs, const std::pair<float, Color4uc>& rhs) const noexcept
		{
			if (lhs.first != rhs.first) return lhs.first < rhs.first;
			if (lhs.second[0] != rhs.second[0]) return lhs.second[0] < rhs.second[0];
			if (lhs.second[1] != rhs.second[1]) return lhs.second[1] < rhs.second[1];
			return lhs.second[2] < rhs.second[2]; // comparision logic
		}
	};

	/**
	 * Palette de couleur.
	 * Elle se résume en un dégradé de couleurs.
	 */
	class Palette : public PaletteInterface
	{
	protected:
		std::set <std::pair<float, Color4uc>, Compare> m_gradient;
		std::string m_name;
		double m_begin, m_end;
		int m_from, m_to;
		bool m_autoscale, m_hilow, m_threshold;
		float m_filterMin, m_filterMax;
		
		static uint32_t m_seedRand;

	public:

		Palette();

		/**
		 * \brief Constructeur.
		 * \param _color_begin Couleur de début.
		 * \param _color_end Couleur de fin.
		 */
		Palette(const Color4uc _color_begin, const Color4uc _color_end, const std::string & = "", const bool = false);

		Palette(const Palette&);

		void setPalette(PaletteInterface*);
		void setPalette(const Palette&);

		std::set <std::pair<float, Color4uc>>::iterator getElement(unsigned int);
		std::set <std::pair<float, Color4uc>>::const_iterator getElement(unsigned int) const;

		/**
		 * \brief Applique une couleur à une position. Si la position existe déjà, change la couleur, sinon rajoute une nouvelle position.
		 * \param _position Position [0.0, 1.0].
		 * \param _color Couleur.
		 */
		void setColor(float _position, Color4uc _color);
		/**
		 * \brief Supprime la ième couleur du dégradé.
		 * \param _index Index de la couleur.
		 */
		void removeColorAt(unsigned int _index);
		/**
		 * \brief retourne la ième couleur du dégradé.
		 * \param _index Index de la couleur.
		 * \return Couleur à cette position.
		 */
		Color4uc colorAt(unsigned int _index) const;
		/**
		 * \brief Fixe la ième couleur du dégradé.
		 * \param _index Index de la couleur.
		 * \param _color Couleur.
		 */
		void setColorAt(unsigned int _index, Color4uc _color);
		/**
		 * \brief Retourne la position de la ième couleur.
		 * \param _index Index de la couleur.
		 * \return Position.
		 */
		float colorPosition(unsigned int _index) const;
		/**
		 * \brief Fixe la position de la ième couleur.
		 * \param _index Index de la couleur.
		 * \param _position Nouvelle position.
		 */
		void setColorPosition(unsigned int _index, float _position);

		void getGradientInfos(std::vector <float>&, std::vector <Color4uc>&) const;

		PaletteInterface* copy() const;
		void copy(Palette*);

		const unsigned int memorySize() const;
		const Color4uc getColor(const float) const;
		const Color4uc getColorLUT(const float) const;
		const Color4uc getColorNoInterpolation(const float) const;


		inline void setBounds(const int _from, const int _to) { m_from = _from; m_to = _to; }
		inline const int getBegin() const { return m_from; }
		inline const int getEnd() const { return m_to; }

		inline const std::string& getName() const { return m_name; }
		inline void setName(const std::string& _name) { m_name = _name; }

		void setFilterMinMax(const float _min, const float _max) { m_filterMin = _min; m_filterMax = _max; }
		const float getFilterMin() const { return m_filterMin; }
		const float getFilterMax() const { return m_filterMax; }

		void setHiLow(const bool _val) { m_hilow = _val; }
		const bool isHiLow() const { return m_hilow; }

		void setThreshold(const bool _val) { m_threshold = _val; }
		virtual const bool isThreshold() const { return m_threshold; }

		inline const bool null() const { return m_gradient.empty(); }
		const size_t size() const { return m_gradient.size(); }

		static Palette getStaticLut(const std::string&);
		static Palette* getStaticLutPtr(const std::string&);
		static Palette* getMonochromePalette(const int, const int, const int);
		static Color4uc getColor(const std::string&);
	};
}

#endif //_PALETTE_H_

