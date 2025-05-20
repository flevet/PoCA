/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ImagesList.hpp
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

#ifndef ImagesList_hpp__
#define ImagesList_hpp__


#include <General/BasicComponentList.hpp>

namespace poca::core {
	class ImageInterface;

	class ImagesList : public BasicComponentList {
	public:
		ImagesList(ImageInterface*, const std::string & = "");
		~ImagesList();

		BasicComponentInterface* copy();
		virtual void copyComponentsPtr(BasicComponentList*);

		void addImage(ImageInterface*, const std::string & = "");
		ImageInterface* currentImage();
		ImageInterface* getImage(const uint32_t);
		uint32_t currentImageIndex() const;

		void eraseCurrentImage() { eraseImage(m_currentComponent); }
		void eraseImage(const uint32_t);

		const std::string& currentName() const;
		const std::string& getName(const uint32_t _index) const;
		void setName(const uint32_t _index, const std::string& _name);
		void setCurrentName(const std::string& _name);


	protected:
		std::vector <std::string> m_names;
	};
}
#endif

