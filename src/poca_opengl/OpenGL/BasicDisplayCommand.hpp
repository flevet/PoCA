/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      BasicDisplayCommand.hpp
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

#ifndef BasicDisplayCommand_h__
#define BasicDisplayCommand_h__

#include <General/Command.hpp>

class QOpenGLFramebufferObject;

namespace poca::core {
	class BasicComponent;
}

namespace poca::opengl {

	class BasicDisplayCommand : public poca::core::Command {
	public:
		BasicDisplayCommand(poca::core::BasicComponentInterface*, const std::string&);
		BasicDisplayCommand(const BasicDisplayCommand&);
		~BasicDisplayCommand();

		//Command
		const poca::core::CommandInfos saveParameters() const { return poca::core::CommandInfos(); }
		void execute(poca::core::CommandInfo*);
		poca::core::Command* copy();
		poca::core::CommandInfo createCommand(const std::string&, const nlohmann::json&);

		void updatePickingFBO(const int _w, const int _h);
		void pick(const int, const int, const bool);

	protected:
		poca::core::BasicComponentInterface* m_component;

		QOpenGLFramebufferObject* m_pickFBO;
		int m_wImage, m_hImage, m_idSelection;
		bool m_pickingEnabled;
	};
}

#endif

