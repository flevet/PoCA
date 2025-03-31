/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      BasicDisplayCommand.cpp
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

#include <QtGui/QOpenGLFramebufferObject>
#include <QtGui/QImage>
#include <glm/gtc/matrix_transform.hpp>

#include <General/BasicComponent.hpp>

#include "BasicDisplayCommand.hpp"

namespace poca::opengl {

	BasicDisplayCommand::BasicDisplayCommand(poca::core::BasicComponentInterface* _component, const std::string& _name) : poca::core::Command(_name), m_pickFBO(NULL), m_wImage(0), m_hImage(0), m_idSelection(-1), m_pickingEnabled(true)
	{
		m_component = _component;
	}

	BasicDisplayCommand::BasicDisplayCommand(const BasicDisplayCommand& _o) :Command(_o)
	{
		m_component = _o.m_component;
	}

	BasicDisplayCommand::~BasicDisplayCommand()
	{
		if (m_pickFBO != NULL)
			delete m_pickFBO;
		m_pickFBO = NULL;
	}

	void BasicDisplayCommand::execute(poca::core::CommandInfo* _infos)
	{
		if (_infos->nameCommand == "updatePickingBuffer") {
			int w = _infos->getParameter<int>("width"), h = _infos->getParameter<int>("height");
			updatePickingFBO(w, h);
		}
		else if (_infos->nameCommand == "pick") {
			if (!m_pickingEnabled) return;
			int x = _infos->getParameter<int>("x"), y = _infos->getParameter<int>("y");
			bool saveImage = _infos->getParameter<bool>("saveImage");
			pick(x, y, saveImage);
		}
		else if (_infos->nameCommand == "freeGPU") {
			if (m_pickFBO != NULL)
				delete m_pickFBO;
			m_pickFBO = NULL;
		}
		else if(_infos->nameCommand == "togglePicking"){
			bool val = _infos->getParameter<bool>("togglePicking");
			m_pickingEnabled = val;
		}
		else if (_infos->nameCommand == "setIDObjectPicked") {
			int id = _infos->getParameter<int>("setIDObjectPicked");
			m_idSelection = id;
		}
	}

	poca::core::CommandInfo BasicDisplayCommand::createCommand(const std::string& _nameCommand, const nlohmann::json& _parameters)
	{
		if (_nameCommand == "updatePickingBuffer") {
			int w, h;
			bool complete = _parameters.contains("width");
			if (complete)
				w = _parameters["width"].get<int>();
			complete &= _parameters.contains("height");
			if (complete) {
				h = _parameters["height"].get<int>();
				return poca::core::CommandInfo(false, _nameCommand, "width", w , "height", h);
			}
		}
		else if (_nameCommand == "pick") {
			int x, y;
			bool saveImage = false;
			if(_parameters.contains("saveImage"))
				saveImage = _parameters["saveImage"].get<bool>();
			bool complete = _parameters.contains("x");
			if (complete)
				x = _parameters["x"].get<int>();
			complete &= _parameters.contains("y");
			if (complete) {
				y = _parameters["y"].get<int>();
				return poca::core::CommandInfo(false, _nameCommand, "x", x , "y", y, "saveImage", saveImage);
			}
		}
		else if (_nameCommand == "freeGPU") {
			return poca::core::CommandInfo(false, "freeGPU");
		}
		else if (_nameCommand == "togglePicking") {
			bool val = _parameters.get<bool>();
			return poca::core::CommandInfo(false, _nameCommand, val);
		}
		else if (_nameCommand == "setIDObjectPicked") {
			int val = _parameters.get<int>();
			return poca::core::CommandInfo(false, _nameCommand, val);
		}

		return poca::core::CommandInfo();
	}

	void BasicDisplayCommand::pick(const int _x, const int _y, const bool _saveImage)
	{
		int sizeImage = m_wImage * m_hImage;
		if (m_pickFBO == NULL) return;
		m_pickFBO->bind();
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		float pixel, * pixs = NULL;
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glReadPixels(_x, m_hImage - _y - 1, 1, 1, GL_RED, GL_FLOAT, &pixel);
		if (_saveImage) {
			pixs = new float[sizeImage];
			glReadPixels(0, 0, m_wImage, m_hImage, GL_RED, GL_FLOAT, pixs);
		}
		glReadBuffer(GL_NONE);
		m_pickFBO->release();

		m_idSelection = (int)(pixel - 1);

		if (_saveImage) {
			QImage image2 = QImage(m_wImage, m_hImage, QImage::Format_Grayscale16);
			for (int j = 0; j < m_hImage; ++j) {
				quint16* dst = (quint16*)(image2.bits() + (m_hImage - j - 1) * image2.bytesPerLine());
				for (int i = 0; i < m_wImage; ++i) {
					int index = i + j * m_wImage, index2 = index, index3 = m_wImage - i - 1;
					dst[i] = (quint16)pixs[index2];
				}
			}
			QString name = QString("e:/pick_obj.png");
			image2.save(name);
			quint16* tmp = (quint16*)image2.bits();
			quint16 val = tmp[m_hImage * _y + _x];
			std::cout << "For [" << _x << ", " << _y << "], val in image = " << val << std::endl;
			delete[] pixs;
		}
	}

	poca::core::Command* BasicDisplayCommand::copy()
	{
		return new BasicDisplayCommand(*this);
	}

	void BasicDisplayCommand::updatePickingFBO(const int _w, const int _h)
	{
		m_wImage = _w;
		m_hImage = _h;
		if (m_pickFBO != NULL)
			delete m_pickFBO;
		m_pickFBO = new QOpenGLFramebufferObject(m_wImage, m_hImage, QOpenGLFramebufferObject::Depth, GL_TEXTURE_2D, GL_RED);
		glBindTexture(GL_TEXTURE_2D, m_pickFBO->texture());
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_wImage, m_hImage, 0, GL_RED, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
}

