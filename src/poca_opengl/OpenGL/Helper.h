/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Helper.h
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

#ifndef Helper_h__
#define Helper_h__

#include <Windows.h>
#include <gl/glew.h>
#include <gl/GL.h>

#include <QtGui/QImage>

#include <Interfaces/PaletteInterface.hpp>
#include <OpenGL/GLBuffer.hpp>

namespace poca::opengl {
	void TEXT_2_PPM_file(const GLuint, int, int, const std::string&);

	class HelperSingleton {
	public:
		static HelperSingleton* instance();
		static void deleteInstance();
		static void setHelperSingleton(poca::opengl::HelperSingleton*);
		~HelperSingleton();

		const GLuint generateLutTexture(poca::core::PaletteInterface* = NULL);
		void generateTexture(const GLuint, const QImage&);

		poca::opengl::QuadSingleGLBuffer <float>& getEllipsoidBuffer();
		poca::opengl::QuadSingleGLBuffer <float>& getEllipsoidNormalsBuffer();
		poca::opengl::QuadSingleGLBuffer <GLushort>& getEllipsoidIndicesBuffer();
		uint32_t getNbIndicesUnitSphere() const { return m_nbIndicesUnitSphere; }

	protected:
		HelperSingleton();

		void computeUnitSphere();

	protected:
		static HelperSingleton* m_instance;
		std::map <std::string, GLuint> m_textures;

		//Unit sphere
		poca::opengl::QuadSingleGLBuffer <float> m_ellipsoidVerticesBuffer, m_ellipsoidNormalsBuffer;
		poca::opengl::QuadSingleGLBuffer <GLushort> m_ellipsoidIndicesBuffer;
		uint32_t m_nbIndicesUnitSphere;
	};
}
#endif

