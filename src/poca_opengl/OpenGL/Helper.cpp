/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Helper.cpp
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

#include <qmath.h>

#include "Helper.h"

namespace poca::opengl {
	HelperSingleton* HelperSingleton::m_instance = 0;

	HelperSingleton* HelperSingleton::instance()
	{
		if (m_instance == 0)
			m_instance = new HelperSingleton;
		return m_instance;
	}

	void HelperSingleton::setHelperSingleton(poca::opengl::HelperSingleton* _lds)
	{
		m_instance = _lds;
	}

	void HelperSingleton::deleteInstance()
	{
		if (m_instance != 0)
			delete m_instance;
		m_instance = 0;
	}

	HelperSingleton::HelperSingleton()
	{

	}

	HelperSingleton::~HelperSingleton()
	{
	}

	const GLuint HelperSingleton::generateLutTexture(poca::core::PaletteInterface* _pal)
	{
		if (m_textures.find(_pal->getName()) == m_textures.end()) {
			GLuint textureLutID;
			glGenTextures(1, &textureLutID);
			//Creation of the texture for the LUT
			unsigned int sizeLut = 512;
			unsigned int cpt = 0;
			std::vector <float> lutValues(sizeLut * 4);
			float stepLut = 1. / (float)(sizeLut - 1);
			for (float val = 0.f; val <= 1.f; val += stepLut) {
				poca::core::Color4uc c = _pal->getColorLUT(val);
				lutValues[cpt++] = (float)c[0] / 255.f;
				lutValues[cpt++] = (float)c[1] / 255.f;
				lutValues[cpt++] = (float)c[2] / 255.f;
				lutValues[cpt++] = (float)c[3] / 255.f;
			}
			glBindTexture(GL_TEXTURE_1D, textureLutID);
			glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, sizeLut, 0, GL_RGBA, GL_FLOAT, lutValues.data());
			glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glBindTexture(GL_TEXTURE_1D, 0);
			m_textures[_pal->getName()] = textureLutID;
			return textureLutID;
		}
		else if (_pal->getName() == "Random") {
			unsigned int sizeLut = 10000;
			unsigned int cpt = 0;
			std::vector <float> lutValues(sizeLut * 4);
			float stepLut = 1. / (float)(sizeLut - 1);
			for (float val = 0.f; val <= 1.f; val += stepLut) {
				poca::core::Color4uc c = _pal->getColorLUT(val);
				lutValues[cpt++] = (float)c[0] / 255.f;
				lutValues[cpt++] = (float)c[1] / 255.f;
				lutValues[cpt++] = (float)c[2] / 255.f;
				lutValues[cpt++] = (float)c[3] / 255.f;
			}
			glBindTexture(GL_TEXTURE_1D, m_textures[_pal->getName()]);
			glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, sizeLut, 0, GL_RGBA, GL_FLOAT, lutValues.data());
			glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glBindTexture(GL_TEXTURE_1D, 0);
		}
		return m_textures[_pal->getName()];
	}

	void HelperSingleton::generateTexture(const GLuint _idTexture, const QImage& _image)
	{
		glBindTexture(GL_TEXTURE_2D, _idTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, _image.width(), _image.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, _image.bits());
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	poca::opengl::QuadSingleGLBuffer <float>& HelperSingleton::getEllipsoidBuffer()
	{ 
		if (m_ellipsoidVerticesBuffer.empty())
			computeUnitSphere();
		return m_ellipsoidVerticesBuffer;
	}

	poca::opengl::QuadSingleGLBuffer <float>& HelperSingleton::getEllipsoidNormalsBuffer()
	{
		if (m_ellipsoidNormalsBuffer.empty())
			computeUnitSphere();
		return m_ellipsoidNormalsBuffer;
	}

	poca::opengl::QuadSingleGLBuffer <GLushort>& HelperSingleton::getEllipsoidIndicesBuffer()
	{ 
		if (m_ellipsoidIndicesBuffer.empty())
			computeUnitSphere();
		return m_ellipsoidIndicesBuffer;
	}

	void HelperSingleton::computeUnitSphere()
	{
		unsigned int rings = 50, sectors = 50;
		float const R = 1. / (float)(rings - 1);
		float const S = 1. / (float)(sectors - 1);
		uint32_t r, s, cpt = 0;
		float radius = 1.;
		std::vector<float> verticesUnitSphere(rings * sectors * 3), normalsUnitSphere(rings * sectors * 3);
		std::vector<GLfloat>::iterator v = verticesUnitSphere.begin(), n = normalsUnitSphere.begin();
		for (r = 0; r < rings; r++) for (s = 0; s < sectors; s++) {
			float const y = sin(-M_PI_2 + M_PI * r * R);
			float const x = cos(2 * M_PI * s * S) * sin(M_PI * r * R);
			float const z = sin(2 * M_PI * s * S) * sin(M_PI * r * R);
			*v++ = x;
			*v++ = y;
			*v++ = z;

			poca::core::Vec3mf normal(x, y, z);
			normal.normalize();
			*n++ = normal.x();
			*n++ = normal.y();
			*n++ = normal.z();
		}
		m_nbIndicesUnitSphere = rings * sectors * 4;
		std::vector<GLushort> indicesUnitSphere(m_nbIndicesUnitSphere);
		std::vector<GLushort>::iterator i = indicesUnitSphere.begin();
		for (r = 0; r < rings - 1; r++) for (s = 0; s < sectors - 1; s++) {
			*i++ = r * sectors + s;
			*i++ = r * sectors + (s + 1);
			*i++ = (r + 1) * sectors + (s + 1);
			*i++ = (r + 1) * sectors + s;
		}

		m_ellipsoidVerticesBuffer.generateBuffer(verticesUnitSphere.size(), 3, GL_FLOAT);
		m_ellipsoidVerticesBuffer.updateBuffer(verticesUnitSphere.data());

		m_ellipsoidNormalsBuffer.generateBuffer(normalsUnitSphere.size(), 3, GL_FLOAT);
		m_ellipsoidNormalsBuffer.updateBuffer(normalsUnitSphere.data());

		m_ellipsoidIndicesBuffer.generateBuffer(indicesUnitSphere.size(), 4, GL_UNSIGNED_INT, GL_ELEMENT_ARRAY_BUFFER);
		m_ellipsoidIndicesBuffer.updateBuffer(indicesUnitSphere.data());
	}

	void TEXT_2_PPM_file(const GLuint _idTexture, int width, int height, const std::string& _filename)
	{
		int     output_width, output_height;

		output_width = width;
		output_height = height;

		glBindTexture(GL_TEXTURE_2D, _idTexture);

		/// READ THE PIXELS VALUES from FBO AND SAVE TO A .PPM FILE
		int             i, j, k;
		unsigned char* buffer = new unsigned char[output_width * output_height * 3];
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, buffer);
		FILE* out = fopen(_filename.c_str(), "w");
		short  TGAhead[] = { 0, 2, 0, 0, 0, 0, output_width, output_height, 24 };
		fwrite(&TGAhead, sizeof(TGAhead), 1, out);
		k = 0;
		for (i = 0; i < output_width; i++)
		{
			for (j = 0; j < output_height; j++)
			{
				fprintf(out, "%u %u %u ", (unsigned int)buffer[k], (unsigned int)buffer[k + 1],
					(unsigned int)buffer[k + 2]);
				k = k + 3;
			}
			fprintf(out, "\n");
		}
		fclose(out);
		glBindTexture(GL_TEXTURE_2D, 0);
		delete[] buffer;
	}
}

