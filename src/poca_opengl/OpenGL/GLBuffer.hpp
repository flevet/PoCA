/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      GLBuffer.hpp
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

#ifndef GLBuffer_h__
#define GLBuffer_h__

#include <Windows.h>
#include <gl/GL.h>
#include <vector>

/**
 *  @file glassert.h
 *  @brief debugging tool for opengl calls
 *
 *  @def glAssert

	This file provide macros in order to check easily openGl API calls.
	use it on every call to ensure better correctness of your programm :
	@code
	#include "glassert.h"

	glAssert( glmyAPICall() )
	@endcode

	@warning never use a glAssert() on a begin() directive.
	glGetError is not allowed inside a glBegin gEnd block !

	On can track down opengl errors with GL_CHECK_ERRORS(); which is usefull if
	the project don't use glAssert() everywhere. In this case a glAssert() can
	be triggered by previous error and won't give valuable insight on where the
	problem is located and from what opengl's primitive call.
*/

#ifndef NDEBUG
#include <iostream>
#include <cassert>

#define __TO_STR(x) __EVAL_STR(x)
#define __EVAL_STR(x) #x

#define glAssert(code) do{code; int l = __LINE__;\
   GLuint err = glGetError(); \
                if (err != GL_NO_ERROR)\
                { \
                  std::cerr << "OpenGL error : " << __FILE__ << "\n";\
                  std::cerr << "line : " << l << "\n";\
                  std::cerr << "Source code : " << __TO_STR(code) << "\n";\
                  std::cerr << "Message : " << (const char*)gluErrorString(err) << "("<<err<<")" << "\n";\
                  assert(false); \
              }\
}while(false)

// -----------------------------------------------------------------------------

#define GL_CHECK_ERRORS() \
do{ GLuint err = glGetError(); \
                if (err != GL_NO_ERROR)\
                { \
                  std::cerr << "OpenGL error : " << __FILE__ << "\n";\
                  std::cerr << "line : " << __LINE__ << "\n";\
                  std::cerr << "Source code : " << __TO_STR(code) << "\n";\
                  std::cerr << "Message : " << (const char*)gluErrorString(err) << "("<<err<<")" << "\n";\
                  assert(false); \
              }\
}while(false)

#else

#define __TO_STR(x) __EVAL_STR(x)
#define __EVAL_STR(x) #x

#define glAssert(code) \
    code

//#define GL_CHECK_ERRORS()
#define GL_CHECK_ERRORS() \
do{ GLuint err = glGetError(); \
                if (err != GL_NO_ERROR)\
                { \
                  std::cerr << "OpenGL error : " << __FILE__ << "\n";\
                  std::cerr << "line : " << __LINE__ << "\n";\
                  std::cerr << "Source code : " << __TO_STR(code) << "\n";\
                  std::cerr << "Message : " << (const char*)gluErrorString(err) << "("<<err<<")" << "\n";\
              }\
}while(false)
#endif

namespace poca::opengl {

	template< class T >
	class SingleGLBuffer {
	public:
		SingleGLBuffer() :m_bufferVertex(0), m_bufferIndices(0) {}
		~SingleGLBuffer();

		void freeGPUMemory();
		bool empty() const;
		void generateBuffer(const size_t, const int, const int, const GLenum = GL_ARRAY_BUFFER);
		void updateBuffer(const T*);
		void updateBuffer(const std::vector < T >&);
		void updateIndices(const std::vector <uint32_t>&);
		void bindBuffer(uint32_t) const;
		void bindBuffer(uint32_t, void*) const;

		void checkSizeBuffer(const size_t);

		inline GLuint getBufferVertex() const { return m_bufferVertex; }
		inline GLuint getBufferIndices() const { return m_bufferIndices; }
		inline const size_t getNbElements() const { return m_nbElems; }
		inline const size_t getNbIndices() const { return m_nbIndices; }
		inline GLenum getMode() const { return m_mode; }
		inline size_t getDim() const { return m_dim; }

	protected:
		GLenum m_target;
		GLuint m_bufferVertex, m_bufferIndices;
		size_t m_nbElems, m_nbIndices;
		size_t m_bufferLength, m_nbPrimitive, m_dim;
		int m_typeBuffer;
		GLenum m_mode;
	};

	template< class T >
	SingleGLBuffer<T>::~SingleGLBuffer()
	{
		if (m_bufferVertex != 0)
			glDeleteBuffers(1, &m_bufferVertex);
		if (m_bufferIndices != 0)
			glDeleteBuffers(1, &m_bufferIndices);
		m_bufferVertex = m_bufferIndices = 0;
	}

	template< class T >
	void SingleGLBuffer<T>::freeGPUMemory()
	{
		if (m_bufferVertex != 0)
			glDeleteBuffers(1, &m_bufferVertex);
		if (m_bufferIndices != 0)
			glDeleteBuffers(1, &m_bufferIndices);
		m_bufferVertex = m_bufferIndices = 0;
	}

	template< class T >
	bool SingleGLBuffer<T>::empty() const
	{
		return m_bufferVertex == 0 || m_nbElems == 0;
	}

	template< class T >
	void SingleGLBuffer<T>::generateBuffer(const size_t _size, const int _dim, const int _typeBuffer, const GLenum _target)
	{
		freeGPUMemory();
		m_nbElems = _size;
		m_dim = _dim;
		m_typeBuffer = _typeBuffer;
		m_target = _target;
		glGenBuffers(1, &m_bufferVertex);
	}

	template< class T >
	void SingleGLBuffer<T>::updateBuffer(const T* _data)
	{
		glBindBuffer(m_target, m_bufferVertex);
		glBufferData(m_target, m_nbElems * sizeof(T), (void*)_data, GL_STATIC_DRAW);
		glBindBuffer(m_target, 0);
	}

	template< class T >
	void SingleGLBuffer<T>::updateBuffer(const std::vector < T >& _data)
	{
		checkSizeBuffer(_data.size());
		if (m_nbElems == 0) return;
		glBindBuffer(m_target, m_bufferVertex);
		glBufferData(m_target, m_nbElems * sizeof(T), (void*)(&_data[0]), GL_STATIC_DRAW);
		glBindBuffer(m_target, 0);
	}

	template< class T >
	void SingleGLBuffer<T>::updateIndices(const std::vector < uint32_t >& _data)
	{
		if(m_bufferIndices == 0)
			glGenBuffers(1, &m_bufferIndices);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_bufferIndices);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, _data.size() * sizeof(uint32_t), (void*)(&_data[0]), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		m_nbIndices = _data.size();
	}

	template< class T >
	void SingleGLBuffer<T>::bindBuffer(uint32_t _attribPointerIndex) const
	{
		GL_CHECK_ERRORS();
		glBindBuffer(m_target, m_bufferVertex);
		GL_CHECK_ERRORS();
		glVertexAttribPointer(
			_attribPointerIndex,                  // attribute 0. No particular reason for 0, but must match the layout in the ShaderReader.
			(GLint)m_dim,                  // size
			m_typeBuffer,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);
		GL_CHECK_ERRORS();
		if (m_bufferIndices != 0)
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_bufferIndices);
		GL_CHECK_ERRORS();
	}

	template< class T >
	void SingleGLBuffer<T>::bindBuffer(uint32_t _attribPointerIndex, void* _offset) const
	{
		glBindBuffer(m_target, m_bufferVertex);
		glVertexAttribPointer(
			_attribPointerIndex,                  // attribute 0. No particular reason for 0, but must match the layout in the ShaderReader.
			(GLint)m_dim,                  // size
			m_typeBuffer,           // type
			GL_FALSE,           // normalized?
			sizeof(T),                  // stride
			_offset//(void*)0            // array buffer offset
		);
		glVertexAttribDivisor(_attribPointerIndex, 1);

		if(m_bufferIndices != 0)
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_bufferIndices);
	}

	template< class T >
	void SingleGLBuffer<T>::checkSizeBuffer(const size_t _newSize)
	{
		if (m_nbElems == _newSize) return;
		m_nbElems = _newSize;
	}

	template< class T >
	class FeatureSingleGLBuffer : public SingleGLBuffer <T> {
	public:
		FeatureSingleGLBuffer() :SingleGLBuffer() { m_nbPrimitive = 1; m_mode = GL_POINTS; }
		~FeatureSingleGLBuffer() {}
	};

	template< class T >
	class PointSingleGLBuffer : public SingleGLBuffer <T> {
	public:
		PointSingleGLBuffer() :SingleGLBuffer() { m_nbPrimitive = 1; m_mode = GL_POINTS; }
		~PointSingleGLBuffer() {}
	};

	template< class T >
	class LineSingleGLBuffer : public SingleGLBuffer <T> {
	public:
		LineSingleGLBuffer() :SingleGLBuffer() { m_nbPrimitive = 2; m_mode = GL_LINES; }
		~LineSingleGLBuffer() {}
	};

	template< class T >
	class TriangleSingleGLBuffer : public SingleGLBuffer <T> {
	public:
		TriangleSingleGLBuffer() :SingleGLBuffer() { m_nbPrimitive = 3; m_mode = GL_TRIANGLES; }
		~TriangleSingleGLBuffer() {}
	};

	template< class T >
	class QuadSingleGLBuffer : public SingleGLBuffer <T> {
	public:
		QuadSingleGLBuffer() :SingleGLBuffer() { m_nbPrimitive = 4; m_mode = GL_QUADS; }
		~QuadSingleGLBuffer() {}
	};

	template< class T >
	class LinesAdjacencySingleGLBuffer : public SingleGLBuffer <T> {
	public:
		LinesAdjacencySingleGLBuffer() :SingleGLBuffer() { m_nbPrimitive = 1; m_mode = GL_LINES_ADJACENCY; }
		~LinesAdjacencySingleGLBuffer() {}
	};

	template< class T >
	class LineStripAdjacencySingleGLBuffer : public SingleGLBuffer <T> {
	public:
		LineStripAdjacencySingleGLBuffer() :SingleGLBuffer() { m_nbPrimitive = 1; m_mode = GL_LINE_STRIP_ADJACENCY; }
		~LineStripAdjacencySingleGLBuffer() {}
	};

	template< class T >
	class GLBuffer {
	public:
		GLBuffer() :m_buffers(NULL), m_indiceBuffers(NULL){}
		~GLBuffer();

		void freeGPUMemory();
		bool empty() const;
		void generateBuffer(const size_t, const int, const int, const int);
		void updateBuffer(const T*);
		void updateBuffer(const std::vector < T >&);
		void updateBuffer(std::vector <std::vector < T >>);
		void bindBuffer(unsigned int, unsigned int) const;

		void checkSizeBuffer(const size_t);

		inline const std::vector <size_t>& getSizeBuffers() const { return m_sizeStrides; }
		inline size_t getNbBuffers() const { return m_sizeStrides.size(); }
		inline GLuint* getBuffers() const { return m_buffers; }
		inline GLuint getBuffer(unsigned int _index) const { return m_buffers[_index]; }
		inline const size_t getNbElements() const { return m_nbElems; }
		inline GLenum getMode() const { return m_mode; }

	protected:
		std::vector <size_t> m_sizeStrides;
		GLuint* m_buffers, * m_indiceBuffers;
		size_t m_nbElems;
		size_t m_bufferLength, m_nbPrimitive, m_dim;
		int m_typeBuffer;
		GLenum m_mode;
	};

	template< class T >
	GLBuffer<T>::~GLBuffer()
	{
		if (m_buffers != 0)
			glDeleteBuffers((GLsizei)m_sizeStrides.size(), m_buffers);
		if (m_indiceBuffers != 0)
			glDeleteBuffers((GLsizei)m_sizeStrides.size(), m_indiceBuffers);
		m_sizeStrides.clear();
		m_buffers = m_indiceBuffers = NULL;
	}

	template< class T >
	void GLBuffer<T>::freeGPUMemory()
	{
		if (m_buffers != 0)
			glDeleteBuffers((GLsizei)m_sizeStrides.size(), m_buffers);
		if (m_indiceBuffers != 0)
			glDeleteBuffers((GLsizei)m_sizeStrides.size(), m_indiceBuffers);
		m_sizeStrides.clear();
		m_buffers = m_indiceBuffers = NULL;
	}

	template< class T >
	bool GLBuffer<T>::empty() const
	{
		return m_buffers == NULL;
	}

	template< class T >
	void GLBuffer<T>::generateBuffer(const size_t _size, const int _bufferLength, const int _dim, const int _typeBuffer)
	{
		freeGPUMemory();
		m_nbElems = _size;
		m_dim = _dim;
		m_typeBuffer = _typeBuffer;
		m_bufferLength = _bufferLength * m_nbPrimitive;
		int numChunks = ceil((double)m_nbElems / (double)m_bufferLength);
		m_sizeStrides.resize(numChunks);
		for (unsigned int n = 0; n < m_sizeStrides.size(); n++)
			m_sizeStrides[n] = (n < m_sizeStrides.size() - 1) ? m_bufferLength : m_nbElems - n * m_bufferLength;
		m_buffers = new GLuint[m_sizeStrides.size()];
		glGenBuffers((GLsizei)m_sizeStrides.size(), m_buffers);
	}

	template< class T >
	void GLBuffer<T>::updateBuffer(const T* _data)
	{
		for (size_t chunk = 0, currentPos = 0; chunk < m_sizeStrides.size(); chunk++) {
			glBindBuffer(GL_ARRAY_BUFFER, m_buffers[chunk]);
			glBufferData(GL_ARRAY_BUFFER, m_sizeStrides[chunk] * sizeof(T), (void*)(_data + currentPos), GL_STATIC_DRAW);
			currentPos += m_sizeStrides[chunk];
		}
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	template< class T >
	void GLBuffer<T>::updateBuffer(const std::vector < T >& _data)
	{
		checkSizeBuffer(_data.size());
		for (size_t chunk = 0, currentPos = 0; chunk < m_sizeStrides.size(); chunk++) {
			glBindBuffer(GL_ARRAY_BUFFER, m_buffers[chunk]);
			glBufferData(GL_ARRAY_BUFFER, m_sizeStrides[chunk] * sizeof(T), (void*)(&_data[0] + currentPos), GL_STATIC_DRAW);
			currentPos += m_sizeStrides[chunk];
		}
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	template< class T >
	void GLBuffer<T>::updateBuffer(std::vector <std::vector < T >> _data)
	{
		if (_data.size() != m_sizeStrides.size()) {
			std::cout << "Problem with creation of OpenGL buffer" << std::endl;
			return;
		}
		for (unsigned int chunk = 0, currentPos = 0; chunk < m_sizeStrides.size(); chunk++) {
			glBindBuffer(GL_ARRAY_BUFFER, m_buffers[chunk]);
			glBufferData(GL_ARRAY_BUFFER, m_sizeStrides[chunk] * sizeof(T), _data[chunk].data() /*+ currentPos*/, GL_STATIC_DRAW);
			//currentPos += m_sizeStrides[chunk];
		}
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	template< class T >
	void GLBuffer<T>::bindBuffer(unsigned int _chunk, unsigned int _attribPointerIndex) const
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_buffers[_chunk]);
		glVertexAttribPointer(
			_attribPointerIndex,                  // attribute 0. No particular reason for 0, but must match the layout in the ShaderReader.
			(GLint)m_dim,                  // size
			m_typeBuffer,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);
	}

	template< class T >
	void GLBuffer<T>::checkSizeBuffer(const size_t _newSize)
	{
		if (m_nbElems == _newSize) return;
		m_nbElems = _newSize;
		int numChunks = ceil((double)m_nbElems / (double)m_bufferLength);
		if (m_sizeStrides.size() == numChunks) {
			for (unsigned int n = 0; n < m_sizeStrides.size(); n++)
				m_sizeStrides[n] = (n < m_sizeStrides.size() - 1) ? m_bufferLength : m_nbElems - n * m_bufferLength;
		}
		else {
			freeGPUMemory();
			m_sizeStrides.resize(numChunks);
			for (unsigned int n = 0; n < m_sizeStrides.size(); n++)
				m_sizeStrides[n] = (n < m_sizeStrides.size() - 1) ? m_bufferLength : m_nbElems - n * m_bufferLength;
			m_buffers = new GLuint[m_sizeStrides.size()];
			glGenBuffers((GLsizei)m_sizeStrides.size(), m_buffers);
		}
	}

	template< class T >
	class FeatureGLBuffer : public GLBuffer <T> {
	public:
		FeatureGLBuffer() :GLBuffer() { m_nbPrimitive = 1; m_mode = GL_POINTS; }
		~FeatureGLBuffer() {}
	};

	template< class T >
	class PointGLBuffer : public GLBuffer <T> {
	public:
		PointGLBuffer() :GLBuffer() { m_nbPrimitive = 1; m_mode = GL_POINTS; }
		~PointGLBuffer() {}
	};

	template< class T >
	class LineGLBuffer : public GLBuffer <T> {
	public:
		LineGLBuffer() :GLBuffer() { m_nbPrimitive = 2; m_mode = GL_LINES; }
		~LineGLBuffer() {}
	};

	template< class T >
	class TriangleGLBuffer : public GLBuffer <T> {
	public:
		TriangleGLBuffer() :GLBuffer() { m_nbPrimitive = 3; m_mode = GL_TRIANGLES; }
		~TriangleGLBuffer() {}
	};

	template< class T >
	class TriangleStripGLBuffer : public GLBuffer <T> {
	public:
		TriangleStripGLBuffer() :GLBuffer() { m_nbPrimitive = 1; m_mode = GL_TRIANGLE_STRIP; }
		~TriangleStripGLBuffer() {}
	};
}

#endif // GLBuffer_h__

