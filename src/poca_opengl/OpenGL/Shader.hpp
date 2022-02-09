/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Shader.hpp
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

#ifndef SHADER_H
#define SHADER_H

#include <Windows.h>
#include <gl/glew.h>
#include <gl/GL.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

namespace poca::opengl {

	class Shader
	{
	public:
		GLuint ID;
		GLuint vertex, fragment, geometry = 0;

		static float MIN_VALUE_FEATURE_SHADER;

		Shader() :ID(0), vertex(0), fragment(0)
		{
		}

		// Constructor generates the shader on the fly
		Shader(const GLchar* vertexPath, const GLchar* fragmentPath, const GLchar* geometryPath = nullptr)
		{
			createAndLinkProgram(vertexPath, fragmentPath, geometryPath);
		}

		const bool alreadyInitialized() const {
			return ID != 0;
		}

		void createAndLinkProgram(const GLchar* vertexPath, const GLchar* fragmentPath, const GLchar* geometryPath = nullptr)
		{
			createWithoutLinkProgram(vertexPath, fragmentPath, geometryPath);
			linkProgram();
		}

		void createWithoutLinkProgram(const GLchar* vertexPath, const GLchar* fragmentPath, const GLchar* geometryPath = nullptr)
		{
			// 1. Retrieve the vertex/fragment source code from filePath
			std::string vertexCode;
			std::string fragmentCode;
			std::string geometryCode;
			std::ifstream vShaderFile;
			std::ifstream fShaderFile;
			std::ifstream gShaderFile;
			// ensures ifstream objects can throw exceptions:
			vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
			fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
			gShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
			try
			{
				// Open files
				vShaderFile.open(vertexPath);
				fShaderFile.open(fragmentPath);
				std::stringstream vShaderStream, fShaderStream;
				// Read file's buffer contents into streams
				vShaderStream << vShaderFile.rdbuf();
				fShaderStream << fShaderFile.rdbuf();
				// close file handlers
				vShaderFile.close();
				fShaderFile.close();
				// Convert stream into string
				vertexCode = vShaderStream.str();
				fragmentCode = fShaderStream.str();
				// If geometry shader path is present, also load a geometry shader
				if (geometryPath != nullptr)
				{
					gShaderFile.open(geometryPath);
					std::stringstream gShaderStream;
					gShaderStream << gShaderFile.rdbuf();
					gShaderFile.close();
					geometryCode = gShaderStream.str();
				}
			}
			catch (std::ifstream::failure e)
			{
				std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
			}
			const GLchar* vShaderCode = vertexCode.c_str();
			const GLchar* fShaderCode = fragmentCode.c_str();
			// 2. Compile shaders
			// Vertex Shader
			vertex = glCreateShader(GL_VERTEX_SHADER);
			glShaderSource(vertex, 1, &vShaderCode, NULL);
			glCompileShader(vertex);
			checkCompileErrors(vertex, "VERTEX");
			// Fragment Shader
			fragment = glCreateShader(GL_FRAGMENT_SHADER);
			glShaderSource(fragment, 1, &fShaderCode, NULL);
			glCompileShader(fragment);
			checkCompileErrors(fragment, "FRAGMENT");
			// If geometry shader is given, compile geometry shader
			if (geometryPath != nullptr)
			{
				const GLchar* gShaderCode = geometryCode.c_str();
				geometry = glCreateShader(GL_GEOMETRY_SHADER);
				glShaderSource(geometry, 1, &gShaderCode, NULL);
				glCompileShader(geometry);
				checkCompileErrors(geometry, "GEOMETRY");
			}
			// Shader Program
			this->ID = glCreateProgram();
			glAttachShader(this->ID, vertex);
			glAttachShader(this->ID, fragment);
			if (geometryPath != nullptr)
				glAttachShader(this->ID, geometry);
		}

		void createAndLinkProgramFromStr(const GLchar* vs, const GLchar* fs, const GLchar* gs = nullptr)
		{
			vertex = glCreateShader(GL_VERTEX_SHADER);
			glShaderSource(vertex, 1, &vs, NULL);
			glCompileShader(vertex);
			checkCompileErrors(vertex, "VERTEX");
			// Fragment Shader
			fragment = glCreateShader(GL_FRAGMENT_SHADER);
			glShaderSource(fragment, 1, &fs, NULL);
			glCompileShader(fragment);
			checkCompileErrors(fragment, "FRAGMENT");
			// If geometry shader is given, compile geometry shader
			if (gs != nullptr)
			{
				geometry = glCreateShader(GL_GEOMETRY_SHADER);
				glShaderSource(geometry, 1, &gs, NULL);
				glCompileShader(geometry);
				checkCompileErrors(geometry, "GEOMETRY");
			}
			// Shader Program
			this->ID = glCreateProgram();
			glAttachShader(this->ID, vertex);
			glAttachShader(this->ID, fragment);
			if (gs != nullptr)
				glAttachShader(this->ID, geometry);
			linkProgram();
		}

		void linkProgram()
		{
			glLinkProgram(this->ID);
			checkCompileErrors(this->ID, "PROGRAM");
			// Delete the shaders as they're linked into our program now and no longer necessery
			glDeleteShader(vertex);
			glDeleteShader(fragment);
			if (geometry != 0)
				glDeleteShader(geometry);
		}

		GLint get_attrib(const char* name) {
			GLint attribute = glGetAttribLocation(this->ID, name);
			if (attribute == -1)
				fprintf(stderr, "Could not bind attribute %s\n", name);
			return attribute;
		}

		GLint get_uniform(const char* name) {
			GLint uniform = glGetUniformLocation(this->ID, name);
			if (uniform == -1)
				fprintf(stderr, "Could not bind uniform %s\n", name);
			return uniform;
		}

		// activate the shader
		// ------------------------------------------------------------------------
		void use() const
		{
			glUseProgram(ID);
		}

		void release() const
		{
			glUseProgram(0);
		}

		void destroy()
		{
			glDeleteProgram(ID);
		}
		// utility uniform functions
		// ------------------------------------------------------------------------
		void setBool(const std::string& name, bool value) const
		{
			glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
		}
		// ------------------------------------------------------------------------
		void setInt(const std::string& name, int value) const
		{
			glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
		}
		// ------------------------------------------------------------------------
		void setUInt(const std::string& name, int value) const
		{
			glUniform1ui(glGetUniformLocation(ID, name.c_str()), value);
		}
		// ------------------------------------------------------------------------
		void setFloat(const std::string& name, float value) const
		{
			glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
		}
		// ------------------------------------------------------------------------
		void setVec2(const std::string& name, const glm::vec2& value) const
		{
			glUniform2fv(glGetUniformLocation(ID, name.c_str()), 1, glm::value_ptr(value));
		}
		void setVec2(const std::string& name, float x, float y) const
		{
			glUniform2f(glGetUniformLocation(ID, name.c_str()), x, y);
		}
		// ------------------------------------------------------------------------
		void setVec3(const std::string& name, const glm::vec3& value) const
		{
			glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, glm::value_ptr(value));
		}
		void setVec3(const std::string& name, float x, float y, float z) const
		{
			glUniform3f(glGetUniformLocation(ID, name.c_str()), x, y, z);
		}
		// ------------------------------------------------------------------------
		void setVec4(const std::string& name, const glm::vec4& value) const
		{
			glUniform4fv(glGetUniformLocation(ID, name.c_str()), 1, glm::value_ptr(value));
		}
		void setVec4(const std::string& name, float x, float y, float z, float w)
		{
			glUniform4f(glGetUniformLocation(ID, name.c_str()), x, y, z, w);
		}
		void setUVec4(const std::string& name, const glm::uvec4& value) const
		{
			glUniform4uiv(glGetUniformLocation(ID, name.c_str()), 1, glm::value_ptr(value));
		}
		// ------------------------------------------------------------------------
		void setMat2(const std::string& name, const glm::mat2& mat) const
		{
			glUniformMatrix2fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat));
		}
		// ------------------------------------------------------------------------
		void setMat3(const std::string& name, const glm::mat3& mat) const
		{
			glUniformMatrix3fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat));
		}
		// ------------------------------------------------------------------------
		void setMat4(const std::string& name, const glm::mat4& mat) const
		{
			glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat));
		}

	private:
		void checkCompileErrors(GLuint shader, std::string type)
		{
			GLint success;
			GLchar infoLog[1024];
			if (type != "PROGRAM")
			{
				glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
				if (!success)
				{
					glGetShaderInfoLog(shader, 1024, NULL, infoLog);
					std::cout << "| ERROR::::SHADER-COMPILATION-ERROR of type: " << type << "|\n" << infoLog << "\n| -- --------------------------------------------------- -- |" << std::endl;
				}
			}
			else
			{
				glGetProgramiv(shader, GL_LINK_STATUS, &success);
				if (!success)
				{
					glGetProgramInfoLog(shader, 1024, NULL, infoLog);
					std::cout << "| ERROR::::PROGRAM-LINKING-ERROR of type: " << type << "|\n" << infoLog << "\n| -- --------------------------------------------------- -- |" << std::endl;
				}
			}
		}
	};
}

#endif

