/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#ifndef ShaderSource_hpp__
#define ShaderSource_hpp__
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
namespace poca::opengl {
	// File shaders share small source modules; includes resolve beside the including file.
	inline std::string expandShaderIncludes(const std::string& _source, const std::filesystem::path& _path, unsigned int _depth = 0) {
		if (_depth > 8) throw std::runtime_error("Shader include depth exceeded: " + _path.u8string());
		std::istringstream input(_source); std::ostringstream output; std::string line;
		while (std::getline(input, line)) {
			if (line.compare(0, 10, "#include \"") != 0) { output << line << '\n'; continue; }
			const auto end = line.find('"', 10);
			if (end == std::string::npos) throw std::runtime_error("Malformed shader include: " + _path.u8string());
			const auto include = _path.parent_path() / line.substr(10, end - 10);
			std::ifstream file(include, std::ios::binary);
			if (!file) throw std::runtime_error("Cannot read shader include: " + include.u8string());
			std::ostringstream contents; contents << file.rdbuf();
			output << expandShaderIncludes(contents.str(), include, _depth + 1);
		}
		return output.str();
	}
}
#endif
