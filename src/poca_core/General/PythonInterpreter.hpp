/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      PythonInterpreter.hpp
*/

#ifndef PythonInterpreter_h__
#define PythonInterpreter_h__

#ifndef NO_PYTHON

#include <string>
#include <vector>

#include "json.hpp"

namespace poca::core {
	class PythonInterpreter {
	public:
		static PythonInterpreter* instance();
		static void deleteInstance();
		static void setPythonInterpreterSingleton(poca::core::PythonInterpreter*);

		~PythonInterpreter();

		struct PythonFeatureInput {
			std::string component;
			std::string feature;
			const std::vector<float>* values = nullptr;
		};
		int executePocaScript(nlohmann::json&, const std::vector<PythonFeatureInput>&, const char*, const char*);
		const nlohmann::json& lastResponse() const { return m_lastResponse; }

	protected:
		PythonInterpreter();

	private:
		static PythonInterpreter* m_instance;

		nlohmann::json m_lastResponse;
	};
}
#endif
#endif
