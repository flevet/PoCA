/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      PythonInterpreter.hpp
*/

#ifndef PythonInterpreter_h__
#define PythonInterpreter_h__

#ifndef NO_PYTHON

#include <QtCore/QVector>
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

		int applyFunctionWith1ArrayParameterAnd1DArrayReturned(QVector <double>&, const QVector <double>&, const char*, const char*);
		int applyFunctionWith2ArraysParameterAnd1DArrayReturned(QVector <double>&, const QVector <double>&, const QVector <double>&, const char*, const char*);

		int applyFunctionWithNArraysParameterAndNArrayReturned(QVector <QVector <double>>&, const QVector <QVector <double>>&, const char*, const char*);
		int applyFunctionWithNArraysParameterAnd1ArrayReturned(QVector <double>&, const QVector <QVector <double>>&, const char*, const char*);
		int applyFunctionWithNFloatArraysParameterAnd1ArrayReturned(QVector <double>&, const std::vector<const std::vector<float>*>&, const char*, const char*);

		struct PythonFeatureInput {
			std::string component;
			std::string feature;
			const std::vector<float>* values = nullptr;
		};
		int executePocaScript(nlohmann::json&, const std::vector<PythonFeatureInput>&, const char*, const char*);
		const nlohmann::json& lastResponse() const { return m_lastResponse; }

	protected:
		PythonInterpreter();

		// External mode does not initialize/link an in-process Python interpreter.
		// It only validates the Python parameters configured in PoCA.
		int initialize();

	private:
		static PythonInterpreter* m_instance;

		bool m_initialized;
		std::string m_pythonRootPath;
		std::string m_pythonScriptsPath;
		nlohmann::json m_lastResponse;

		int executeExternalPython(QVector <QVector <double>>&, const QVector <QVector <double>>&, const char*, const char*);
		std::string resolveScriptPath(const char*) const;
	};
}
#endif
#endif
