/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      PythonInterpreter.cpp
*
* External Python execution backend using QProcess + Windows named shared memory.
*/

#ifndef NO_PYTHON

#include <QtCore/QByteArray>
#include <QtCore/QCoreApplication>
#include <QtCore/QDir>
#include <QtCore/QFileInfo>
#include <QtCore/QProcess>
#include <QtCore/QProcessEnvironment>
#include <QtCore/QString>
#include <QtWidgets/QMessageBox>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#error "External Python shared-memory mode is currently implemented for Windows named shared memory."
#endif

#include <General/Engine.hpp>

#include "../General/PythonInterpreter.hpp"
#include "../General/Misc.h"

namespace {
	struct WinSharedMemory {
		std::string name;
		HANDLE handle = NULL;
		void* data = nullptr;
		size_t size = 0;

		~WinSharedMemory() { close(); }

		bool create(const std::string& _name, size_t _size)
		{
			name = _name;
			size = _size;
			const DWORD sizeLow = static_cast<DWORD>(_size & 0xffffffffULL);
			const DWORD sizeHigh = static_cast<DWORD>((_size >> 32) & 0xffffffffULL);
			handle = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, sizeHigh, sizeLow, name.c_str());
			if (handle == NULL) return false;
			data = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, _size);
			if (data == nullptr) {
				CloseHandle(handle);
				handle = NULL;
				return false;
			}
			return true;
		}

		bool open(const std::string& _name, size_t _size)
		{
			name = _name;
			size = _size;
			handle = OpenFileMappingA(FILE_MAP_READ, FALSE, name.c_str());
			if (handle == NULL) return false;
			data = MapViewOfFile(handle, FILE_MAP_READ, 0, 0, _size);
			if (data == nullptr) {
				CloseHandle(handle);
				handle = NULL;
				return false;
			}
			return true;
		}

		void close()
		{
			if (data != nullptr) {
				UnmapViewOfFile(data);
				data = nullptr;
			}
			if (handle != NULL) {
				CloseHandle(handle);
				handle = NULL;
			}
		}
	};

	std::string normalisePath(std::string _path)
	{
		std::replace(_path.begin(), _path.end(), '\\', '/');
		while (!_path.empty() && (_path.back() == '/' || _path.back() == '\\'))
			_path.pop_back();
		return _path;
	}

	std::string quoteForJson(const std::string& s)
	{
		std::ostringstream os;
		os << '"';
		for (char c : s) {
			switch (c) {
			case '\\': os << "\\\\"; break;
			case '"': os << "\\\""; break;
			case '\n': os << "\\n"; break;
			case '\r': os << "\\r"; break;
			case '\t': os << "\\t"; break;
			default: os << c; break;
			}
		}
		os << '"';
		return os.str();
	}

	std::string makeSharedMemoryName(size_t index)
	{
		const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		std::ostringstream os;
		os << "poca_" << GetCurrentProcessId() << "_" << now << "_in_" << index;
		return os.str();
	}
}

namespace poca::core {
	PythonInterpreter* PythonInterpreter::m_instance = 0;

	PythonInterpreter* PythonInterpreter::instance()
	{
		if (m_instance == 0)
			m_instance = new PythonInterpreter;
		return m_instance;
	}

	void PythonInterpreter::deleteInstance()
	{
		if (m_instance != 0)
			delete m_instance;
		m_instance = 0;
	}

	void PythonInterpreter::setPythonInterpreterSingleton(poca::core::PythonInterpreter* _pint)
	{
		m_instance = _pint;
	}

	PythonInterpreter::PythonInterpreter() : m_initialized(false)
	{
	}

	PythonInterpreter::~PythonInterpreter()
	{
		// Nothing to finalize: Python runs in a child process.
	}

	int PythonInterpreter::initialize()
	{
		if (m_initialized)
			return EXIT_SUCCESS;

		nlohmann::json& parameters = poca::core::Engine::instance()->getGlobalParameters();
		std::vector <std::string> names = { "python_path", "python_dll_path", "python_lib_path", "python_packages_path", "python_scripts_path" };
		std::vector <std::string> paths(names.size());
		if (!parameters.contains("PythonParameters")) {
			QMessageBox msgBox;
			msgBox.setText("Please make sure that the Python paths have been initialized (in Menu >> Plugins >> Python).");
			msgBox.exec();
			return EXIT_FAILURE;
		}
		for (auto n = 0; n < names.size(); n++)
			if (parameters["PythonParameters"].contains(names[n]))
				paths[n] = parameters["PythonParameters"][names[n]].get<std::string>();

		if (paths[0].empty() || paths[4].empty())
			return EXIT_FAILURE;

		m_pythonRootPath = normalisePath(paths[0]);
		m_pythonScriptsPath = normalisePath(paths[4]);
		m_initialized = true;
		return EXIT_SUCCESS;
	}

	std::string PythonInterpreter::resolveScriptPath(const char* _moduleName) const
	{
		std::string script = _moduleName ? _moduleName : "";
		script = normalisePath(script);
		if (script.size() >= 3 && script.substr(script.size() - 3) == ".py") {
			QFileInfo fi(QString::fromStdString(script));
			if (fi.isAbsolute()) return script;
			return m_pythonScriptsPath + "/" + script;
		}

		// Existing callers pass a module name such as "nena".
		return m_pythonScriptsPath + "/" + script + ".py";
	}

	int PythonInterpreter::executeExternalPython(QVector <QVector <double>>& _res, const QVector <QVector <double>>& _data, const char* _moduleName, const char* _funcName)
	{
		if (initialize() == EXIT_FAILURE)
			return EXIT_FAILURE;

		const std::string pythonExe = m_pythonRootPath + "/python.exe";
		const std::string workerScript = m_pythonScriptsPath + "/poca_external_worker.py";
		const std::string userScript = resolveScriptPath(_moduleName);

		if (!QFileInfo::exists(QString::fromStdString(pythonExe))) {
			std::cerr << "Python executable does not exist: " << pythonExe << std::endl;
			return EXIT_FAILURE;
		}
		if (!QFileInfo::exists(QString::fromStdString(workerScript))) {
			std::cerr << "PoCA Python worker does not exist: " << workerScript << std::endl;
			return EXIT_FAILURE;
		}
		if (!QFileInfo::exists(QString::fromStdString(userScript))) {
			std::cerr << "Python script does not exist: " << userScript << std::endl;
			return EXIT_FAILURE;
		}

		std::vector <std::unique_ptr<WinSharedMemory>> inputShms;
		inputShms.reserve(_data.size());

		std::ostringstream request;
		request << "{\"inputs\":[";
		for (int i = 0; i < _data.size(); ++i) {
			const QVector <double>& values = _data[i];
			const size_t nbytes = static_cast<size_t>(values.size()) * sizeof(double);
			std::unique_ptr<WinSharedMemory> shm(new WinSharedMemory());
			const std::string shmName = makeSharedMemoryName(static_cast<size_t>(i));
			if (!shm->create(shmName, std::max<size_t>(nbytes, 1))) {
				std::cerr << "Could not create input shared memory segment: " << shmName << std::endl;
				return EXIT_FAILURE;
			}
			if (nbytes > 0)
				std::memcpy(shm->data, values.constData(), nbytes);

			if (i > 0) request << ",";
			request << "{\"name\":" << quoteForJson(shmName)
				<< ",\"dtype\":\"float64\""
				<< ",\"shape\":[" << values.size() << "]"
				<< ",\"nbytes\":" << nbytes << "}";
			inputShms.push_back(std::move(shm));
		}
		request << "]}\n";

		QProcess process;
		QStringList args;
		args << QString::fromStdString(workerScript)
			 << "--script" << QString::fromStdString(userScript)
			 << "--function" << QString::fromStdString(_funcName ? _funcName : "");

		QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
		env.insert(QStringLiteral("PYTHONNOUSERSITE"), QStringLiteral("1"));
		env.insert(QStringLiteral("PYTHONUNBUFFERED"), QStringLiteral("1"));
		env.insert(QStringLiteral("PATH"), QString::fromStdString(m_pythonRootPath + ";" + m_pythonRootPath + "/Library/bin;" + m_pythonRootPath + "/DLLs;") + env.value(QStringLiteral("PATH")));
		process.setProcessEnvironment(env);
		process.setProcessChannelMode(QProcess::SeparateChannels);
		process.start(QString::fromStdString(pythonExe), args);
		if (!process.waitForStarted(10000)) {
			std::cerr << "Could not start external Python process: " << pythonExe << std::endl;
			return EXIT_FAILURE;
		}

		process.write(QByteArray::fromStdString(request.str()));
		process.waitForBytesWritten(10000);

		QByteArray line;
		while (!line.contains('\n')) {
			if (!process.waitForReadyRead(30000)) {
				std::cerr << "External Python process did not answer. STDERR:\n"
					<< process.readAllStandardError().toStdString() << std::endl;
				process.kill();
				return EXIT_FAILURE;
			}
			line += process.readAllStandardOutput();
		}

		nlohmann::json response;
		try {
			response = nlohmann::json::parse(line.toStdString());
		}
		catch (const nlohmann::json::exception& e) {
			std::cerr << "Could not parse external Python response: " << e.what() << "\nResponse was:\n" << line.toStdString() << std::endl;
			process.kill();
			return EXIT_FAILURE;
		}

		if (!response.value("ok", false)) {
			std::cerr << "External Python error: " << response.value("error", std::string("unknown error")) << std::endl;
			if (response.contains("traceback"))
				std::cerr << response["traceback"].get<std::string>() << std::endl;
			process.write("done\n");
			process.waitForFinished(5000);
			return EXIT_FAILURE;
		}

		_res.clear();
		for (const auto& output : response["outputs"]) {
			const std::string name = output["name"].get<std::string>();
			const size_t nbytes = output["nbytes"].get<size_t>();
			std::vector<size_t> shape = output["shape"].get<std::vector<size_t>>();
			size_t nbValues = 1;
			for (const auto dim : shape) nbValues *= dim;
			if (nbytes != nbValues * sizeof(double)) {
				std::cerr << "Unexpected output size from Python shared memory segment." << std::endl;
				process.write("done\n");
				process.waitForFinished(5000);
				return EXIT_FAILURE;
			}

			WinSharedMemory outputShm;
			if (!outputShm.open(name, std::max<size_t>(nbytes, 1))) {
				std::cerr << "Could not open output shared memory segment: " << name << std::endl;
				process.write("done\n");
				process.waitForFinished(5000);
				return EXIT_FAILURE;
			}

			QVector <double> values(static_cast<int>(nbValues));
			if (nbytes > 0)
				std::memcpy(values.data(), outputShm.data, nbytes);
			_res.push_back(values);
		}

		process.write("done\n");
		process.waitForBytesWritten(5000);
		process.waitForFinished(10000);

		const QByteArray stderrData = process.readAllStandardError();
		if (!stderrData.isEmpty())
			std::cerr << stderrData.toStdString() << std::endl;

		return EXIT_SUCCESS;
	}

	int PythonInterpreter::applyFunctionWith1ArrayParameterAnd1DArrayReturned(QVector <double>& _res, const QVector <double>& _data, const char* _moduleName, const char* _funcName)
	{
		QVector <QVector <double>> in, out;
		in.push_back(_data);
		int result = executeExternalPython(out, in, _moduleName, _funcName);
		if (result == EXIT_SUCCESS && !out.empty()) _res = out[0];
		return result;
	}

	int PythonInterpreter::applyFunctionWith2ArraysParameterAnd1DArrayReturned(QVector <double>& _res, const QVector <double>& _data1, const QVector <double>& _data2, const char* _moduleName, const char* _funcName)
	{
		QVector <QVector <double>> in, out;
		in.push_back(_data1);
		in.push_back(_data2);
		int result = executeExternalPython(out, in, _moduleName, _funcName);
		if (result == EXIT_SUCCESS && !out.empty()) _res = out[0];
		return result;
	}

	int PythonInterpreter::applyFunctionWithNArraysParameterAndNArrayReturned(QVector <QVector <double>>& _res, const QVector <QVector <double>>& _data, const char* _moduleName, const char* _funcName)
	{
		return executeExternalPython(_res, _data, _moduleName, _funcName);
	}

	int PythonInterpreter::applyFunctionWithNArraysParameterAnd1ArrayReturned(QVector <double>& _res, const QVector <QVector <double>>& _data, const char* _moduleName, const char* _funcName)
	{
		QVector <QVector <double>> out;
		int result = executeExternalPython(out, _data, _moduleName, _funcName);
		if (result == EXIT_SUCCESS && !out.empty()) _res = out[0];
		return result;
	}

	int PythonInterpreter::applyFunctionWithNFloatArraysParameterAnd1ArrayReturned(QVector <double>& _res, const std::vector<const std::vector<float>*>& _data, const char* _moduleName, const char* _funcName)
	{
		// Keep PythonWidget source-compatible while avoiding the old QVector<double>
		// conversion path. Inputs are exported as float32 shared-memory arrays.
		if (initialize() == EXIT_FAILURE)
			return EXIT_FAILURE;

		const std::string pythonExe = m_pythonRootPath + "/python.exe";
		const std::string workerScript = m_pythonScriptsPath + "/poca_external_worker.py";
		const std::string userScript = resolveScriptPath(_moduleName);

		if (!QFileInfo::exists(QString::fromStdString(pythonExe))) {
			std::cerr << "Python executable does not exist: " << pythonExe << std::endl;
			return EXIT_FAILURE;
		}
		if (!QFileInfo::exists(QString::fromStdString(workerScript))) {
			std::cerr << "PoCA Python worker does not exist: " << workerScript << std::endl;
			return EXIT_FAILURE;
		}
		if (!QFileInfo::exists(QString::fromStdString(userScript))) {
			std::cerr << "Python script does not exist: " << userScript << std::endl;
			return EXIT_FAILURE;
		}

		std::vector <std::unique_ptr<WinSharedMemory>> inputShms;
		inputShms.reserve(_data.size());

		std::ostringstream request;
		request << "{\"inputs\":[";
		for (size_t i = 0; i < _data.size(); ++i) {
			const std::vector<float>* values = _data[i];
			const size_t nbValues = values != nullptr ? values->size() : 0;
			const size_t nbytes = nbValues * sizeof(float);
			std::unique_ptr<WinSharedMemory> shm(new WinSharedMemory());
			const std::string shmName = makeSharedMemoryName(i);
			if (!shm->create(shmName, std::max<size_t>(nbytes, 1))) {
				std::cerr << "Could not create input shared memory segment: " << shmName << std::endl;
				return EXIT_FAILURE;
			}
			if (nbytes > 0 && values != nullptr)
				std::memcpy(shm->data, values->data(), nbytes);

			if (i > 0) request << ",";
			request << "{\"name\":" << quoteForJson(shmName)
				<< ",\"dtype\":\"float32\""
				<< ",\"shape\":[" << nbValues << "]"
				<< ",\"nbytes\":" << nbytes << "}";
			inputShms.push_back(std::move(shm));
		}
		request << "]}\n";

		QProcess process;
		QStringList args;
		args << QString::fromStdString(workerScript)
			 << "--script" << QString::fromStdString(userScript)
			 << "--function" << QString::fromStdString(_funcName ? _funcName : "");

		QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
		env.insert(QStringLiteral("PYTHONNOUSERSITE"), QStringLiteral("1"));
		env.insert(QStringLiteral("PYTHONUNBUFFERED"), QStringLiteral("1"));
		env.insert(QStringLiteral("PATH"), QString::fromStdString(m_pythonRootPath + ";" + m_pythonRootPath + "/Library/bin;" + m_pythonRootPath + "/DLLs;") + env.value(QStringLiteral("PATH")));
		process.setProcessEnvironment(env);
		process.setProcessChannelMode(QProcess::SeparateChannels);
		process.start(QString::fromStdString(pythonExe), args);
		if (!process.waitForStarted(10000)) {
			std::cerr << "Could not start external Python process: " << pythonExe << std::endl;
			return EXIT_FAILURE;
		}

		process.write(QByteArray::fromStdString(request.str()));
		process.waitForBytesWritten(10000);

		QByteArray line;
		while (!line.contains('\n')) {
			if (!process.waitForReadyRead(30000)) {
				std::cerr << "External Python process did not answer. STDERR:\n"
					<< process.readAllStandardError().toStdString() << std::endl;
				process.kill();
				return EXIT_FAILURE;
			}
			line += process.readAllStandardOutput();
		}

		nlohmann::json response;
		try {
			response = nlohmann::json::parse(line.toStdString());
		}
		catch (const nlohmann::json::exception& e) {
			std::cerr << "Could not parse external Python response: " << e.what() << "\nResponse was:\n" << line.toStdString() << std::endl;
			process.kill();
			return EXIT_FAILURE;
		}

		if (!response.value("ok", false)) {
			std::cerr << "External Python error: " << response.value("error", std::string("unknown error")) << std::endl;
			if (response.contains("traceback"))
				std::cerr << response["traceback"].get<std::string>() << std::endl;
			process.write("done\n");
			process.waitForFinished(5000);
			return EXIT_FAILURE;
		}

		_res.clear();
		if (response.contains("outputs") && !response["outputs"].empty()) {
			const auto& output = response["outputs"][0];
			const std::string name = output["name"].get<std::string>();
			const size_t nbytes = output["nbytes"].get<size_t>();
			std::vector<size_t> shape = output["shape"].get<std::vector<size_t>>();
			size_t nbValues = 1;
			for (const auto dim : shape) nbValues *= dim;
			if (nbytes != nbValues * sizeof(double)) {
				std::cerr << "Unexpected output size from Python shared memory segment." << std::endl;
				process.write("done\n");
				process.waitForFinished(5000);
				return EXIT_FAILURE;
			}

			WinSharedMemory outputShm;
			if (!outputShm.open(name, std::max<size_t>(nbytes, 1))) {
				std::cerr << "Could not open output shared memory segment: " << name << std::endl;
				process.write("done\n");
				process.waitForFinished(5000);
				return EXIT_FAILURE;
			}

			_res.resize(static_cast<int>(nbValues));
			if (nbytes > 0)
				std::memcpy(_res.data(), outputShm.data, nbytes);
		}

		process.write("done\n");
		process.waitForBytesWritten(5000);
		process.waitForFinished(10000);

		const QByteArray stderrData = process.readAllStandardError();
		if (!stderrData.isEmpty())
			std::cerr << stderrData.toStdString() << std::endl;

		return EXIT_SUCCESS;
	}
	int PythonInterpreter::executePocaScript(nlohmann::json& _response, const std::vector<PythonInterpreter::PythonFeatureInput>& _inputs, const char* _moduleName, const char* _funcName)
	{
		_response = nlohmann::json::object();
		m_lastResponse = nlohmann::json::object();
		if (initialize() == EXIT_FAILURE)
			return EXIT_FAILURE;

		const std::string pythonExe = m_pythonRootPath + "/python.exe";
		const std::string workerScript = m_pythonScriptsPath + "/poca_external_worker.py";
		const std::string userScript = resolveScriptPath(_moduleName);

		if (!QFileInfo::exists(QString::fromStdString(pythonExe)) || !QFileInfo::exists(QString::fromStdString(workerScript)) || !QFileInfo::exists(QString::fromStdString(userScript))) {
			std::cerr << "Python external mode path error. python=" << pythonExe << ", worker=" << workerScript << ", script=" << userScript << std::endl;
			return EXIT_FAILURE;
		}

		std::vector <std::unique_ptr<WinSharedMemory>> inputShms;
		inputShms.reserve(_inputs.size());

		std::ostringstream request;
		request << "{\"api\":\"poca\",\"inputs\":[";
		for (size_t i = 0; i < _inputs.size(); ++i) {
			const PythonInterpreter::PythonFeatureInput& input = _inputs[i];
			const std::vector<float>* values = input.values;
			const size_t nbValues = values != nullptr ? values->size() : 0;
			const size_t nbytes = nbValues * sizeof(float);
			std::unique_ptr<WinSharedMemory> shm(new WinSharedMemory());
			const std::string shmName = makeSharedMemoryName(i);
			if (!shm->create(shmName, std::max<size_t>(nbytes, 1))) {
				std::cerr << "Could not create input shared memory segment: " << shmName << std::endl;
				return EXIT_FAILURE;
			}
			if (nbytes > 0 && values != nullptr)
				std::memcpy(shm->data, values->data(), nbytes);

			if (i > 0) request << ",";
			request << "{\"name\":" << quoteForJson(shmName)
				<< ",\"component\":" << quoteForJson(input.component)
				<< ",\"feature\":" << quoteForJson(input.feature)
				<< ",\"dtype\":\"float32\""
				<< ",\"shape\":[" << nbValues << "]"
				<< ",\"nbytes\":" << nbytes << "}";
			inputShms.push_back(std::move(shm));
		}
		request << "]}\n";

		QProcess process;
		QStringList args;
		args << QString::fromStdString(workerScript)
			 << "--script" << QString::fromStdString(userScript)
			 << "--function" << QString::fromStdString(_funcName ? _funcName : "");

		QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
		env.insert(QStringLiteral("PYTHONNOUSERSITE"), QStringLiteral("1"));
		env.insert(QStringLiteral("PYTHONUNBUFFERED"), QStringLiteral("1"));
		env.insert(QStringLiteral("PATH"), QString::fromStdString(m_pythonRootPath + ";" + m_pythonRootPath + "/Library/bin;" + m_pythonRootPath + "/DLLs;") + env.value(QStringLiteral("PATH")));
		process.setProcessEnvironment(env);
		process.setProcessChannelMode(QProcess::SeparateChannels);
		process.start(QString::fromStdString(pythonExe), args);
		if (!process.waitForStarted(10000)) {
			std::cerr << "Could not start external Python process: " << pythonExe << std::endl;
			return EXIT_FAILURE;
		}

		process.write(QByteArray::fromStdString(request.str()));
		process.waitForBytesWritten(10000);

		QByteArray line;
		while (!line.contains('\n')) {
			if (!process.waitForReadyRead(30000)) {
				std::cerr << "External Python process did not answer. STDERR:\n" << process.readAllStandardError().toStdString() << std::endl;
				process.kill();
				return EXIT_FAILURE;
			}
			line += process.readAllStandardOutput();
		}

		nlohmann::json response;
		try { response = nlohmann::json::parse(line.toStdString()); }
		catch (const nlohmann::json::exception& e) {
			std::cerr << "Could not parse external Python response: " << e.what() << "\nResponse was:\n" << line.toStdString() << std::endl;
			process.kill();
			return EXIT_FAILURE;
		}

		if (!response.value("ok", false)) {
			std::cerr << "External Python error: " << response.value("error", std::string("unknown error")) << std::endl;
			if (response.contains("traceback")) std::cerr << response["traceback"].get<std::string>() << std::endl;
			process.write("done\n");
			process.waitForFinished(5000);
			return EXIT_FAILURE;
		}

		// Copy returned shared-memory arrays into the JSON object as in-memory vectors.
		if (response.contains("actions")) {
			for (auto& action : response["actions"]) {
				if (action.contains("values_shm")) {
					auto payload = action["values_shm"];
					const std::string name = payload["name"].get<std::string>();
					const size_t nbytes = payload["nbytes"].get<size_t>();
					std::vector<size_t> shape = payload["shape"].get<std::vector<size_t>>();
					size_t nbValues = 1; for (const auto dim : shape) nbValues *= dim;
					if (nbytes != nbValues * sizeof(double)) {
						std::cerr << "Unexpected output size from Python shared memory segment." << std::endl;
						process.write("done\n"); process.waitForFinished(5000); return EXIT_FAILURE;
					}
					WinSharedMemory outputShm;
					if (!outputShm.open(name, std::max<size_t>(nbytes, 1))) {
						std::cerr << "Could not open output shared memory segment: " << name << std::endl;
						process.write("done\n"); process.waitForFinished(5000); return EXIT_FAILURE;
					}
					std::vector<double> values(nbValues);
					if (nbytes > 0) std::memcpy(values.data(), outputShm.data, nbytes);
					action.erase("values_shm");
					action["values"] = values;
				}
			}
		}

		process.write("done\n");
		process.waitForBytesWritten(5000);
		process.waitForFinished(10000);

		const QByteArray stderrData = process.readAllStandardError();
		if (!stderrData.isEmpty()) std::cerr << stderrData.toStdString() << std::endl;

		_response = response;
		m_lastResponse = response;
		return EXIT_SUCCESS;
	}
}
#endif
