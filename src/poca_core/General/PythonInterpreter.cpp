/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      PythonInterpreter.cpp
*
* External Python execution backend using QProcess + Windows named shared memory.
*/

#ifndef NO_PYTHON

#include <QtCore/QByteArray>
#include <QtCore/QFileInfo>
#include <QtCore/QProcess>
#include <QtCore/QProcessEnvironment>
#include <QtCore/QString>

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

#include "../General/PythonInterpreter.hpp"

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

	PythonInterpreter::PythonInterpreter()
	{
	}

	PythonInterpreter::~PythonInterpreter()
	{
		// Nothing to finalize: Python runs in a child process.
	}

	int PythonInterpreter::describePocaScript(nlohmann::json& _response, const char* _pythonExecutable, const char* _scriptPath)
	{
		_response = nlohmann::json::object();
		const std::string pythonExe = normalisePath(_pythonExecutable ? _pythonExecutable : "");
		const std::string userScript = normalisePath(_scriptPath ? _scriptPath : "");
		if (pythonExe.empty() || userScript.empty()) {
			std::cerr << "Python external mode requires both python.exe and a script path." << std::endl;
			return EXIT_FAILURE;
		}

		QFileInfo userScriptInfo(QString::fromStdString(userScript));
		const std::string workerScript = normalisePath(userScriptInfo.absolutePath().toStdString()) + "/poca_external_worker.py";
		if (!QFileInfo::exists(QString::fromStdString(pythonExe)) || !QFileInfo::exists(QString::fromStdString(workerScript)) || !QFileInfo::exists(QString::fromStdString(userScript))) {
			std::cerr << "Python external mode path error. python=" << pythonExe << ", worker=" << workerScript << ", script=" << userScript << std::endl;
			return EXIT_FAILURE;
		}

		QProcess process;
		QStringList args;
		args << QString::fromStdString(workerScript)
			 << "--script" << QString::fromStdString(userScript)
			 << "--function" << QStringLiteral("run")
			 << "--describe";

		QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
		env.insert(QStringLiteral("PYTHONNOUSERSITE"), QStringLiteral("1"));
		env.insert(QStringLiteral("PYTHONUNBUFFERED"), QStringLiteral("1"));
		QFileInfo pythonInfo(QString::fromStdString(pythonExe));
		const std::string pythonRootPath = normalisePath(pythonInfo.absolutePath().toStdString());
		env.insert(QStringLiteral("PATH"), QString::fromStdString(pythonRootPath + ";" + pythonRootPath + "/Library/bin;" + pythonRootPath + "/DLLs;") + env.value(QStringLiteral("PATH")));
		process.setProcessEnvironment(env);
		process.setProcessChannelMode(QProcess::SeparateChannels);
		process.start(QString::fromStdString(pythonExe), args);
		if (!process.waitForStarted(10000)) {
			std::cerr << "Could not start external Python process: " << pythonExe << std::endl;
			return EXIT_FAILURE;
		}

		process.write("{\"api\":\"poca_describe\"}\n");
		process.waitForBytesWritten(10000);

		QByteArray line;
		while (!line.contains('\n')) {
			if (!process.waitForReadyRead(30000)) {
				std::cerr << "External Python describe did not answer. STDERR:\n" << process.readAllStandardError().toStdString() << std::endl;
				process.kill();
				return EXIT_FAILURE;
			}
			line += process.readAllStandardOutput();
		}
		try { _response = nlohmann::json::parse(line.toStdString()); }
		catch (const nlohmann::json::exception& e) {
			std::cerr << "Could not parse external Python describe response: " << e.what() << "\nResponse was:\n" << line.toStdString() << std::endl;
			process.kill();
			return EXIT_FAILURE;
		}
		process.waitForFinished(5000);
		const QByteArray stderrData = process.readAllStandardError();
		if (!stderrData.isEmpty()) std::cerr << stderrData.toStdString() << std::endl;
		return _response.value("ok", false) ? EXIT_SUCCESS : EXIT_FAILURE;
	}

	int PythonInterpreter::executePocaScript(nlohmann::json& _response, const std::vector<PythonInterpreter::PythonFeatureInput>& _inputs, const char* _pythonExecutable, const char* _scriptPath)
	{
		_response = nlohmann::json::object();
		m_lastResponse = nlohmann::json::object();
		const std::string pythonExe = normalisePath(_pythonExecutable ? _pythonExecutable : "");
		const std::string userScript = normalisePath(_scriptPath ? _scriptPath : "");
		if (pythonExe.empty() || userScript.empty()) {
			std::cerr << "Python external mode requires both python.exe and a script path." << std::endl;
			return EXIT_FAILURE;
		}

		QFileInfo userScriptInfo(QString::fromStdString(userScript));
		const std::string workerScript = normalisePath(userScriptInfo.absolutePath().toStdString()) + "/poca_external_worker.py";

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
			 << "--function" << QStringLiteral("run");

		QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
		env.insert(QStringLiteral("PYTHONNOUSERSITE"), QStringLiteral("1"));
		env.insert(QStringLiteral("PYTHONUNBUFFERED"), QStringLiteral("1"));
		QFileInfo pythonInfo(QString::fromStdString(pythonExe));
		const std::string pythonRootPath = normalisePath(pythonInfo.absolutePath().toStdString());
		env.insert(QStringLiteral("PATH"), QString::fromStdString(pythonRootPath + ";" + pythonRootPath + "/Library/bin;" + pythonRootPath + "/DLLs;") + env.value(QStringLiteral("PATH")));
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
