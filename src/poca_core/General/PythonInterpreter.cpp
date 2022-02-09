/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      PythonInterpreter.cpp
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

#ifndef NO_PYTHON

#ifdef _DEBUG
#undef _DEBUG
#include <python.h>
#define _DEBUG
#else
#include <python.h>
#endif
#include "numpy/arrayobject.h"

#include <QtWidgets/QMessageBox>
#include <iostream>

#include <DesignPatterns/StateSoftwareSingleton.hpp>

#include "../General/PythonInterpreter.hpp"

namespace poca::core {
#if PY_MAJOR_VERSION >= 3 
	int
		init_numpy2()
	{
		import_array();
	}
#else 
	void
		init_numpy2()
	{
		import_array();
	}
#endif 

	const std::size_t ENV_BUF_SIZE = 2048; // Enough for your PATH?

	void PrintFullPath(const char* partialPath)
	{
		printf("Connecting to Python.\n");
		char full[_MAX_PATH];
		if (_fullpath(full, partialPath, _MAX_PATH) != NULL)
			printf("Full path is: %s\n", full);
		else
			printf("Invalid path\n");
	}

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

	PythonInterpreter::PythonInterpreter(): m_initialized(false)
	{
	}

	PythonInterpreter::~PythonInterpreter()
	{
		if(m_initialized)
			Py_Finalize();
	}

	int PythonInterpreter::initialize()
	{
		if (m_initialized)
			return EXIT_SUCCESS;

		poca::core::StateSoftwareSingleton* sss = poca::core::StateSoftwareSingleton::instance();
		nlohmann::json& parameters = sss->getParameters();
		std::vector <std::string> names = { "python_path", "python_dll_path", "python_lib_path", "python_packages_path", "python_scripts_path" };
		std::vector <std::string> paths(names.size());
		if (!parameters.contains("PythonParameters")) {
			QMessageBox msgBox;
			msgBox.setText("Please make sure that the Python paths have been initialized (in Menu >> Plugins >> Python).");
			msgBox.exec();
			return EXIT_FAILURE;
		}
		for(auto n = 0; n < names.size(); n++)
			if(parameters["PythonParameters"].contains(names[n]))
				paths[n] = parameters["PythonParameters"][names[n]].get<std::string>();

		for (const auto& path : paths)
			if (path.empty()) {
				return EXIT_FAILURE;
			}

		// Get current directory
		PrintFullPath(".\\");

		//Add needed path to environment variable PATH
		char buf[ENV_BUF_SIZE];
		std::size_t bufsize = ENV_BUF_SIZE;
		std::string pathToPython = paths[0];
		int e = getenv_s(&bufsize, buf, bufsize, "PATH");
		if (e) {
			//std::cerr << "`getenv_s` failed, returned " << e << '\n';
			//exit(EXIT_FAILURE);
		}
		std::string env_path, orig_path = buf;
		env_path = pathToPython + ";";
		env_path += orig_path;
		e = _putenv_s("PATH", env_path.c_str());
		if (e) {
			std::cerr << "`_putenv_s` failed, returned " << e << std::endl;
			return EXIT_FAILURE;
		}

		//Add PYTHONPATH and PYTHONHOME env variables
		std::string pythonpath = paths[1] + ";" + paths[2] + ";" + paths[3] + ";" + paths[4] + ";";
		_putenv_s("PYTHONPATH", pythonpath.c_str());
		std::string pythonhome = pathToPython;
		_putenv_s("PYTHONHOME", pythonhome.c_str());

		Py_InitializeEx(0);
		init_numpy2();

		m_initialized = true;
		return EXIT_SUCCESS;
	}

	int PythonInterpreter::applyFunctionWith1ArrayParameterAnd1DArrayReturned(QVector <double>& _res, const QVector <double>& _data, const char* _moduleName, const char* _funcName)
	{
		int result = EXIT_FAILURE;
		result = initialize();
		if (result == EXIT_FAILURE)
			return result;
		result = EXIT_FAILURE;

		const int SIZE = _data.size();
		npy_intp dims[1] = { SIZE };
		const int ND = 1;
		long double* c_arr = new long double[SIZE];

		for (int i = 0; i < SIZE; i++) {
			c_arr[i] = _data[i];
		}

		// Convert it to a NumPy array.
		PyObject* pArray = PyArray_SimpleNewFromData(ND, dims, NPY_LONGDOUBLE, reinterpret_cast<void*>(c_arr));
		if (pArray) {
			// import mymodule
			PyObject* pName = PyUnicode_FromString(_moduleName);
			if (pName) {
				PyObject* pModule = PyImport_Import(pName);
				if (pModule) {
					// import function
					PyObject* pFunc = PyObject_GetAttrString(pModule, _funcName);
					if (pFunc) {
						if (!PyCallable_Check(pFunc)) {
							std::cerr << _moduleName << "." << _funcName
								<< " is not callable." << std::endl;
						}
						else {
							PyObject* pReturn = PyObject_CallFunctionObjArgs(pFunc, pArray, NULL);
							if (pReturn) {
								PyArrayObject* np_ret = reinterpret_cast<PyArrayObject*>(pReturn);
								if (PyArray_NDIM(np_ret) != 1) {
									std::cerr << _moduleName << "." << _funcName
										<< " returned array with wrong dimension." << std::endl;
								}
								else {
									// Convert back to C++ array and print.
									int len = PyArray_SHAPE(np_ret)[0];
									long double* c_out;
									c_out = reinterpret_cast<long double*>(PyArray_DATA(np_ret));
									_res.resize(len);
									for (int i = 0; i < len; i++) {
										_res[i] = c_out[i];
									}
								}

								Py_DECREF(pReturn);
							}
						}
						Py_DECREF(pFunc);
					}
					Py_DECREF(pModule);
				}
				else
					PyErr_Print();
				Py_DECREF(pName);
			}
			Py_DECREF(pArray);
		}

		delete[] c_arr;
		result = EXIT_SUCCESS;

		if (PyErr_CheckSignals())
			PyErr_PrintEx(1);

		return result;
	}

	int PythonInterpreter::applyFunctionWith2ArraysParameterAnd1DArrayReturned(QVector <double>& _res, const QVector <double>& _data1, const QVector <double>& _data2, const char* _moduleName, const char* _funcName)
	{
		int result = EXIT_FAILURE;
		result = initialize();
		if (result == EXIT_FAILURE)
			return result;
		result = EXIT_FAILURE;

		const int SIZE1 = _data1.size(), SIZE2 = _data2.size();
		npy_intp dims1[1] = { SIZE1 }, dims2[1] = { SIZE2 };
		const int ND = 1;

		long double* c_arr1 = new long double[_data1.size()];
		for (int i = 0; i < SIZE1; i++) {
			c_arr1[i] = _data1[i];
		}
		long double* c_arr2 = new long double[_data2.size()];
		for (int i = 0; i < SIZE2; i++) {
			c_arr2[i] = _data2[i];
		}

		// Convert it to a NumPy array.
		PyObject* pArray1 = PyArray_SimpleNewFromData(ND, dims1, NPY_LONGDOUBLE, reinterpret_cast<void*>(c_arr1));
		PyObject* pArray2 = PyArray_SimpleNewFromData(ND, dims2, NPY_LONGDOUBLE, reinterpret_cast<void*>(c_arr2));
		if (pArray1 && pArray2) {
			// import mymodule
			PyObject* pName = PyUnicode_FromString(_moduleName);
			if (pName) {
				PyObject* pModule = PyImport_Import(pName);
				if (pModule) {
					// import function
					PyObject* pFunc = PyObject_GetAttrString(pModule, _funcName);
					if (pFunc) {
						if (!PyCallable_Check(pFunc)) {
							std::cerr << _moduleName << "." << _funcName
								<< " is not callable." << std::endl;
						}
						else {
							PyObject* pReturn = PyObject_CallFunctionObjArgs(pFunc, pArray1, pArray2, NULL);
							if (pReturn) {
								PyArrayObject* np_ret = reinterpret_cast<PyArrayObject*>(pReturn);
								if (PyArray_NDIM(np_ret) != 1) {
									std::cerr << _moduleName << "." << _funcName
										<< " returned array with wrong dimension." << std::endl;
								}
								else {
									// Convert back to C++ array and print.
									int len = PyArray_SHAPE(np_ret)[0];
									long double* c_out;
									c_out = reinterpret_cast<long double*>(PyArray_DATA(np_ret));
									_res.resize(len);
									for (int i = 0; i < len; i++) {
										_res[i] = c_out[i];
									}
								}

								Py_DECREF(pReturn);
							}
						}
						Py_DECREF(pFunc);
					}
					Py_DECREF(pModule);
				}
				else
					PyErr_Print();
				Py_DECREF(pName);
			}
			Py_DECREF(pArray1);
			Py_DECREF(pArray2);
		}

		delete[] c_arr1;
		delete[] c_arr2;
		result = EXIT_SUCCESS;

		if (PyErr_CheckSignals())
			PyErr_PrintEx(1);

		return result;
	}

	int PythonInterpreter::applyFunctionWithNArraysParameterAndNArrayReturned(QVector <QVector <double>>& _res, const QVector <QVector <double>>& _data, const char* _moduleName, const char* _funcName)
	{
		int result = EXIT_FAILURE;
		result = initialize();
		if (result == EXIT_FAILURE)
			return result;
		result = EXIT_FAILURE;

		if (_data.size() > 6)
			return result;

		const int ND = 1;
		std::vector <int> sizes(_data.size());
		std::vector <long double*> c_arrs(_data.size());
		std::vector <PyObject*> pArrays(_data.size());
		for (unsigned int n = 0; n < sizes.size(); n++) {
			sizes[n] = _data[n].size();
			c_arrs[n] = new long double[sizes[n]];
			for (int i = 0; i < sizes[n]; i++) {
				c_arrs[n][i] = _data[n][i];
			}
			npy_intp dims[1] = { sizes[n] };
			pArrays[n] = PyArray_SimpleNewFromData(ND, dims, NPY_LONGDOUBLE, reinterpret_cast<void*>(c_arrs[n]));
		}
		if (pArrays[0]) {
			// import mymodule
			PyObject* pName = PyUnicode_FromString(_moduleName);
			if (pName) {
				PyObject* pModule = PyImport_Import(pName);
				if (pModule) {
					// import function
					PyObject* pFunc = PyObject_GetAttrString(pModule, _funcName);
					if (pFunc) {
						if (!PyCallable_Check(pFunc)) {
							std::cerr << _moduleName << "." << _funcName
								<< " is not callable." << std::endl;
						}
						else {
							PyObject* pReturn;
							switch (_data.size()) {
							case 1:
								pReturn = PyObject_CallFunctionObjArgs(pFunc, pArrays[0], NULL);
								break;
							case 2:
								pReturn = PyObject_CallFunctionObjArgs(pFunc, pArrays[0], pArrays[1], NULL);
								break;
							case 3:
								pReturn = PyObject_CallFunctionObjArgs(pFunc, pArrays[0], pArrays[1], pArrays[2], NULL);
								break;
							case 4:
								pReturn = PyObject_CallFunctionObjArgs(pFunc, pArrays[0], pArrays[1], pArrays[2], pArrays[3], NULL);
								break;
							case 5:
								pReturn = PyObject_CallFunctionObjArgs(pFunc, pArrays[0], pArrays[1], pArrays[2], pArrays[3], pArrays[4], NULL);
								break;
							case 6:
								pReturn = PyObject_CallFunctionObjArgs(pFunc, pArrays[0], pArrays[1], pArrays[2], pArrays[3], pArrays[4], pArrays[5], NULL);
								break;
							}
							if (pReturn) {
								PyArrayObject* np_ret = reinterpret_cast<PyArrayObject*>(pReturn);
								if (PyArray_NDIM(np_ret) == 0) {
									std::cerr << _moduleName << "." << _funcName
										<< " returned array with wrong dimension." << std::endl;
								}
								else {
									// Convert back to C++ array and print.
									long double* c_out;
									c_out = reinterpret_cast<long double*>(PyArray_DATA(np_ret));
									unsigned int cpt = 0;

									_res.resize(PyArray_NDIM(np_ret));
									for (unsigned int n = 0; n < _res.size(); n++) {
										int len = PyArray_SHAPE(np_ret)[n];
										_res[n].resize(len);
										for (int i = 0; i < len; i++) {
											_res[n][i] = c_out[cpt++];
										}
									}
								}

								Py_DECREF(pReturn);
							}
						}
						Py_DECREF(pFunc);
					}
					Py_DECREF(pModule);
				}
				else
					PyErr_Print();
				Py_DECREF(pName);
			}
			for (unsigned int n = 0; n < sizes.size(); n++)
				Py_DECREF(pArrays[n]);
		}

		for (unsigned int n = 0; n < sizes.size(); n++)
			delete[] c_arrs[n];
		result = EXIT_SUCCESS;

		if (PyErr_CheckSignals())
			PyErr_PrintEx(1);

		return result;
	}

	int PythonInterpreter::applyFunctionWithNArraysParameterAnd1ArrayReturned(QVector <double>& _res, const QVector <QVector <double>>& _data, const char* _moduleName, const char* _funcName)
	{
		int result = EXIT_FAILURE;
		result = initialize();
		if (result == EXIT_FAILURE)
			return result;
		result = EXIT_FAILURE;

		if (_data.size() > 6)
			return result;

		const int ND = 1;
		std::vector <int> sizes(_data.size());
		std::vector <long double*> c_arrs(_data.size());
		std::vector <PyObject*> pArrays(_data.size());
		for (unsigned int n = 0; n < sizes.size(); n++) {
			sizes[n] = _data[n].size();
			c_arrs[n] = new long double[sizes[n]];
			for (int i = 0; i < sizes[n]; i++) {
				c_arrs[n][i] = _data[n][i];
			}
			npy_intp dims[1] = { sizes[n] };
			pArrays[n] = PyArray_SimpleNewFromData(ND, dims, NPY_DOUBLE, reinterpret_cast<void*>(c_arrs[n]));
		}

		if (pArrays[0]) {
			// import mymodule
			PyObject* pName = PyUnicode_FromString(_moduleName);
			if (pName) {
				PyObject* pModule = PyImport_Import(pName);
				if (pModule) {
					// import function
					PyObject* pFunc = PyObject_GetAttrString(pModule, _funcName);
					if (pFunc) {
						if (!PyCallable_Check(pFunc)) {
							std::cerr << _moduleName << "." << _funcName
								<< " is not callable." << std::endl;
						}
						else {
							PyObject* pReturn;
							switch (_data.size()) {
							case 1:
								pReturn = PyObject_CallFunctionObjArgs(pFunc, pArrays[0], NULL);
								break;
							case 2:
								pReturn = PyObject_CallFunctionObjArgs(pFunc, pArrays[0], pArrays[1], NULL);
								break;
							case 3:
								pReturn = PyObject_CallFunctionObjArgs(pFunc, pArrays[0], pArrays[1], pArrays[2], NULL);
								break;
							case 4:
								pReturn = PyObject_CallFunctionObjArgs(pFunc, pArrays[0], pArrays[1], pArrays[2], pArrays[3], NULL);
								break;
							case 5:
								pReturn = PyObject_CallFunctionObjArgs(pFunc, pArrays[0], pArrays[1], pArrays[2], pArrays[3], pArrays[4], NULL);
								break;
							case 6:
								pReturn = PyObject_CallFunctionObjArgs(pFunc, pArrays[0], pArrays[1], pArrays[2], pArrays[3], pArrays[4], pArrays[5], NULL);
								break;
							}
							if (pReturn) {
								PyArrayObject* np_ret = reinterpret_cast<PyArrayObject*>(pReturn);
								if (PyArray_NDIM(np_ret) != 1) {
									std::cerr << _moduleName << "." << _funcName
										<< " returned array with wrong dimension." << std::endl;
								}
								else {
									// Convert back to C++ array and print.
									int len = PyArray_SHAPE(np_ret)[0];
									long double* c_out;
									c_out = reinterpret_cast<long double*>(PyArray_DATA(np_ret));
									_res.resize(len);
									for (int i = 0; i < len; i++) {
										_res[i] = c_out[i];
									}
								}
								PyErr_Print();
								Py_DECREF(pReturn);
							}
							else
								PyErr_Print();
						}
						PyErr_Print();
						Py_DECREF(pFunc);
					}
					else
						PyErr_Print();
					Py_DECREF(pModule);
				}
				else
					PyErr_Print();
				Py_DECREF(pName);
			}
			else
				PyErr_Print();
			for (unsigned int n = 0; n < sizes.size(); n++)
				Py_DECREF(pArrays[n]);
		}

		for (unsigned int n = 0; n < sizes.size(); n++)
			delete[] c_arrs[n];
		result = EXIT_SUCCESS;

		if (PyErr_CheckSignals())
			PyErr_PrintEx(1);

		return result;
	}
}
#endif

