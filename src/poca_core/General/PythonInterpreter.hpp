/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      PythonInterpreter.hpp
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

#ifndef PythonInterpreter_h__
#define PythonInterpreter_h__

#ifndef NO_PYTHON

#include <QtCore/QVector>

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

	protected:
		PythonInterpreter();

		int initialize();

	private:
		static PythonInterpreter* m_instance;

		bool m_initialized;
	};
}
#endif
#endif

