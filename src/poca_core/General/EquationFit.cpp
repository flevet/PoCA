/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      EquationFit.cpp
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

#include <math.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <QtCore/qmath.h>
#include <iostream>

#include "EquationFit.hpp"
#include "../Fit/lmcurve.h"

namespace poca::core {
	double leeFunction2(double n, const double* p) {
		return pow(p[0], n) * (1 - p[0]);
	}

	double expDecayHalfLife2(double t, const double* p) {
		return p[0] + p[1] * exp(-t / p[2]);
	}

	double expDecayValue2(double t, const double* p) {
		return p[0] * exp(-t * p[1]);
	}

	EquationFit::EquationFit()
	{
	}

	EquationFit::EquationFit(const EquationFit& _o) : m_ts(_o.m_ts), m_values(_o.m_values), m_fitsValues(_o.m_fitsValues), m_paramsEqn(_o.m_paramsEqn), m_typeEqn(_o.m_typeEqn), m_nbParamEqn(_o.m_nbParamEqn), m_eqn(_o.m_eqn), m_rSqrTest(_o.m_rSqrTest)
	{
		m_function = _o.m_function;
	}

	EquationFit& EquationFit::operator=(const EquationFit& _o)
	{
		m_function = _o.m_function;
		m_rSqrTest = _o.m_rSqrTest;
		m_nbParamEqn = _o.m_nbParamEqn;
		m_eqn = _o.m_eqn;
		m_typeEqn = _o.m_typeEqn;

		m_ts = _o.m_ts;
		m_values = _o.m_values;
		m_fitsValues = _o.m_fitsValues;
		m_paramsEqn = _o.m_paramsEqn;

		return *this;
	}

	EquationFit::EquationFit(const std::vector<float>& _ts, const std::vector<float>& _values, const int _typeEqn)
	{
		setEquation(_ts, _values, _typeEqn);
	}

	EquationFit::EquationFit(const std::vector<double>& _ts, const std::vector<double>& _values, const int _typeEqn)
	{
		setEquation(_ts, _values, _typeEqn);
	}

	void EquationFit::setEquation(const std::vector<float>& _ts, const std::vector<float>& _values, const int _typeEqn)
	{
		std::vector <double> tsTmp(_ts.begin(), _ts.end()), valuesTmp(_values.begin(), _values.end());
		setEquation(tsTmp, valuesTmp, _typeEqn);
	}

	void EquationFit::setEquation(const std::vector<double>& _ts, const std::vector<double>& _values, const int _typeEqn)
	{
		m_typeEqn = _typeEqn;

		m_ts = _ts;
		m_values = _values;

		switch (m_typeEqn)
		{
		case LeeFunction:
		{
			m_nbParamEqn = 1;
			m_paramsEqn.resize(m_nbParamEqn);
			m_paramsEqn[0] = m_values[0];
			m_function = &leeFunction2;
			break;
		}
		case ExpDecayHalLife:
		{
			m_nbParamEqn = 3;
			m_paramsEqn.resize(m_nbParamEqn);
			m_paramsEqn[0] = m_values[m_ts.size() - 1]; m_paramsEqn[1] = m_values[0]; m_paramsEqn[2] = 2;
			m_function = &expDecayHalfLife2;
			break;
		}
		case ExpDecayValue:
		{
			m_nbParamEqn = 2;
			m_paramsEqn.resize(m_nbParamEqn);
			m_paramsEqn[0] = m_values[0]; m_paramsEqn[1] = 2;
			m_function = &expDecayValue2;
			break;
		}
		default:
			break;
		}

		lm_control_struct control = lm_control_double;
		lm_status_struct status;
		control.verbosity = 9;
		//printf( "Fitting ...\n" );
		lmcurve(m_nbParamEqn, m_paramsEqn.data(), m_ts.size(), m_ts.data(), m_values.data(), m_function, &control, &status);
		printf( "Results:\n" );
		printf( "status after %d function evaluations:\n  %s\n", status.nfev, lm_infmsg[status.outcome] );
		printf("obtained parameters:\n");
		for ( int i = 0; i < m_nbParamEqn; ++i)
			printf("  par[%i] = %12g\n", i, m_paramsEqn[i]);
		printf("obtained norm:\n  %12g\n", status.fnorm );
		m_fitsValues.resize(m_ts.size());
		//printf("fitting data as follows:\n");
		for (int i = 0; i < m_ts.size(); ++i) {
			//printf( "  t[%2d]=%4g y=%6g fit=%10g residue=%12g\n", i, m_ts[i], m_values[i], m_function( m_ts[i], m_paramsEqn ), m_values[i] - m_function( m_ts[i], m_paramsEqn ) );
			m_fitsValues[i] = m_function(m_ts[i], m_paramsEqn.data());
		}


		switch (m_typeEqn) {
		case LeeFunction:
			m_eqn = ("y = " + std::to_string(m_paramsEqn[0]) + "^x*( 1 - " + std::to_string(m_paramsEqn[0]) + ")");
			break;
		case ExpDecayHalLife:
			m_eqn = ("y = " + std::to_string(m_paramsEqn[0]) + " + " + std::to_string(m_paramsEqn[1]) + "*exp(-x/" + std::to_string(m_paramsEqn[2]) + ")");
			break;
		case ExpDecayValue:
			m_eqn = ("y = " + std::to_string(m_paramsEqn[0]) + "*exp(-" + std::to_string(m_paramsEqn[1]) + "x)");
			break;
		default:
			break;
		}
	}

	EquationFit::~EquationFit()
	{
	}

	double EquationFit::getFitValues(const double _t)
	{
		return m_function(_t, m_paramsEqn.data());
	}

	double EquationFit::getFitValuesFunctionN(const double _t, const unsigned int _indexSubFunction)
	{
		return m_function(_t, m_paramsEqn.data());
	}
}

