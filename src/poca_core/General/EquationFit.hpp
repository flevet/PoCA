/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      EquationFit.hpp
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

#ifndef EquationFit_h__
#define EquationFit_h__

#include <vector>
#include <string>

namespace poca::core {
	class EquationFit {
	public:
		enum EquationType
		{
			Gaussian = 0,
			ExpDecayValue = 1,
			LeeFunction = 2,
			ExpDecayHalLife = 3,
		};

		EquationFit();
		EquationFit(const EquationFit&);
		EquationFit(const std::vector<float>&, const std::vector<float>&, const int);
		EquationFit(const std::vector<double>&, const std::vector<double>&, const int);
		~EquationFit();

		EquationFit& operator=(const EquationFit&);

		inline const std::vector<double>& getTs() const { return m_ts; }
		inline const std::vector<double>& getValues() const { return m_values; }
		inline const std::vector<double>& getFitValues() const { return m_fitsValues; }
		inline const std::vector<double>& getParams() const { return m_paramsEqn; }
		inline int getNbParam() const { return m_nbParamEqn; }
		inline size_t getNbTs() const { return m_ts.size(); }
		inline int typeEqn() const { return m_typeEqn; }
		inline const std::string& getEquation() const { return m_eqn; }
		inline const double getError() const { return m_rSqrTest; }

		inline const double getFittedValue(const double _xs) const { return m_function(_xs, m_paramsEqn.data()); }

		void setEquation(const std::vector<float>&, const std::vector<float>&, const int);
		void setEquation(const std::vector<double>&, const std::vector<double>&, const int);

		double getFitValues(const double);
		double getFitValuesFunctionN(const double, const unsigned int);

	protected:
		std::vector<double> m_values, m_fitsValues, m_paramsEqn, m_ts;
		double m_rSqrTest;
		int m_nbParamEqn, m_typeEqn;
		std::string m_eqn;

		double(*m_function)(double, const double*);
	};
}
#endif // EquationFit_h__

