/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DetectionSetCommandContext.hpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*/

#ifndef DetectionSetCommandContext_h__
#define DetectionSetCommandContext_h__

#include <General/Command.hpp>

namespace poca::core {
	class EquationFit;

	struct CleanEquationsContext {
		poca::core::EquationFit* blinks = nullptr;
		poca::core::EquationFit* tons = nullptr;
		poca::core::EquationFit* toffs = nullptr;
		uint32_t nbEmissionBursts = 0;
		uint32_t nbOriginalLocs = 0;
		uint32_t nbSupressedLocs = 0;
		uint32_t nbAddedLocs = 0;
		uint32_t nbUncorrectedLocs = 0;
		uint32_t darkTime = 0;
	};
}

#endif
