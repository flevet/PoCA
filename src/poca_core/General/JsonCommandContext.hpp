/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      JsonCommandContext.hpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*/

#ifndef JsonCommandContext_h__
#define JsonCommandContext_h__

#include <General/Command.hpp>

namespace poca::core {
	struct JsonFileContext {
		nlohmann::json* file = nullptr;
	};
}

#endif
