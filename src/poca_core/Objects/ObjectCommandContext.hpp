/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      ObjectCommandContext.hpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*/

#ifndef ObjectCommandContext_h__
#define ObjectCommandContext_h__

#include <General/Command.hpp>

namespace poca::core {
	class MyObjectInterface;

	struct CreatedObjectContext {
		poca::core::MyObjectInterface* object = nullptr;
	};

	struct TargetObjectContext {
		poca::core::MyObjectInterface* object = nullptr;
	};
}

#endif
