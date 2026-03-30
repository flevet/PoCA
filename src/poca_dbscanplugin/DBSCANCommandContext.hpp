/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DBSCANCommandContext.hpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*/

#ifndef DBSCANCommandContext_h__
#define DBSCANCommandContext_h__

#include <General/Command.hpp>

class DBSCANCommand;

struct DBSCANCommandContext {
	DBSCANCommand* command = nullptr;
};

#endif
