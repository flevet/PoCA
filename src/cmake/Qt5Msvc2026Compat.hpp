#ifndef POCA_QT5_MSVC2026_COMPAT_HPP
#define POCA_QT5_MSVC2026_COMPAT_HPP

// Qt 5.15.2 defines QT_MAKE_CHECKED_ARRAY_ITERATOR as
// stdext::make_checked_array_iterator() for MSVC. MSVC 14.51 (VS 2026)
// removed that long-deprecated stdext helper. Qt 6 already avoids it on
// recent MSVC toolsets. Include qglobal first so Qt defines its macro once,
// then replace only that macro with the no-op form used on other platforms.
#if defined(_MSC_VER) && _MSC_VER >= 1950
#include <QtCore/qglobal.h>
#ifdef QT_MAKE_CHECKED_ARRAY_ITERATOR
#undef QT_MAKE_CHECKED_ARRAY_ITERATOR
#endif
#define QT_MAKE_CHECKED_ARRAY_ITERATOR(x, y) (x)
#endif

#endif // POCA_QT5_MSVC2026_COMPAT_HPP
