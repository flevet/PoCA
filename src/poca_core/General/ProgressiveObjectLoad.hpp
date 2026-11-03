/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#pragma once
#include <QtCore/QObject>
#include <General/json.hpp>
namespace poca::core {
	class MyObjectInterface;
	// Optional GUI lifecycle. All signals and methods run on the application thread.
	// A ready root is Engine-owned; detach() revokes every borrowed object pointer.
	class ProgressiveObjectLoad : public QObject {
		Q_OBJECT
	public:
		using QObject::QObject;
		virtual void start() = 0;
		virtual void cancel() = 0;
		virtual void detach() = 0;
		virtual nlohmann::json openingParameters() const = 0;
	signals:
		void objectReady(poca::core::MyObjectInterface*);
		void childAttached(poca::core::MyObjectInterface*, poca::core::MyObjectInterface*);
		void collectionCompleted(poca::core::MyObjectInterface*);
		void progressChanged(qulonglong ready, qulonglong total, qulonglong failed, const QString& state);
		void finished(const QString& error);
	};
}
