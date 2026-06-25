#ifndef TestRegistry_hpp__
#define TestRegistry_hpp__

#include <functional>
#include <string>
#include <vector>

namespace poca::core {
	class MyObjectInterface;

	struct TestActionDescriptor {
		using ObjectFactory = std::function<MyObjectInterface*()>;

		std::string id;
		std::string menuPath;
		std::string label;
		std::string statusTip;
		bool requiresCuda{ false };
		ObjectFactory createObject;
	};

	class TestRegistry {
	public:
		void add(const TestActionDescriptor&);
		void clear();

		const std::vector<TestActionDescriptor>& descriptors() const;
		const TestActionDescriptor* descriptor(const std::string&) const;

	private:
		std::vector<TestActionDescriptor> m_descriptors;
	};
}

#endif
