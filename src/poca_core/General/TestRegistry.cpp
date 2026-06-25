#include <General/TestRegistry.hpp>

#include <algorithm>

namespace poca::core {
	void TestRegistry::add(const TestActionDescriptor& _descriptor)
	{
		auto it = std::find_if(m_descriptors.begin(), m_descriptors.end(), [&_descriptor](const TestActionDescriptor& _current) {
			return _current.id == _descriptor.id;
		});
		if (it != m_descriptors.end()) {
			*it = _descriptor;
			return;
		}
		m_descriptors.push_back(_descriptor);
	}

	void TestRegistry::clear()
	{
		m_descriptors.clear();
	}

	const std::vector<TestActionDescriptor>& TestRegistry::descriptors() const
	{
		return m_descriptors;
	}

	const TestActionDescriptor* TestRegistry::descriptor(const std::string& _id) const
	{
		auto it = std::find_if(m_descriptors.begin(), m_descriptors.end(), [&_id](const TestActionDescriptor& _current) {
			return _current.id == _id;
		});
		return it == m_descriptors.end() ? nullptr : &(*it);
	}
}
