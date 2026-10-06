/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#include "Engine.hpp"
#include "PluginList.hpp"
#include "../Interfaces/BasicComponentInterface.hpp"
#include "../Objects/MyObject.hpp"
#include "../Objects/MyObjectDisplayCommand.hpp"
#include <set>
#include <stdexcept>

namespace poca::core {
	MyObjectInterface* Engine::createObject(const std::string& _dir, const std::string& _name,
		std::vector<std::unique_ptr<BasicComponentInterface>> _components)
	{
		if (_components.empty()) throw std::invalid_argument("Whole-dataset object has no components");
		std::set<std::string> names;
		for (const auto& component : _components)
			if (!component || !names.insert(component->getName()).second)
				throw std::invalid_argument("Whole-dataset object has null or duplicate components");
		auto object = std::make_unique<MyObject>();
		object->setDir(_dir); object->setName(_name);
		size_t dimension = 2;
		auto display = std::make_unique<MyObjectDisplayCommand>(object.get());
		object->addCommand(display.get()); display.release();
		// MyObject inserts at the front; retain the loader's component order.
		for (auto it = _components.rbegin(); it != _components.rend(); ++it) {
			dimension = (std::max)(dimension,size_t((*it)->dimension()));
			addComponentToObject(object.get(),it->get());
			it->release();
		}
		object->setDimension(dimension);
		m_plugins->addCommands(object.get());
		// No registration until validation, component ownership and commands succeed.
		m_datasets.push_back(std::make_tuple(object.get(),nullptr));
		m_currentDataset = &m_datasets.back();
		return object.release();
	}
}
