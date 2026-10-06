/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#include "Engine.hpp"
#include "PluginList.hpp"
#include "../Interfaces/BasicComponentInterface.hpp"
#include "../Objects/MyObject.hpp"
#include "../Objects/MyMultipleObject.hpp"
#include "../Objects/MyObjectDisplayCommand.hpp"
#include <set>
#include <stdexcept>

namespace poca::core {
	std::unique_ptr<MyObjectInterface> Engine::assembleObject(const std::string& _dir, const std::string& _name,
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
		return object;
	}

	std::unique_ptr<MyObjectInterface> Engine::assembleMultipleObject(std::vector<std::unique_ptr<MyObjectInterface>> _children,
		std::vector<std::unique_ptr<BasicComponentInterface>> _components)
	{
		if (_children.empty()) throw std::invalid_argument("Multiple dataset has no children");
		std::vector<MyObjectInterface*> children;
		for (const auto& child : _children) {
			if (!child || dynamic_cast<MyMultipleObject*>(child.get()))
				throw std::invalid_argument("Multiple dataset requires owned MyObject children");
			children.push_back(child.get());
		}
		// Constructor success is the ownership handoff; failure retains all owners.
		// No grid recomputation: the loader restores the persisted arrangement.
		auto object = std::make_unique<MyMultipleObject>(children,false,false);
		for (auto& child : _children) child.release();
		std::set<std::string> names;
		for (const auto& component : _components)
			if (!component || !names.insert(component->getName()).second)
				throw std::invalid_argument("Multiple dataset has null or duplicate aggregate components");
		for (auto it = _components.rbegin(); it != _components.rend(); ++it) {
			addComponentToObject(object.get(),it->get());
			it->release();
		}
		auto display = std::make_unique<MyObjectDisplayCommand>(object.get());
		object->addCommand(display.get()); display.release();
		m_plugins->addCommands(object.get());
		return object;
	}

	MyObjectInterface* Engine::registerObject(std::unique_ptr<MyObjectInterface> _object)
	{
		if (!_object) throw std::invalid_argument("Missing assembled dataset");
		// Children stay owned by MyMultipleObject and are never registered separately.
		m_datasets.push_back(std::make_tuple(_object.get(),nullptr));
		m_currentDataset = &m_datasets.back();
		return _object.release();
	}

	MyObjectInterface* Engine::createObject(const std::string& _dir, const std::string& _name,
		std::vector<std::unique_ptr<BasicComponentInterface>> _components)
	{
		return registerObject(assembleObject(_dir,_name,std::move(_components)));
	}
}
