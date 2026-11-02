/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#pragma once
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <glm/glm.hpp>
#include <General/Vec6.hpp>
namespace poca::core {
	class MyObjectInterface;
	class BasicComponentInterface;
	// Borrowed references live only for one synchronous display dispatch.
	// No GL objects, image/geometry plugin types or serialized identities.
	class CollectionRenderPlan {
	public:
		enum class Disposition { Pending, Disabled, Culled, Deferred, Batched, Legacy };
		struct Item {
			MyObjectInterface* child{};
			BasicComponentInterface* component{};
			BasicComponentInterface* element{};
			size_t componentIndex{}, subIndex{};
			glm::mat4 localModel{1.f}, effectiveModel{1.f};
			BoundingBox localBounds, parentBounds, worldBounds;
			glm::vec4 screenBounds{-1.f,-1.f,1.f,1.f};
			bool selected{}, visible{}, visibilityKnown{}, handled{};
			uint64_t resourceIdentity{}, resourceVersion{};
			glm::uvec3 activeDimensions{0};
			int level{-1};
			bool forceLevelZero{};
			Disposition disposition{Disposition::Pending};
		};
		struct Component {
			MyObjectInterface* child{};
			BasicComponentInterface* source{};
			std::string family;
			size_t first{}, count{};
			bool handled{};
		};
		struct Child {
			MyObjectInterface* source{};
			size_t originalIndex{};
			float depth{};
			bool validDepth{};
		};
		void build(const std::vector<MyObjectInterface*>&, const glm::mat4& objectModel,
			const glm::mat4& parentRenderModel, const glm::mat4& viewProjection, const glm::mat4& view, bool collectItems = true);
		const std::vector<Child>& children() const { return m_children; }
		const std::vector<Component>& components() const { return m_components; }
		std::vector<Item>& items() { return m_items; }
		const std::vector<Item>& items() const { return m_items; }
		bool handled(MyObjectInterface*, BasicComponentInterface*) const;
		void claim(size_t componentIndex);
		void markHandled(size_t itemIndex);
		bool elementHandled(BasicComponentInterface* element) const { return m_handledElements.count(element)!=0; }
		void claimCompleteComponents();
		static BoundingBox transformedBounds(const BoundingBox&, const glm::mat4&);
		static glm::vec4 projectedBounds(const BoundingBox&, const glm::mat4&);
	private:
		struct Key {
			MyObjectInterface* child; BasicComponentInterface* component;
			bool operator==(const Key& other) const { return child == other.child && component == other.component; }
		};
		struct Hash { size_t operator()(const Key&) const; };
		std::vector<Child> m_children;
		std::vector<Component> m_components;
		std::vector<Item> m_items;
		std::unordered_map<Key,size_t,Hash> m_lookup;
		std::unordered_set<BasicComponentInterface*> m_handledElements;
	};
}
