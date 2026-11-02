/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#include "CollectionRenderPlan.hpp"
#include "BasicComponentList.hpp"
#include "PerformanceProfiler.hpp"
#include <Interfaces/MyObjectInterface.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
namespace poca::core {
	size_t CollectionRenderPlan::Hash::operator()(const Key& key) const {
		return std::hash<void*>{}(key.child) ^ (std::hash<void*>{}(key.component) << 1);
	}
	BoundingBox CollectionRenderPlan::transformedBounds(const BoundingBox& box, const glm::mat4& matrix) {
		auto result = BoundingBox::initBBox();
		for (int corner = 0; corner < 8; ++corner) {
			const auto p = matrix * glm::vec4(box[corner&1 ? 3:0],box[corner&2 ? 4:1],box[corner&4 ? 5:2],1.f);
			result.addPointBBox(p.x,p.y,p.z);
		}
		return result;
	}
	glm::vec4 CollectionRenderPlan::projectedBounds(const BoundingBox& box, const glm::mat4& mvp) {
		glm::vec4 result((std::numeric_limits<float>::max)(),(std::numeric_limits<float>::max)(),
			-(std::numeric_limits<float>::max)(),-(std::numeric_limits<float>::max)());
		for (int corner = 0; corner < 8; ++corner) {
			const auto p = mvp * glm::vec4(box[corner&1 ? 3:0],box[corner&2 ? 4:1],box[corner&4 ? 5:2],1.f);
			if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.w) || p.w <= 1e-6f)
				return glm::vec4(-1.f,-1.f,1.f,1.f); // Conservative near-plane/unknown coverage.
			result.x=std::min(result.x,p.x/p.w); result.y=std::min(result.y,p.y/p.w);
			result.z=std::max(result.z,p.x/p.w); result.w=std::max(result.w,p.y/p.w);
		}
		return result;
	}
	void CollectionRenderPlan::build(const std::vector<MyObjectInterface*>& children, const glm::mat4& objectModel,
		const glm::mat4& parentRenderModel, const glm::mat4& viewProjection, const glm::mat4& view, bool collectItems) {
		PerformanceProfiler::RenderTimer timer(PerformanceProfiler::RenderMetric::CollectionPreparationUs);
		m_children.clear(); m_components.clear(); m_items.clear(); m_lookup.clear(); m_handledElements.clear();
		for (size_t childIndex=0; childIndex<children.size(); ++childIndex) {
			auto* child=children[childIndex]; if (!child) continue;
			auto localBounds=BoundingBox::initBBox(); bool haveBounds=false;
			const auto localModel=child->getModelMatrix(), effectiveModel=parentRenderModel*localModel;
			for (auto* component:child->getComponents()) {
				if (!component) continue;
				auto* list=dynamic_cast<BasicComponentList*>(component);
				if (list && !list->nbComponents()) continue;
				const auto box=component->boundingBox();
				for (int corner=0; corner<8; ++corner) {
					const float x=box[corner&1 ? 3:0], y=box[corner&2 ? 4:1], z=box[corner&4 ? 5:2];
					if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
						localBounds.addPointBBox(x,y,z); haveBounds=true;
					}
				}
				if (!collectItems) continue; // Legacy mode needs only its existing child depth order.
				const size_t index=m_components.size();
				m_components.push_back({child,component,component->getName(),m_items.size(),0,false});
				m_lookup.emplace(Key{child,component},index);
				const size_t count=list ? list->nbComponents():1;
				for (size_t sub=0; sub<count; ++sub) {
					auto* element=list ? static_cast<BasicComponentInterface*>(list->getComponent(uint32_t(sub))):component;
					if (!element) continue;
					Item item; item.child=child; item.component=component; item.element=element;
					item.componentIndex=index; item.subIndex=sub; item.localModel=localModel; item.effectiveModel=effectiveModel;
					item.localBounds=element->boundingBox();
					item.parentBounds=transformedBounds(item.localBounds,localModel);
					item.worldBounds=transformedBounds(item.localBounds,objectModel*localModel);
					item.screenBounds=projectedBounds(item.localBounds,viewProjection*effectiveModel);
					item.selected=element->isSelected(); m_items.push_back(item);
				}
				m_components[index].count=m_items.size()-m_components[index].first;
			}
			if (!haveBounds) localBounds=child->boundingBox();
			const auto center=glm::vec4((localBounds[0]+localBounds[3])*.5f,(localBounds[1]+localBounds[4])*.5f,(localBounds[2]+localBounds[5])*.5f,1.f);
			const float depth=(view*effectiveModel*center).z;
			m_children.push_back({child,childIndex,depth,std::isfinite(depth)});
		}
		PerformanceProfiler::renderCount(PerformanceProfiler::RenderMetric::CollectionCandidateItems,m_items.size());
		for (const auto& item:m_items) if (item.selected) PerformanceProfiler::renderCount(PerformanceProfiler::RenderMetric::CollectionSelectedItems);
		std::stable_sort(m_children.begin(),m_children.end(),[](const Child& a,const Child& b) {
			if (a.validDepth!=b.validDepth) return a.validDepth;
			if (a.validDepth && a.depth!=b.depth) return a.depth<b.depth;
			return a.originalIndex<b.originalIndex;
		});
	}
	bool CollectionRenderPlan::handled(MyObjectInterface* child, BasicComponentInterface* component) const {
		const auto found=m_lookup.find({child,component});
		return found!=m_lookup.end() && m_components[found->second].handled;
	}
	void CollectionRenderPlan::markHandled(size_t index) {
		auto& item=m_items.at(index);
		if (item.disposition==Disposition::Pending) throw std::logic_error("Cannot handle pending render item");
		item.handled=true; m_handledElements.insert(item.element);
	}
	void CollectionRenderPlan::claimCompleteComponents() {
		for (auto& component:m_components) {
			bool complete=true;
			for (size_t i=component.first;i<component.first+component.count;++i) complete=complete && m_items[i].handled;
			component.handled=complete;
		}
	}
	void CollectionRenderPlan::claim(size_t index) {
		auto& component=m_components.at(index);
		for (size_t i=component.first; i<component.first+component.count; ++i)
			if (!m_items[i].handled) throw std::logic_error("Cannot claim incompletely rendered component");
		component.handled=true;
	}
}
