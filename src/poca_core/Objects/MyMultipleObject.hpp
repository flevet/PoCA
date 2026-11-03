/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MyMultipleObject.hpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*
* Homepage:  https://github.com/flevet/PoCA
*
* PoCA is a free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 3 of the License, or (at your option) any later version.
*
* The algorithms that underlie PoCA have required considerable
* development. They are described in the original SR-Tesseler paper,
* doi:10.1038/nmeth.3579. If you use PoCA as part of work (visualization, 
* manipulation, quantification) towards a scientific publication, please include 
* a citation to the original paper.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with this program; if not, write to the Free Software Foundation,
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

#ifndef MyMultipleObject_h__
#define MyMultipleObject_h__

#include "DesignPatterns/Subject.hpp"
#include "General/CommandableObject.hpp"
#include "Interfaces/MyObjectInterface.hpp"
#include "General/BasicComponent.hpp"
#include "OpenGL/Camera.hpp"

#include <map>
#include <memory>
#include <optional>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Objects/MyObject.hpp>
#include <General/ProgressiveLoadProfile.hpp>

class MyMultipleObject : public poca::core::MyObject {
public:
	struct HierarchyNode {
		std::string label;
		std::string levelName;
		int parentIndex{ -1 };
		std::vector<size_t> children;
		std::vector<size_t> objectIndices;
		std::map<std::string, std::string> metadata;
	};

	// Persistence assembly may retain supplied transforms without grid placement.
	MyMultipleObject(std::vector<poca::core::MyObjectInterface*>, const bool = false, const bool = true);
	~MyMultipleObject();

	float getX() const;
	float getY() const;
	float getZ() const;
	float getWidth() const;
	float getHeight() const;
	float getThick() const;

	void setWidth(const float);
	void setHeight(const float);
	void setThick(const float);

	void executeCommand(poca::core::CommandInfo*);
	void executeCommand(poca::core::CommandInfo*, const poca::core::CommandExecutionContext&);
	void executeCommand(poca::core::CommandInfo*, const poca::core::CommandExecutionContext&, poca::core::CommandExecutionResult&);
	poca::core::CommandInfo createCommand(const std::string&, const nlohmann::json&) override;

	const poca::core::BoundingBox boundingBox() const;

	const size_t dimension() const;

	virtual void executeCommandOnSpecificComponent(const std::string&, poca::core::CommandInfo*);
	virtual void executeCommandOnSpecificComponent(const std::string&, poca::core::CommandInfo*, const poca::core::CommandExecutionContext&);
	virtual void executeCommandOnSpecificComponent(const std::string&, poca::core::CommandInfo*, const poca::core::CommandExecutionContext&, poca::core::CommandExecutionResult&);
	virtual void executeGlobalCommand(poca::core::CommandInfo*);
	virtual void executeGlobalCommand(poca::core::CommandInfo*, const poca::core::CommandExecutionContext&);
	virtual void executeGlobalCommand(poca::core::CommandInfo*, const poca::core::CommandExecutionContext&, poca::core::CommandExecutionResult&);

	void resetModelMatrices(const bool = true);
	void recomputeGrid();

	const size_t nbColors() const { return m_colors.size(); }
	poca::core::MyObjectInterface* getObject(const size_t _index) { return m_colors[_index]; }
	const poca::core::MyObjectInterface* getObject(const size_t _index) const { return m_colors[_index]; }
	poca::core::MyObjectInterface* currentObject() { return m_colors[m_currentColor]; }
	size_t currentObjectID() const { return m_currentColor; }
	void setCurrentObject(const size_t _idx) {
		if (_idx >= m_colors.size()) throw std::out_of_range("Current child is not ready");
		if (m_loadState == LoadState::Loading && _idx != m_currentColor) m_loadingSelectionChanged=true;
		m_currentColor = _idx;
	}

	inline void setGridBBoxes(const std::vector <poca::core::BoundingBox>& _bboxes) { m_gridBBoxes = _bboxes; }
	inline const std::vector <poca::core::BoundingBox>& getGridBBoxes() const { return m_gridBBoxes; }
	inline std::vector <poca::core::BoundingBox>& getGridBBoxes() { return m_gridBBoxes; }

	bool hasHierarchy() const { return !m_hierarchy.empty(); }
	void clearHierarchy();
	size_t addHierarchyNode(const std::string&, const std::string& = "", int = -1);
	void attachObjectToHierarchyNode(const size_t, const size_t);
	void setHierarchyNodeMetadata(const size_t, const std::map<std::string, std::string>&);
	const std::vector<HierarchyNode>& hierarchy() const { return m_hierarchy; }
	std::vector<size_t> collectObjectIndicesForHierarchyNode(const size_t, const bool = true) const;
	void setSelectedObjectIndices(const std::vector<size_t>&);
	const std::vector<size_t>& selectedObjectIndices() const { return m_selectedObjectIndices; }
	bool hasSelectedObjectIndices() const { return !m_selectedObjectIndices.empty(); }
	void setBatchComponentRendering(const bool _batch) { m_batchComponentRendering = _batch; }
	bool batchComponentRendering() const { return m_batchComponentRendering && isLoadComplete(); }
	// Only the existing image provider tolerates an append-only ready prefix.
	bool imageComponentRendering() const { return m_batchComponentRendering; }
	void setLoadingProfile(std::shared_ptr<poca::core::ProgressiveLoadProfile> profile) { m_loadingProfile=std::move(profile); }
	std::shared_ptr<poca::core::ProgressiveLoadProfile> loadingProfile() const { return m_loadingProfile; }
	// Move out all owned children, leaving a safely destructible empty container.
	std::vector<std::unique_ptr<poca::core::MyObjectInterface>> releaseObjects();

	enum class LoadState { Complete, Loading, Cancelled, Failed };
	struct LoadingCommandRejection { std::string message; };
	void beginProgressiveLoad(const std::vector<std::string>&, const std::optional<poca::core::BoundingBox>&);
	void appendObject(size_t, std::unique_ptr<poca::core::MyObjectInterface>&&);
	void finishProgressiveLoad();
	void stopProgressiveLoad(LoadState, const std::string&, size_t);
	bool isLoadComplete() const { return m_loadState == LoadState::Complete; }
	LoadState loadState() const { return m_loadState; }
	size_t failedChildren() const { return m_failedChildren; }
	bool loadingSelectionChanged() const { return m_loadingSelectionChanged; }
	size_t expectedChildren() const { return m_expectedNames.empty() ? m_colors.size() : m_expectedNames.size(); }
	const std::string& loadDiagnostic() const { return m_loadDiagnostic; }
	uint64_t childrenRevision() const { return m_childrenRevision; }
	const std::optional<poca::core::BoundingBox>& pendingLoadBounds() const { return m_pendingBounds; }
	void requireCompleteCollection() const;
	void setLoadingCancellation(std::function<void()> callback) { m_loadingCancellation=std::move(callback); }

protected:
	static void initializeMeshDecorations(poca::core::BasicComponentInterface*);
	std::string incompleteLoadMessage() const;
	bool checkLoadingCommand(const poca::core::CommandInfo*, poca::core::CommandExecutionResult&) const;
	LoadState m_loadState{ LoadState::Complete };
	std::vector<std::string> m_expectedNames;
	std::optional<poca::core::BoundingBox> m_pendingBounds;
	std::string m_loadDiagnostic;
	uint64_t m_childrenRevision{ 0 };
	size_t m_failedChildren{ 0 };
	bool m_loadingSelectionChanged{ false };
	std::function<void()> m_loadingCancellation;
	std::shared_ptr<poca::core::ProgressiveLoadProfile> m_loadingProfile;
	std::vector <poca::core::MyObjectInterface*> m_colors;
	size_t m_currentColor;

	std::vector <poca::core::BoundingBox> m_gridBBoxes;
	bool m_gridSelected{ true };
	bool m_batchComponentRendering{ false };
	std::vector <HierarchyNode> m_hierarchy;
	std::vector<size_t> m_selectedObjectIndices;
};

#endif // SMLMObject_h__
