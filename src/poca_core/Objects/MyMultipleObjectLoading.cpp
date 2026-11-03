/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#include "MyMultipleObject.hpp"
#include <General/BasicComponentList.hpp>
#include <QtCore/QCoreApplication>
#include <QtCore/QThread>
#include <stdexcept>

namespace {
	void requireAttachmentThread() {
		auto* app = QCoreApplication::instance();
		if (app && QThread::currentThread() != app->thread())
			throw std::logic_error("MyMultipleObject attachment must run on the application thread");
	}

}
void MyMultipleObject::beginProgressiveLoad(const std::vector<std::string>& names, const std::optional<poca::core::BoundingBox>& bounds)
{
	requireAttachmentThread();
	if (!isLoadComplete() || names.size() <= m_colors.size() || m_colors.empty())
		throw std::logic_error("Progressive loading requires a nonempty ordered prefix of the complete collection");
	for (size_t i=0; i<m_colors.size(); ++i)
		if (m_colors[i]->getName() != names[i]) throw std::invalid_argument("Progressive prefix identity mismatch");
	m_colors.reserve(names.size());
	m_expectedNames = names; m_pendingBounds = bounds; m_loadState = LoadState::Loading;
	m_failedChildren=0; m_loadingSelectionChanged=false; m_loadDiagnostic.clear();
	if (names.size() > 1 && m_colors.size() == 1)
		for (auto* component : m_colors.front()->getComponents()) initializeMeshDecorations(component);
}
void MyMultipleObject::appendObject(size_t index, std::unique_ptr<poca::core::MyObjectInterface>&& child)
{
	requireAttachmentThread();
	if (m_loadState != LoadState::Loading || index != m_colors.size() || index >= m_expectedNames.size())
		throw std::logic_error("Progressive attachment must commit the next manifest index");
	if (!child || dynamic_cast<MyMultipleObject*>(child.get()) || child->getName() != m_expectedNames[index])
		throw std::invalid_argument("Progressive attachment child identity/type mismatch");
	for (auto* component : child->getComponents()) initializeMeshDecorations(component);
	m_colors.push_back(child.get()); child.release();
	++m_childrenRevision;
	// CollectionRenderPlan is frame-local. Per-child commands and GL resources stay untouched.
	// Progressive GUI owns the coalesced refresh; append never invalidates existing GPU data.
}
void MyMultipleObject::finishProgressiveLoad()
{
	requireAttachmentThread();
	if (m_loadState != LoadState::Loading || m_colors.size() != m_expectedNames.size())
		throw std::logic_error("Cannot complete a collection with pending manifest indices");
	m_loadState = LoadState::Complete; m_pendingBounds.reset(); m_loadDiagnostic.clear(); m_loadingCancellation={};
	++m_childrenRevision;
}
void MyMultipleObject::stopProgressiveLoad(LoadState state, const std::string& cause, size_t failedChildren)
{
	requireAttachmentThread();
	if (m_loadState != LoadState::Loading || (state != LoadState::Cancelled && state != LoadState::Failed))
		throw std::logic_error("Invalid progressive collection stop transition");
	if (failedChildren > expectedChildren()-m_colors.size())
		throw std::logic_error("Failed child count exceeds remaining manifest slots");
	m_loadState = state; m_failedChildren=failedChildren; m_loadDiagnostic = cause; m_pendingBounds.reset();
}
std::string MyMultipleObject::incompleteLoadMessage() const
{
	return "Collection is incomplete (" + std::to_string(m_colors.size()) + "/" +
		std::to_string(expectedChildren()) + " datasets ready). Analyses, layout changes and whole-dataset export require a completed load. " + m_loadDiagnostic;
}
void MyMultipleObject::requireCompleteCollection() const
{
	if (!isLoadComplete()) throw std::runtime_error(incompleteLoadMessage());
}
bool MyMultipleObject::checkLoadingCommand(const poca::core::CommandInfo* command, poca::core::CommandExecutionResult& result) const
{
	if (isLoadComplete() || !command) return true;
	const auto& name = command->nameCommand;
	for (const char* allowed : {"display","displayOverlay","createDisplay","freeGPU","updatePickingBuffer","pick",
		"requestLodUpdate","updateTransform","sortWRTCameraPosition","cameraMouseEvent","OpenGLContextInitialized",
		"displaySSAO","drawSSAO","getBoundingBox","boundingBox","updateDisplay","updateInfosObject"})
		if (name == allowed) return true;
	const auto message=incompleteLoadMessage();
	command->errorMessage(message);
	result.set(LoadingCommandRejection{message});
	return false;
}
