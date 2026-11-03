/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#include "MainWindow.hpp"
#include "MdiChild.hpp"
#include <General/Engine.hpp>
#include <General/ProgressiveObjectLoad.hpp>
#include <DesignPatterns/MacroRecorderSingleton.hpp>
#include <Objects/MyMultipleObject.hpp>
#include <OpenGL/Camera.hpp>
#include <QtCore/QPointer>
#include <QtCore/QElapsedTimer>
#include <QtWidgets/QAction>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <memory>
#include <iostream>

namespace {
	struct LoadingView {
		QPointer<poca::opengl::Camera> camera;
		QPointer<MdiChild> window;
	};
}
bool MainWindow::openProgressiveFile(const QString& filename, poca::core::CommandInfo* command)
{
	auto* engine=poca::core::Engine::instance();
	poca::core::ProgressiveObjectLoad* load=nullptr;
	try { load=engine->createProgressiveLoad(filename,this); }
	catch (const std::exception& error) {
		QMessageBox::warning(this,tr("OME-Zarr loading"),QString::fromUtf8(error.what()));
		command->setRecordable(false);
		return true;
	}
	if (!load) return false;
	const auto opening=*command;
	command->setRecordable(false);
	auto view=std::make_shared<LoadingView>();
	QElapsedTimer started; started.start();
	auto* panel=new QWidget(this);
	auto* row=new QHBoxLayout(panel); row->setContentsMargins(4,0,4,0);
	auto* label=new QLabel(tr("Loading OME-Zarr: reading metadata"),panel);
	auto* progress=new QProgressBar(panel); progress->setRange(0,0);
	auto* cancel=new QPushButton(tr("Cancel loading"),panel);
	row->addWidget(label); row->addWidget(progress); row->addWidget(cancel);
	statusBar()->addPermanentWidget(panel);
	connect(cancel,&QPushButton::clicked,load,[load,cancel] { cancel->setEnabled(false); load->cancel(); });
	connect(load,&poca::core::ProgressiveObjectLoad::objectReady,this,[this,load,view,opening,engine,started](poca::core::MyObjectInterface* root) {
		try {
			auto* camera=dynamic_cast<poca::opengl::Camera*>(createWindows(root));
			if (!camera) throw std::runtime_error("Could not create the progressive collection camera");
			view->camera=camera; view->window=m_currentMdi;
			engine->addCameraToObject(root,camera);
			connect(camera,&QObject::destroyed,load,[load] { load->detach(); });
			auto firstFrame=std::make_shared<bool>(true);
			connect(camera,&poca::opengl::Camera::frameSwapped,camera,[root,started,firstFrame] {
				if (!*firstFrame) return;
				*firstFrame=false;
				std::cout << "[PoCA][Zarr][Progressive] first submitted frame ms=" << started.elapsed()
					<< " ready children=" << root->nbColors() << std::endl;
			});
			// Constructor framing uses the complete available metadata envelope, once.
			camera->makeCurrent();
			poca::core::CommandInfo display(false,"createDisplay"); root->executeGlobalCommand(&display);
			camera->doneCurrent(); camera->update();
			auto recorded=opening;
			const auto parameters=load->openingParameters();
			for (const auto& item : parameters.items()) recorded.addParameter(item.key(),item.value());
			poca::core::MacroRecorderSingleton::instance()->addCommand("MainWindow",&recorded);
			refreshLoadingControls();
		}
		catch (const std::exception& error) {
			load->detach();
			if (view->window) view->window->close();
			else engine->removeObject(root);
			QMessageBox::warning(this,tr("Cannot create OME-Zarr viewer"),QString::fromUtf8(error.what()));
		}
	});
	connect(load,&poca::core::ProgressiveObjectLoad::childAttached,this,[this,view](poca::core::MyObjectInterface* root,poca::core::MyObjectInterface* child) {
		if (!view->camera) return;
		view->camera->makeCurrent();
		poca::core::CommandInfo display(false,"createDisplay"); child->executeGlobalCommand(&display);
		view->camera->doneCurrent(); view->camera->update();
		if (m_currentMdi && m_currentMdi->getWidget()->getObject() == root) {
			refreshObjectsPanel(); refreshPropertiesPanel();
		}
	});
	connect(load,&poca::core::ProgressiveObjectLoad::collectionCompleted,this,[this,view](poca::core::MyObjectInterface* root) {
		if (!view->camera) return;
		view->camera->makeCurrent();
		poca::core::CommandInfo display(false,"createDisplay");
		for (auto* component : root->getComponents()) component->executeCommand(&display);
		view->camera->doneCurrent(); view->camera->update();
		if (m_currentMdi && m_currentMdi->getWidget()->getObject() == root) {
			root->notify("LoadObjCharacteristicsAllWidgets");
			refreshObjectsPanel(); refreshPropertiesPanel(); refreshLoadingControls();
		}
		if (view->window) view->window->setWindowTitle(QString::fromStdString(root->getName()));
	});
	connect(load,&poca::core::ProgressiveObjectLoad::progressChanged,this,
		[label,progress,view](qulonglong ready,qulonglong total,qulonglong failed,const QString& state) {
			const auto pending=total >= ready+failed ? total-ready-failed : 0;
			label->setText(tr("OME-Zarr: %1 / %2 ready; %3 pending; %4 failed - %5").arg(ready).arg(total).arg(pending).arg(failed).arg(state));
			if (total) { progress->setRange(0,int(total)); progress->setValue(int(ready)); }
			if (view->window && view->camera) {
				auto* root=view->camera->getObject();
				const auto* multiple=dynamic_cast<MyMultipleObject*>(root);
				view->window->setWindowTitle(QString::fromStdString(root->getName())+
					(multiple && multiple->isLoadComplete() ? QString{} : tr(" [Incomplete: %1/%2 - %3]").arg(ready).arg(total).arg(state)));
			}
		});
	connect(load,&poca::core::ProgressiveObjectLoad::finished,this,[this,load,panel,view](const QString& error) {
		statusBar()->removeWidget(panel); panel->deleteLater();
		if (view->camera && m_currentMdi && m_currentMdi->getWidget()->getObject() == view->camera->getObject()) {
			refreshObjectsPanel(); refreshPropertiesPanel(); refreshLoadingControls();
		}
		if (!error.isEmpty()) QMessageBox::warning(this,tr("OME-Zarr loading stopped"),error);
		load->deleteLater();
	});
	load->start();
	return true;
}
void MainWindow::refreshLoadingControls()
{
	auto* object=m_currentMdi && m_currentMdi->getWidget() ? m_currentMdi->getWidget()->getObject() : nullptr;
	const auto* multiple=dynamic_cast<MyMultipleObject*>(object);
	const bool complete=!multiple || multiple->isLoadComplete();
	m_tabWidget->setEnabled(complete);
	for (auto* action : {m_plusAct,m_duplicateAct,m_colocAct,m_reloadDatasetAct})
		if (action) action->setEnabled(object && complete);
	refreshObjectControls();
}
