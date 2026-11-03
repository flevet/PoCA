/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#include "MainWindow.hpp"
#include "MdiChild.hpp"
#include <General/Engine.hpp>
#include <General/ProgressiveObjectLoad.hpp>
#include <DesignPatterns/MacroRecorderSingleton.hpp>
#include <Objects/MyMultipleObject.hpp>
#include <OpenGL/Camera.hpp>
#include <QtCore/QPointer>
#include <QtCore/QTimer>
#include <QtGui/QFontMetrics>
#include <General/ProgressiveLoadProfile.hpp>
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
		QPointer<QTimer> refresh;
		bool panelsDirty{false}, completed{false};
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
	view->refresh=new QTimer(this);
	view->refresh->setSingleShot(true); view->refresh->setInterval(100);
	connect(view->refresh,&QTimer::timeout,this,[this,view] {
		if (!view->camera) return;
		auto* root=view->camera->getObject();
		// Reuse the camera gate shared with image streaming; camera/input updates stay immediate.
		view->camera->requestImageStreamRedraw();
		if (view->panelsDirty && m_currentMdi && m_currentMdi->getWidget()->getObject() == root) {
			refreshObjectsPanel(); refreshPropertiesPanel();
			view->panelsDirty=false;
		}
	});
	auto* panel=new QWidget(this);
	auto* row=new QHBoxLayout(panel); row->setContentsMargins(4,0,4,0);
	auto* label=new QLabel(tr("OME-Zarr metadata: reading root manifest"),panel);
	label->setFixedWidth(440);
	const auto showStatus=[label](const QString& text) {
		label->setToolTip(text);
		label->setText(label->fontMetrics().elidedText(text,Qt::ElideRight,label->width()));
	};
	auto* progress=new QProgressBar(panel); progress->setRange(0,0); progress->setFixedWidth(150);
	auto* cancel=new QPushButton(tr("Cancel loading"),panel);
	row->addWidget(label); row->addWidget(progress); row->addWidget(cancel);
	statusBar()->addPermanentWidget(panel);
	connect(cancel,&QPushButton::clicked,load,[load,cancel] { cancel->setEnabled(false); load->cancel(); });
	connect(load,&poca::core::ProgressiveObjectLoad::objectReady,this,[this,load,view,opening,engine](poca::core::MyObjectInterface* root) {
		try {
			auto* camera=dynamic_cast<poca::opengl::Camera*>(createWindows(root));
			if (!camera) throw std::runtime_error("Could not create the progressive collection camera");
			view->camera=camera; view->window=m_currentMdi;
			engine->addCameraToObject(root,camera);
			connect(camera,&QObject::destroyed,load,[load] { load->detach(); });
			// Existing render commands lazily create only visible/selected resources on paint.
			connect(camera,&poca::opengl::Camera::frameSwapped,camera,[view] {
				if (view->camera) {
					if (auto profile=poca::core::ProgressiveLoadProfile::forObject(view->camera->getObject())) profile->frameSubmitted();
				}
				if (view->completed && view->camera) {
					if (auto profile=poca::core::ProgressiveLoadProfile::forObject(view->camera->getObject()); profile && profile->hasCompletedScene()) profile->report("first_completed_collection_frame");
				}
			});
			camera->requestImageStreamRedraw();
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
	connect(load,&poca::core::ProgressiveObjectLoad::childAttached,this,[view](poca::core::MyObjectInterface*,poca::core::MyObjectInterface*) {
		if (!view->camera) return;
		view->panelsDirty=true;
		if (!view->refresh->isActive()) view->refresh->start();
	});
	connect(load,&poca::core::ProgressiveObjectLoad::collectionCompleted,this,[this,view](poca::core::MyObjectInterface* root) {
		if (!view->camera) return;
		view->completed=true; view->panelsDirty=true;
		if (!view->refresh->isActive()) view->refresh->start();
		if (m_currentMdi && m_currentMdi->getWidget()->getObject() == root) {
			root->notify("LoadObjCharacteristicsAllWidgets"); refreshLoadingControls();
		}
		if (view->window) view->window->setWindowTitle(QString::fromStdString(root->getName()));
	});
	connect(load,&poca::core::ProgressiveObjectLoad::preflightProgressChanged,panel,
		[showStatus,progress](qulonglong completed,qulonglong total,const QString& stage) {
			showStatus(total ? tr("OME-Zarr metadata: %1 / %2 scanned - %3").arg(completed).arg(total).arg(stage)
				: tr("OME-Zarr metadata: %1").arg(stage));
			if (total) { progress->setRange(0,int(total)); progress->setValue(int(completed)); }
			else progress->setRange(0,0);
			progress->setFormat(total ? tr("%v / %m scanned") : QString{});
		},Qt::QueuedConnection);
	connect(load,&poca::core::ProgressiveObjectLoad::progressChanged,panel,
		[showStatus,progress,view](qulonglong ready,qulonglong total,qulonglong failed,const QString& state) {
			const auto pending=total >= ready+failed ? total-ready-failed : 0;
			showStatus(tr("OME-Zarr: %1 / %2 ready; %3 pending; %4 failed - %5").arg(ready).arg(total).arg(pending).arg(failed).arg(state));
			progress->setFormat(tr("%v / %m ready"));
			if (total) { progress->setRange(0,int(total)); progress->setValue(int(ready)); }
			if (view->window && view->camera) {
				auto* root=view->camera->getObject();
				const auto* multiple=dynamic_cast<MyMultipleObject*>(root);
				view->window->setWindowTitle(QString::fromStdString(root->getName())+
					(multiple && multiple->isLoadComplete() ? QString{} : tr(" [Incomplete: %1/%2 - %3]").arg(ready).arg(total).arg(state)));
			}
		},Qt::QueuedConnection);
	connect(load,&poca::core::ProgressiveObjectLoad::finished,this,[this,load,panel,view](const QString& error) {
		statusBar()->removeWidget(panel); panel->deleteLater();
		view->refresh->stop(); view->refresh->deleteLater();
		if (view->camera) view->camera->requestImageStreamRedraw();
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
