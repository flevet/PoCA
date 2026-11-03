/* Software: PoCA; Copyright: Florian Levet (2026); License: LGPL v3 */
#include <Windows.h>
#include <GL/glew.h>
#include "MdiChild.hpp"
#include <General/Engine.hpp>
#include <General/TestRegistry.hpp>
#include <General/Image.hpp>
#include <General/ImagesList.hpp>
#include <General/CollectionRenderPlan.hpp>
#include <Objects/MyObject.hpp>
#include <Objects/MyMultipleObject.hpp>
#include <QtWidgets/QApplication>
#include <QtCore/QElapsedTimer>
#include <QtCore/QSignalBlocker>
#include <QtCore/QThread>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <iostream>
#include <array>
#include <memory>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {
	void requireReset(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
	class LoadingWindowProbe : public MdiChild {
	public:
		explicit LoadingWindowProbe(poca::opengl::Camera* camera) : MdiChild(camera) {
			setAttribute(Qt::WA_DeleteOnClose,false); // The fixture owns this window throughout event processing.
		}
		void seedVolumeSlider() {
			const QSignalBlocker blocked(m_tSlider);
			m_tSlider->setValue(0); // Positive absolute Z used to emit a hidden slice command on reset.
		}

	};
	class ProgressiveResetFixture {
	public:
		void run(bool labels, bool batching, bool perspective, bool storedPyramid) {
			requireReset(qEnvironmentVariableIntValue("POCA_ZARR_SUPPRESS_RENDERING") != 1,
				"Disable POCA_ZARR_SUPPRESS_RENDERING for visibility regression tests");
			m_labels=labels; m_storedPyramid=storedPyramid;
			std::vector<std::string> names;
			for (size_t i=0;i<83;++i) { m_pending.push_back(child(i)); names.push_back(m_pending.back()->getName()); }
			auto bounds=poca::core::BoundingBox::initBBox();
			for (const auto& object:m_pending) {
				const auto box=poca::core::CollectionRenderPlan::transformedBounds(object->getBasicComponent("ImagesList")->boundingBox(),object->getModelMatrix());
				bounds.addPointBBox(box[0],box[1],box[2]); bounds.addPointBBox(box[3],box[4],box[5]);
			}
			std::vector<std::unique_ptr<poca::core::MyObjectInterface>> first;
			first.push_back(std::move(m_pending[0]));
			auto* engine=poca::core::Engine::instance();
			auto owned=engine->assembleMultipleObject(std::move(first),{},batching);
			m_root=dynamic_cast<MyMultipleObject*>(owned.get());
			configureObject(*m_root);
			m_root->beginProgressiveLoad(names,bounds);
			auto camera=std::make_unique<poca::opengl::Camera>(m_root,3);
			m_camera=camera.get();
			checkCameraState();
			if (perspective) m_camera->setProjectionType(poca::opengl::Camera::Perspective);
			m_window=std::make_unique<LoadingWindowProbe>(camera.get()); camera.release();
			engine->registerObject(std::move(owned)); // MdiChild now owns camera and removes registered root.
			m_area.addSubWindow(m_window.get()); m_area.resize(640,480);
			m_window->resize(620,450); m_area.show(); m_window->show();
			QElapsedTimer initialization; initialization.start();
			while (!m_camera->isValid() && initialization.elapsed()<10000) {
				QApplication::processEvents(); QThread::msleep(5);
			}
			requireReset(m_camera->isValid(),"Reset visibility fixture requires a working OpenGL camera");
			m_camera->setOpenGLContextInitialized(true);
			checkCameraState();
			requireReset(m_camera->getCurrentCrop()==bounds,"Initial camera lost preflight collection envelope");
			// No reset, rotation or selection change before this framebuffer assertion.
			requireImagePixels("fresh_initializeGL",1);
			if (storedPyramid) requireReset(m_nativeReads->load()>0,"Initial display did not read a stored pyramid level");
			checkStreamingCounters();
			for (size_t target:{size_t(1),size_t(10),size_t(40),size_t(83)}) {
				while (m_root->nbColors()<target) {
					const auto index=m_root->nbColors(); m_root->appendObject(index,std::move(m_pending[index]));
				}
				if (target==83) m_root->finishProgressiveLoad();
				checkpoint(target);
			}
			std::cout << "Reset Viewer pixel regression passed: " << (labels ? "LABEL":"RAW")
				<< " batching=" << batching << " perspective=" << perspective << " stored_pyramid=" << storedPyramid << " checkpoints=1,10,40,83" << std::endl;
		}
	private:
		static void configureObject(poca::core::MyObjectInterface& object) {
			for (auto* command:object.getCommands()) {
				for (const char* key:{"displayBoundingBox","displayGrid","displayTransformGizmo","fontDisplay","displayScaleBar","useSSAO"})
					if (command->hasParameter(key)) command->loadParameters(poca::core::CommandInfo(false,key,false));
				if (command->hasParameter("colorBakground"))
					command->loadParameters(poca::core::CommandInfo(false,"colorBakground",std::array<unsigned char,4>{0,0,0,255}));
			}
		}
		std::unique_ptr<poca::core::MyObjectInterface> child(size_t index) {
			auto image=std::make_unique<poca::core::Image<uint16_t>>(m_labels ? poca::core::LABEL:poca::core::RAW);
			if (m_storedPyramid) {
				image->setPixelReloadCallback([](std::vector<uint16_t>&) { throw std::runtime_error("Visibility fixture materialized scientific pixels"); });
				image->initializeStorageBacked(128,128,128,{uint16_t(0),uint16_t(1000)},true,0,1000);
				const auto read=[](const poca::core::Region3D&,void* target,size_t bytes) {
					std::fill_n(static_cast<uint16_t*>(target),bytes/sizeof(uint16_t),uint16_t(1000)); return true;
				};
				image->setRegionReaderCallback(read);
				image->setNativePyramidLevels({{128,128,128},{64,64,64},{8,8,8}});
				const auto reads=m_nativeReads;
				image->setNativePyramidRegionReaderCallback([read,reads](uint32_t level,const poca::core::Region3D& region,void* target,size_t bytes) {
					if (level>0) ++*reads;
					return read(region,target,bytes);
				});
			}
			else {
				image->pixels().assign(8*8*8,uint16_t(1000)); image->pixels().front()=0;
				image->finalizeImage(8,8,8);
			}
			image->setBoundingBox(0,0,12,8,8,20);
			auto images=std::make_unique<poca::core::ImagesList>(std::move(image),"bright fixture");
			// Only the first image contributes, so asynchronous refinement of other
			// children cannot produce a false-positive framebuffer difference.
			images->getImage(0)->setSelected(index==0);
			std::vector<std::unique_ptr<poca::core::BasicComponentInterface>> components;
			components.push_back(std::move(images));
			auto object=poca::core::Engine::instance()->assembleObject("","reset "+std::to_string(index),std::move(components));
			configureObject(*object);
			auto* concrete=dynamic_cast<poca::core::MyObject*>(object.get());
			concrete->setTranslationVector(glm::vec3(-12.f*float(index%10)-15.f,-12.f*float(index/10)+9.f,3.f));
			concrete->setRotationMatrix(glm::rotate(glm::mat4(1),.15f+.01f*float(index%5),glm::vec3(0,1,0)));
			concrete->updateModelMatrixFromTransform();
			return object;
		}
		poca::core::ImageInterface* firstImage() {
			return static_cast<poca::core::ImagesList*>(m_root->getObject(0)->getBasicComponent("ImagesList"))->getImage(0);
		}
		static size_t pixelDifference(const QImage& before,const QImage& after) {
			requireReset(!before.isNull() && before.size()==after.size(),"Invalid final framebuffer readback");
			size_t result=0;
			for (int y=0;y<after.height();++y) for (int x=0;x<after.width();++x)
				if ((before.pixel(x,y)&0x00ffffff)!=(after.pixel(x,y)&0x00ffffff)) ++result;
			return result;
		}
		void checkCameraState() {
			const auto& state=m_camera->getStateCamera();
			for (const auto& matrix:{state.m_matrix,state.m_matrixView})
				for (int c=0;c<4;++c) for (int r=0;r<4;++r)
					requireReset(std::isfinite(matrix[c][r]),"Camera view is nonfinite");
			for (const auto& vector:{state.m_eye,state.m_center,state.m_up,state.m_translationModel})
				for (int a=0;a<3;++a) requireReset(std::isfinite(vector[a]),"Camera vector is nonfinite");
			requireReset(glm::length(glm::cross(state.m_eye-state.m_center,state.m_up))>0.f,"Camera orientation is degenerate");
			requireReset(std::isfinite(m_camera->getDistanceOrtho()) && m_camera->getDistanceOrtho()>0.f &&
				std::isfinite(m_camera->getCameraDistance()) && m_camera->getCameraDistance()>0.f,"Camera fit is invalid");
		}
		void requireImagePixels(const char* phase,size_t ready) {
			QElapsedTimer timer; timer.start(); size_t pixels=0;
			while (timer.elapsed()<15000 && pixels<8) {
				QApplication::processEvents();
				const auto frame=m_camera->grabFramebuffer();
				requireReset(!frame.isNull(),"Initial/moving camera framebuffer is unavailable");
				pixels=0;
				for (int y=0;y<frame.height();++y) for (int x=0;x<frame.width();++x)
					if (frame.pixel(x,y)&0x00ffffff) ++pixels;
				if (pixels<8) QThread::msleep(5);
			}
			requireReset(pixels>=8,"No non-background image pixels before Reset Viewer");
			// Confirm those pixels came from the selected image, including at startup.
			const auto visible=m_camera->grabFramebuffer();
			auto* image=firstImage(); const auto selected=image->isSelected();
			image->setSelected(false); const auto hidden=m_camera->grabFramebuffer(); image->setSelected(selected);
			requireReset(pixelDifference(hidden,visible)>=8,"Framebuffer pixels did not come from the selected image");
			checkCameraState();
			for (int c=0;c<4;++c) for (int r=0;r<4;++r)
				requireReset(std::isfinite(m_camera->getProjectionMatrix()[c][r]),"Camera projection is nonfinite");
			std::cout << "Visibility phase=" << phase << " ready=" << ready << " non_background_pixels=" << pixels
				<< " elapsed_ms=" << timer.elapsed() << std::endl;
		}
		void checkStreamingCounters() {
			using Profiler=poca::core::PerformanceProfiler;
			using Metric=Profiler::RenderMetric;
			std::array<uint64_t,size_t(Metric::Count)> totals{};
			for (const auto& record:Profiler::renderInstance().renderRecords()) if (record.cameraId==m_camera->renderCameraId())
				for (size_t i=0;i<totals.size();++i) totals[i]+=record.values[i];
			requireReset(totals[size_t(Metric::VisibleImages)]>0 && totals[size_t(Metric::LodRequestAttempts)]>0 &&
				totals[size_t(Metric::Texture3DChunks)]>0,"Fresh camera did not start visible image streaming/uploads");
			std::cout << "Fresh camera visible_images=" << totals[size_t(Metric::VisibleImages)]
				<< " culled_images=" << totals[size_t(Metric::CulledImages)]
				<< " lod_request_attempts=" << totals[size_t(Metric::LodRequestAttempts)]
				<< " texture3d_upload_chunks=" << totals[size_t(Metric::Texture3DChunks)] << std::endl;
		}
		void checkpoint(size_t ready) {
			requireImagePixels("progressive",ready);
			auto* image=firstImage();
			const auto model=m_root->getObject(0)->getModelMatrix();
			const auto selected=image->isSelected();
			const auto frame=image->currentFrame();
			const auto box=m_root->boundingBox();
			auto& state=m_camera->getStateCamera();
			state.m_rotationSum=glm::angleAxis(.6f,glm::normalize(glm::vec3(1,2,3)));
			m_camera->updateCameraEyeUp(true,true);
			requireImagePixels("rotated",ready);
			state.m_translationModel+=glm::vec3(.5f,-.5f,.25f);
			requireImagePixels("translated",ready);
			m_window->seedVolumeSlider();
			m_camera->resetProjection(); m_window->resetViewer(); // MainWindow::resetViewer sequence.
			requireReset(!m_camera->isCropped() && image->currentFrame()==frame,"3D Reset Viewer silently selected a slice/crop");
			requireReset(m_root->boundingBox()==box && m_camera->getCurrentCrop()==box,"Reset changed/lost partial collection envelope");
			requireReset(bool(m_root->pendingLoadBounds())==(ready<83),"Partial envelope lifetime changed at completion");
			image->setSelected(false);
			QApplication::processEvents();
			const auto without=m_camera->grabFramebuffer();
			image->setSelected(selected);
			QElapsedTimer recovery; recovery.start();
			size_t changed=0;
			while (recovery.elapsed()<15000 && changed<8) {
				QApplication::processEvents();
				requireReset(m_window->isVisible(),"Reset visibility fixture window closed before acceptance");
				changed=pixelDifference(without,m_camera->grabFramebuffer());
				if (changed<8) QThread::msleep(5);
			}
			requireReset(changed>=8,"No actual image contribution after Reset Viewer/texture recovery");
			requireReset(image->isSelected()==selected && image->currentFrame()==frame &&
				m_root->getObject(0)->getModelMatrix()==model,"Reset/recovery altered scientific view selection or child transform");
			std::cout << "Reset ready=" << ready << " final_image_rgb_pixels=" << changed << " recovery_ms=" << recovery.elapsed() << std::endl;
		}
		bool m_labels{}, m_storedPyramid{};
		std::shared_ptr<std::atomic<size_t>> m_nativeReads{std::make_shared<std::atomic<size_t>>(0)};
		MyMultipleObject* m_root{};
		poca::opengl::Camera* m_camera{};
		std::vector<std::unique_ptr<poca::core::MyObjectInterface>> m_pending;
		QMdiArea m_area;
		std::unique_ptr<LoadingWindowProbe> m_window;
	};
	poca::core::MyObjectInterface* runProgressiveResetPixels() {
		auto& profiler=poca::core::PerformanceProfiler::renderInstance();
		auto& engine=*poca::core::Engine::instance();
		struct RestoreDiagnostics {
			poca::core::Engine& engine; poca::core::PerformanceProfiler& profiler;
			bool verbose, metrics; std::vector<std::string> types;
			~RestoreDiagnostics() {
				engine.clearVerboseTypes();
				for (const auto& type:types) engine.addVerboseType(type);
				engine.setVerbose(verbose); profiler.setRenderMetricsEnabled(metrics);
			}
		} restore{engine,profiler,engine.verboseEnabled(),profiler.renderMetricsEnabled(),engine.verboseTypes()};
		engine.clearVerboseTypes(); engine.addVerboseType("renderPerformance"); engine.setVerbose(true);
		profiler.setRenderMetricsEnabled(true);
		for (bool labels:{false,true}) for (bool batching:{false,true})
			for (bool perspective:{false,true}) for (bool storedPyramid:{false,true})
				ProgressiveResetFixture().run(labels,batching,perspective,storedPyramid);
		return nullptr;
	}
}
void registerMdiChildLoadingTests(poca::core::TestRegistry& registry)
{
	registry.add({"progressive_reset_pixels","camera","Progressive Reset Viewer (actual image pixels)",
		"Fresh initializeGL and Reset pixels, RAW/LABEL, shared/individual, ortho/perspective, resident/native pyramid at 1/10/40/83; requires image plugin and OpenGL",false,runProgressiveResetPixels});
}