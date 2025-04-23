/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MdiChild.cpp
*
* Copyright: Florian Levet (2020-2022)
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

#include <Windows.h>
#include <QtGui/QIcon>
#include <QtWidgets/QLayout>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QFileDialog>
#include <float.h>
#include <dtv.h>

#include <OpenGL/Camera.hpp>
#include <General/Engine.hpp>
#include <Plot/Icons.hpp>

#include "../Widgets/MdiChild.hpp"

MdiChild::MdiChild(poca::opengl::CameraInterface* _widget, QWidget * _parent /*= 0*/, Qt::WindowFlags _flags /*= 0 */ ):QMdiSubWindow( _parent, _flags )
{
	this->setObjectName( "MdiChild" );
	m_widget = _widget;
	m_camera = dynamic_cast<poca::opengl::Camera*>(m_widget);
	poca::core::MyObjectInterface* object = m_widget->getObject();
	const poca::core::BoundingBox& bbox = object->boundingBox();
	QWidget* w = dynamic_cast <QWidget*>(m_widget);
	//m_testButton = new QPushButton(QString("Start Game"), w);
	int maxSize = 50;
	QGridLayout* layoutTop = new QGridLayout;
	m_3DButton = new QPushButton();
	m_3DButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_3DButton->setMaximumSize(QSize(maxSize, maxSize));
	m_3DButton->setIcon(QIcon(QPixmap(poca::plot::threeDIcon)));
	m_3DButton->setToolTip("3D view");
	m_3DButton->setCheckable(true);
	m_3DButton->setChecked(true);
	QObject::connect(m_3DButton, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));
	m_2DtButton = new QPushButton();
	m_2DtButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_2DtButton->setMaximumSize(QSize(maxSize, maxSize));
	m_2DtButton->setIcon(QIcon(QPixmap(poca::plot::twoDTIcon)));
	m_2DtButton->setToolTip("2D+t view");
	m_2DtButton->setCheckable(true);
	m_2DtButton->setChecked(false);
	QObject::connect(m_2DtButton, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));
	m_playButton = new QPushButton();
	m_playButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	m_playButton->setMaximumSize(QSize(maxSize, maxSize));
	m_playButton->setIcon(QIcon(QPixmap(poca::plot::playIcon)));
	m_playButton->setToolTip("Play");
	m_playButton->setCheckable(true);
	m_playButton->setChecked(false);
	QObject::connect(m_playButton, SIGNAL(clicked(bool)), this, SLOT(actionNeeded(bool)));
	QWidget* emptyWLine = new QWidget;
	emptyWLine->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	QButtonGroup* bgroup = new QButtonGroup;
	bgroup->addButton(m_2DtButton);
	bgroup->addButton(m_3DButton);
	m_minT = ceil(bbox[2]);
	m_maxT = ceil(bbox[5] - 1);
	m_interval = m_maxT - m_minT;
	m_tLabel = new QLabel(QString::number(m_minT).rightJustified(5, '0'));
	m_tSlider = new QSlider(Qt::Vertical);
	m_tSlider->setMinimum(0);
	m_tSlider->setMaximum(m_interval);
	m_tSlider->setSliderPosition(0);
	m_tSlider->setSingleStep(1);
	m_tSlider->setTickInterval(1);
	m_tSlider->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Expanding);
	m_tSlider->setEnabled(false);
	QObject::connect(m_tSlider, SIGNAL(valueChanged(int)), SLOT(actionNeeded(int)));
	layoutTop->addWidget(m_3DButton, 0, 0, 1, 1);
	layoutTop->addWidget(m_2DtButton, 0, 1, 1, 1);
	layoutTop->addWidget(m_playButton, 0, 2, 1, 1);
	layoutTop->addWidget(emptyWLine, 0, 3, 1, 1);
	layoutTop->addWidget(m_tLabel, 1, 0, 1, 2);
	layoutTop->addWidget(m_tSlider, 2, 0, 1, 1);
	m_topW = new QWidget(w);
	m_topW->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
	m_topW->setLayout(layoutTop);
	m_topW->move(0, 0);

	if (object->dimension() == 2)
		m_topW->hide();
	//m_testButton->winId(); // add this
	//
	m_playButton->hide();
	m_tLabel->hide();
	m_tSlider->hide();
	if(w)
		setWidget(w);

	setAttribute( Qt::WA_DeleteOnClose );
	setFocusPolicy(Qt::StrongFocus);
	setMinimumSize( 100, 100 );

	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_widget);
	if(cam)
		QObject::connect(cam, SIGNAL(clickInsideWindow()), this, SLOT(baseWidgetWasClicked()));
}

MdiChild::~MdiChild()
{
	poca::core::Engine* engine = poca::core::Engine::instance();
	poca::core::MyObjectInterface* obj = m_widget->getObject();
	delete m_widget;
	poca::core::Engine::instance()->removeObject(obj);
	emit(setCurrentMdi( NULL ) );
}

void MdiChild::resizeEvent( QResizeEvent * _event )
{
	const QSize& size = _event->size();
	QWidget * wid = this->parentWidget();
	int x = this->parentWidget()->x(), y = this->parentWidget()->y(), w = this->parentWidget()->width(), h = this->parentWidget()->height();
	QMdiSubWindow::resizeEvent( _event );
	m_topW->setGeometry(0, 0, size.width(), 50);
	m_topW->updateGeometry();
}

void MdiChild::moveEvent( QMoveEvent * _event )
{
	QMdiSubWindow::moveEvent( _event );
}

QSize MdiChild::sizeHint() const
{
	std::array <int, 2> size = m_widget->sizeHintInterface();
	return QSize(size[0], size[1]);
}

void MdiChild::resizeWindow()
{
	std::array <int, 2> sizeArray = m_widget->sizeHintInterface();
	QSize size = QSize(sizeArray[0], sizeArray[1]), parentSize = parentWidget()->size();
	int wRemaining = parentSize.width() - this->x(), hRemaining = parentSize.height() - this->y();
	if( size.width() < wRemaining && size.height() < hRemaining ){
		this->resize( size );
	}
	else{
		int sizeW = ( size.width() < wRemaining ) ? size.width() : wRemaining;
		int sizeH = ( size.height() < hRemaining ) ? size.height() : hRemaining;
		this->resize( sizeW, sizeH );
	}
}

void MdiChild::resizeWindow( const float _factor, const float _maxImageW, const float _maxImageH )
{
	QSize size = this->size(), parentSize = parentWidget()->size();
	size *= _factor;
	float wRemaining = parentSize.width() - this->x(), hRemaining = parentSize.height() - this->y();
	if( size.width() < wRemaining && size.height() < hRemaining ){
		this->resize( size );
	}
	else{
		int sizeW = ( size.width() < wRemaining ) ? size.width() : wRemaining;
		int sizeH = ( size.height() < hRemaining ) ? size.height() : hRemaining;
		this->resize( sizeW, sizeH );
	}
}

bool MdiChild::eventFilter( QObject * _obj, QEvent * _event )
{
	return QMdiSubWindow::eventFilter( _obj, _event );
}

void MdiChild::mousePressEvent( QMouseEvent * _event )
{
	QMdiSubWindow::mousePressEvent( _event );
	emit(setCurrentMdi( this ) );
}

void MdiChild::baseWidgetWasClicked()
{
	emit(setCurrentMdi( this ) );
}

void MdiChild::actionNeeded()
{
}

void MdiChild::actionNeeded(int _val)
{
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_widget);
	if (cam == NULL)
		return;
	QObject* sender = QObject::sender();
	if (sender == m_tSlider) {
		auto plane = getPlane(_val);
		std::cout << _val << " - > " << plane << std::endl;
		poca::core::BoundingBox bbox = cam->getCurrentCrop();
		bbox[2] = plane;
		bbox[5] = plane;
		cam->zoomToBoundingBox(bbox, false);
		m_tLabel->setText(QString::number(plane).rightJustified(5, '0'));
	}
}

void MdiChild::actionNeeded(bool _val)
{
	poca::opengl::Camera* cam = dynamic_cast <poca::opengl::Camera*>(m_widget);
	if (cam == NULL)
		return;
	QObject* sender = QObject::sender();
	if (sender == m_3DButton) {
		m_tSlider->setEnabled(!_val);
		cam->fixPlane(poca::opengl::Camera::Plane_XY, false);
		cam->zoomToBoundingBox(cam->getObject()->boundingBox(), false);
		m_playButton->hide();
		m_tLabel->hide();
		m_tSlider->hide();
	}
	else if (sender == m_2DtButton) {
		m_tSlider->setEnabled(_val);
		cam->fixPlane(poca::opengl::Camera::Plane_XY, true);
		auto plane = getPlane(m_tSlider->value());
		poca::core::BoundingBox bbox = cam->getCurrentCrop();
		bbox[2] = plane;
		bbox[5] = plane;
		cam->zoomToBoundingBox(bbox, false);
		m_playButton->show();
		m_tLabel->show();
		m_tSlider->show();
	}
	else if (sender == m_playButton) {
		if (_val) {
			if (m_timer == NULL) {
				m_timer = new QTimer(this);
				connect(m_timer, SIGNAL(timeout()), this, SLOT(playFrame()));
			}
			m_timer->start(10);
		}
		else {
			m_timer->stop();
			std::vector <QImage>& frames = m_camera->getMovieFrames();
			QString filename("movie.mp4");
			filename = QFileDialog::getSaveFileName(NULL, QObject::tr("Save movie..."), filename, QString("mp4 files (*.mp4)"), 0, QFileDialog::DontUseNativeDialog);
			if (filename.isEmpty()) return;

			const std::vector <QImage>& _frames = cam->getMovieFrames();

			atg_dtv::Encoder encoder;
			atg_dtv::Encoder::VideoSettings settings{};

			// Output filename
			settings.fname = filename.toStdString();

			// Input dimensions
			settings.inputWidth = _frames[0].width();
			settings.inputHeight = _frames[0].height();

			// Output dimensions
			settings.width = _frames[0].width();
			settings.height = _frames[0].height();

			// Encoder settings
			settings.hardwareEncoding = true;
			settings.bitRate = 16000000;
			settings.frameRate = 24;

			const int FrameCount = _frames.size();

			auto start = std::chrono::steady_clock::now();

			std::cout << "==============================================\n";
			std::cout << " Direct to Video (DTV) Sample Application\n\n";

			encoder.run(settings, 2);

			for (int i = 0; i < FrameCount; ++i) {
				if ((i + 1) % 100 == 0 || i >= FrameCount - 10) {
					std::cout << "Frame: " << (i + 1) << "/" << FrameCount << "\n";
				}

				const int sin_i = std::lroundf(255 * (0.5 + 0.5 * std::sin(i * 0.01)));

				atg_dtv::Frame* frame = encoder.newFrame(true);
				if (frame == nullptr) break;
				if (encoder.getError() != atg_dtv::Encoder::Error::None) break;

				const int lineWidth = frame->m_lineWidth;
				for (int y = 0; y < settings.inputHeight; ++y) {
					uint8_t* row = &frame->m_rgb[y * lineWidth];
					for (int x = 0; x < settings.inputWidth; ++x) {
						const int index = x * 3;
						QRgb color = _frames[i].pixel(x, y);
						row[index + 0] = qRed(color); // r
						row[index + 1] = qGreen(color); // g
						row[index + 2] = qBlue(color);   // b
					}
				}

				/*QString paddedNumber = QString::number(i).rightJustified(5, '0');
				bool res = _frames[i].save(QString("e:/poca_") + paddedNumber + QString(".png"));
				if (!res)
					std::cout << "Problem with saving" << std::endl;*/

				encoder.submitFrame();

				bool res = _frames[i].save(QString("d:/poca_%1.jpg").arg(QString::number(i + 1).rightJustified(3, '0')));
				if (!res)
					std::cout << "Problem with saving" << std::endl;
			}

			encoder.commit();
			encoder.stop();

			auto end = std::chrono::steady_clock::now();

			const double elapsedSeconds =
				std::chrono::duration<double>(end - start).count();

			std::cout << "==============================================\n";
			if (encoder.getError() == atg_dtv::Encoder::Error::None) {
				std::cout << "Encoding took: " << elapsedSeconds << " seconds" << "\n";
				std::cout << "Real-time framerate: " << FrameCount / elapsedSeconds << " FPS" << "\n";
			}
			else {
				std::cout << "Encoding failed\n";
			}
			frames.clear();
		}
	}
}

int32_t MdiChild::getPlane(int _val)
{
	return m_minT + _val;
}

void MdiChild::setFrame(int _val)
{

}

void MdiChild::playFrame()
{
	auto val = m_tSlider->value() + 1;
	if (val > m_interval)
		val = 0;
	m_tSlider->setValue(val);
	m_camera->makeCurrent();
	m_camera->drawElementsOffscreen();
}

MyMdiArea::MyMdiArea( QWidget * _w ):QMdiArea( _w )
{

}

MyMdiArea::~MyMdiArea()
{

}

bool MyMdiArea::eventFilter( QObject * _receiver, QEvent * _event )
{
	return QMdiArea::eventFilter( _receiver, _event );
}




