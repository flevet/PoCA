/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      MdiChild.hpp
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

#ifndef MdiChild_h__
#define MdiChild_h__

#include <QtWidgets/QMdiSubWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/Qlabel>
#include <QtGui/QResizeEvent>
#include <QtGui/QFocusEvent>
#include <QtWidgets/QMdiArea>

#include <OpenGL/Camera.hpp>

class MyMdiArea: public QMdiArea{
	Q_OBJECT
public:
	MyMdiArea( QWidget * = 0 );
	~MyMdiArea();

protected:
	bool eventFilter( QObject *, QEvent * );
};

class MdiChild: public QMdiSubWindow{
	Q_OBJECT
public:
	MdiChild(poca::opengl::CameraInterface*, QWidget * = 0, Qt::WindowFlags = Qt::WindowFlags());
	~MdiChild();

	inline poca::opengl::CameraInterface* getWidget() {return m_widget;}
	QSize sizeHint() const;

signals:
	void setCurrentMdi( MdiChild * );

public slots:
	void resizeWindow();
	void resizeWindow( const float, const float, const float );
	void baseWidgetWasClicked();

	void actionNeeded();
	void actionNeeded(int);
	void actionNeeded(bool);

	void setFrame(int);

protected slots:
	void playFrame();

protected:
	void resizeEvent( QResizeEvent * );
	void moveEvent( QMoveEvent * );
	bool eventFilter( QObject *, QEvent * );
	void mousePressEvent( QMouseEvent * );
	void mouseMoveEvent(QMouseEvent*);

	int32_t getPlane(int);

protected:
	poca::opengl::CameraInterface * m_widget;
	poca::opengl::Camera* m_camera{ NULL };
	QPushButton* m_2DtButton, * m_3DButton, * m_playButton;
	QSlider* m_tSlider;
	QWidget* m_topW, * m_emptyForSliderW;
	QLabel* m_tLabel;
	QTimer* m_timer{ NULL };
	int32_t m_minT, m_maxT, m_interval;
};

#endif // MdiChild_h__

