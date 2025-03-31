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
#include <float.h>

#include <OpenGL/Camera.hpp>
#include <General/Engine.hpp>

#include "../Widgets/MdiChild.hpp"

MdiChild::MdiChild(poca::opengl::CameraInterface* _widget, QWidget * _parent /*= 0*/, Qt::WindowFlags _flags /*= 0 */ ):QMdiSubWindow( _parent, _flags )
{
	this->setObjectName( "MdiChild" );
	m_widget = _widget;
	QWidget* w = dynamic_cast <QWidget*>(m_widget);
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
	QWidget * wid = this->parentWidget();
	int x = this->parentWidget()->x(), y = this->parentWidget()->y(), w = this->parentWidget()->width(), h = this->parentWidget()->height();
	QMdiSubWindow::resizeEvent( _event );
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


