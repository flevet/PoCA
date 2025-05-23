/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      main.cpp
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

#include <wchar.h>
#include <QtWidgets/QApplication>
#include <QtWidgets/QSplashScreen>
#include <time.h>

#include "Widgets/MainWindow.hpp"

void myMessageOutput(QtMsgType type, const QMessageLogContext& context, const QString& msg)
{
}

int main(int argc, char *argv[])
{
	// Set the environment variable in code BEFORE QApplication is created
	qputenv("QT_AUTO_SCREEN_SCALE_FACTOR", "0");

	qInstallMessageHandler(myMessageOutput); // Install the handler
	QCoreApplication::setAttribute(Qt::AA_UseDesktopOpenGL);
	QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
	QApplication a(argc, argv);
	a.setStyle("fusion");

	QPixmap pixmap("./images/splash.png");
	QSplashScreen splash(pixmap);
	splash.show();

	srand((unsigned)time(NULL));

	MainWindow mainWin;
	mainWin.showMaximized();
	mainWin.show();
	splash.finish(&mainWin);

	return a.exec();
}

