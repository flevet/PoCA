# PoCA: Point Cloud Analyst
PoCA is the direct continuation of SR-Tesseler and Coloc-Tesseler. SR-Tesseler has been released in 2015 and, since then, I worked in parralel on extending SR-Tesseler and developing Coloc-Tesseler. As some point, SR-Tesseler became a monster when speaking of coding, and doing any modification was painful. Similarly, I needed to clear the Coloc-Tesseler code to be able to make it available. I therefore decided last summer to re-code completely the platform. PoCA will be the result of this decision and integrate SR-, Coloc-Tesseler and much more in a single software platform.

What's in it:

* 3D Visualization
* Optimized 2D and 3D Voronoi diagrams (less than a minute processing for 4,000,000 localizations, even in 3D)
* Advanced interaction through a picking mechanism
* Creation of a "colocalization object" by linking two opened datasets
* DBSCAN should work in 2D and 3D

Known issues:

* Using OpenGL has rendering engine seems to sometimes break the native Qt rendering. It can result in having the whole PoCA window rendered as black. Usually, forcing redraw of the window resolves part of the problem (still, at least rendering of one feature such as the Voronoi stays broken)
* Only one set of objects can be currently created. 
* PoCA will certainly crash if the user does something I did not planned (clicking on a button while no dataset is opened for instance)

# Compilation
There are CMakeLists.txt in each project to help with compilation. Nevertheless, some of the libraries are not automatically found and need to be manually added in cmake-gui.
Required libraries:

* CGAL (Tested with version 5.3, https://www.cgal.org/)
* Boost (Tested with version 1.74.0, https://www.boost.org/)
* Qt (Tested with version 5.15.2, https://www.qt.io/)
* CUDA (Tested with version 11.5, https://developer.nvidia.com/cuda-zone)
* TBB (Tested with version 2020.3, https://github.com/oneapi-src/oneTBB/releases/tag/v2020.3)
* Eigen3 (https://eigen.tuxfamily.org/)
* GLM (Tested with version 0.9.9.8, https://github.com/g-truc/glm)
* GLEW (Tested with version 2.1.0, http://glew.sourceforge.net/)
* Python (Tested with version 3.7.4)

# Binaries
As of right now, only binaries for Windows are available (https://github.com/flevet/PoCA/releases).

I'm also working on the manual.

If upon executing poca.exe Windows asks for dlls such as "VCRUNTIME140_1.dll", you may need to install the "microsoft visual c++ 2019 redistributable package (x64)": https://docs.microsoft.com/en-GB/cpp/windows/latest-supported-vc-redist?view=msvc-160

For having access to the Voronoi 3D construction, you will need an NVidia card with the latest drivers installed as well as CUDA (10.2 for instance, may work with newest versions): https://developer.nvidia.com/cuda-10.2-download-archive
