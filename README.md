# PoCA: Point Clouds Analyst
PoCA is the direct continuation of SR-Tesseler and Coloc-Tesseler. SR-Tesseler has been released in 2015 and, since then, I worked in parralel on extending SR-Tesseler and developing Coloc-Tesseler. As some point, SR-Tesseler became a monster when speaking of coding, and doing any modification was painful. Similarly, I needed to clear the Coloc-Tesseler code to be able to make it available. I therefore decided last summer to re-code completely the platform. PoCA will be the result of this decision and integrate SR-, Coloc-Tesseler and much more in a single software platform.

I released a very preliminary beta for the tutorial I did at I2K2020. I just added a first version of the manual (quite incomplete) to explain briefly most of the commands.

What's in it:

* 3D Visualization
* Optimized 2D and 3D Voronoi diagrams (less than a minute processing for 4,000,000 localizations, even in 3D)
* Advanced interaction through a picking mechanism
* Creation of a "colocalization object" by linking two opened datasets
* DBSCAN should work in 2D and 3D

Known issues:

* Using OpenGL has rendering engine seems to sometimes break the native Qt rendering. It can result in having the whole PoCA window rendered as black. Usually, forcing redraw of the window resolves part of the problem (still, at least rendering of one feature such as the Voronoi stays broken)
* Coloc-Tesseler is still experimental, the Manders coefficients does not seem to work and the Spearmann ones are very slow.
* Only one set of objects can be currently created. 
* PoCA will certainly crash if the user does something I did not planned (clicking on a button while no dataset is opened for instance)

Missing features:

* No export of the objects stats is available yet

I still think that people could find it interesting in its current state, to see where we will go with this platform. Also, PoCA is ongoing heavy development right now so expect frequent updates. 
