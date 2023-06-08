# PoCA: Point Cloud Analyst

## Introduction
PoCA is a a powerful stand-alone software designed to ease the manipulation and quantification of multidimensional and multicolor SMLM point cloud data. It is built around a custom-made Open-GL-based rendering engine that provides full user interactive control of SMLM point cloud data, both for visualization and manipulation. It combines the strengths of both C++ and Python programming languages, providing access to efficient and optimized C++ computer graphics algorithms and Python ecosystem. It is designed for improving users and developers’ experience, by integrating a user-friendly GUI, a macro recorder, and the capability to execute Python code easily. PoCA is the result of a decade of developments and the legacy of SR-Tesseler and Coloc-Tesseler, software solutions that were swiftly adopted by the community.

<p align="center">
	<img src="https://poca-smlm.github.io/images/poca.gif" width="1000">
</p>

**If you use it, please cite it:**

* Florian Levet & Jean-Baptiste Sibarita. 
*PoCA: a software platform for point cloud data visualization and quantification*. [Nature Methods 20, 629–630 (2023) doi:10.1038/s41592-023-01811-4](https://doi.org/10.1038/s41592-023-01811-4)

* Florian Levet, Eric Hosy, Adel Kechkar, Corey Butler, Anne Beghin, Daniel Choquet, Jean-Baptiste Sibarita. 
*SR-Tesseler: a method to segment and quantify localization-based super-resolution microscopy data*. [Nature Methods 12 (11), 1065-71 (2015) doi:10.1038/nmeth.3579](https://doi.org/10.1038/nmeth.3579)

* Florian Levet, Guillaume Julien, Rémi Galland, Corey Butler, Anne Beghin, Anaël Chazeau, Philipp Hoess, Jonas Ries, Grégory Giannone, Jean-Baptiste Sibarita. 
*A tessellation-based colocalization analysis approach for single-molecule localization microscopy*.
[Nature Communications 10, 2379 (2019) doi:10.1038/s41467-019-10007-4](https://doi.org/10.1038/s41467-019-10007-4)

PoCA is developed by [Florian Levet](https://www.researchgate.net/profile/Florian-Levet), researcher in the [Quantitative Imaging of the Cell team](https://www.iins.u-bordeaux.fr/SIBARITA), headed by [Jean-Baptiste Sibarita](https://www.researchgate.net/profile/Jean-Baptiste-Sibarita). FL and JBS are part of the [Interdisciplinary Insitute for Neuroscience](https://www.iins.u-bordeaux.fr/). FL is part of the [Bordeaux Imaging Center](https://www.bic.u-bordeaux.fr/).

If you search for support, please open a thread on the [image.sc](https://image.sc/) forum or raise an [issue](https://github.com/flevet/PoCA/issues) here.

## Overview
* [Installation and compilation](https://poca-smlm.github.io/installation.html)
* [PoCA main interface](https://poca-smlm.github.io/images/poca_windows.png)
* [Opening localization files](https://poca-smlm.github.io/opening.html)
* [Manipulating point clouds](https://poca-smlm.github.io/manipulating.html)
	- [Visualization](https://poca-smlm.github.io/manipulating.html#visualization)
	- [Filtering](https://poca-smlm.github.io/manipulating.html#filtering)
	- [Cropping](https://poca-smlm.github.io/manipulating.html#cropping)
	- [Picking](https://poca-smlm.github.io/manipulating.html#picking)
* Quantification techniques
	- [Delaunay triangulation](https://poca-smlm.github.io/delaunay.html)
	- [Voronoi diagram](https://poca-smlm.github.io/voronoi.html)
	- [DBSCAN](https://poca-smlm.github.io/dbscan.html)
* [Objects](https://poca-smlm.github.io/objects.html)
* Colocalization analysis
	- [Prerequesite](https://poca-smlm.github.io/prerequesite_coloc.html)
	- [Coloc-Tesseler](https://poca-smlm.github.io/coloc-tesseler.html)
	- [Object colocalization](https://poca-smlm.github.io/objects_colocalization.html)
* [ROIs](https://poca-smlm.github.io/rois.html)
* [Macros](https://poca-smlm.github.io/macros.html)
* [Executing Python scripts](https://poca-smlm.github.io/python.html)
* [How to cite](https://poca-smlm.github.io/citations.html)


## Use cases
* [Creation of clusters with a Voronoi diagram](https://poca-smlm.github.io/useCase_clustering_voronoi.html)
* [Creation of clusters with a Delaunay triangulation](https://poca-smlm.github.io/useCase_clustering_delaunay.html)
* [Colocalization analysis with Coloc-Tesseler](https://poca-smlm.github.io/useCase_coloc_tess.html)
* [Colocalization analysis with object colocalization](https://poca-smlm.github.io/useCase_coloc_objs.html)