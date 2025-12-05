/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      CGAL_helpers.cpp
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

#include "CGAL_helpers.hpp"

namespace poca::geometry {
    void laplacian_smooth(Surface_mesh_3_double& mesh, int iterations, double lambda)
    {
        // Temporary storage for updated positions
        Surface_mesh_3_double::Property_map<vertex_descriptor, Point_3_double> new_positions;
        bool created;
        boost::tie(new_positions, created) =
            mesh.add_property_map<vertex_descriptor, Point_3_double>("v:new_pos", Point_3_double(0, 0, 0));

        auto vpm = get(CGAL::vertex_point, mesh);

        for (int it = 0; it < iterations; ++it) {
            // Compute new positions
            for (vertex_descriptor v : vertices(mesh)) {
                // Skip isolated vertices
                if (CGAL::halfedge(v, mesh) == Surface_mesh_3_double::null_halfedge())
                    continue;

                Point_3_double p = vpm[v];
                Kernel::Vector_3 sum(0.0, 0.0, 0.0);
                int valence = 0;

                for (auto hv : CGAL::halfedges_around_target(v, mesh)) {
                    vertex_descriptor vn = source(hv, mesh);
                    sum = sum + (vpm[vn] - p);
                    ++valence;
                }

                if (valence > 0) {
                    Kernel::Vector_3 avg = sum / static_cast<double>(valence);
                    // new position = p + lambda * average(neighbors - p)
                    new_positions[v] = p + lambda * avg;
                }
                else {
                    // No neighbors, keep original
                    new_positions[v] = p;
                }
            }

            // Commit new positions
            for (vertex_descriptor v : vertices(mesh)) {
                vpm[v] = new_positions[v];
            }
        }

        mesh.remove_property_map(new_positions);
    }
}