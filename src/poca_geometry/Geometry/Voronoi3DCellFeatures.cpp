/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      Voronoi3DCellFeatures.cpp
*
* Copyright: Florian Levet (2020-2026)
*
* License:   LGPL v3
*/

#include "Voronoi3DCellFeatures.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cfloat>
#include <functional>
#include <limits>
#include <numeric>
#include <unordered_set>

#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <Eigen/Dense>

namespace poca::geometry {
	namespace {
		constexpr double PI = 3.141592653589793238462643383279502884;

		inline Eigen::Vector3d toEigen(const Point_3_inexact& p)
		{
			return Eigen::Vector3d(CGAL::to_double(p.x()), CGAL::to_double(p.y()), CGAL::to_double(p.z()));
		}

		inline Eigen::Vector3d toEigen(const Point_3_double& p)
		{
			return Eigen::Vector3d(CGAL::to_double(p.x()), CGAL::to_double(p.y()), CGAL::to_double(p.z()));
		}

		inline Eigen::Vector3d toEigen(const poca::core::Vec3mf& p)
		{
			return Eigen::Vector3d(p.x(), p.y(), p.z());
		}

		inline Eigen::Matrix3d outer(const Eigen::Vector3d& u, const Eigen::Vector3d& v)
		{
			return u * v.transpose();
		}

		inline double signedTetraVolume6(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c)
		{
			return a.dot(b.cross(c));
		}

		inline Eigen::Matrix3d tetraSecondMomentOrigin(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c, const double vSigned)
		{
			Eigen::Matrix3d m = Eigen::Matrix3d::Zero();
			const double wDiag = vSigned / 10.0;
			const double wCross = vSigned / 20.0;
			m += wDiag * (outer(a, a) + outer(b, b) + outer(c, c));
			m += wCross * (outer(a, b) + outer(b, a) + outer(a, c) + outer(c, a) + outer(b, c) + outer(c, b));
			return m;
		}

		inline float finiteFloat(const double v, const float fallback = 0.f)
		{
			return std::isfinite(v) ? static_cast<float>(v) : fallback;
		}

		inline double safeLog(const double v)
		{
			return std::log(std::max(v, 1e-300));
		}

		inline double zscore(const double v, const double mean, const double sd)
		{
			const double denom = sd > std::numeric_limits<double>::epsilon() ? sd : 1.0;
			return (v - mean) / denom;
		}

		struct RunningStats {
			double sum = 0.0, sum2 = 0.0;
			size_t n = 0;
			void add(const double v) { if (std::isfinite(v)) { sum += v; sum2 += v * v; ++n; } }
			double mean() const { return n > 0 ? sum / static_cast<double>(n) : 0.0; }
			double sd() const { const double m = mean(); return n > 1 ? std::sqrt(std::max(0.0, sum2 / static_cast<double>(n) - m * m)) : 1.0; }
		};

		struct CellMetrics {
			double volume = 0.0;
			double surfaceArea = 0.0;
			double sphericity = 0.0;
			double compactness = 0.0;
			double seedCentroidDist = 0.0;
			double seedCentroidDistNorm = 0.0;
			double covarianceAnisotropy = 0.0;
			double covarianceLinearity = 0.0;
			double covariancePlanarity = 0.0;
			double covarianceSphericity = 0.0;
			double normalAnisotropy = 0.0;
			double normalPlanarity = 0.0;
			double bboxAspectRatio = 0.0;
			double distToBox = 0.0;
			double distToBoxNorm = 0.0;
			uint32_t nbFaces = 0;
			uint32_t nbVertices = 0;
			Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
			Eigen::Vector3d anisotropyVector = Eigen::Vector3d::Zero();
			Eigen::Vector3d covEigenvalues = Eigen::Vector3d::Zero();
			Eigen::Vector3d normalEigenvalues = Eigen::Vector3d::Zero();
			Eigen::Vector3d principalAxis = Eigen::Vector3d::Zero();
		};

		inline double distanceToBoxPlanesInside(const Eigen::Vector3d& p, const poca::core::BoundingBox* box)
		{
			if (box == nullptr) return 0.0;
			const bool inside = p.x() >= (*box)[0] && p.x() <= (*box)[3] && p.y() >= (*box)[1] && p.y() <= (*box)[4] && p.z() >= (*box)[2] && p.z() <= (*box)[5];
			if (!inside) return 0.0;
			const double dx = std::min(p.x() - (*box)[0], (*box)[3] - p.x());
			const double dy = std::min(p.y() - (*box)[1], (*box)[4] - p.y());
			const double dz = std::min(p.z() - (*box)[2], (*box)[5] - p.z());
			return std::min(dx, std::min(dy, dz));
		}

		void finalizeMoments(CellMetrics& out, const double v6Sum, const Eigen::Vector3d& m1, const Eigen::Matrix3d& m2, Eigen::Matrix3d normalTensor, const Eigen::Vector3d& seed, const poca::core::BoundingBox* clipBox)
		{
			const double vSigned = v6Sum / 6.0;
			if (!std::isfinite(vSigned) || std::abs(vSigned) <= std::numeric_limits<double>::epsilon())
				return;

			out.volume = std::abs(vSigned);
			out.centroid = m1 / vSigned;
			out.anisotropyVector = out.centroid - seed;
			out.seedCentroidDist = out.anisotropyVector.norm();

			const double req = std::cbrt(3.0 * out.volume / (4.0 * PI));
			out.seedCentroidDistNorm = req > 0.0 ? out.seedCentroidDist / req : 0.0;
			out.sphericity = (out.surfaceArea > 0.0 && out.volume > 0.0) ? std::pow(PI, 1.0 / 3.0) * std::pow(6.0 * out.volume, 2.0 / 3.0) / out.surfaceArea : 0.0;
			out.compactness = (out.surfaceArea > 0.0) ? (36.0 * PI * out.volume * out.volume) / (out.surfaceArea * out.surfaceArea * out.surfaceArea) : 0.0;

			Eigen::Matrix3d cov = (m2 / vSigned) - (out.centroid * out.centroid.transpose());
			cov = 0.5 * (cov + cov.transpose());
			Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
			if (solver.info() == Eigen::Success) {
				Eigen::Vector3d ev = solver.eigenvalues();
				std::array<double, 3> e = { std::max(0.0, ev(0)), std::max(0.0, ev(1)), std::max(0.0, ev(2)) };
				std::sort(e.begin(), e.end(), std::greater<double>());
				const double l1 = e[0], l2 = e[1], l3 = e[2];
				out.covEigenvalues = Eigen::Vector3d(l1, l2, l3);
				out.principalAxis = solver.eigenvectors().col(2).normalized();
				out.covarianceAnisotropy = (l3 > 0.0) ? std::sqrt(l1 / l3) : 0.0;
				out.covarianceLinearity = (l1 > 0.0) ? (l1 - l2) / l1 : 0.0;
				out.covariancePlanarity = (l1 > 0.0) ? (l2 - l3) / l1 : 0.0;
				out.covarianceSphericity = (l1 > 0.0) ? l3 / l1 : 0.0;
			}

			if (out.surfaceArea > 0.0) {
				normalTensor /= out.surfaceArea;
				Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solverN(normalTensor);
				if (solverN.info() == Eigen::Success) {
					Eigen::Vector3d ev = solverN.eigenvalues();
					std::array<double, 3> e = { std::max(0.0, ev(0)), std::max(0.0, ev(1)), std::max(0.0, ev(2)) };
					std::sort(e.begin(), e.end(), std::greater<double>());
					out.normalEigenvalues = Eigen::Vector3d(e[0], e[1], e[2]);
					out.normalAnisotropy = e[2] > 0.0 ? e[0] / e[2] : 0.0;
					out.normalPlanarity = e[0] > 0.0 ? (e[1] - e[2]) / e[0] : 0.0;
				}
			}

			out.distToBox = distanceToBoxPlanesInside(seed, clipBox);
			out.distToBoxNorm = req > 0.0 ? out.distToBox / req : 0.0;
		}

		CellMetrics computePolyhedronMetrics(const Polyhedron_3_inexact& poly, const Eigen::Vector3d& seed, const poca::core::BoundingBox* clipBox)
		{
			CellMetrics out;
			if (poly.empty()) return out;

			out.nbFaces = static_cast<uint32_t>(poly.size_of_facets());
			out.nbVertices = static_cast<uint32_t>(poly.size_of_vertices());
			Eigen::Vector3d m1 = Eigen::Vector3d::Zero();
			Eigen::Matrix3d m2 = Eigen::Matrix3d::Zero();
			Eigen::Matrix3d normalTensor = Eigen::Matrix3d::Zero();
			double v6Sum = 0.0;
			double xmin = DBL_MAX, ymin = DBL_MAX, zmin = DBL_MAX, xmax = -DBL_MAX, ymax = -DBL_MAX, zmax = -DBL_MAX;

			for (auto vi = poly.vertices_begin(); vi != poly.vertices_end(); ++vi) {
				const Eigen::Vector3d p = toEigen(vi->point());
				xmin = std::min(xmin, p.x()); ymin = std::min(ymin, p.y()); zmin = std::min(zmin, p.z());
				xmax = std::max(xmax, p.x()); ymax = std::max(ymax, p.y()); zmax = std::max(zmax, p.z());
			}

			for (auto fi = poly.facets_begin(); fi != poly.facets_end(); ++fi) {
				std::vector<Eigen::Vector3d> vertices;
				auto h = fi->facet_begin();
				do { vertices.push_back(toEigen(h->vertex()->point())); } while (++h != fi->facet_begin());
				for (size_t i = 1; i + 1 < vertices.size(); ++i) {
					const Eigen::Vector3d& a = vertices[0]; const Eigen::Vector3d& b = vertices[i]; const Eigen::Vector3d& c = vertices[i + 1];
					const Eigen::Vector3d cross = (b - a).cross(c - a);
					const double triArea = 0.5 * cross.norm();
					out.surfaceArea += triArea;
					if (triArea > 0.0) normalTensor += triArea * outer(cross.normalized(), cross.normalized());
					const double v6 = signedTetraVolume6(a, b, c);
					v6Sum += v6;
					const double vSigned = v6 / 6.0;
					m1 += vSigned * (a + b + c) * 0.25;
					m2 += tetraSecondMomentOrigin(a, b, c, vSigned);
				}
			}

			finalizeMoments(out, v6Sum, m1, m2, normalTensor, seed, clipBox);
			const double dx = xmax - xmin, dy = ymax - ymin, dz = zmax - zmin;
			const double minD = std::max(std::numeric_limits<double>::epsilon(), std::min(dx, std::min(dy, dz)));
			out.bboxAspectRatio = std::max(dx, std::max(dy, dz)) / minD;
			return out;
		}

		CellMetrics computeMeshMetrics(Surface_mesh_3_double mesh, const Eigen::Vector3d& seed, const poca::core::BoundingBox* clipBox)
		{
			namespace PMP = CGAL::Polygon_mesh_processing;
			CellMetrics out;
			if (mesh.number_of_vertices() == 0 || mesh.number_of_faces() == 0) return out;
			try { PMP::orient_to_bound_a_volume(mesh); } catch (...) {}
			out.nbFaces = static_cast<uint32_t>(mesh.number_of_faces());
			out.nbVertices = static_cast<uint32_t>(mesh.number_of_vertices());
			try { out.surfaceArea = PMP::area(mesh); } catch (...) { out.surfaceArea = 0.0; }
			Eigen::Vector3d m1 = Eigen::Vector3d::Zero();
			Eigen::Matrix3d m2 = Eigen::Matrix3d::Zero();
			Eigen::Matrix3d normalTensor = Eigen::Matrix3d::Zero();
			double v6Sum = 0.0;
			double xmin = DBL_MAX, ymin = DBL_MAX, zmin = DBL_MAX, xmax = -DBL_MAX, ymax = -DBL_MAX, zmax = -DBL_MAX;

			for (auto v : mesh.vertices()) {
				const Eigen::Vector3d p = toEigen(mesh.point(v));
				xmin = std::min(xmin, p.x()); ymin = std::min(ymin, p.y()); zmin = std::min(zmin, p.z());
				xmax = std::max(xmax, p.x()); ymax = std::max(ymax, p.y()); zmax = std::max(zmax, p.z());
			}
			for (auto f : mesh.faces()) {
				std::vector<Eigen::Vector3d> vertices;
				auto h = mesh.halfedge(f);
				auto start = h;
				do { vertices.push_back(toEigen(mesh.point(mesh.target(h)))); h = mesh.next(h); } while (h != start);
				for (size_t i = 1; i + 1 < vertices.size(); ++i) {
					const Eigen::Vector3d& a = vertices[0]; const Eigen::Vector3d& b = vertices[i]; const Eigen::Vector3d& c = vertices[i + 1];
					const Eigen::Vector3d cross = (b - a).cross(c - a);
					const double triArea = 0.5 * cross.norm();
					if (triArea > 0.0) normalTensor += triArea * outer(cross.normalized(), cross.normalized());
					const double v6 = signedTetraVolume6(a, b, c);
					v6Sum += v6;
					const double vSigned = v6 / 6.0;
					m1 += vSigned * (a + b + c) * 0.25;
					m2 += tetraSecondMomentOrigin(a, b, c, vSigned);
				}
			}
			if (out.surfaceArea <= 0.0) {
				// fallback to accumulated triangulation area if PMP::area failed
				for (auto f : mesh.faces()) {
					std::vector<Eigen::Vector3d> vertices;
					auto h = mesh.halfedge(f); auto start = h;
					do { vertices.push_back(toEigen(mesh.point(mesh.target(h)))); h = mesh.next(h); } while (h != start);
					for (size_t i = 1; i + 1 < vertices.size(); ++i) out.surfaceArea += 0.5 * (vertices[i] - vertices[0]).cross(vertices[i + 1] - vertices[0]).norm();
				}
			}

			finalizeMoments(out, v6Sum, m1, m2, normalTensor, seed, clipBox);
			const double dx = xmax - xmin, dy = ymax - ymin, dz = zmax - zmin;
			const double minD = std::max(std::numeric_limits<double>::epsilon(), std::min(dx, std::min(dy, dz)));
			out.bboxAspectRatio = std::max(dx, std::max(dy, dz)) / minD;
			return out;
		}

		void add(std::map<std::string, std::vector<float>>& values, const std::string& name, const size_t n)
		{
			values[name] = std::vector<float>(n, 0.f);
		}

		Voronoi3DCellFeatures::FeatureSet fillFeatureSet(const std::vector<CellMetrics>& metrics, const std::vector<double>& inputVolumes, const std::vector<Eigen::Vector3d>& seeds, const poca::core::MyArrayUInt32& neighbors, const std::vector<bool>& borderLocs)
		{
			Voronoi3DCellFeatures::FeatureSet out;
			const size_t n = metrics.size();
			const std::vector<std::string> names = {
				"cellSurfaceArea", "cellSphericity", "cellCompactness", "cellNbFaces", "cellNbVertices", "cellBboxAspectRatio",
				"cellCentroidX", "cellCentroidY", "cellCentroidZ",
				"anisotropyVectorX", "anisotropyVectorY", "anisotropyVectorZ", "anisotropyVectorNorm", "anisotropyVectorNormEqRadius",
				"covAnisotropy", "covLinearity", "covPlanarity", "covSphericity", "covEigenValue1", "covEigenValue2", "covEigenValue3",
				"principalAxisX", "principalAxisY", "principalAxisZ",
				"normalAnisotropy", "normalPlanarity", "normalTensorEigenValue1", "normalTensorEigenValue2", "normalTensorEigenValue3",
				"minkowskiVolumeAnisotropy", "minkowskiSurfaceAnisotropy", "minkowskiSurfacePlanarity",
				"logVol", "offsetNorm", "sphericity", "anisotropy", "area", "isBorder", "cellBorderDistance", "cellBorderDistanceEqRadius",
				"localLogVolumeCV", "cvLogVol", "localLogVolumeZScore", "localAnisotropyVectorAlignment", "localPrincipalAxisAlignment", "anisotropyVectorDivergence",
				"anisotropyVectorConvergence", "cavityRimScore", "voidScore"
			};
			for (const auto& name : names) add(out.values, name, n);

			RunningStats stLogVol, stOffset, stLogAniso, stSphericity, stInvDist;
			for (size_t i = 0; i < n; ++i) {
				const bool isBorder = i < borderLocs.size() && borderLocs[i];
				if (isBorder) continue;
				const double vol = inputVolumes[i] > 0.0 ? inputVolumes[i] : metrics[i].volume;
				stLogVol.add(safeLog(vol));
				stOffset.add(metrics[i].seedCentroidDistNorm);
				stLogAniso.add(safeLog(metrics[i].covarianceAnisotropy));
				stSphericity.add(metrics[i].sphericity);
				const double inv = metrics[i].distToBoxNorm > 0.0 ? 1.0 / metrics[i].distToBoxNorm : 0.0;
				stInvDist.add(inv);
			}
			const double meanLogVol = stLogVol.mean(), sdLogVol = stLogVol.sd();
			const double meanOffset = stOffset.mean(), sdOffset = stOffset.sd();
			const double meanLogAniso = stLogAniso.mean(), sdLogAniso = stLogAniso.sd();
			const double meanSph = stSphericity.mean(), sdSph = stSphericity.sd();
			const double meanInvDist = stInvDist.mean(), sdInvDist = stInvDist.sd();

			for (size_t i = 0; i < n; ++i) {
				const double vol = inputVolumes[i] > 0.0 ? inputVolumes[i] : metrics[i].volume;
				out.values["cellSurfaceArea"][i] = finiteFloat(metrics[i].surfaceArea);
				out.values["cellSphericity"][i] = finiteFloat(metrics[i].sphericity);
				out.values["cellCompactness"][i] = finiteFloat(metrics[i].compactness);
				out.values["cellNbFaces"][i] = static_cast<float>(metrics[i].nbFaces);
				out.values["cellNbVertices"][i] = static_cast<float>(metrics[i].nbVertices);
				out.values["cellBboxAspectRatio"][i] = finiteFloat(metrics[i].bboxAspectRatio);
				out.values["cellCentroidX"][i] = finiteFloat(metrics[i].centroid.x());
				out.values["cellCentroidY"][i] = finiteFloat(metrics[i].centroid.y());
				out.values["cellCentroidZ"][i] = finiteFloat(metrics[i].centroid.z());
				out.values["anisotropyVectorX"][i] = finiteFloat(metrics[i].anisotropyVector.x());
				out.values["anisotropyVectorY"][i] = finiteFloat(metrics[i].anisotropyVector.y());
				out.values["anisotropyVectorZ"][i] = finiteFloat(metrics[i].anisotropyVector.z());
				out.values["anisotropyVectorNorm"][i] = finiteFloat(metrics[i].seedCentroidDist);
				out.values["anisotropyVectorNormEqRadius"][i] = finiteFloat(metrics[i].seedCentroidDistNorm);
				out.values["covAnisotropy"][i] = finiteFloat(metrics[i].covarianceAnisotropy);
				out.values["covLinearity"][i] = finiteFloat(metrics[i].covarianceLinearity);
				out.values["covPlanarity"][i] = finiteFloat(metrics[i].covariancePlanarity);
				out.values["covSphericity"][i] = finiteFloat(metrics[i].covarianceSphericity);
				out.values["covEigenValue1"][i] = finiteFloat(metrics[i].covEigenvalues.x());
				out.values["covEigenValue2"][i] = finiteFloat(metrics[i].covEigenvalues.y());
				out.values["covEigenValue3"][i] = finiteFloat(metrics[i].covEigenvalues.z());
				out.values["principalAxisX"][i] = finiteFloat(metrics[i].principalAxis.x());
				out.values["principalAxisY"][i] = finiteFloat(metrics[i].principalAxis.y());
				out.values["principalAxisZ"][i] = finiteFloat(metrics[i].principalAxis.z());
				out.values["normalAnisotropy"][i] = finiteFloat(metrics[i].normalAnisotropy);
				out.values["normalPlanarity"][i] = finiteFloat(metrics[i].normalPlanarity);
				out.values["normalTensorEigenValue1"][i] = finiteFloat(metrics[i].normalEigenvalues.x());
				out.values["normalTensorEigenValue2"][i] = finiteFloat(metrics[i].normalEigenvalues.y());
				out.values["normalTensorEigenValue3"][i] = finiteFloat(metrics[i].normalEigenvalues.z());
				out.values["minkowskiVolumeAnisotropy"][i] = finiteFloat(metrics[i].covarianceAnisotropy);
				out.values["minkowskiSurfaceAnisotropy"][i] = finiteFloat(metrics[i].normalAnisotropy);
				out.values["minkowskiSurfacePlanarity"][i] = finiteFloat(metrics[i].normalPlanarity);
				out.values["logVol"][i] = finiteFloat(safeLog(vol));
				out.values["offsetNorm"][i] = finiteFloat(metrics[i].seedCentroidDistNorm);
				out.values["sphericity"][i] = finiteFloat(metrics[i].sphericity);
				out.values["anisotropy"][i] = finiteFloat(metrics[i].covarianceAnisotropy);
				out.values["area"][i] = finiteFloat(metrics[i].surfaceArea);
				out.values["isBorder"][i] = (i < borderLocs.size() && borderLocs[i]) ? 1.f : 0.f;
				out.values["cellBorderDistance"][i] = finiteFloat(metrics[i].distToBox);
				out.values["cellBorderDistanceEqRadius"][i] = finiteFloat(metrics[i].distToBoxNorm);
				const double inv = metrics[i].distToBoxNorm > 0.0 ? 1.0 / metrics[i].distToBoxNorm : 0.0;
				out.values["voidScore"][i] = finiteFloat(zscore(safeLog(vol), meanLogVol, sdLogVol) + zscore(metrics[i].seedCentroidDistNorm, meanOffset, sdOffset) + zscore(safeLog(metrics[i].covarianceAnisotropy), meanLogAniso, sdLogAniso) - zscore(metrics[i].sphericity, meanSph, sdSph) - zscore(inv, meanInvDist, sdInvDist));
			}

			const auto& firsts = neighbors.getFirstElements();
			const auto& neighs = neighbors.getData();
			if (firsts.size() >= n + 1) {
				for (size_t i = 0; i < n; ++i) {
					double sum = safeLog(inputVolumes[i] > 0.0 ? inputVolumes[i] : metrics[i].volume);
					double sum2 = sum * sum;
					double alignVec = 0.0, alignAxis = 0.0, div = 0.0;
					uint32_t cpt = 1, cptAlignVec = 0, cptAlignAxis = 0, cptDiv = 0;
					for (uint32_t idx = firsts[i]; idx < firsts[i + 1]; ++idx) {
						const uint32_t j = neighs[idx];
						if (j == std::numeric_limits<uint32_t>::max() || j >= n) continue;
						const double lj = safeLog(inputVolumes[j] > 0.0 ? inputVolumes[j] : metrics[j].volume);
						sum += lj; sum2 += lj * lj; ++cpt;
						const double ni = metrics[i].anisotropyVector.norm(), nj = metrics[j].anisotropyVector.norm();
						if (ni > std::numeric_limits<double>::epsilon() && nj > std::numeric_limits<double>::epsilon()) { alignVec += std::abs(metrics[i].anisotropyVector.dot(metrics[j].anisotropyVector) / (ni * nj)); ++cptAlignVec; }
						const double ai = metrics[i].principalAxis.norm(), aj = metrics[j].principalAxis.norm();
						if (ai > std::numeric_limits<double>::epsilon() && aj > std::numeric_limits<double>::epsilon()) { alignAxis += std::abs(metrics[i].principalAxis.dot(metrics[j].principalAxis) / (ai * aj)); ++cptAlignAxis; }
						Eigen::Vector3d dir = seeds[j] - seeds[i];
						const double d = dir.norm();
						if (d > std::numeric_limits<double>::epsilon()) { dir /= d; div += (metrics[j].anisotropyVector - metrics[i].anisotropyVector).dot(dir) / d; ++cptDiv; }
					}
					if (cpt > 1) {
						const double mean = sum / static_cast<double>(cpt);
						const double var = std::max(0.0, sum2 / static_cast<double>(cpt) - mean * mean);
						const double cv = (std::abs(mean) > std::numeric_limits<double>::epsilon()) ? std::sqrt(var) / std::abs(mean) : 0.0;
						out.values["localLogVolumeCV"][i] = finiteFloat(cv);
						out.values["cvLogVol"][i] = finiteFloat(cv);
						out.values["localLogVolumeZScore"][i] = finiteFloat(zscore(out.values["logVol"][i], mean, std::sqrt(var)));
					}
					out.values["localAnisotropyVectorAlignment"][i] = cptAlignVec > 0 ? finiteFloat(alignVec / static_cast<double>(cptAlignVec)) : 0.f;
					out.values["localPrincipalAxisAlignment"][i] = cptAlignAxis > 0 ? finiteFloat(alignAxis / static_cast<double>(cptAlignAxis)) : 0.f;
					out.values["anisotropyVectorDivergence"][i] = cptDiv > 0 ? finiteFloat(div / static_cast<double>(cptDiv)) : 0.f;
					out.values["anisotropyVectorConvergence"][i] = cptDiv > 0 ? finiteFloat(-div / static_cast<double>(cptDiv)) : 0.f;
				}
			}

			RunningStats stVoidScore, stLocalLogVolZ, stConvergence;
			for (size_t i = 0; i < n; ++i) {
				const bool isBorder = i < borderLocs.size() && borderLocs[i];
				if (isBorder) continue;
				stVoidScore.add(out.values["voidScore"][i]);
				stLocalLogVolZ.add(out.values["localLogVolumeZScore"][i]);
				stConvergence.add(out.values["anisotropyVectorConvergence"][i]);
			}
			const double meanVoidScore = stVoidScore.mean(), sdVoidScore = stVoidScore.sd();
			const double meanLocalLogVolZ = stLocalLogVolZ.mean(), sdLocalLogVolZ = stLocalLogVolZ.sd();
			const double meanConvergence = stConvergence.mean(), sdConvergence = stConvergence.sd();
			for (size_t i = 0; i < n; ++i) {
				const bool isBorder = i < borderLocs.size() && borderLocs[i];
				if (isBorder) {
					out.values["cavityRimScore"][i] = 0.f;
					continue;
				}
				out.values["cavityRimScore"][i] = finiteFloat(
					zscore(out.values["voidScore"][i], meanVoidScore, sdVoidScore) +
					zscore(out.values["localLogVolumeZScore"][i], meanLocalLogVolZ, sdLocalLogVolZ) +
					zscore(out.values["anisotropyVectorConvergence"][i], meanConvergence, sdConvergence));
			}
			return out;
		}
	}

	Voronoi3DCellFeatures::FeatureSet Voronoi3DCellFeatures::compute(
		const std::vector<Polyhedron_3_inexact>& polyhedrons,
		const std::vector<float>& volumes,
		const float* xs, const float* ys, const float* zs,
		const poca::core::MyArrayUInt32& neighbors,
		const std::vector<bool>& borderLocs)
	{
		const size_t n = polyhedrons.size();
		if (n == 0 || xs == nullptr || ys == nullptr || zs == nullptr) return Voronoi3DCellFeatures::FeatureSet();
		std::vector<CellMetrics> metrics(n);
		std::vector<Eigen::Vector3d> seeds(n);
		std::vector<double> inputVolumes(n, 0.0);
		for (size_t i = 0; i < n; ++i) {
			seeds[i] = Eigen::Vector3d(xs[i], ys[i], zs[i]);
			metrics[i] = computePolyhedronMetrics(polyhedrons[i], seeds[i], nullptr);
			inputVolumes[i] = i < volumes.size() ? static_cast<double>(volumes[i]) : metrics[i].volume;
		}
		return fillFeatureSet(metrics, inputVolumes, seeds, neighbors, borderLocs);
	}

	Voronoi3DCellFeatures::FeatureSet Voronoi3DCellFeatures::compute(
		const std::vector<Surface_mesh_3_double>& meshes,
		const std::vector<poca::core::Vec3mf>& seedsIn,
		const poca::core::MyArrayUInt32& neighbors,
		const std::vector<bool>& borderLocs,
		const poca::core::BoundingBox* clipBox)
	{
		const size_t n = std::min(meshes.size(), seedsIn.size());
		if (n == 0) return Voronoi3DCellFeatures::FeatureSet();
		std::vector<CellMetrics> metrics(n);
		std::vector<Eigen::Vector3d> seeds(n);
		std::vector<double> inputVolumes(n, 0.0);
		for (size_t i = 0; i < n; ++i) {
			seeds[i] = toEigen(seedsIn[i]);
			metrics[i] = computeMeshMetrics(meshes[i], seeds[i], clipBox);
			inputVolumes[i] = metrics[i].volume;
		}
		return fillFeatureSet(metrics, inputVolumes, seeds, neighbors, borderLocs);
	}

	Voronoi3DCellFeatures::FeatureSet Voronoi3DCellFeatures::compute(
		const std::vector<Surface_mesh_3_double>& meshes,
		const std::vector<poca::core::Vec3mf>& seedsIn,
		const poca::core::MyArrayUInt32& neighbors,
		const std::vector<uint32_t>& borderIndices,
		const poca::core::BoundingBox* clipBox)
	{
		std::vector<bool> borderLocs(meshes.size(), false);
		for (const uint32_t idx : borderIndices)
			if (idx < borderLocs.size()) borderLocs[idx] = true;
		return compute(meshes, seedsIn, neighbors, borderLocs, clipBox);
	}
}
