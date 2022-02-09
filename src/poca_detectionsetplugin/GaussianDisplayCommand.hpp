/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      GaussianDisplayCommand.hpp
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

#ifndef GaussianDisplayCommand_h__
#define GaussianDisplayCommand_h__

#include <General/Command.hpp>
#include <Geometry/DetectionSet.hpp>
#include <OpenGL/Camera.hpp>
#include <OpenGL/GLBuffer.hpp>

class QOpenGLFramebufferObject;
class DetectionSetDisplayCommand;

class GaussianDisplayCommand : public poca::core::Command {
public:
	GaussianDisplayCommand(poca::geometry::DetectionSet*);
	GaussianDisplayCommand(const GaussianDisplayCommand&);
	~GaussianDisplayCommand();

	void execute(poca::core::CommandInfo*);
	void freeGPUMemory();

	poca::core::Command* copy();
	const poca::core::CommandInfos saveParameters() const { return poca::core::CommandInfos(); }
	poca::core::CommandInfo createCommand(const std::string&, const nlohmann::json&);

protected:
	void display(poca::opengl::Camera*, const bool);
	void createDisplay();
	void generateFeatureBuffer(poca::core::HistogramInterface* = NULL);
	void sortWrtCameraPosition(const glm::vec3&, const glm::vec3&);

protected:
	poca::geometry::DetectionSet* m_dset;
	float m_minX;
	std::string m_nameSigmaXY = "sigmaXY", m_nameSigmaZ = "sigmaZ";

	GLuint m_textureLutID;
	GLfloat m_minOriginalFeature, m_maxOriginalFeature, m_actualValueFeature, m_alphaValue;
	poca::opengl::PointSingleGLBuffer <poca::core::Vec3mf> m_pointBuffer, m_uncertaintiesBuffer;
	poca::opengl::FeatureSingleGLBuffer <float> m_featureBuffer;

	DetectionSetDisplayCommand* m_dc;
};

#endif

