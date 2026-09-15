/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DetectionSetMultiObjectDisplayCommand.hpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*/

#ifndef DetectionSetMultiObjectDisplayCommand_h__
#define DetectionSetMultiObjectDisplayCommand_h__

#include <QtCore/QString>

#include <vector>

#include <General/Vec3.hpp>
#include <General/Vec4.hpp>
#include <glm/mat4x4.hpp>
#include <OpenGL/BasicDisplayCommand.hpp>
#include <OpenGL/GLBuffer.hpp>

class MyMultipleObject;
class DetectionSetDisplayCommand;

class DetectionSetMultiObjectDisplayCommand : public poca::opengl::BasicDisplayCommand {
public:
	DetectionSetMultiObjectDisplayCommand(MyMultipleObject*);
	DetectionSetMultiObjectDisplayCommand(const DetectionSetMultiObjectDisplayCommand&);
	~DetectionSetMultiObjectDisplayCommand();

	void execute(poca::core::CommandInfo*);
	void execute(poca::core::CommandInfo*, const poca::core::CommandExecutionContext&, poca::core::CommandExecutionResult&);
	poca::core::Command* copy();

	void freeGPUMemory();

protected:
	struct ObjectPointRange {
		size_t objectIndex = 0, first = 0, count = 0;
	};
	bool canBatch() const;
	bool rebuild();
	bool refreshTransformBuffers();
	bool updateFeatureBuffer();
	bool updateFeatureBuffer(const std::vector<size_t>&);
	bool updateHistogramState();
	bool refreshLutTexture();
	bool transformBuffersDirty() const;
	DetectionSetDisplayCommand* referenceDisplayCommand() const;
	void display(poca::opengl::Camera*, const bool, const bool, poca::core::CommandExecutionResult&);
	void drawElements(poca::opengl::Camera*, const bool, DetectionSetDisplayCommand*);

protected:
	MyMultipleObject* m_object;
	GLuint m_textureLutID;
	GLfloat m_minOriginalFeature, m_maxOriginalFeature, m_currentMinOriginalFeature, m_currentMaxOriginalFeature, m_actualValueFeature;
	bool m_isScaleLUT;

	poca::opengl::PointSingleGLBuffer<poca::core::Vec3mf> m_pointBuffer, m_normalBuffer;
	poca::opengl::FeatureSingleGLBuffer<float> m_featureBuffer;
	poca::opengl::PointSingleGLBuffer<poca::core::Color4D> m_colorBuffer;
	std::vector<ObjectPointRange> m_objectPointRanges;
	std::vector<glm::mat4> m_cachedTransforms;
};

#endif
