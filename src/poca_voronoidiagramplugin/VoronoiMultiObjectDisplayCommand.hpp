/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      VoronoiMultiObjectDisplayCommand.hpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*/

#ifndef VoronoiMultiObjectDisplayCommand_h__
#define VoronoiMultiObjectDisplayCommand_h__

#include <QtCore/QString>

#include <OpenGL/BasicDisplayCommand.hpp>
#include <OpenGL/GLBuffer.hpp>

class MyMultipleObject;

class VoronoiDiagramDisplayCommand;

class VoronoiMultiObjectDisplayCommand : public poca::opengl::BasicDisplayCommand {
public:
	VoronoiMultiObjectDisplayCommand(MyMultipleObject*);
	VoronoiMultiObjectDisplayCommand(const VoronoiMultiObjectDisplayCommand&);
	~VoronoiMultiObjectDisplayCommand();

	std::vector<poca::core::CommandSpec> commandSpecs() const;
	void execute(poca::core::CommandInfo*);
	void execute(poca::core::CommandInfo*, const poca::core::CommandExecutionContext&, poca::core::CommandExecutionResult&);
	poca::core::Command* copy();

	void freeGPUMemory();

protected:
	bool updateFeatureBuffers();
	void display(poca::opengl::Camera*, const bool, poca::core::CommandExecutionResult&);
	void drawElements(poca::opengl::Camera*, VoronoiDiagramDisplayCommand*);

	bool rebuild();
	bool canBatch() const;
	VoronoiDiagramDisplayCommand* referenceDisplayCommand() const;
	void generateBoundingBoxSelection(const int);

protected:
	MyMultipleObject* m_object;

	bool m_hasCells, m_is3D;
	GLuint m_textureLutID;
	GLfloat m_minOriginalFeature, m_maxOriginalFeature;

	poca::opengl::PointSingleGLBuffer<poca::core::Vec3mf> m_pointBuffer;
	poca::opengl::FeatureSingleGLBuffer<float> m_locsFeatureBuffer;

	poca::opengl::TriangleGLBuffer<poca::core::Vec3mf> m_triangleBuffer;
	poca::opengl::TriangleGLBuffer<float> m_triangleFeatureBuffer;
	poca::opengl::LineSingleGLBuffer<poca::core::Vec3mf> m_lineBuffer, m_lineNormalBuffer;
	poca::opengl::LineSingleGLBuffer<float> m_lineFeatureBuffer;
	poca::opengl::LineGLBuffer<poca::core::Vec3mf> m_boundingBoxSelection;
};

#endif
