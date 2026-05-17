/*
* Software:  PoCA: Point Cloud Analyst
*
* File:      DelaunayTriangulationMultiObjectDisplayCommand.hpp
*
* Copyright: Florian Levet (2020-2025)
*
* License:   LGPL v3
*/

#ifndef DelaunayTriangulationMultiObjectDisplayCommand_h__
#define DelaunayTriangulationMultiObjectDisplayCommand_h__

#include <QtCore/QString>

#include <General/Vec3.hpp>
#include <OpenGL/BasicDisplayCommand.hpp>
#include <OpenGL/GLBuffer.hpp>

class MyMultipleObject;
class DelaunayTriangulationDisplayCommand;

class DelaunayTriangulationMultiObjectDisplayCommand : public poca::opengl::BasicDisplayCommand {
public:
	DelaunayTriangulationMultiObjectDisplayCommand(MyMultipleObject*);
	DelaunayTriangulationMultiObjectDisplayCommand(const DelaunayTriangulationMultiObjectDisplayCommand&);
	~DelaunayTriangulationMultiObjectDisplayCommand();

	std::vector<poca::core::CommandSpec> commandSpecs() const;
	void execute(poca::core::CommandInfo*);
	void execute(poca::core::CommandInfo*, const poca::core::CommandExecutionContext&, poca::core::CommandExecutionResult&);
	poca::core::Command* copy();

	void freeGPUMemory();

protected:
	bool canBatch() const;
	bool rebuild();
	bool updateFeatureBuffer();
	DelaunayTriangulationDisplayCommand* referenceDisplayCommand() const;
	void display(poca::opengl::Camera*, const bool, poca::core::CommandExecutionResult&);
	void drawElements(poca::opengl::Camera*, DelaunayTriangulationDisplayCommand*);
	void generateBoundingBoxSelection(const int);

protected:
	MyMultipleObject* m_object;
	GLuint m_textureLutID;
	GLfloat m_minOriginalFeature, m_maxOriginalFeature, m_actualValueFeature;

	poca::opengl::TriangleGLBuffer<poca::core::Vec3mf> m_triangleBuffer;
	poca::opengl::TriangleGLBuffer<float> m_featureBuffer;
	poca::opengl::LineGLBuffer<poca::core::Vec3mf> m_boundingBoxSelection;
};

#endif
