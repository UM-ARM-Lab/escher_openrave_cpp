#ifndef DRAWING_HPP
#define DRAWING_HPP

// #include <openrave/plugin.h>
// #include <boost/bind.hpp>

class ContactState;
class RobotProperties;
class TorsoPoseState;
class MapCell3D;

class DrawingHandler{
	std::vector< OpenRAVE::GraphHandlePtr > graphptrs;
	std::vector< std::vector<OpenRAVE::RaveVector<float > >* > region_boundary_pointers;
	OpenRAVE::EnvironmentBasePtr penv;
	std::vector< OpenRAVE::RaveVector<float> > foot_corners;
	std::vector< OpenRAVE::RaveVector<float> > hand_corners;
public:
	DrawingHandler(OpenRAVE::EnvironmentBasePtr _penv);
	DrawingHandler(OpenRAVE::EnvironmentBasePtr _penv, std::shared_ptr<RobotProperties> _robot_properties);
	void ClearHandler();
	// void DrawBodyPath(Node* current); // Draw the upperbody path in thr door planning, postpone this implementation.(DrawPaths)
	void DrawGridPath(std::shared_ptr<MapCell3D> current_state); // Draw the Dijkstra grid path
	void DrawContactPath(std::shared_ptr<ContactState> current_state); // Draw the contact path given the final state(DrawStances)
	void DrawTorsoPath(std::shared_ptr<TorsoPoseState> current_state);
	void DrawContacts(std::shared_ptr<ContactState> current_state); // Draw the contacts of one node(DrawStance)
	// void DrawContact(enum contact_type,contact_transform); // Draw one contact.(DrawContact)
	void DrawLocation(OpenRAVE::RaveTransformMatrix<OpenRAVE::dReal> transform, OpenRAVE::RaveVector<float> color); // Draw a point at the location(DrawLocation)
	void DrawLocation(OpenRAVE::RaveVector<float> location, OpenRAVE::RaveVector<float> color); // Draw a point at the location(DrawLocation)
	void DrawLocation(Translation3D location, Vector3D color); // Draw a point at the location(DrawLocation)
	void DrawArrow(OpenRAVE::RaveVector<OpenRAVE::dReal> location, OpenRAVE::RaveVector<OpenRAVE::dReal> arrow, OpenRAVE::RaveVector<float> color); // Draw an arrow at the location(DrawArrow)
	void DrawArrow(Translation3D location, Vector3D arrow, Vector3D color); // Draw an arrow at the location(DrawArrow)
	void DrawTransform(OpenRAVE::RaveTransformMatrix<OpenRAVE::dReal> transform); // Draw the transform in 3 axes(DrawOrientation)
	void DrawManipulatorPoses(OpenRAVE::RobotBasePtr robot); // Draw the manipulator poses given robot object(DrawManipulatorPoses)
	void DrawGoalRegion(OpenRAVE::RaveTransformMatrix<OpenRAVE::dReal> transform, double radius); // Draw the region with given transform and radius.(DrawRegion)
	void DrawRegion(OpenRAVE::RaveVector<OpenRAVE::dReal> center, OpenRAVE::RaveVector<OpenRAVE::dReal> normal, double radius, float line_width); // Draw the region with given center, normal and radius.(DrawContactRegion)

	void DrawLineSegment(Translation3D from_vec, Translation3D to_vec, std::array<float,4> color = {0,0,0,1}); // Draw a line segment given two ends(DrawLineStrips)
	void DrawLineSegment(OpenRAVE::RaveVector<OpenRAVE::dReal> from_vec, OpenRAVE::RaveVector<OpenRAVE::dReal> to_vec, OpenRAVE::RaveVector<float> color = OpenRAVE::RaveVector<float>(0,0,0,1)); // Draw a line segment given two ends(DrawLineStrips)
	// void DrawSurface(Tri_mesh trimesh); // Draw the trimesh surface.(DrawSurface)
	// void DrawObjectPath(Node* current); // Draw the manipulated object path, postpone implementation.(DrawObjectPath)
	~DrawingHandler();
};

#endif
