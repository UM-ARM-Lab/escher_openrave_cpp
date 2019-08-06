#include "Utilities.hpp"


// using namespace OpenRAVE;

DrawingHandler::DrawingHandler(OpenRAVE::EnvironmentBasePtr _penv):penv(_penv)
{
    foot_corners.resize(4);
    foot_corners[0] = OpenRAVE::RaveVector<OpenRAVE::dReal>(FOOT_HEIGHT/2,FOOT_WIDTH/2,0.01);
    foot_corners[1] = OpenRAVE::RaveVector<OpenRAVE::dReal>(-FOOT_HEIGHT/2,FOOT_WIDTH/2,0.01);
    foot_corners[2] = OpenRAVE::RaveVector<OpenRAVE::dReal>(-FOOT_HEIGHT/2,-FOOT_WIDTH/2,0.01);
    foot_corners[3] = OpenRAVE::RaveVector<OpenRAVE::dReal>(FOOT_HEIGHT/2,-FOOT_WIDTH/2,0.01);

    hand_corners.resize(4);
    hand_corners[0] = OpenRAVE::RaveVector<OpenRAVE::dReal>(-0.01,HAND_HEIGHT/2,HAND_WIDTH/2);
    hand_corners[1] = OpenRAVE::RaveVector<OpenRAVE::dReal>(-0.01,-HAND_HEIGHT/2,HAND_WIDTH/2);
    hand_corners[2] = OpenRAVE::RaveVector<OpenRAVE::dReal>(-0.01,-HAND_HEIGHT/2,-HAND_WIDTH/2);
    hand_corners[3] = OpenRAVE::RaveVector<OpenRAVE::dReal>(-0.01,HAND_HEIGHT/2,-HAND_WIDTH/2);
}

DrawingHandler::DrawingHandler(OpenRAVE::EnvironmentBasePtr _penv, std::shared_ptr<RobotProperties> _robot_properties):penv(_penv)
{
    float foot_h = _robot_properties->foot_h_;
    float foot_w = _robot_properties->foot_w_;
    float hand_h = _robot_properties->hand_h_;
    float hand_w = _robot_properties->hand_w_;

    foot_corners.resize(4);
    foot_corners[0] = OpenRAVE::RaveVector<OpenRAVE::dReal>(foot_h/2.0, -foot_w/2.0, 0.01);
    foot_corners[1] = OpenRAVE::RaveVector<OpenRAVE::dReal>(foot_h/2.0, foot_w/2.0, 0.01);
    foot_corners[2] = OpenRAVE::RaveVector<OpenRAVE::dReal>(-foot_h/2.0, foot_w/2.0, 0.01);
    foot_corners[3] = OpenRAVE::RaveVector<OpenRAVE::dReal>(-foot_h/2.0, -foot_w/2.0, 0.01);

    hand_corners.resize(4);
    hand_corners[0] = OpenRAVE::RaveVector<OpenRAVE::dReal>(-0.01, hand_h/2.0, hand_w/2.0);
    hand_corners[1] = OpenRAVE::RaveVector<OpenRAVE::dReal>(-0.01, -hand_h/2.0, hand_w/2.0);
    hand_corners[2] = OpenRAVE::RaveVector<OpenRAVE::dReal>(-0.01, -hand_h/2.0, -hand_w/2.0);
    hand_corners[3] = OpenRAVE::RaveVector<OpenRAVE::dReal>(-0.01, hand_h/2.0, -hand_w/2.0);
}

void DrawingHandler::ClearHandler()
{
    graphptrs.clear();
}

// void DrawingHandler::DrawBodyPath(Node* current) // Draw the upperbody path in thr door planning, postpone this implementation.(DrawPaths)
// {

// }

void DrawingHandler::DrawGridPath(std::shared_ptr<MapCell3D> current_state) // Draw the Dijkstra grid path, postpone implementation.
{
    std::shared_ptr<MapCell3D> c = current_state;

    while(true)
    {
        if(c->is_root_)
        {
            break;
        }

        GridPositions3D current_grid_position = c->getPositions();
        GridPositions3D parent_grid_position = c->parent_->getPositions();

        Translation3D current_position(current_grid_position[0], current_grid_position[1], 0.2);
        Translation3D parent_position(parent_grid_position[0], parent_grid_position[1], 0.2);

        DrawLineSegment(current_position, parent_position, {1,0,0,1});

        c = c->parent_;
    }
}

void DrawingHandler::DrawContactPath(std::shared_ptr<ContactState> current_state) // Draw the contact path given the final state(DrawStances)
{
    std::shared_ptr<ContactState> c = current_state;

    while(true)
    {
        DrawContacts(c);

        if(c->is_root_)
        {
            break;
        }

        c = c->parent_;
    }
}

void DrawingHandler::DrawTorsoPath(std::shared_ptr<TorsoPoseState> current_state)
{
    std::shared_ptr<TorsoPoseState> c = current_state;

    while(true)
    {
        if(c->is_root_)
        {
            break;
        }

        DrawLineSegment(c->pose_.getXYZ(), c->parent_->pose_.getXYZ(), {1,0,0,1});

        c = c->parent_;
    }
}

void DrawingHandler::DrawContacts(std::shared_ptr<ContactState> current_state) // Draw the contacts of one node(DrawStance)
{

    std::shared_ptr<Stance> current_stance = current_state->stances_vector_[0];

    // draw left foot pose
    if(current_stance->ee_contact_status_[ContactManipulator::L_LEG])
    {
        OpenRAVE::RaveTransformMatrix<OpenRAVE::dReal> left_foot_transform = current_stance->left_foot_pose_.GetRaveTransformMatrix();

        std::vector< OpenRAVE::RaveVector<float> > transformed_left_foot_corners(5);

        for(unsigned int i = 0; i < transformed_left_foot_corners.size(); i++)
        {
            // std::cout << foot_corners[i][0] << " " << foot_corners[i][1] << " " << foot_corners[i][2] << std::endl;
            transformed_left_foot_corners[i] = left_foot_transform * foot_corners[i%4];
        }

        graphptrs.push_back(penv->drawlinestrip(&(transformed_left_foot_corners[0].x), transformed_left_foot_corners.size(), sizeof(transformed_left_foot_corners[0]), 5, OpenRAVE::RaveVector<float>(1,0,0,1)));
        // DrawTransform(OpenRAVE::RaveTransform(left_foot_transform));
    }


    // draw right foot pose
    if(current_stance->ee_contact_status_[ContactManipulator::R_LEG])
    {
        OpenRAVE::RaveTransformMatrix<OpenRAVE::dReal> right_foot_transform = current_stance->right_foot_pose_.GetRaveTransformMatrix();

        std::vector< OpenRAVE::RaveVector<float> > transformed_right_foot_corners(5);

        for(unsigned int i = 0; i < transformed_right_foot_corners.size(); i++)
        {
            transformed_right_foot_corners[i] = right_foot_transform * foot_corners[i%4];
        }

        graphptrs.push_back(penv->drawlinestrip(&(transformed_right_foot_corners[0].x), transformed_right_foot_corners.size(), sizeof(transformed_right_foot_corners[0]), 5, OpenRAVE::RaveVector<float>(0,1,0,1)));
        // DrawTransform(OpenRAVE::RaveTransform(right_foot_transform));
    }

    // draw left hand pose
    if(current_stance->ee_contact_status_[ContactManipulator::L_ARM])
    {
        OpenRAVE::RaveTransformMatrix<OpenRAVE::dReal> left_hand_transform = current_stance->left_hand_pose_.GetRaveTransformMatrix();

        std::vector< OpenRAVE::RaveVector<float> > transformed_left_hand_corners(5);

        for(unsigned int i = 0; i < transformed_left_hand_corners.size(); i++)
        {
            transformed_left_hand_corners[i] = left_hand_transform * hand_corners[i%4];
        }

        graphptrs.push_back(penv->drawlinestrip(&(transformed_left_hand_corners[0].x), transformed_left_hand_corners.size(), sizeof(transformed_left_hand_corners[0]), 5, OpenRAVE::RaveVector<float>(0,0,1,1)));
        // DrawTransform(OpenRAVE::RaveTransform(left_hand_transform));
    }


    // draw right hand pose
    if(current_stance->ee_contact_status_[ContactManipulator::R_ARM])
    {
        OpenRAVE::RaveTransformMatrix<OpenRAVE::dReal> right_hand_transform = current_stance->right_hand_pose_.GetRaveTransformMatrix();

        std::vector< OpenRAVE::RaveVector<float> > transformed_right_hand_corners(5);

        for(unsigned int i = 0; i < transformed_right_hand_corners.size(); i++)
        {
            transformed_right_hand_corners[i] = right_hand_transform * hand_corners[i%4];
        }

        graphptrs.push_back(penv->drawlinestrip(&(transformed_right_hand_corners[0].x), transformed_right_hand_corners.size(), sizeof(transformed_right_hand_corners[0]), 5, OpenRAVE::RaveVector<float>(1,1,0,1)));
        // DrawTransform(OpenRAVE::RaveTransform(right_hand_transform));
    }
}

// void DrawingHandler::DrawContact(enum contact_type,contact_transform); // Draw one contact.(DrawContact)

void DrawingHandler::DrawLocation(OpenRAVE::RaveTransformMatrix<OpenRAVE::dReal> transform, OpenRAVE::RaveVector<float> color) // Draw a point at the location(DrawLocation)
{
    float trans_x_float = (float)transform.trans[0];
    graphptrs.push_back(penv->plot3(&(trans_x_float), 1, 0, 0.02, color, 1));
}

void DrawingHandler::DrawLocation(OpenRAVE::RaveVector<float> location, OpenRAVE::RaveVector<float> color) // Draw a point at the location(DrawLocation)
{
    graphptrs.push_back(penv->plot3(&(location.x), 1, 0, 0.02, color, 1));
}

void DrawingHandler::DrawLocation(Translation3D location, Vector3D color) // Draw a point at the location(DrawLocation)
{
    OpenRAVE::RaveVector<float> location_ravevector(location[0], location[1], location[2]);
    OpenRAVE::RaveVector<float> color_ravevector(color[0], color[1], color[2], 1);
    DrawLocation(location_ravevector, color_ravevector);
}

void DrawingHandler::DrawArrow(OpenRAVE::RaveVector<OpenRAVE::dReal> location, OpenRAVE::RaveVector<OpenRAVE::dReal> arrow, OpenRAVE::RaveVector<float> color) // Draw an arrow at the location(DrawArrow)
{
    graphptrs.push_back(penv->drawarrow(location, location+arrow, 0.005, color));
}

void DrawingHandler::DrawArrow(Translation3D location, Vector3D arrow, Vector3D color) // Draw an arrow at the location(DrawArrow)
{
    OpenRAVE::RaveVector<OpenRAVE::dReal> location_ravevector(location[0], location[1], location[2]);
    OpenRAVE::RaveVector<OpenRAVE::dReal> arrow_ravevector(arrow[0], arrow[1], arrow[2]);
    OpenRAVE::RaveVector<float> color_ravevector(color[0], color[1], color[2], 1);

    if(arrow[0] != 0 || arrow[1] != 0 || arrow[2] != 0)
    {
        graphptrs.push_back(penv->drawarrow(location_ravevector, location_ravevector+arrow_ravevector, 0.005, color_ravevector));
    }
}

void DrawingHandler::DrawTransform(OpenRAVE::RaveTransformMatrix<OpenRAVE::dReal> transform) // Draw the transform in 3 axes(DrawOrientation)
{
    OpenRAVE::RaveVector<OpenRAVE::dReal> from_vec = transform.trans;
    OpenRAVE::RaveVector<OpenRAVE::dReal> to_vec_x = from_vec + 0.2 * OpenRAVE::RaveVector<OpenRAVE::dReal>(transform.m[0],transform.m[4],transform.m[8]);
    OpenRAVE::RaveVector<OpenRAVE::dReal> to_vec_y = from_vec + 0.2 * OpenRAVE::RaveVector<OpenRAVE::dReal>(transform.m[1],transform.m[5],transform.m[9]);
    OpenRAVE::RaveVector<OpenRAVE::dReal> to_vec_z = from_vec + 0.2 * OpenRAVE::RaveVector<OpenRAVE::dReal>(transform.m[2],transform.m[6],transform.m[10]);

    graphptrs.push_back(penv->drawarrow(from_vec, to_vec_x, 0.005, OpenRAVE::RaveVector<float>(1, 0, 0)));
    graphptrs.push_back(penv->drawarrow(from_vec, to_vec_y, 0.005, OpenRAVE::RaveVector<float>(0, 1, 0)));
    graphptrs.push_back(penv->drawarrow(from_vec, to_vec_z, 0.005, OpenRAVE::RaveVector<float>(0, 0, 1)));
}

void DrawingHandler::DrawManipulatorPoses(OpenRAVE::RobotBasePtr robot) // Draw the manipulator poses given robot object(DrawManipulatorPoses)
{
    std::vector< boost::shared_ptr<OpenRAVE::RobotBase::Manipulator> > manipulators = robot->GetManipulators();

    for(unsigned int i = 0; i < manipulators.size(); i++)
    {
        boost::shared_ptr<OpenRAVE::RobotBase::Manipulator> manipulator = manipulators[i];
        DrawTransform(OpenRAVE::RaveTransformMatrix<OpenRAVE::dReal>(manipulator->GetTransform()));
    }
}

void DrawingHandler::DrawGoalRegion(OpenRAVE::RaveTransformMatrix<OpenRAVE::dReal> transform, double radius) // Draw the region with given transform and radius.(DrawRegion)
{
    OpenRAVE::RaveVector<OpenRAVE::dReal> center = transform.trans;
    OpenRAVE::RaveVector<OpenRAVE::dReal> x_vector = OpenRAVE::RaveVector<OpenRAVE::dReal>(transform.m[0],transform.m[4],transform.m[8]);
    OpenRAVE::RaveVector<OpenRAVE::dReal> y_vector = OpenRAVE::RaveVector<OpenRAVE::dReal>(transform.m[1],transform.m[5],transform.m[9]);

    DrawRegion(center, OpenRAVE::RaveVector<OpenRAVE::dReal>(0,0,1), radius, 5.0); // Draw the region with given center, normal and radius.(DrawContactRegion)

    std::vector< OpenRAVE::RaveVector<OpenRAVE::dReal> > arrow_points(5);

    arrow_points[0] = center - radius*2.0/3.0 * x_vector;
    arrow_points[1] = center + radius*2.0/3.0 * x_vector;
    arrow_points[2] = center + radius/2.0 * y_vector;
    arrow_points[3] = center + radius*2.0/3.0 * x_vector;
    arrow_points[4] = center - radius/2.0 * y_vector;

    float arrow_point_x_float = (float)arrow_points[0].x;

    graphptrs.push_back(penv->drawlinestrip(&(arrow_point_x_float),arrow_points.size(),sizeof(arrow_points[0]),5.0,OpenRAVE::RaveVector<float>(0,0,0,0)));
}

void DrawingHandler::DrawRegion(OpenRAVE::RaveVector<OpenRAVE::dReal> center, OpenRAVE::RaveVector<OpenRAVE::dReal> normal, double radius, float line_width) // Draw the region with given center, normal and radius.(DrawContactRegion)
{
    OpenRAVE::RaveVector<OpenRAVE::dReal> x_vector;
    if(normal.x == 0 && normal.y == 0)
    {
        x_vector = OpenRAVE::RaveVector<OpenRAVE::dReal>(1,0,0);
    }
    else
    {
        x_vector = OpenRAVE::RaveVector<OpenRAVE::dReal>(normal.y,-normal.x,0);
        x_vector = x_vector.normalize3();
    }

    OpenRAVE::RaveVector<OpenRAVE::dReal> y_vector = normal.cross(x_vector);

    std::vector<OpenRAVE::RaveVector<float> >* region_boundary_points_float = new std::vector<OpenRAVE::RaveVector<float > >;
    // std::vector< OpenRAVE::RaveVector<float> > region_boundary_points_float;
    region_boundary_points_float->resize(37);
    OpenRAVE::RaveVector<OpenRAVE::dReal> region_boundary_point;

    for(unsigned int i = 0; i < 37; i++)
    {
        region_boundary_point = center + std::cos(i*10*(M_PI / 180))*radius*x_vector + std::sin(i*10*(M_PI / 180))*radius*y_vector;
        (*region_boundary_points_float)[i] = {region_boundary_point.x, region_boundary_point.y,
                                              region_boundary_point.z, region_boundary_point.w}; // truncate OpenRAVE::dReals to floats
    }

    region_boundary_pointers.push_back(region_boundary_points_float);

    graphptrs.push_back(penv->drawlinestrip((float *)region_boundary_points_float->data(),region_boundary_points_float->size(),sizeof((*region_boundary_points_float)[0]),line_width,OpenRAVE::RaveVector<float>(0,0,0,1)));
    graphptrs.push_back(penv->drawarrow(center, center + 0.1 * normal, 0.005, OpenRAVE::RaveVector<float>(1,0,0, 1)));
}

void DrawingHandler::DrawLineSegment(Translation3D from_vec, Translation3D to_vec, std::array<float,4> color)
{
    OpenRAVE::RaveVector<OpenRAVE::dReal> from_vec_ravevector = OpenRAVE::RaveVector<OpenRAVE::dReal>(from_vec[0],from_vec[1],from_vec[2]);
    OpenRAVE::RaveVector<OpenRAVE::dReal> to_vec_ravevector = OpenRAVE::RaveVector<OpenRAVE::dReal>(to_vec[0],to_vec[1],to_vec[2]);
    OpenRAVE::RaveVector<float> color_ravevector = OpenRAVE::RaveVector<float>(color[0],color[1],color[2],color[3]);

    DrawLineSegment(from_vec_ravevector, to_vec_ravevector, color_ravevector);
}

void DrawingHandler::DrawLineSegment(OpenRAVE::RaveVector<OpenRAVE::dReal> from_vec, OpenRAVE::RaveVector<OpenRAVE::dReal> to_vec, OpenRAVE::RaveVector<float> color) // Draw a line segment given two ends(DrawLineStrips)
{
    std::vector<float> line_endpoints = {(float)from_vec[0],(float)from_vec[1],(float)from_vec[2],(float)to_vec[0],(float)to_vec[1],(float)to_vec[2]};
    graphptrs.push_back(penv->drawlinestrip(&(line_endpoints[0]),line_endpoints.size()/3,sizeof(line_endpoints[0])*3,3.0,color));
}

// void DrawingHandler::DrawSurface(TrimeshSurface trimesh) // Draw the trimesh surface.(DrawSurface)
// {
//     float r = static_cast<float> (std::rand()) / static_cast<float> (RAND_MAX);
//     float g = static_cast<float> (std::rand()) / static_cast<float> (RAND_MAX);
//     float b = static_cast<float> (std::rand()) / static_cast<float> (RAND_MAX);

//     float total_rgb = std::sqrt(r*r+g*g+b*b);
//     r = r/total_rgb;
//     g = g/total_rgb;
//     b = b/total_rgb;

//     // DrawOrientation(trimesh.transform_matrix);

//     // for boundary in struct.boundaries:

//     //     boundaries_point = np.zeros((2,3),dtype=float)
//     //     boundaries_point[0:1,:] = np.atleast_2d(np.array(struct.vertices[boundary[0]]))
//     //     boundaries_point[1:2,:] = np.atleast_2d(np.array(struct.vertices[boundary[1]]))

//     //     draw_handles.append(env.drawlinestrip(points = boundaries_point,linewidth = 5.0,colors = np.array((r,g,b))))

//     // graphptrs.push_back(penv->drawtrimesh())
//     // _penv->drawtrimesh(&vpoints[0],sizeof(float)*3,pindices,numTriangles,OpenRAVE::RaveVector<float>(1,0.5,0.5,1))
//     // draw_handles.append(env.drawtrimesh(trimesh.kinbody->GetLinks()[0].GetCollisionData().vertices,struct.kinbody.GetLinks()[0].GetCollisionData().indices,OpenRAVE::RaveVector<float>(r,g,b,1.0)))

// }

// void DrawingHandler::DrawObjectPath(Node* current) // Draw the manipulated object path, postpone implementation.(DrawObjectPath)
// {

// }

DrawingHandler::~DrawingHandler()
{
    for(int i = 0; i < region_boundary_pointers.size(); ++i)
    {
        delete region_boundary_pointers[i];
    }
}