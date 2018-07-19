

#include "Utilities.hpp"

#include <qhull/qhull.h>
#include <qhull/mem.h>
#include <qhull/qset.h>
#include <qhull/geom.h>
#include <qhull/merge.h>
#include <qhull/poly.h>
#include <qhull/io.h>
#include <qhull/stat.h>

/// function for computing a 6D convex hull
int convexHull6D(coordT* pointsIn, int numPointsIn, std::vector< std::vector<double> >& facet_coefficients);

/// TODO: Document
void GetSupportPointsForLink(OpenRAVE::RobotBase::LinkPtr p_link, OpenRAVE::Vector tool_dir, OpenRAVE::Transform result_tf, std::vector<OpenRAVE::Vector>& contacts);
std::vector<OpenRAVE::Vector> GetSupportPoints(OpenRAVE::RobotBase::ManipulatorPtr p_manip);

void GetFrictionCone(OpenRAVE::Vector &center, OpenRAVE::Vector &direction, OpenRAVE::dReal mu, NEWMAT::Matrix *mat, int offset_r, int offset_c, OpenRAVE::Transform temp_tf);
void GetASurf(OpenRAVE::RobotBase::ManipulatorPtr p_manip, OpenRAVE::Transform cone_tf, NEWMAT::Matrix *mat, int offset_r);
void GetAStance(OpenRAVE::Transform cone_tf, NEWMAT::Matrix* mat, int offset_r);

/// compute the surface support cone for the given manipulator
NEWMAT::ReturnMatrix GetSurfaceCone(string& manipname, OpenRAVE::dReal mu);

/// compute the GIWC for giwc stability
NEWMAT::ReturnMatrix GetGIWCSpanForm(std::vector<std::string>& manip_ids, std::vector<OpenRAVE::dReal>& friction_coeffs);
void GetGIWC(std::vector<std::string>& manip_ids, std::vector<OpenRAVE::dReal>& friction_coeffs, std::vector<OpenRAVE::dReal>& ikparams);