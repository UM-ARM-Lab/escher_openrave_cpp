#include "Utilities.hpp"

std::map<string,NEWMAT::Matrix> _computed_contact_surface_cones;

int convexHull6D(coordT* pointsIn, int numPointsIn, std::vector< std::vector<double> >& facet_coefficients) {

    char flags[250];
    int exitcode;
    facetT *facet, *newFacet;
    int curlong, totlong;
    facet_coefficients.clear();

    float min_value = 0.000001;

    sprintf (flags, "qhull QJ Pp s Tc ");

    // cout<<"input points."<<endl;
    // for(int i = 0; i < numPointsIn; i++)
    // {
    //     cout<<pointsIn[i*6+0]<<" "<<pointsIn[i*6+1]<<" "<<pointsIn[i*6+2]<<" "<<pointsIn[i*6+3]<<" "<<pointsIn[i*6+4]<<" "<<pointsIn[i*6+5]<<endl;
    // }

    exitcode= qh_new_qhull (6, numPointsIn, pointsIn, false, flags, NULL, stderr);

    FORALLfacets {
        facet->seen = 0;
    }

    facet = qh facet_list;
    int facet_numbers = qh num_facets;

    set<string> facet_hash_set;
    stringstream coeff_string;
    int rounded_coeff;

    // cout<<"facet parameters"<<endl;

    for(int i = 0; i < facet_numbers; i++)
    {
        // get the facet equations.
        coordT* normal = facet->normal;
        coordT offset = facet->offset;

        string hash = "";

        if(fabs(offset) < 0.0001)
        {
            std::vector<double> coeff(7);
            if(fabs(offset) < min_value)
            {
                coeff[0] = 0;
            }
            else
            {
                coeff[0] = offset;
            }
            rounded_coeff = round(coeff[0]*1000);
            if(rounded_coeff == 0) rounded_coeff = 0; // eliminate negative 0
            coeff_string << rounded_coeff;
            hash = hash + coeff_string.str();
            coeff_string.str(std::string());
            coeff_string.clear();

            for(int i = 0; i < 6; i++)
            {
                if(fabs(normal[i]) < min_value)
                {
                    coeff[i+1] = 0;
                }
                else
                {
                    coeff[i+1] = normal[i];
                }
                rounded_coeff = round(coeff[i+1]*1000);
                if(rounded_coeff == 0) rounded_coeff = 0; // eliminate negative 0
                coeff_string << rounded_coeff;
                hash = hash + "," + coeff_string.str();
                coeff_string.str(std::string());
                coeff_string.clear();
            }

            if(facet_hash_set.find(hash) == facet_hash_set.end())
            {
                facet_coefficients.push_back(coeff);
                facet_hash_set.insert(hash);
                // cout<<hash<<endl;
                // cout<<coeff[0]<<" "<<coeff[1]<<" "<<coeff[2]<<" "<<coeff[3]<<" "<<coeff[4]<<" "<<coeff[5]<<" "<<coeff[6]<<endl;
            }

        }
        if(i != facet_numbers-1)
        {
            newFacet = (facetT*)facet->next;
            facet = newFacet;
        }

    }

    qh_freeqhull(!qh_ALL);
    qh_memfreeshort (&curlong, &totlong);

    return exitcode;
}

/// Returns the support points relative to the world frame
void GetSupportPointsForLink(OpenRAVE::RobotBase::LinkPtr p_link, OpenRAVE::Vector tool_dir, OpenRAVE::Transform result_tf, std::vector<OpenRAVE::Vector>& contacts) {
    OpenRAVE::Transform tf = result_tf.inverse() * p_link->GetTransform();

    OpenRAVE::AABB aabb = p_link->ComputeLocalAABB();

    // If any extent is 0, the link has no volume and is assumed to be a virtual link
    if (aabb.extents.x <= 0 || aabb.extents.y <= 0 || aabb.extents.z <= 0) {
        return;
    }

    if(strcmp(p_link->GetName().c_str(), "l_foot") != 0 &&
       strcmp(p_link->GetName().c_str(), "r_foot") != 0 &&
       strcmp(p_link->GetName().c_str(), "l_palm") != 0 &&
       strcmp(p_link->GetName().c_str(), "r_palm") != 0 &&
       strcmp(p_link->GetName().c_str(), "L_ANKLE_AA") != 0 &&
       strcmp(p_link->GetName().c_str(), "R_ANKLE_AA") != 0)
    {
        return;
    }

    // Iterates over the 8 combinations of (+ or -) for each of the 3 dimensions
    for (int neg_mask = 0; neg_mask < 8; neg_mask++) {
        OpenRAVE::Vector contact;
        bool is_invalid = false;

        // Iterate over x, y, and z (compiler probably unrolls this loop)
        for (int axis = 0; axis < 3; axis++) {
            bool neg = !!(neg_mask&(1<<axis));
            // A point will be "invalid" if it is opposite the local tool direction
            is_invalid = is_invalid || (neg && (tool_dir[axis] > 0.85)) || (!neg && (tool_dir[axis] < -0.85));
            if (is_invalid) break;

            contact[axis] = aabb.pos[axis] + (neg ? -1 : 1)*aabb.extents[axis];
        }

        if (!is_invalid) {
            OpenRAVE::Vector contact_t = tf * contact;
            contacts.push_back(contact_t);

//            return; //! TEMP
        }
    }

}

/// Returns the support points of the given manipulator in the WORLD frame
std::vector<OpenRAVE::Vector> GetSupportPoints(OpenRAVE::RobotBase::ManipulatorPtr p_manip) {
    // Get all rigidly attached links -- in my case, the end effector link is a virtual
    // link with 0 volume. The actual physical ee link is rigidly attached to it.
    std::vector<OpenRAVE::RobotBase::LinkPtr> attached_links;
    p_manip->GetEndEffector()->GetRigidlyAttachedLinks(attached_links);

    // Other manipulator info
    OpenRAVE::Transform world_to_manip = p_manip->GetTransform().inverse();
    OpenRAVE::Vector tool_dir = p_manip->GetLocalToolDirection();

    std::vector<OpenRAVE::Vector> contacts;
    for (int i = 0; i < attached_links.size(); i++) {
        const char* link_name = attached_links[i]->GetName().c_str();

        // Transforms the tool_dir into the link frame
        GetSupportPointsForLink(attached_links[i], world_to_manip * attached_links[i]->GetTransform() * tool_dir, p_manip->GetTransform(), contacts);

    }
    return contacts;
}

const int CONE_DISCRETIZATION_RESOLUTION = 4;
void GetFrictionCone(OpenRAVE::Vector& center, OpenRAVE::Vector& direction, OpenRAVE::dReal mu, NEWMAT::Matrix* mat, int offset_r, int offset_c, OpenRAVE::Transform temp_tf) {
    // This sets `a` (for axis) to the index of `direction` that is nonzero, sets `c` (cos) to the first index that
    // is zero, and `s` (sin) to the second that is zero. Formulas derived from a truth table.
    // NOTE: 1-based indexing
    int a = direction[0] ? 1 : direction[1] ? 2 : 3;
    int c = 1 + (a == 1); // 1 if `a` is not 1, 2 otherwise.
    int s = 3 - (a == 3); // 3 if `a` is not 3, 2 otherwise

    OpenRAVE::dReal step = M_PI * 2.0 / CONE_DISCRETIZATION_RESOLUTION;
    OpenRAVE::dReal angle = 0;
    // NOTE 1-based indexing
    for (int i = 1; i <= CONE_DISCRETIZATION_RESOLUTION; i++) {
        // a-column will be -1 or 1. The -1 multiplication is because friction force occurs in the opposite direction
        // (*mat)(offset_r + i, offset_c + a) = center[a-1] + direction[a-1] * -1;
        // (*mat)(offset_r + i, offset_c + s) = center[s-1] + round(mu * sin(angle) * 10000) * 0.0001;
        // (*mat)(offset_r + i, offset_c + c) = center[c-1] + round(mu * cos(angle) * 10000) * 0.0001;
        (*mat)(offset_r + i, offset_c + a) = direction[a-1] * -1;
        (*mat)(offset_r + i, offset_c + s) = round(mu * sin(angle) * 10000) * 0.0001;
        (*mat)(offset_r + i, offset_c + c) = round(mu * cos(angle) * 10000) * 0.0001;
        angle += step;
    }

}

void GetASurf(OpenRAVE::RobotBase::ManipulatorPtr p_manip, OpenRAVE::Transform cone_to_manip, NEWMAT::Matrix *mat, int offset_r) {
    OpenRAVE::Transform manip_to_world = p_manip->GetTransform(); // For testing
    OpenRAVE::TransformMatrix tf_matrix = OpenRAVE::TransformMatrix(cone_to_manip); //TransformMatrix(cone_to_manip);

    // First 3 columns are just the tf_matrix
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            (*mat)(offset_r + r + 1, c + 1) =  tf_matrix.m[ r*4 + c ];
        }
    }

    // (Notes originally written for Python)
    // To calculate v, we take the origin of the friction cone in the world frame with cone_tf.trans,
    // then transform it to the surface frame by multipling by T_W_s = T_s_W.I. This gives the displacement from
    // surface frame to contact frame. To get the displacement from contact frame to surface frame, it is multiplied
    // by -1 to reverse direction b/c disp(a, b) = -disp(b, a).
    OpenRAVE::Vector v = (cone_to_manip.trans);
    v *= -1;
    // NOTE The above math messes up the 4th element of v, which must be 1 for affine transforms
    v[3] = 1;

    // Last 3 columns are the cross-product-equivalent matrix for r
    (*mat)(offset_r + 1, 4) =  0.0;
    (*mat)(offset_r + 1, 5) = -v.z;
    (*mat)(offset_r + 1, 6) =  v.y;
    (*mat)(offset_r + 2, 4) =  v.z;
    (*mat)(offset_r + 2, 5) =  0.0;
    (*mat)(offset_r + 2, 6) = -v.x;
    (*mat)(offset_r + 3, 4) = -v.y;
    (*mat)(offset_r + 3, 5) =  v.x;
    (*mat)(offset_r + 3, 6) =  0.0;
}

void GetAStance(OpenRAVE::Transform tf, NEWMAT::Matrix* mat, int offset_r) {
    // Note: This AStance computes the transpose of the AStance from the GIWC paper,
    // because of the way the cone representation is done here
    // TransformMatrix m = TransformMatrix(tf.inverse());
    OpenRAVE::TransformMatrix m = OpenRAVE::TransformMatrix(tf);

    // Create -R matrix
    NEWMAT::Matrix negR_T(3, 3);
    negR_T << -m.m[0] << -m.m[4] << -m.m[8]
           << -m.m[1] << -m.m[5] << -m.m[9]
           << -m.m[2] << -m.m[6] << -m.m[10];

    // Create the transpose of the cross-product-equivalent matrix of the transform's translation component
    NEWMAT::Matrix crossP_T(3, 3);
    crossP_T <<  0.0        <<  tf.trans.z << -tf.trans.y
             << -tf.trans.z <<  0.0        <<  tf.trans.x
             <<  tf.trans.y << -tf.trans.x <<  0.0       ;

    // Create TRANSPOSE OF matrix [    -R        0 ]
    //                            [ [p]x * -R   -R ]
    (*mat).SubMatrix(offset_r + 1, offset_r + 3, 1, 3) = negR_T;
    // Computes transpose of multiplication by using (negR * crossP)_T = negR_T * crossP_T
    (*mat).SubMatrix(offset_r + 1, offset_r + 3, 4, 6) = negR_T * crossP_T;
    (*mat).SubMatrix(offset_r + 4, offset_r + 6, 1, 3) = 0.0;
    (*mat).SubMatrix(offset_r + 4, offset_r + 6, 4, 6) = negR_T;

    // for(int i = 1; i <= mat->Nrows(); i++)
    // {
    //     for(int j = 1; j <= mat->Ncols(); j++)
    //     {
    //         (*mat)(i,j) = 0.001 * round((*mat)(i,j) * 1000.0);
    //     }
    // }
    negR_T.ReleaseAndDelete();
    crossP_T.ReleaseAndDelete();
}

NEWMAT::ReturnMatrix GetSurfaceCone(OpenRAVE::RobotBasePtr robot, string& manipname, OpenRAVE::dReal mu) {

    if(_computed_contact_surface_cones.count(manipname) != 0){
        return _computed_contact_surface_cones.find(manipname)->second;
    }
    else{
        OpenRAVE::RobotBase::ManipulatorPtr p_manip = robot->GetManipulator(manipname);
        OpenRAVE::Vector manip_dir = p_manip->GetLocalToolDirection();

        std::vector<OpenRAVE::Vector> support_points = GetSupportPoints(p_manip);
        int num_points = support_points.size();

        // Calculate combined friction cone matrix
        int rows = CONE_DISCRETIZATION_RESOLUTION*num_points;
        int cols = 3*num_points;

        NEWMAT::Matrix f_cones_diagonal(rows, cols);
        f_cones_diagonal = 0.0; // All non-filled spots should be 0

        // Fill the diagonals
        for (int i = 0; i < num_points; i++) {
            GetFrictionCone(support_points[i], manip_dir, mu, &f_cones_diagonal, i*CONE_DISCRETIZATION_RESOLUTION, i*3, p_manip->GetTransform());
        }

        // Calculate A_surf matrix
        NEWMAT::Matrix a_surf_stacked(cols, 6); // TODO: Rename `cols` because it's the rows here

        for (int i = 0; i < num_points; i++) {
            // Cone transform has no rotation relative to the manipulator's transform and is translated by
            // the vector contained in support_points
            OpenRAVE::Transform cone_tf;
            cone_tf.trans = support_points[i];
            GetASurf(p_manip, cone_tf, &a_surf_stacked, 3*i);
        }

        NEWMAT::Matrix mat = f_cones_diagonal * a_surf_stacked; // Dot product

        // omit redundant rows, useless, no redundant rows
        dd_ErrorType err;
        dd_MatrixPtr contact_span_cdd = dd_CreateMatrix(mat.Nrows(), mat.Ncols()+1);
        for (int r = 0; r < mat.Nrows(); r++) {
        // First element of each row indicates whether it's a point or ray. These are all rays, indicated by 0.
            dd_set_si(contact_span_cdd->matrix[r][0], 0.0);
            for (int c = 0; c < mat.Ncols(); c++) {
                dd_set_si(contact_span_cdd->matrix[r][c+1], mat(r+1, c+1));
            }
        }

        // dd_rowset redundant_rows = dd_RedundantRows(contact_span_cdd,&err);

        // int redundant_rows_num = sizeof(redundant_rows) / sizeof(unsigned long);

        // NEWMAT::Matrix reduced_mat(0,6);

        // for(int i = 0; i < mat.Nrows(); i++)
        // {
        //     bool redundant = false;
        //     for(int j = 0; j < redundant_rows_num; j++)
        //     {
        //         if(i == redundant_rows[j])
        //         {
        //             redundant = true;
        //             break;
        //         }

        //     }

        //     if(!redundant)
        //     {
        //         reduced_mat &= mat.Row(i+1);
        //     }
        // }

        // _computed_contact_surface_cones.insert(std::pair<string,NEWMAT::Matrix>(manipname,reduced_mat));
        _computed_contact_surface_cones.insert(std::pair<string,NEWMAT::Matrix>(manipname,mat));

        dd_FreeMatrix(contact_span_cdd);
        f_cones_diagonal.ReleaseAndDelete();
        a_surf_stacked.ReleaseAndDelete();
        mat.Release();
        // reduced_mat.Release();
        // return reduced_mat;
        return mat;
    }
}

NEWMAT::ReturnMatrix GetGIWCSpanForm(OpenRAVE::RobotBasePtr robot, std::vector<std::string>& manip_ids, std::vector<OpenRAVE::dReal>& friction_coeffs) {
    int num_manips = manip_ids.size();
    std::vector<NEWMAT::Matrix> matrices;
    int total_rows = 0;

    for (int i = 0; i < num_manips; i++) {
        matrices.push_back(GetSurfaceCone(robot, manip_ids[i], friction_coeffs[i]));
        total_rows += matrices.back().Nrows();
    }

    // RAVELOG_INFO("num_manips: %d\n",num_manips);

    // Calculate combined surface cone matrix
    NEWMAT::Matrix s_cones_diagonal(total_rows, 6*num_manips);
    s_cones_diagonal = 0.0;

    int current_row_offset = 0;
    for (int i = 0; i < num_manips; i++) {
        s_cones_diagonal.SubMatrix(current_row_offset+1, current_row_offset+matrices[i].Nrows(), (6*i)+1, (6*i)+6) = matrices[i];
        current_row_offset += matrices[i].Nrows();
    }

    // Calculate A_stance matrix
    NEWMAT::Matrix a_stance_stacked(6 * num_manips, 6);

    for (int i = 0; i < num_manips; i++) {
        GetAStance(robot->GetManipulator(manip_ids[i])->GetTransform(), &a_stance_stacked, 6*i);
    }

    NEWMAT::Matrix mat = s_cones_diagonal * a_stance_stacked; // Dot product

    // RAVELOG_INFO("s_cones_diagonal Rows: %d, Col: %d\n",s_cones_diagonal.Nrows(),s_cones_diagonal.Ncols());
    // cout<<s_cones_diagonal<<endl;
    // RAVELOG_INFO("a_stance_stacked Rows: %d, Col: %d\n",a_stance_stacked.Nrows(),a_stance_stacked.Ncols());
    // cout<<a_stance_stacked<<endl;
    // RAVELOG_INFO("mat Rows: %d, Col: %d\n",mat.Nrows(),mat.Ncols());

    s_cones_diagonal.ReleaseAndDelete();
    a_stance_stacked.ReleaseAndDelete();
    for(unsigned int i = 0; i < matrices.size(); i++)
    {
        matrices[i].ReleaseAndDelete();
    }
    mat.Release();
    return mat;
}

void GetGIWC(OpenRAVE::RobotBasePtr robot, std::vector<std::string>& manip_ids, std::vector<OpenRAVE::dReal>& mus, std::vector<OpenRAVE::dReal>& ikparams) {

    // unsigned long start = timeGetTime();

    // cout<<"Beginning of GetGIWC."<<endl;
    // int a;
    // a = getchar();

    dd_set_global_constants();

    // cout<<"dd_set_global_constraint."<<endl;
    // int b;
    // b = getchar();

    dd_ErrorType err;

    NEWMAT::Matrix giwc_span = GetGIWCSpanForm(robot, manip_ids, mus);
    // RAVELOG_INFO("giwc_span_cdd Rows: %d, Col: %d\n",giwc_span.Nrows(),giwc_span.Ncols());


    // int al;
    // std::cin>>al;

    // unsigned long after_dd_getspanform = timeGetTime();
    // RAVELOG_INFO("after_dd_getspanform: %d\n",(after_dd_getspanform-start));

    // graphptrs.clear();
    // draw_cone(GetEnv(), giwc_span, OpenRAVE::Transform(), giwc_span.Nrows(), giwc_span.Ncols());

    // dd_MatrixPtr giwc_span_cdd = dd_CreateMatrix(giwc_span.Nrows(), giwc_span.Ncols()+1);
    // giwc_span_cdd->representation = dd_Generator;

    // unsigned long after_dd_creatematrix = timeGetTime();
    // RAVELOG_INFO("after_dd_creatematrix: %d\n",(after_dd_creatematrix-after_dd_getspanform));

    // cout<<"giwc_span_cdd."<<endl;
    // int c;
    // c = getchar();

    std::vector<coordT> pointsIn((giwc_span.Nrows()+1)*6);
    pointsIn[0] = 0;
    pointsIn[1] = 0;
    pointsIn[2] = 0;
    pointsIn[3] = 0;
    pointsIn[4] = 0;
    pointsIn[5] = 0;

    // TODO: Is there a better way than doing this?
    for (int r = 0; r < giwc_span.Nrows(); r++) {
        // First element of each row indicates whether it's a point or ray. These are all rays, indicated by 0.
        // dd_set_si(giwc_span_cdd->matrix[r][0], 0.000001);
        for (int c = 0; c < giwc_span.Ncols(); c++) {
            // It's legal to multiply an entire row by the same value (here 1e4)
            // This rounds everything down to a fixed precision int
            // dd_set_si(giwc_span_cdd->matrix[r][c+1], (long) (giwc_span(r+1, c+1) * 1e4));
            // dd_set_si(giwc_span_cdd->matrix[r][c+1], round(giwc_span(r+1, c+1) * 10000));
            pointsIn[(r+1)*6 + c] = 0.0001 * round(giwc_span(r+1, c+1) * 10000);
        }
    }

    // unsigned long after_dd_set_si = timeGetTime();
    // RAVELOG_INFO("after_dd_set_si: %d\n",(after_dd_set_si-after_dd_getspanform));


    int numPointsIn = giwc_span.Nrows()+1;
    std::vector< std::vector<double> > facet_coefficients;
    convexHull6D(&pointsIn[0], numPointsIn, facet_coefficients);

    // cout<<"dd_set_si."<<endl;
    // int d;
    // d = getchar();

    // unsigned long after_qhull = timeGetTime();
    // RAVELOG_INFO("after_qhull: %d\n",(after_qhull-after_dd_set_si));

    // dd_PolyhedraPtr poly = dd_DDMatrix2Poly2(giwc_span_cdd, dd_MaxCutoff, &err);
    // if (err != dd_NoError) {
    //     RAVELOG_INFO("CDD Error: ");
    //     dd_WriteErrorMessages(stdout, err);
    //     throw OPENRAVE_EXCEPTION_FORMAT("CDD Error: %d", err, ORE_InvalidState);
    // }

    // cout<<"poly."<<endl;
    // int e;
    // e = getchar();

    // unsigned long after_dd_matrix2poly2 = timeGetTime();
    // RAVELOG_INFO("after_dd_matrix2poly2: %d\n",(after_dd_matrix2poly2-after_dd_set_si));

    // dd_MatrixPtr giwc_face_cdd = dd_CopyInequalities(poly);

    // RAVELOG_INFO("Ustance Size:(%d,%d)",giwc_face_cdd->rowsize,giwc_face_cdd->colsize);

    // cout<<"giwc_face_cdd."<<endl;
    // int f;
    // f = getchar();

    ikparams.push_back(facet_coefficients.size());
    for (int row = 0; row < facet_coefficients.size(); row++) {
        // Note this skips element 0 of each row, which should always be 0
        for (int col = 1; col < 7; col++) {
            ikparams.push_back(-facet_coefficients[row][col]);
        }
    }

    giwc_span.ReleaseAndDelete();
    // dd_FreeMatrix(giwc_face_cdd);
    // cout<<"Free giwc_face_cdd."<<endl;
    // int g;
    // g = getchar();

    // dd_FreePolyhedra(poly);
    // cout<<"Free poly."<<endl;
    // int h;
    // h = getchar();

    // dd_FreeMatrix(giwc_span_cdd);
    // cout<<"Free giwc_span_cdd."<<endl;
    // int i;
    // i = getchar();

    dd_free_global_constants();
    // cout<<"Free global constraint."<<endl;
    // int j;
    // j = getchar();

}