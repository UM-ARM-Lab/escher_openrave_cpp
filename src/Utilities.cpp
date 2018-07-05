#include "Utilities.hpp"
// euclidean distance btwn two points in a 2D coordinate system
float euclideanDistance2D(const Translation2D& q, const Translation2D& p)
{
	return (q-p).norm();
}

// euclidean distance btwn two points in a 3D coordinate system
float euclideanDistance3D(const Translation3D& q, const Translation3D& p)
{
	return (q-p).norm();
}

TransformationMatrix constructTransformationMatrix(float m00, float m01, float m02, float m03, float m10, float m11, float m12, float m13, float m20, float m21, float m22, float m23)
{
    TransformationMatrix T;

    T(0,0) = m00; T(0,1) = m01; T(0,2) = m02; T(0,3) = m03;
    T(1,0) = m10; T(1,1) = m11; T(1,2) = m12; T(1,3) = m13;
    T(2,0) = m20; T(2,1) = m21; T(2,2) = m22; T(2,3) = m23;
    T(3,0) = 0;   T(3,1) = 0;   T(3,2) = 0;   T(3,3) = 1;

    return T;
}

TransformationMatrix inverseTransformationMatrix(TransformationMatrix T)
{
	RotationMatrix R = T.block<3,3>(0,0);
	Translation3D t = T.block<3,1>(0,3);

	TransformationMatrix T_inverse;
	T_inverse.block(0,0,3,3) = R.transpose();
	T_inverse.block(0,3,3,1) = -R.transpose()*t;
	T_inverse(3,0) = 0;
	T_inverse(3,1) = 0;
	T_inverse(3,2) = 0;
	T_inverse(3,3) = 1;

	return T_inverse;
}

RotationMatrix RPYToSO3(const RPYTF& e) // TODO: need to change to the more robust version
{
	float roll_in_rad = e.roll_ * DEG2RAD;
	float pitch_in_rad = e.pitch_ * DEG2RAD;
	float yaw_in_rad = e.yaw_ * DEG2RAD;

	float roll_x = cos(roll_in_rad);
	float roll_y = sin(roll_in_rad);

	float pitch_x = cos(pitch_in_rad);
	float pitch_y = sin(pitch_in_rad);

	float yaw_x = cos(yaw_in_rad);
	float yaw_y = sin(yaw_in_rad);

	RotationMatrix rot_mat;
	rot_mat(0,0) = pitch_x * yaw_x;
	rot_mat(0,1) = -pitch_x * yaw_y;
	rot_mat(0,2) = pitch_y;
	rot_mat(1,0) = roll_x * yaw_y + yaw_x * roll_y * pitch_y;
	rot_mat(1,1) = roll_x * yaw_x - roll_y * pitch_y * yaw_y;
	rot_mat(1,2) = -pitch_x * roll_y;
	rot_mat(2,0) = roll_y * yaw_y - roll_x * yaw_x * pitch_y;
	rot_mat(2,1) = yaw_x * roll_y + roll_x * pitch_y * yaw_y;
	rot_mat(2,2) = roll_x * pitch_x;

	return rot_mat;
}

TransformationMatrix XYZRPYToSE3(const RPYTF& e)
{
	float roll_in_rad = e.roll_ * DEG2RAD;
	float pitch_in_rad = e.pitch_ * DEG2RAD;
	float yaw_in_rad = e.yaw_ * DEG2RAD;

	float roll_x = cos(roll_in_rad);
	float roll_y = sin(roll_in_rad);

	float pitch_x = cos(pitch_in_rad);
	float pitch_y = sin(pitch_in_rad);

	float yaw_x = cos(yaw_in_rad);
	float yaw_y = sin(yaw_in_rad);

	return constructTransformationMatrix(pitch_x * yaw_x, -pitch_x * yaw_y, pitch_y, e.x_,
	                                     roll_x * yaw_y + yaw_x * roll_y * pitch_y, roll_x * yaw_x - roll_y * pitch_y * yaw_y, -pitch_x * roll_y, e.y_,
									     roll_y * yaw_y - roll_x * yaw_x * pitch_y, yaw_x * roll_y + roll_x * pitch_y * yaw_y, yaw_x * roll_y + roll_x * pitch_y * yaw_y, e.z_);

}

RPYTF SO3ToRPY(const RotationMatrix& R)
{
	float epsilon = 0.0001;
	float roll, pitch, yaw;

    if(std::fabs(R(0,0)) > epsilon and std::fabs(R(2,2)) > epsilon)
	{
        roll = std::atan2(-R(1,2),R(2,2)) * RAD2DEG;
        pitch = std::atan2(R(0,2),std::sqrt(R(1,2)*R(1,2)+R(2,2)*R(2,2))) * RAD2DEG;
        yaw = std::atan2(-R(0,1),R(0,0)) * RAD2DEG;
	}
    else if(std::fabs(R(0,0)) > epsilon and std::fabs(R(2,2)) <= epsilon)
	{
        pitch = std::atan2(R(0,2),std::sqrt(R(1,2)*R(1,2)+R(2,2)*R(2,2))) * RAD2DEG;
        yaw = std::atan2(-R(0,1),R(0,0)) * RAD2DEG;

        float yaw_rad = yaw * DEG2RAD;

        if(std::fabs(std::cos(yaw_rad)) > std::fabs(std::sin(yaw_rad)))
		{
            if(R(2,1) / std::cos(yaw_rad) > 0)
                roll = 90;
            else
                roll = -90;
		}
		else
		{
            if(R(2,0) / std::sin(yaw_rad) > 0)
                roll = 90;
            else
                roll = -90;
		}
	}
    else if(std::fabs(R(0,0)) <= epsilon and std::fabs(R(2,2)) > epsilon)
	{
        roll = std::atan2(-R(1,2),R(2,2)) * RAD2DEG;
        pitch = std::atan2(R(0,2),std::sqrt(R(1,2)*R(1,2)+R(2,2)*R(2,2))) * RAD2DEG;

        float roll_rad = roll * DEG2RAD;

        if(std::fabs(std::cos(roll_rad)) > std::fabs(std::sin(roll_rad)))
		{
            if(R(1,0) / std::cos(roll_rad) > 0)
                yaw = 90;
            else
                yaw = -90;
		}
		else
		{
            if(R(2,0) / std::sin(roll_rad) > 0)
                yaw = 90;
            else
                yaw = -90;
		}
	}
    else if(std::fabs(R(0,0)) <= epsilon and std::fabs(R(2,2)) <= epsilon)
	{

        if(std::fabs(R(0,2))-1 <= epsilon)
		{
            if(R(0,2) > 0)
                pitch = 90;
            else
                pitch = -90;

            if(std::fabs(R(1,0)) > epsilon)
			{
                roll = std::atan2(R(2,0),R(1,0)) * RAD2DEG;
				double R_1_0 = R(1,0);
                yaw = std::asin(std::max(std::min(R_1_0/std::cos(roll*DEG2RAD),1.0),-1.0)) * RAD2DEG;
			}
			else
			{
                if(pitch == 90)
				{
                    if(R(2,0) < R(1,1))
					{
                        roll = -90;
                        yaw = 90;
					}
					else
					{
                        roll = 90;
                        yaw = 90;
					}
				}
				else
				{
                    if(R(2,0) > 0)
					{
                        roll = 90;
                        yaw = 90;
					}
                    else
					{
                        roll = -90;
                        yaw = 90;
					}
				}
			}
		}
        else
		{
            pitch = -std::asin(R(1,1)/R(2,0)) * RAD2DEG;
            roll = -std::asin(R(1,2)/std::cos(pitch*DEG2RAD)) * RAD2DEG;
            yaw = -std::asin(R(0,1)/std::cos(pitch*DEG2RAD)) * RAD2DEG;
		}
	}

	return RPYTF(0, 0, 0, roll, pitch, yaw);
}

RPYTF SE3ToXYZRPY(const TransformationMatrix& T)
{
	RotationMatrix R = T.block(0,0,3,3);

	RPYTF rpy = SO3ToRPY(R);

	return RPYTF(T(0,3), T(1,3), T(2,3), rpy.roll_, rpy.pitch_, rpy.yaw_);
}

Translation2D gridPositions2DToTranslation2D(GridPositions2D positions)
{
	return Translation2D(positions[0],positions[1]);
}

GridPositions2D translation2DToGridPositions2D(Translation2D positions)
{
	return {positions[0],positions[1]};
}

float getFirstTerminalAngle(float angle)
{
	float result_angle = angle;
    while(result_angle < -180 or result_angle >= 180)
	{
		if(result_angle < -180)
		{
			result_angle += 360;
		}
		else if(result_angle >= 180)
		{
			result_angle -= 360;
		}
	}

    return result_angle;
}

float getAngleDifference(float angle1, float angle2)
{
	return getFirstTerminalAngle(angle1 - angle2);
}

float getAngleMean(float angle1, float angle2)
{
	// return the closest mean angle in -180 ~ 180
	float mean1 = (angle1+angle2)/2.0;
	float mean2 = mean1 + 180;

	if(std::fabs(getAngleDifference(mean1,angle1)) < std::fabs(getAngleDifference(mean2,angle1)))
	{
		return mean1;
	}
	else
	{
		return mean2;
	}
}

std::array<float,4> HSVToRGB(std::array<float,4> hsv)
{
	std::array<float,4> rgb;

	float h = hsv[0];
	float s = hsv[1];
	float v = hsv[2];

	if(h > 360 || h < 0 || s < 0 || s > 1 || v < 0 || v > 1)
	{
		// RAVELOG_WARN("Invalid input of HSV value: (%6.3f,%5.3f,%5.3f). Set each field to its limit.\n",h,s,v);
	}

	// h = std::max(std::min(h,float(360.0)),float(0.0));
	s = std::max(std::min(s,float(1.0)),float(0.0));
	v = std::max(std::min(v,float(1.0)),float(0.0));

	while(h < 0)
	{
		h = h + 360;
	}
	while(h > 360)
	{
		h = h - 360;
	}

	int hi = ((int)floor(h/60)) % 6;

	float f = h/60.0 - hi;

	float p = v*(1-s);
	float q = v*(1-f*s);
	float t = v*(1-(1-f)*s);

	switch(hi)
	{
		case 0:
			rgb[0] = v; rgb[1] = t; rgb[2] = p;
			break;
		case 1:
			rgb[0] = q; rgb[1] = v; rgb[2] = p;
			break;
		case 2:
			rgb[0] = p; rgb[1] = v; rgb[2] = t;
			break;
		case 3:
			rgb[0] = p; rgb[1] = q; rgb[2] = v;
			break;
		case 4:
			rgb[0] = t; rgb[1] = p; rgb[2] = v;
			break;
		case 5:
			rgb[0] = v; rgb[1] = p; rgb[2] = q;
			break;
		default:
			std::cout << "Impossible" << std::endl;
			break;
	}


	rgb[3] = hsv[3];

	return rgb;
}

bool isValidPosition(Translation2D p)
{
	if(p[0] == -99.0)
	{
		return false;
	}
	else
	{
		return true;
	}
}

bool isValidPosition(Translation3D p)
{
	if(p[0] == -99.0)
	{
		return false;
	}
	else
	{
		return true;
	}
}

RPYTF transformPoseFromOpenraveToSL(RPYTF& e)
{
    RPYTF transformed_e(0, 0, 0, 0, 0, 0);
    RotationMatrix R = RPYToSO3(e);

    float foot_position_offset = 0.035;
    transformed_e.x_ = e.x_ - foot_position_offset * R(0,0);
    transformed_e.y_ = e.y_ - foot_position_offset * R(1,0);
    transformed_e.z_ = e.z_ - foot_position_offset * R(2,0);

    TransformationMatrix SL_openrave_transform;
    RotationMatrix SL_openrave_rotation;
    SL_openrave_transform << 0, -1, 0, 0,
                             1,  0, 0, 0,
                             0,  0, 1, -1.05,
                             0, 0, 0, 1;
    SL_openrave_rotation = SL_openrave_transform.block(0,0,3,3);

    Translation3D transformed_position = (SL_openrave_transform * Translation3D(transformed_e.x_, transformed_e.y_, transformed_e.z_).homogeneous()).block(0,0,3,1);
    transformed_e.x_ = transformed_position[0];
    transformed_e.y_ = transformed_position[1];
    transformed_e.z_ = transformed_position[2];

    RPYTF rpy = SO3ToRPY(SL_openrave_rotation * RPYToSO3(e) * SL_openrave_rotation.transpose());
    transformed_e.roll_ = rpy.roll_;
    transformed_e.pitch_ = rpy.pitch_;
    transformed_e.yaw_ = rpy.yaw_;

    return transformed_e;
}

Eigen::Vector3d rotateVectorFromOpenraveToSL(Vector3D& t)
{
    RotationMatrix SL_openrave_rotation;
    SL_openrave_rotation << 0, -1, 0,
                             1,  0, 0,
                             0,  0, 1;

    Eigen::Vector3d rotated_vector = (SL_openrave_rotation * t).cast<double>();

    return rotated_vector;
}

Vector3D rotateVectorFromSLToOpenrave(Eigen::Vector3d& t)
{
    RotationMatrix openrave_SL_rotation;
    openrave_SL_rotation <<  0,  1, 0,
                             -1, 0, 0,
                             0,  0, 1;

    Vector3D rotated_vector = (openrave_SL_rotation * t.cast<float>());

    return rotated_vector;
}

Eigen::Vector3d transformPositionFromOpenraveToSL(Translation3D& t)
{
    TransformationMatrix SL_openrave_transform;
    SL_openrave_transform << 0, -1, 0, 0,
                             1,  0, 0, 0,
                             0,  0, 1, -1.05,
                             0, 0, 0, 1;

    Eigen::Vector3d transformed_position = (SL_openrave_transform * t.homogeneous()).block(0,0,3,1).cast<double>();

    return transformed_position;
}

Translation3D transformPositionFromSLToOpenrave(Eigen::Vector3d& t)
{
    TransformationMatrix openrave_SL_transform;
    openrave_SL_transform <<  0, 1, 0, 0,
                             -1, 0, 0, 0,
                              0, 0, 1, 1.05,
                              0, 0, 0, 1;

    Translation3D transformed_position = (openrave_SL_transform * t.cast<float>().homogeneous()).block(0,0,3,1);

    return transformed_position;
}
