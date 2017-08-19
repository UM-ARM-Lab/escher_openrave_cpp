#include "Utilities.hpp"
// euclidean distance btwn two points in a 2D coordinate system
float euclideanDistance2D(const Translation2D& q, const Translation2D& p)
{
	return sqrt(pow(q[0] - p[0], 2) + pow(q[1] - p[1], 2));
}

// euclidean distance btwn two points in a 3D coordinate system
float euclideanDistance3D(const Translation3D& q, const Translation3D& p)
{
	return sqrt(pow(q[0] - p[0], 2) + pow(q[1] - p[1], 2) + pow(q[2] - p[2], 2));
}

TransformationMatrix constructTransformationMatrix(float m00, float m01, float m02, float m03, float m10, float m11, float m12, float m13, float m20, float m21, float m22, float m23)
{
    TransformationMatrix T;
    
    T(0,0) = m00; T(0,1) = m01; T(0,2) = m02; T(0,3) = m03;
    T(1,0) = m00; T(1,1) = m01; T(1,2) = m02; T(1,3) = m03;
    T(2,0) = m00; T(2,1) = m01; T(2,2) = m02; T(2,3) = m03;
    T(3,0) = 0;   T(3,1) = 0;   T(3,2) = 0;   T(3,3) = 1;

    return T;
}

RotationMatrix RPYToSO3(const RPY_tf& e) // TODO: need to change to the more robust version
{
	float roll_in_rad = e.roll * (M_PI / 180);
	float pitch_in_rad = e.pitch * (M_PI / 180);
	float yaw_in_rad = e.yaw * (M_PI / 180);

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

TransformationMatrix XYZRPYToSE3(const RPY_tf& e)
{
	float roll_in_rad = e.roll * (M_PI / 180);
	float pitch_in_rad = e.pitch * (M_PI / 180);
	float yaw_in_rad = e.yaw * (M_PI / 180);

	float roll_x = cos(roll_in_rad);
	float roll_y = sin(roll_in_rad);

	float pitch_x = cos(pitch_in_rad);
	float pitch_y = sin(pitch_in_rad);

	float yaw_x = cos(yaw_in_rad);
	float yaw_y = sin(yaw_in_rad);

	return constructTransformationMatrix(pitch_x * yaw_x, -pitch_x * yaw_y, pitch_y, e.x,
	                                     roll_x * yaw_y + yaw_x * roll_y * pitch_y, roll_x * yaw_x - roll_y * pitch_y * yaw_y, -pitch_x * roll_y, e.y,
									     roll_y * yaw_y - roll_x * yaw_x * pitch_y, yaw_x * roll_y + roll_x * pitch_y * yaw_y, yaw_x * roll_y + roll_x * pitch_y * yaw_y, e.z);

}

Translation2D gridPositions2DToTranslation2D(GridPositions2D positions)
{
	return Translation2D(positions[0],positions[1]);
}

GridPositions2D translation2DToGridPositions2D(Translation2D positions)
{
	return {positions[0],positions[1]};
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