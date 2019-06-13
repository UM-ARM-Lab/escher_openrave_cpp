import math
import numpy as np

# com_dict
# x
# 0: (-Inf, -0.3)
# 1: [-0.3, -0.2)
# 2: [-0.2, -0.1)
# 3: [-0.1, 0.0)
# 4: [0.0, 0.1)
# 5: [0.1, 0.2)
# 6: [0.2, Inf)
# y
# 0: (-Inf, -0.1)
# 1: [-0.1, 0.0)
# 2: [0.0, 0.1)
# 3: [0.1, 0.2)
# 4: [0.2, Inf)
# z
# 0: (-Inf, 0.8)
# 1: [0.8, 0.9)
# 2: [0.9, 1.0)
# 3: [1.0, 1.1)
# 4: [1.1, Inf)
def com_index(x):
    idxx = max(min(int(math.floor(x[-6] * 10) + 4), 6), 0)
    idxy = max(min(int(math.floor(x[-5] * 10) + 2), 4), 0)
    idxz = max(min(int(math.floor(x[-4] * 10) - 7), 4), 0)
    return (idxx, idxy, idxz)


def adjust_com(com_before_adjustment, original_frame, new_frame):
    """
    "original_frame" is mean_feet_pose, which has 6 entries

    "new_frame" is torse pose, which has 3 entries
    """
    original_x = original_frame[0]
    original_y = original_frame[1]
    original_z = original_frame[2]
    original_yaw = original_frame[5]
    original_yaw_in_radian = original_yaw / 180.0 * np.pi

    global_com = np.zeros_like(com_before_adjustment)
    global_com[:, 0] = original_x - com_before_adjustment[:, 1] * np.sin(original_yaw_in_radian) + com_before_adjustment[:, 0] * np.cos(original_yaw_in_radian)
    global_com[:, 1] = original_y + com_before_adjustment[:, 1] * np.cos(original_yaw_in_radian) + com_before_adjustment[:, 0] * np.sin(original_yaw_in_radian)
    global_com[:, 2] = original_z + com_before_adjustment[:, 2]

    new_x = new_frame[0]
    new_y = new_frame[1]
    new_z = 0
    new_yaw = new_frame[2]
    new_yaw_in_radian = new_yaw / 180.0 * np.pi
    rotation_matrix = np.array([[np.cos(-new_yaw_in_radian), -np.sin(-new_yaw_in_radian)],
                                [np.sin(-new_yaw_in_radian), np.cos(-new_yaw_in_radian)]])
    com_after_adjustment = np.copy(global_com)
    com_after_adjustment[:, 0:2] = np.matmul(rotation_matrix, (global_com - np.array([new_x, new_y, new_z]))[:, 0:2].T).T
    return com_after_adjustment


def discretize_torso_pose(arr, resolution):
    new_arr = np.zeros_like(arr, dtype=int)
    indices = np.argwhere(np.absolute(arr) > resolution/2.0).reshape(-1,)
    values = arr[indices]
    values -= np.sign(values) * resolution/2.0
    new_arr[indices] = np.sign(values) * np.ceil(np.round(np.absolute(values) / resolution, 1)).astype(int)
    return new_arr


def discretize_x(X, X_BOUNDARY_1, X_BOUNDARY_2):
    """
    (-Inf, X_BOUNDARY_1): 0
    [X_BOUNDARY_1, X_BOUNDARY_2): 1
    [X_BOUNDARY_2, Inf): 2
    """
    new_X = np.zeros_like(X, dtype=int)
    new_X[np.argwhere(X >= X_BOUNDARY_1).reshape(-1,)] = 1
    new_X[np.argwhere(X >= X_BOUNDARY_2).reshape(-1,)] = 2
    return new_X


def adjust_p2(p1, p2):
    """
    p1: [p1x, p1y, p1yaw]
    p2: [p2x, p2y, p2yaw]
    """
    p1_yaw_in_radian = p1[2] / 180.0 * np.pi
    rotation_matrix = np.array([[np.cos(-p1_yaw_in_radian), -np.sin(-p1_yaw_in_radian)],
                                [np.sin(-p1_yaw_in_radian), np.cos(-p1_yaw_in_radian)]])
    p2_xy_after_adjustment = np.matmul(rotation_matrix, np.array([[p2[0] - p1[0]], [p2[1] - p1[1]]]))
    return [p2_xy_after_adjustment[0][0], p2_xy_after_adjustment[1][0], p2[2] - p1[2]]



def main():
    # original_frame = [math.sqrt(3), 1.0, -0.5, 0, 0, -30.0]
    # com_before_adjustment = np.array([[0.0, -2.0 / math.sqrt(3), 0.7],
    #                                   [-2.0, 0.0, 0.5]])
    # new_frame = [0.0, 0.0, 90.0]
    # com_after_adjustment = adjust_com(com_before_adjustment, original_frame, new_frame)
    # print(com_after_adjustment)

    # arr = np.array([-75.0, -70.0, -67.5, -60.0, -52.5, -45.0, -37.5, -30.0, -22.5, -15.0, -7.5, 0.0, 7.5, 10.0, 22.5, 30.0, 37.5, 45.0, 52.5, 60.0, 67.5, 75.0, 82.5, 90])
    # new_arr = discretize_torso_pose(arr, 15.0)
    # anws = np.array([-5, -5, -4, -4, -3, -3, -2, -2, -1, -1, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6])
    # assert(np.array_equal(new_arr, anws))
    # arr = np.array([-0.5, -0.4, -0.375, -0.35, -0.225, -0.15, -0.075, 0, 0.075, 0.15, 0.225, 0.30, 0.375, 0.45, 0.5])
    # new_arr = discretize_torso_pose(arr, 0.15)
    # anws = np.array([-3, -3, -2, -2, -1, -1, 0, 0, 0, 1, 1, 2, 2, 3, 3])
    # assert(np.array_equal(new_arr, anws))
    
    # X_BOUNDARY_1 = -0.2
    # X_BOUNDARY_2 = -0.1
    # X = np.array([-0.25, -0.2, -0.15, -0.1, -0.05, 0.5])
    # new_X = discretize_x(X, X_BOUNDARY_1, X_BOUNDARY_2)
    # anws = np.array([0, 1, 1, 2, 2, 2])
    # assert(np.array_equal(new_X, anws))

    p1 = [1.0, 1.0, 30.0]
    p2 = [2.0, 1.0 + 1.0 / math.sqrt(3), 60.0]
    print(adjust_p2(p1, p2))



                

if __name__ == '__main__':
    main()