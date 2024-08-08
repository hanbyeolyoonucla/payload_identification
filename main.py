# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from math import pi
import numpy as np
from spatialmath import SE3
from myrobot import SerialLink


def load_qft_data(file_name):
    raw_data = np.loadtxt(file_name, delimiter=',')
    joints = raw_data[:,:6]
    forces = raw_data[:,6:9]
    torques = raw_data[:,9:]
    return joints, forces, torques


def skew_sym_matrix(vec):
    assert vec.size == 3
    return np.array([[0, -vec[2], vec[1]],
                    [vec[2], 0, -vec[0]],
                    [-vec[1], vec[0], 0]])


def forces_to_F(forces):
    assert forces.shape[1] == 3
    F = np.array([]).reshape(0,6)
    for force in forces:
        F_idx = np.concatenate((skew_sym_matrix(force),np.eye(3)), axis=1)
        F = np.concatenate((F,F_idx), axis=0)
    return F


def joints_to_R(joints):
    # define robot
    nominal_dh = np.array([[0, 0, 0, 0.135],
                           [-pi / 2, 0, -pi / 2, 0],
                           [0, 0.135, 0, 0],
                           [-pi / 2, 0.038, 0, 0.120],
                           [pi / 2, 0, 0, 0],
                           [-pi / 2, 0, pi, 0.070]])
    T_tool = SE3.Rz(pi) * SE3.Tz(0.03482)
    meca = SerialLink(mdh=nominal_dh, T_tool=T_tool)

    R = np.array([]).reshape(0,6)
    for jt in joints:
        R_idx = np.concatenate((meca.fkine(jt*pi/180).R.transpose(), np.eye(3)), axis=1)
        R = np.concatenate((R,R_idx), axis=0)
    return R


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    fname = './data/payload_identification_data.txt'
    joints, forces, torques = load_qft_data(fname)

    # Calculate p = [x y z k1 k2 k3]
    taus = torques.flatten()
    F = forces_to_F(forces)
    p = np.linalg.pinv(F) @ taus
    x, y, z, k1, k2, k3 = p
    print(p)

    # Calculate l = [Lx Ly Lz Fx0 Fy0 Fz0]
    fs = forces.flatten()
    R = joints_to_R(joints)
    l = np.linalg.pinv(R) @ fs
    Lx, Ly, Lz, Fx0, Fy0, Fz0 = l
    print(l)

    # Calculate Tx0 Ty0 Tz0
    Tx0 = k1 - Fz0*y + Fy0*z
    Ty0 = k2 - Fx0*z + Fz0*x
    Tz0 = k3 - Fy0*x + Fx0*y

    # save parameters
    params = np.array([[x, y, z],
                       [Fx0, Fy0, Fz0],
                       [Tx0, Ty0, Tz0],
                       [Lx, Ly, Lz]])
    np.savetxt("./result/payload_params.txt", params)




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
