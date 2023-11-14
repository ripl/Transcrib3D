import json
import os
import numpy as np

# scan_id="scene0000_00"
# path=os.path.join("data","scanrefer_camera_info",scan_id+".anns.json")

# with open(path) as f:
#     data=json.load(f)

# for d in data[0:10]:
#     print("\n")
#     for k,v in d.items():
#         if isinstance(v,dict):
#             print(k,":")
#             for kk,vv in v.items():
#                 print(kk,":",vv)
#         else:
#             print(k,":",v)


def euler_to_rot_matrix_zyx(euler_angles):
    """
    Converts ZYX Euler angles to a 3x3 rotation matrix.
    
    :param euler_angles: A tuple or list of three Euler angles (Z, Y, X).
    :return: A 3x3 rotation matrix.
    """
    z, y, x = euler_angles
    cz, sz = np.cos(z), np.sin(z)
    cy, sy = np.cos(y), np.sin(y)
    cx, sx = np.cos(x), np.sin(x)

    rot_matrix = np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [   -sy,             cy * sx,             cy * cx]
    ])

    return rot_matrix

def euler_to_rot_matrix_xyz(euler_angles):
    """
    Converts XYZ Euler angles to a 3x3 rotation matrix.
    
    :param euler_angles: A tuple or list of three Euler angles (X, Y, Z).
    :return: A 3x3 rotation matrix.
    """
    x, y, z = euler_angles
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)

    rot_matrix = np.array([
        [cy * cz,          -cy * sz,            sy],
        [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx],
        [-cz * sx + cx * sy * sz, sx * sz + cx * cz * sy, cx * cy]
    ])

    return rot_matrix

def euler_to_rot_matrix_yzx(euler_angles):
    """
    Converts YZX Euler angles to a 3x3 rotation matrix.
    
    :param euler_angles: A tuple or list of three Euler angles (Y, Z, X).
    :return: A 3x3 rotation matrix.
    """
    y, z, x = euler_angles
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    cx, sx = np.cos(x), np.sin(x)

    rot_matrix = np.array([
        [cy * cz, sx * sy - cx * cy * sz, cx * sy + cy * sx * sz],
        [     sz,             cx * cz,            -cz * sx],
        [-cz * sy, cy * sx + cx * sy * sz, cx * cy - sx * sy * sz]
    ])

    return rot_matrix


z_axis=np.array([0,0,-1])

position=np.array([2.196494472633714, 2.938295595878993, 2.117838198778193])
rotation=np.array([-1.108686547277176, 0.03518419696438971, 3.124073545308852])
lookat=np.array([2.108596384525299, 0.7030213586986065, 1.004494620487094])

position =  np.array([3.1524491619872834, 4.362509203480364, 1.5642548908498668])
rotation =np.array( [1.2667286113308494, 0.9938004858452627, 0.2571727179379676])
lookat   =np.array([0.6726098209619522, 5.902551174163818, 1.080991506576538])

# R=euler_to_rot_matrix_zyx(rotation)
R=euler_to_rot_matrix_xyz(rotation)
# R=euler_to_rot_matrix_yzx(rotation)


z_axis_trans=np.dot(R,z_axis.reshape(3,1)).reshape(-1)

look_vec=lookat-position

print(z_axis_trans/np.linalg.norm(z_axis_trans))
print(look_vec/np.linalg.norm(look_vec))
