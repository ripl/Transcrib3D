def draw_points(points,title):
    import matplotlib.pyplot as plt
    import numpy as np
    # 提取x、y和z坐标
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # 创建一个3D散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图
    ax.scatter(x, y, z, c='b', marker='o')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    unit=0.5
    len_x=max(x)-min(x)
    len_y=max(y)-min(y)
    len_z=max(z)-min(z)
    n_tick_x=int((len_x+0.2)/unit)+1
    n_tick_y=int((len_y+0.2)/unit)+1
    n_tick_z=int((len_z+0.2)/unit)+1

    ax.set_xticks(np.arange(int(2*min(x)-1)/2,int(2*max(x)+1)/2,unit))
    ax.set_yticks(np.arange(int(2*min(y)-1)/2,int(2*max(y)+1)/2,unit))
    ax.set_zticks(np.arange(int(2*min(z)-1)/2,int(2*max(z)+1)/2,unit))

    ax.set_box_aspect([len_x,len_y,len_z])

    
    ax.set_title(title)

    # 显示图形
    plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def draw_points(points, title):
#     # 提取x、y和z坐标
#     x = points[:, 0]
#     y = points[:, 1]
#     z = points[:, 2]

#     # 创建一个3D散点图
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # 绘制散点图
#     ax.scatter(x, y, z, c='b', marker='o')

#     # 设置坐标轴标签
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

#     ax.set_title(title)

#     # 显示图形
#     plt.show()

#     # 最大范围的数据维度
#     max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()

#     # 添加填充
#     xpad = max_range - (x.max()-x.min())
#     ypad = max_range - (y.max()-y.min())
#     zpad = max_range - (z.max()-z.min())

#     # 重新设置坐标轴的范围
#     ax.set_xlim3d([x.min() - xpad / 2, x.max() + xpad / 2])
#     ax.set_ylim3d([y.min() - ypad / 2, y.max() + ypad / 2])
#     ax.set_zlim3d([z.min() - zpad / 2, z.max() + zpad / 2])
    
#     # 重新显示图形
#     plt.show()


if __name__=='__main__':
    import os
    import numpy as np

    objects_info_folder_path="H:\ScanNet_Data\data\scannet\scans\objects_info_mask3d_90"
    files=os.listdir(objects_info_folder_path)

    for file in files:
        if file.startswith("scene"):
            points=np.load(os.path.join(objects_info_folder_path,file),allow_pickle=True)
            draw_points(points,title=file)