import os
import cv2
import numpy as np
import open3d as o3d

from pathlib import Path
import argparse

class Kitti_Dataset:
    def __init__(self, dir_path, split="training"):
        super(Kitti_Dataset, self).__init__()
        self.dir_path = os.path.join(dir_path, split)
        # calib矫正参数文件夹地址
        self.calib = os.path.join(self.dir_path, "calib")
        # RGB图像的文件夹地址
        self.images = os.path.join(self.dir_path, "image_2")
        # 点云图像文件夹地址
        self.pcs = os.path.join(self.dir_path, "velodyne")
        # 标签文件夹的地址
        if split == 'testing':
            self.labels = os.path.join(self.dir_path, "results")
        else:
            self.labels = os.path.join(self.dir_path, "label_2")

    # 得到当前数据集的大小
    def __len__(self):
        file = []
        for _, _, file in os.walk(self.images):
            pass

        # 返回rgb图片的数量
        return len(file)

    # 得到矫正参数的信息
    def get_calib(self, index):
        # 得到矫正参数文件
        calib_path = os.path.join(self.calib, "{:06d}.txt".format(index))
        with open(calib_path) as f:
            lines = f.readlines()

        lines = list(filter(lambda x: len(x) and x != '\n', lines))
        dict_calib = {}
        for line in lines:
            key, value = line.split(":")
            dict_calib[key] = np.array([float(x) for x in value.split()])
        return Calib(dict_calib)

    def get_rgb(self, index):
        # 首先得到图片的地址
        img_path = os.path.join(self.images, "{:06d}.png".format(index))
        return cv2.imread(img_path)

    def get_pcs(self, index):
        pcs_path = os.path.join(self.pcs, "{:06d}.bin".format(index))
        # 点云的四个数据（x, y, z, r)
        aaa = np.fromfile(pcs_path, dtype=np.float32, count=-1).reshape([-1, 4])
        return aaa[:, :3]

    def get_labels(self, index):
        labels_path = os.path.join(self.labels, "{:06d}.txt".format(index))
        with open(labels_path) as f:
            lines = f.readlines()
        lines = list(filter(lambda x: len(x) > 0 and x != '\n', lines))
    
        return [Object3d(x) for x in lines]


class Object3d:
    def __init__(self, content):
        super(Object3d, self).__init__()
        # content 就是一个字符串，根据空格分隔开来
        lines = content.split()

        # 去掉空字符
        lines = list(filter(lambda x: len(x), lines))

        self.name, self.truncated, self.occluded, self.alpha = lines[0], float(lines[1]), float(lines[2]), float(lines[3])

        self.bbox = [lines[4], lines[5], lines[6], lines[7]]
        self.bbox = np.array([float(x) for x in self.bbox])
        self.dimensions = [lines[8], lines[9], lines[10]]
        self.dimensions = np.array([float(x) for x in self.dimensions])
        self.location = [lines[11], lines[12], lines[13]]
        self.location = np.array([float(x) for x in self.location])
        self.rotation_y = float(lines[14])
        #这一行是模型训练后的label通常最后一行是阈值，可以同个这个过滤掉概率低的object
        #如果只要显示kitti本身则不需要这一行
        #self.ioc = float(lines[15])


class Calib:
    def __init__(self, dict_calib):
        super(Calib, self).__init__()
        self.P0 = dict_calib['P0'].reshape(3, 4)
        self.P1 = dict_calib['P1'].reshape(3, 4)
        self.P2 = dict_calib['P2'].reshape(3, 4)
        self.P3 = dict_calib['P3'].reshape(3, 4)
        self.R0_rect = dict_calib['R0_rect'].reshape(3, 3)
        self.P0 = dict_calib['P0'].reshape(3, 4)
        self.Tr_velo_to_cam = dict_calib['Tr_velo_to_cam'].reshape(3, 4)
        self.Tr_imu_to_velo = dict_calib['Tr_imu_to_velo'].reshape(3, 4)




# 根据偏航角计算旋转矩阵（逆时针旋转）
def rot_y(rotation_y):
    cos = np.cos(rotation_y)
    sin = np.sin(rotation_y)
    R = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    return R

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dataset', type=str, default='../kitti', help='dir for the label data')
    args = parser.parse_args()
    dir_path = Path(args.path_dataset)

    # 读取训练集文件夹
    split = "testing"
    dataset = Kitti_Dataset(dir_path, split=split)

    ids = [os.path.splitext(x)[0] for x in os.listdir(dataset.images)]
    for iter, id in enumerate(ids):
        print(id)
        k = int(id)
        # if iter > 50:
        #     break
        img3_d = dataset.get_rgb(k)
        calib = dataset.get_calib(k)
        obj = dataset.get_labels(k)
        for num in range(len(obj)):
            if obj[num].name == "Car" or obj[num].name == "Pedestrian" or obj[num].name == "Cyclist":
            	#这一行为阈值用来过滤训练概率较低的object
                #if (obj[num].name == "Car" and obj[num].ioc >= 0.7) or obj[num].ioc > 0.5:
                	# step1 得到rot_y旋转矩阵 3*3
                    R = rot_y(obj[num].rotation_y)
                    # 读取obect物体的高宽长信息
                    h, w, l = obj[num].dimensions[0], obj[num].dimensions[1], obj[num].dimensions[2]

                    # step2
                    # 得到该物体的坐标以底面为原点中心所在的物体坐标系下各个点的坐标
                    #     7 -------- 4
                    #    /|         /|
                    #   6 -------- 5 .
                    #   | |        | |
                    #   . 3 -------- 0
                    #   |/   .- - -|/ - - -> (x)
                    #   2 ---|----- 1
                    #        |
                    #        | (y)
                    x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
                    y = [0, 0, 0, 0, -h, -h, -h, -h]
                    z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
                    # 将xyz转化成3*8的矩阵
                    corner_3d = np.vstack([x, y, z])
                    # R * X
                    corner_3d = np.dot(R, corner_3d)

                    # 将该物体移动到相机坐标系下的原点处（涉及到坐标的移动，直接相加就行）
                    corner_3d[0, :] += obj[num].location[0]
                    corner_3d[1, :] += obj[num].location[1]
                    corner_3d[2, :] += obj[num].location[2]

                    # 将3d的bbox转换到2d坐标系中（需要用到内参矩阵)
                    corner_3d = np.vstack((corner_3d, np.zeros((1, corner_3d.shape[-1]))))
                    corner_2d = np.dot(calib.P2, corner_3d)
                    # 在像素坐标系下，横坐标x = corner_2d[0, :] /= corner_2d[2, :]
                    # 纵坐标的值以此类推
                    corner_2d[0, :] /= corner_2d[2, :]
                    corner_2d[1, :] /= corner_2d[2, :]

                    corner_2d = np.array(corner_2d, dtype=np.int32)

                    # 绘制立方体边界框
                    color = [0, 255, 0]
                    # 线宽
                    thickness = 2

                    #绘制3d框
                    for corner_i in range(0, 4):
                        i, j = corner_i, (corner_i + 1) % 4
                        cv2.line(img3_d, (corner_2d[0, i], corner_2d[1, i]), (corner_2d[0, j], corner_2d[1, j]), color, thickness)
                        i, j = corner_i + 4, (corner_i + 1) % 4 + 4
                        cv2.line(img3_d, (corner_2d[0, i], corner_2d[1, i]), (corner_2d[0, j], corner_2d[1, j]), color, thickness)
                        i, j = corner_i, corner_i + 4
                        cv2.line(img3_d, (corner_2d[0, i], corner_2d[1, i]), (corner_2d[0, j], corner_2d[1, j]), color, thickness)

                    # cv2.line(img3_d,(corner_2d[0, 0],corner_2d[1, 0]), (corner_2d[0, 5], corner_2d[1, 5]),color, thickness)
                    # cv2.line(img3_d, (corner_2d[0, 1], corner_2d[1, 1]), (corner_2d[0, 4], corner_2d[1, 4]), color, thickness)

        cv2.imwrite(f'output/demo/rgb_{str(k)}.png', img3_d)
        





