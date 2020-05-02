# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import Augmentor

label_dict = {"others":0, "jieqiu":1, "kuju":2, "luomanni":3, "luoshahong":4, "luoshalv":5, "lvluo":6, "naiyoulv":7, "xibanya":8}
PATCH_CNT_EACH_IMAGE = 500
PATCH_SIZE = 112


def aug_patch(samples_path):
    number_of_pathes = len(os.listdir(samples_path))
    p = Augmentor.Pipeline(samples_path)
    # p.ground_truth(labels_path)
    # p.random_contrast(probability=, min_factor=, max_factor=)
    p.random_contrast(1, 0.5, 1.5)
    # p.random_brightness(probability=, min_factor=, max_factor=)
    p.random_brightness(1, 0.5, 1.5)
    p.flip_left_right(1)
    p.flip_top_bottom(1)
    # p.rotate(probability=, max_left_rotation=, max_right_rotation=)
    # 左右最大旋转20度
    p.rotate(1, 20, 20)
    # 增强之后数量变成原来的5倍
    p.sample(5 * number_of_pathes)


# other地物的label图像
def generate_others_label(rgb_path, label_path):
    # print filename
    other_data = cv2.imread(rgb_path)
    h, w, c = other_data.shape
    mask_data = np.ones((h, w), dtype=np.uint8)
    cv2.imwrite(label_path, 255 * mask_data)


# 绿色生菜的label
def generate_green_label(img_path, label_path):
    image_data = cv2.imread(img_path)
    # print "image data shape", image_data.shape
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    label_image = np.ones(shape=(image_data.shape[0], image_data.shape[1]), dtype=np.uint8)
    image_data = image_data / 255.0
    rr = image_data[:, :, 0]
    gg = image_data[:, :, 1]
    bb = image_data[:, :, 2]

    rgb_sum = rr + gg + bb
    # 各个植被指数的定义 https://www.cnblogs.com/sqwang/archive/2005/03/17/119774.html
    # 超绿超红差分指数
    vi = (2 * gg - rr - bb) - (1.4 * rr - gg)
    # print np.amax(vi), np.amin(vi)
    # <0.5 denotes dark black, delete it to reduce the sun shadows' effects
    vi[rgb_sum < 0.5] = 0.0
    # print np.amax(vi), np.amin(vi)
    # keep the color of lettuce self
    vimax = 0.05
    label_image[vi < vimax] = 0
    # print img_name
    print np.unique(label_image)
    # print label_image.shape
    # saved_path = img_name.split('.')[0] + '_label.png'
    cv2.imwrite(label_path, label_image * 255)

def generate_red_label(image_path, label_path):
    # print image_name
    eps = 1e-4
    image_data = cv2.imread(image_path)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    label_image = np.ones(shape=(image_data.shape[0], image_data.shape[1]), dtype=np.uint8)
    image_data = image_data / 255.0
    rr = image_data[:, :, 0]
    gg = image_data[:, :, 1]
    bb = image_data[:, :, 2]
    rgbsum = rr + gg + bb
    ggrr = (gg - rr) / (gg + rr + eps)
    # print np.unique(ggrr)
    max_ggrr = np.max(ggrr)
    # print max_ggrr
    min_ggrr = np.min(ggrr)
    # print min_ggrr
    ggrr = (ggrr - min_ggrr) / (max_ggrr - min_ggrr)
    # print np.max(ggrr)
    ggrr[ggrr > 0.45] = 1.0
    ggrr[rgbsum < 0.5] = 1.0
    label_image[ggrr == 1] = 0.0
    # print np.unique(label_image)
    # print label_image.shape
    cv2.imwrite(label_path, label_image * 255)


def rand_crop(cls_id, rgb_idx, rgb_path, label_path, blue_path, green_path, nir_path, red_path, rededge_path):
    print "cls_id = ", cls_id
    print "rgb_path = ", rgb_path
    # 读取rgb, label以及五个多光谱图像，获取对应的shape
    rgb_data = cv2.imread(rgb_path)
    if rgb_data is None:
        print "rgb_data is None, ", rgb_path
    rgb_heigt, rgb_width, rgb_depth = rgb_data.shape
    label_data = cv2.imread(label_path, -1)
    if label_data is None:
        print "label_data is None, ", label_path
    h_label, w_label = label_data.shape
    blue_data = cv2.imread(blue_path, -1)
    if blue_data is None:
        print "blue_data is None, ", blue_path
    h_blue, w_blue = blue_data.shape
    green_data = cv2.imread(green_path, -1)
    if green_data is None:
        print "green_data is None, ", green_path
    h_green, w_green = green_data.shape
    red_data = cv2.imread(red_path, -1)
    if red_data is None:
        print "red_data is None, ", red_path
    h_red, w_red = red_data.shape
    nir_data = cv2.imread(nir_path, -1)
    if nir_data is None:
        print "nir_data is None, ", nir_path
    h_nir, w_nir = nir_data.shape
    rededge_data = cv2.imread(rededge_path, -1)
    if rededge_data is None:
        print "rededge_data is None, ", rededge_path
    h_rededge, w_rededge = rededge_data.shape

    # 断言宽、高相同
    assert (rgb_heigt == h_label and rgb_width == w_label)
    assert (rgb_heigt == h_blue and rgb_width == w_blue)
    assert (rgb_heigt == h_green and rgb_width == w_green)
    assert (rgb_heigt == h_nir and rgb_width == w_nir)
    assert (rgb_heigt == h_red and rgb_width == w_red)
    assert (rgb_heigt == h_rededge and rgb_width == w_rededge)

    # 循环产生PATCH_CNT_EACH_IMAGE= 500个patch及label
    patch_cnt = PATCH_CNT_EACH_IMAGE
    if cls_id == 1:
        patch_cnt = 5 * PATCH_CNT_EACH_IMAGE
    for i_patch in range(patch_cnt):
        # 随机设置crop的点坐标
        x_begin = np.random.randint(0, rgb_width-PATCH_SIZE)
        y_begin = np.random.randint(0, rgb_heigt-PATCH_SIZE)
        # print "x_begin = ", x_begin
        # print "y_begin = ", y_begin
        x_end = x_begin + PATCH_SIZE
        y_end = y_begin + PATCH_SIZE

        # 即others地物，那么所有的label值=255
        if cls_id == 0:
            # print "进入"
            # 取patch
            rgb_patch = rgb_data[y_begin:y_end, x_begin:x_end, :]
            label_patch = label_data[y_begin:y_end, x_begin:x_end]
            # 意思是周围填充部分，只有一个数值
            if len(np.unique(rgb_patch)) < 10:
                # print "继续"
                continue
            else:
                # print "写图像"
                cv2.imwrite('samples'+os.sep+str(cls_id)+'-'+str(rgb_idx)+'-'+str(0)+'-'+str(i_patch)+'.png', rgb_patch)
                cv2.imwrite('samples'+os.sep+str(cls_id)+'-'+str(rgb_idx)+'-'+str(1)+'-'+str(i_patch) + '.png', blue_data[y_begin:y_end, x_begin:x_end])
                cv2.imwrite('samples'+os.sep+str(cls_id)+'-'+str(rgb_idx)+'-'+str(2)+'-'+str(i_patch) + '.png', green_data[y_begin:y_end, x_begin:x_end])
                cv2.imwrite('samples'+os.sep+str(cls_id)+'-'+str(rgb_idx)+'-'+str(3)+'-'+str(i_patch) + '.png', red_data[y_begin:y_end, x_begin:x_end])
                cv2.imwrite('samples'+os.sep+str(cls_id)+'-'+str(rgb_idx)+'-'+str(4)+'-'+str(i_patch) + '.png', nir_data[y_begin:y_end, x_begin:x_end])
                cv2.imwrite('samples'+os.sep+str(cls_id)+'-'+str(rgb_idx)+'-'+str(5)+'-'+str(i_patch) + '.png', rededge_data[y_begin:y_end, x_begin:x_end])
                cv2.imwrite('labels' + os.sep + str(cls_id) + '-' + str(rgb_idx) + '-' + str(i_patch) + '.png', label_patch)
        # 生菜，label的值为255和0，其中0是背景
        elif cls_id != 0:
            # print "进入 cls_id ", cls_id
            # 取patch
            rgb_patch = rgb_data[y_begin:y_end, x_begin:x_end, :]
            label_patch = label_data[y_begin:y_end, x_begin:x_end]
            # 只有一个label值
            # if np.amax(rgb_patch) == 0:
            if len(np.unique(rgb_patch)) < 10 or np.amax(label_patch) == 0:
                # print "继续"
                continue
            else:
                # print "写图像"
                cv2.imwrite('samples'+os.sep+str(cls_id)+'-'+str(rgb_idx)+'-'+str(0)+'-'+str(i_patch)+'.png', rgb_patch)
                cv2.imwrite('samples'+os.sep+str(cls_id)+'-'+str(rgb_idx)+'-'+str(1)+'-'+str(i_patch) + '.png', blue_data[y_begin:y_end, x_begin:x_end])
                cv2.imwrite('samples'+os.sep+str(cls_id)+'-'+str(rgb_idx)+'-'+str(2)+'-'+str(i_patch) + '.png', green_data[y_begin:y_end, x_begin:x_end])
                cv2.imwrite('samples'+os.sep+str(cls_id)+'-'+str(rgb_idx)+'-'+str(3)+'-'+str(i_patch) + '.png', red_data[y_begin:y_end, x_begin:x_end])
                cv2.imwrite('samples'+os.sep+str(cls_id)+'-'+str(rgb_idx)+'-'+str(4)+'-'+str(i_patch) + '.png', nir_data[y_begin:y_end, x_begin:x_end])
                cv2.imwrite('samples'+os.sep+str(cls_id)+'-'+str(rgb_idx)+'-'+str(5)+'-'+str(i_patch) + '.png', rededge_data[y_begin:y_end, x_begin:x_end])
                cv2.imwrite('labels' + os.sep + str(cls_id) + '-' + str(rgb_idx) + '-' + str(i_patch) + '.png', label_patch)
        else:
            print "cls_id = ", cls_id, " no cropped"


def main_logic():
    for root, dirs, files in os.walk('mixedsamples', topdown=True):
        if root == 'mixedsamples':
            continue
        else:
            print "root = ", root
            # 生成mask
            # for file in files:
            #     if file.startswith('mask'):
            #         print "file name = ", file
            #         f_path = os.path.join(root, file)
            #         mask_path = os.path.join(root, 'label'+file.split('mask')[1])
            #         if root.endswith('others'):
            #             generate_others_label(f_path, mask_path)
            #         elif root.endswith('luoshahong'):
            #             generate_red_label(f_path, mask_path)
            #         else:
            #             generate_green_label(f_path, mask_path)

            # 随机crop样本，要找到对应的rgb和多光谱
            # 获取类别号，用于对crop的样本和标签命名
            class_id = label_dict[root.split('/')[1]]
            # 统计有多少个rgb图像的名称，用于对crop的样本和标签命名
            rgb_file_list = []
            for file in files:
                if file.startswith('mask'):
                    rgb_file_list.append(file)
            print "rgb_file_list = ", rgb_file_list
            # 统计有多少个rgb图像的个数
            # rgb_file_cnt = len(rgb_file_lixst)
            # 通过rgb图像的名称，找到对应的label图像和五个多光谱图像
            for idx, rgb_name in enumerate(rgb_file_list):
                if rgb_name.startswith('mask'):
                    rgb_path = os.path.join(root, rgb_name)
                    label_path = os.path.join(root, 'label'+rgb_name.split('mask')[1])
                    _, date, name = rgb_name.split('_')
                    blue_path = os.path.join(root, "Tif_" + date + '_' + 'blue' + name)
                    green_path = os.path.join(root, "Tif_" + date + '_' + 'green' + name)
                    red_path = os.path.join(root, "Tif_" + date + '_' + 'red' + name)
                    nir_path = os.path.join(root, "Tif_" + date + '_' + 'nir' + name)
                    rededge_path = os.path.join(root, "Tif_" + date + '_' + 'red edge' + name)
                    # 调用rand_crop
                    rand_crop(class_id, idx, rgb_path, label_path, blue_path, green_path, nir_path, red_path, rededge_path)


if __name__ == '__main__':
    main_logic()