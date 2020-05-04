# -*- coding: utf-8 -*-
"""
计算采收期的每一棵生菜的重量，整个田块的总重量，并划分到不同的重量区间
"""

# from __future__ import print_function
import cv2
import numpy as np
import copy

# import xlsxwriter
# workbook = xlsxwriter.Workbook('RR.xlsx')
# worksheet = workbook.add_worksheet("RR")


# 利用建立的Cell来判定生菜的种植位置
# 采样区域：垄的角度、宽度、生菜的种植密度、飞行高度
# 换个地方飞行，还需要做一个采样区域
cellfile = 'Cell.tif'
imgrgbori = cv2.imread(cellfile, 3)
imgrgbori = imgrgbori / 255.0
GG = imgrgbori[:, :, 2]
# RR:  257.0 0.0
# print "test", GG[9, 28]
print ("GG: ", np.amax(GG), np.amin(GG))
print ("GG.shape", GG.shape)
# print RR
# h_rr, w_rr = GG.shape[0], GG.shape[1]
# for ii in range(h_rr):
#     for jj in range(w_rr):
#         worksheet.write(ii, jj, str(GG[ii][jj]))
# print RR[30:40, 25:30]
# workbook.close()

# 生菜产量
img = cv2.imread("mask_D4.tif", 3)
img = img / 255.0
bb = img[:, :, 0]
gg = img[:, :, 1]
rr = img[:, :, 2]

vi = (2.0 * gg - rr - bb) - (1.4 * rr - gg)
rgbsum = rr + gg + bb
vi[rgbsum < 0.5] = 0.0

vi[vi < 0.02] = 0.


# print vi[6082, 9217]
rows, cols = vi.shape[0], vi.shape[1]
# vi.shape (13557, 11474)
print ("vi.shape", vi.shape)
# 0.49254901960784314 -102.79999999999995
print ("vi", np.amax(vi), np.amin(vi))

totaldataout = np.zeros((rows, cols))
innerdataout = np.zeros((rows, cols))
# 18和59飞行高度及cell的形状确定的参数，cell是最小的计算单元
numcol = int(np.floor(vi.shape[1] / 18))
numrow = int(np.floor(vi.shape[0] / 59))
totalweight = np.zeros((numrow, numcol))
innerweight = np.zeros((numrow, numcol))
# tmpdata = GG
# tpt = 0
print "before recurrence", vi[6109:6111, 4320:4322]
for col in range(numcol - 4):
    for row in range(numrow - 1):
        # tpt += 1
        tmpdata = np.zeros((67, 45))
        # cell_data = vi[(59 * row):(59 * row + 67), (18 * col):(18 * col + 45)]
        # for tmp_i in range(67):
        #     for tmp_j in range(45):
        #         tmpdata[tmp_i, tmp_j] = cell_data[tmp_i, tmp_j]
        tmpdata = copy.deepcopy(vi[(59 * row):(59 * row + 67), (18 * col):(18 * col + 45)])
        tmpdata[GG <= 1] = 0.
        tmpsum = sum(sum(tmpdata))
        # print col, row, tmpsum
        # totalweight大于0的是生菜，大于0的个数如果是100个，则有200棵生菜，数值是生菜的重量
        totalweight[row, col] = (16.94*tmpsum/2.-176.38)
        innerweight[row, col] = (11.19*tmpsum/2.-188.29)
        if totalweight[row, col] < 0 or innerweight[row, col] < 0:
            totalweight[row, col] = 0.
            innerweight[row, col] = 0.
        # 图像显示，和vi同比例
        for i in range(tmpdata.shape[0]):
            for j in range(tmpdata.shape[1]):
                if GG[i, j] > 1:
                    totaldataout[59 * row + i, 18 * col + j] = totalweight[row, col]
                    innerdataout[59 * row + i, 18 * col + j] = innerweight[row, col]
print "after recurrence", vi[6109:6111, 4320:4322]
# 输出结果
# ('totalweight ', 771.7109725490195, 0.0)
print ("totalweight ", np.amax(totalweight), np.amin(totalweight))
print ("innerweight ", np.amax(innerweight), np.amin(innerweight))
# print np.amax(totaldataout), np.amin(totaldataout)
# cv2.imwrite("totaldataout.tif", totaldataout * 255)
# cv2.imwrite("innerdataout.tif", innerdataout * 255)

"""
分级标准：<300, 400~500g, 500~600g, 600~700g, >700g
total_statistical_number_of_each_interval = [0, 0, 0, 0, 0, 0]
"""
total_statistical_numbers = [0, 0, 0, 0, 0]
# inner_statistical_numbers = [0, 0, 0, 0, 0]
zongchanliang_total = 0
zongchanliang_inner = 0
# the number of all lettuce
all_lettuce_number = numcol * numrow * 2

for r_idx in range(numrow):
    for c_idx in range(numcol):
        # 整棵菜
        total_w = totalweight[r_idx, c_idx]
        zongchanliang_total += total_w
        inner_w = innerweight[r_idx, c_idx]
        zongchanliang_inner += inner_w
        if total_w < 350:
            total_statistical_numbers[0] += 1
        elif 400 < total_w < 500:
            total_statistical_numbers[1] += 1
        elif 500 < total_w < 600:
            total_statistical_numbers[2] += 1
        elif 600 < total_w < 700:
            total_statistical_numbers[3] += 1
        elif 700 < total_w:
            total_statistical_numbers[4] += 1

# return total_statistical_numbers, zongchanliang_total, zongchanliang_inner, total_statistical_numbers