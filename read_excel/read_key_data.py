#!/usr/bin/env python
# -*- coding:utf-8 -*-

import xlrd
import xlwt
from datetime import datetime
from xlrd import xldate_as_tuple
import numpy as np

filename = r'山东2018春生产关键数据对比记录.xlsx'

def read_key_data_excel():
    workbook = xlrd.open_workbook(filename)

    sheet_mgr = workbook.sheet_by_name(u'山东育苗')
    nrows = sheet_mgr.nrows
    ncols = sheet_mgr.ncols
    print("nrows = %d, ncols = %d" %(nrows, ncols))
    print sheet_mgr.cell(2,0).ctype 

    all_content = []
    for i in range(nrows):
        row_content = []
        for j in range(ncols):
	    ctype = sheet_mgr.cell(i, j).ctype 
	    cell = sheet_mgr.cell_value(i, j)
  	    if 
	    #if ctype == 2 and cell % 1 == 0:
            #    cell = int(cell)
            #elif ctype == 3:
            #    date = datetime(*xldate_as_tuple(cell, 0))
            #    cell = date.strftime('%Y/%d/%m')
            #elif ctype == 4:
            #    cell = True if cell == 1 else False
            row_content.append(cell)
        all_content.append(row_content)
    return all_content

def write_excel(data):
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('hf', cell_overwrite_ok=True)
    nrows, ncols = np.shape(data)
    for i in range(nrows):
        for j in range(ncols):
            sheet.write(i,j,data[i][j])
    book.save(r'有效数据/val_'+filename.split('.')[0]+'.xls')


if __name__=='__main__':
    data = read_key_data_excel()
    # print data
    print np.shape(data)
    write_excel(data)
