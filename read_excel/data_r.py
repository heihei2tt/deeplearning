#!/usr/bin/env python
# -*- coding:utf-8 -*-

import xlrd
import xlwt
from datetime import datetime
from xlrd import xldate_as_tuple
import numpy as np

filename = r'2018年七级日报.xlsx'
#filename = r'2017年七级秋季日报 -.xlsx'

def read_excel():
    # open file
    workbook = xlrd.open_workbook(filename)
    # get all sheet

    print (workbook.sheet_names())

    sheet_mgr = workbook.sheet_by_name(u'田间管理')
    print (sheet_mgr.name,sheet_mgr.nrows,sheet_mgr.ncols)

    print(type(sheet_mgr.cell(2,0).value))
    print(sheet_mgr.cell(2,0).value)

    nrows = sheet_mgr.nrows
    ncols = sheet_mgr.ncols
    print("nrows = %d, ncols = %d" %(nrows, ncols))

    all_content = []
    for i in range(nrows):
	row_content = []
        if sheet_mgr.cell(i,6).value==u'移栽' or sheet_mgr.cell(i,6).value==u'采收' or sheet_mgr.cell(i,6).value==u'定植' or sheet_mgr.cell(i,6).value==u'定值':
            for j in range(15):
		ctype = sheet_mgr.cell(i,j).ctype
		cell = sheet_mgr.cell_value(i,j)
		if ctype==2 and cell%1==0:
		    cell = int(cell)
		elif ctype==3:
		    date = datetime(*xldate_as_tuple(cell, 0))
                    cell = date.strftime('%Y/%d/%m')
		elif ctype==4:
		    cell = True if cell == 1 else False
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
    book.save(r'有效数据/valid_'+filename.split('.')[0]+'.xls')


if __name__ == '__main__':
    data = read_excel()
    print type(data)
    print np.shape(data)
    write_excel(data)
