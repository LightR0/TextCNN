#!/usr/bin/python
# -*- coding: utf-8 -*-

import os 
import random
import shutil

'''
从THUCNews随机采样一份子集数据，类别为10个，每个类别数据为6500条。
其中label_list为需要的类别列表，可以自行减少。data_num为每个类别数据条数，可以自行增加或减少。
'''

labellist = ["体育", "娱乐", "家居", "教育", "时政", "游戏", "社会", "科技", "股票", "财经"]
datanum = 6500
rawdir = "THUCNews"
newdir = "data/thucnews/"

def copy_datas(raw_dir, new_dir, data_num):
	for category in os.listdir(raw_dir):
		print category
		raw_cat_dir = os.path.join(raw_dir, category)
		new_cat_dir = os.path.join(new_dir, category)
		files = os.listdir(raw_cat_dir)
        
		# 采样文件索引
		random_list = [random.randint(0, len(files)-1) for _ in range(data_num)]
		for text_index in random_list:
			if not os.path.exists(new_cat_dir):
				os.makedirs(new_cat_dir)
			shutil.copy(os.path.join(raw_cat_dir, files[text_index]), new_cat_dir)

if __name__== "__main__":
	copy_datas(rawdir, newdir, datanum)
