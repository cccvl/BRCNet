'''
生成训练文件，此脚本生成ff++，一幅图像+一幅三维图+一个标签
'''

import os
import random

fake_real = ['fake', 'real']

dataset_type = ['train', 'test', 'val']



for data in dataset_type:
    txt_list = []
    for detect in fake_real:
        pncc = '/*/examples_c23/' + data + '_' + detect
        root = 'ff++_c23/' + data

        if detect == 'fake':
            label = '1'
        elif detect == 'real':
            label = '0'
        path_list = os.listdir(pncc)
        path_list.sort()
        for imgs in path_list:
            if '.jpg' not in imgs: continue
            root_path = os.path.join(os.path.join(root, detect), imgs)
            pncc_path = os.path.join(pncc, imgs)
            if not os.path.exists(root_path): break
            if not os.path.exists(pncc_path): break
            txt_list.append(root_path + ' ' + pncc_path + ' ' + label)
    data_csv = open('ff++_c23/' + data + '_aux.csv', 'w')
    for line in txt_list:
        data_csv.write(line + '\n')

    data_csv.close()




            

