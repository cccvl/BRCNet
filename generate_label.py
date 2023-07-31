'''
生成训练文件，此脚本生成ff++，一幅图像+一个标签
'''

import os
import random

file_list = ['train', 'val', 'test']

# 
root_path = './ff++_c23'

fake_real = ['fake', 'real']
        
train_list = []
test_list  = []
val_list   = []
for file in file_list:
    path = os.path.join(root_path, file)
    if file == 'train':
        for child in fake_real:
            for img in os.listdir(os.path.join(path, child)):
                if '.jpg' not in img: continue
                train_list.append(os.path.join(os.path.join(path, child), img) + '_' + child)
    elif file == 'test':
        for child in fake_real:
            for img in os.listdir(os.path.join(path, child)):
                if '.jpg' not in img: continue
                test_list.append(os.path.join(os.path.join(path, child), img) + '_' + child)
    else:
        for child in fake_real:
            for img in os.listdir(os.path.join(path, child)):
                if '.jpg' not in img: continue
                val_list.append(os.path.join(os.path.join(path, child), img) + '_' + child)        


train_csv = open('train.csv', 'w')
test_csv  = open('test.csv', 'w')
val_csv   = open('val.csv', 'w')

for info in train_list:
    if 'fake' in info:
        train_csv.write(info.split('_fake')[0] + ' ' + '1' + '\n')
    elif 'real' in info:
        train_csv.write(info.split('_real')[0] + ' ' + '0' + '\n')

for info in val_list:
    if 'fake' in info:
        val_csv.write(info.split('_fake')[0] + ' ' + '1' + '\n')
    elif 'real' in info:
        val_csv.write(info.split('_real')[0] + ' ' + '0' + '\n')

for info in test_list:
    if 'fake' in info:
        test_csv.write(info.split('_fake')[0] + ' ' + '1' + '\n')
    elif 'real' in info:
        test_csv.write(info.split('_real')[0] + ' ' + '0' + '\n')

    
