'''
从原始视频中提取人脸
'''

import os
import json
import dlib
import cv2

# 使用 Dlib 的正面人脸检测器 frontal_face_detector
detector = dlib.get_frontal_face_detector()

def frame_extraction(path, target, extract_num):
    cam = cv2.VideoCapture(path)
    frames = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    freq = frames//(extract_num + 1)
    num = 0

    # try :
        
    #     # 创建名为data的文件夹
    #     if not os.path.exists( 'data' ):
    #         os.makedirs( 'data' )
    
    # # 如果未创建，则引发错误
    # except OSError:
    #     print ( 'Error: Creating directory of data' )
    # frame
    currentframe = 0
    while ( True ):
        currentframe += 1
        if currentframe%freq == 0 and num < extract_num:
            # reading from frame
            ret, frame = cam.read()
            if not ret:
                print('not res , not image')
                break
            # 如果视频仍然存在，继续创建图像
            source_name = path.split('/')[-4] + '_' + path.split('/')[-1].split('.')[0] + '_'
            name = source_name  + str (currentframe) + '.jpg'
            #print ( 'Creating...' + name)
            temp_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #   将图片传入并检测
            rectangles = detector(temp_img, 1)
            draw = frame.copy()
            for rectangle in rectangles:
                draw = draw[int(rectangle.top()):int(rectangle.bottom()), int(rectangle.left()):int(rectangle.right())]
                break
            if draw.size == 0:
                break
    
            # 写入提取的图像
            cv2.imwrite(os.path.join(target, name), draw)
        
            # 增加计数器，以便显示创建了多少帧
            currentframe = currentframe + 1
            num += 1
        if num >= extract_num:
            break
    
    # 一旦完成释放所有的空间和窗口
    cam.release()
    # cv2.destroyAllWindows()


json_list = ['val.json', 'test.json', 'train.json']
fake_real = ['manipulated_sequences', 'original_sequences']
fake_type = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

# 存储人脸路径
frame_extract_path = '/*/*/*/*'
# 原始数据路径
root_path = '/*/*/*/*/*/C40'
for json_file in json_list:
    with open(os.path.join('/*/', json_file), encoding='utf-8') as a:
        info = json.load(a)
        # 遍历训练/测试/验证集的视频编码
        for file_name in info:
            # 真实数据集和伪造数据集不同的视频读取方式
            for fake_real_path in fake_real:
                if fake_real_path == 'manipulated_sequences':
                    for fake in fake_type:
                        fake_path = os.path.join(os.path.join(root_path, fake_real_path), fake) + '/c40/videos'
                        file1 = file_name[0] + '_' + file_name[1] + '.mp4'
                        file2 = file_name[1] + '_' + file_name[0] + '.mp4'
                        fake_root1 = os.path.join(fake_path, file1)
                        fake_root2 = os.path.join(fake_path, file2)
                        # print(fake_root1)
                        # print(fake_root2)
                        # print('fake done!')
                        frame_extract_path1 = os.path.join(frame_extract_path, json_file.split('.')[0]) + '/fake'
                        frame_extract_path2 = os.path.join(frame_extract_path, json_file.split('.')[0]) + '/fake'
                        frame_extraction(fake_root1, frame_extract_path1, extract_num=10)
                        frame_extraction(fake_root2, frame_extract_path2, extract_num=10)
                elif fake_real_path == 'original_sequences':
                    real_path = os.path.join(os.path.join(root_path, fake_real_path), 'youtube/c40/videos')
                    file1 = file_name[0] + '.mp4'
                    file2 = file_name[1] + '.mp4'
                    real_root1 = os.path.join(real_path, file1)
                    real_root2 = os.path.join(real_path, file2)
                    # print(real_root1)
                    # print(real_root2)
                    # print('real done!')
                    frame_extract_path1 = os.path.join(frame_extract_path, json_file.split('.')[0]) + '/real'
                    frame_extract_path2 = os.path.join(frame_extract_path, json_file.split('.')[0]) + '/real'
                    frame_extraction(real_root1, frame_extract_path1, extract_num=40)
                    frame_extraction(real_root2, frame_extract_path2, extract_num=40)


