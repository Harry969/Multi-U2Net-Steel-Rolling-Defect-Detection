# 对得到的 datasets/VOCdevkit/VOC2007/ 下的数据集进行划分，最终输出四个txt文件
# 保存于 datasets/VOCdevkit/VOC2007/ImageSets/Main 下

import os
import random
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--xml_path', default='VOCData/annotations', type=str, help='input xml label path')

parser.add_argument('--xml_path', default='C:/Users/ASUS/Desktop/mmsegmentation-master/data/VOCdevkit/VOC2012/SegmentationClass', type=str, help='input xml label path')
parser.add_argument('--txt_path', default='C:/Users/ASUS/Desktop/mmsegmentation-master/data/VOCdevkit/VOC2012/ImageSets/Segmentation', type=str, help='output txt label path')

# parser.add_argument('--xml_path', default='E:/资料/2-公司/1_公司项目/13-工程质检/4-用于训练的数据/datasets-3483/for_train/Annotations-20220608', type=str, help='input xml label path')
# parser.add_argument('--txt_path', default='E:/资料/2-公司/1_公司项目/13-工程质检/4-用于训练的数据/datasets-3483/for_train/ImageSets-20220608/Main', type=str, help='output txt label path')
opt = parser.parse_args()

trainval_percent = 0.8  # 0
train_percent = 0.8  # 0
xmlfilepath = opt.xml_path
txtsavepath = opt.txt_path
total_xml = os.listdir(xmlfilepath)
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

num = len(total_xml)
list_index = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list_index, tv)
train = random.sample(trainval, tr)

file_trainval = open(txtsavepath + '/trainval.txt', 'w')
file_test = open(txtsavepath + '/test.txt', 'w')
file_train = open(txtsavepath + '/train.txt', 'w')
file_val = open(txtsavepath + '/val.txt', 'w')

for i in list_index:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        file_trainval.write(name)
        if i in train:
            file_train.write(name)
        else:
            file_val.write(name)
    else:
        file_test.write(name)

file_trainval.close()
file_train.close()
file_val.close()
file_test.close()

