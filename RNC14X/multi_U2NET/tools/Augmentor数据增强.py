# 导入数据增强工具
import Augmentor

# 确定原始图像存储路径以及掩码文件存储路径
p = Augmentor.Pipeline("E:/资料/4-惠州三协/1-语义分割的一个demo/1-数据集/2-data_for_train/8-400张图片-20220719(上)-更改了评估指标/images")
p.ground_truth("E:/资料/4-惠州三协/1-语义分割的一个demo/1-数据集/2-data_for_train/8-400张图片-20220719(上)-更改了评估指标/masks")

# 图像旋转： 按照概率0.8执行，最大左旋角度20，最大右旋角度20
p.rotate(probability=0.8, max_left_rotation=25, max_right_rotation=25)

# 图像左右互换： 按照概率0.5执行
p.flip_left_right(probability=0.7)

# 图像放大缩小： 按照概率0.5执行，面积为原始图0.85倍
p.zoom_random(probability=0.5, percentage_area=0.85)

# 最终扩充的数据样本数
p.sample(200)
