# U2Net训练多类别分割

## 注意：

​		训练模型的时候，调用的是 models 文件夹内的  u2net.py

​		torch2onnx的时候， 调用的是 models 文件夹内的  u2net_onnx.py

​		嵌入了 记忆力机制的 u2net5p_x.py 的那两个脚本，仅能成功调用，模型的性能还未稳定。

## 数据增广

```
tools/Augmentor数据增强.py
tools/img_resize.py
tools/弹性形变.py
tools/随机裁剪.py
tools/图像翻转.py
```

## 数据划分

​	1、目录结构

```
U-2-Net-master
	└── datasets
			└── train_data
	     			├── images               #  所有的原图
	     			├── masks                #  所有的mask
                    └── predict_masks 
    		└── test_data
         			├── images               #  所有的测试原图
         			├── masks                #  所有的mask
         			└── predict_masks        #  u2net_test.py 的推理结果保存路径
```

## 数据标注

​	images+masks

​		原图：datasets/train_data/images

​					datasets/test_data/images

​		类似于纯黑的mask图：datasets/train_data/masks

​												datasets/test_data/masks

## 预训练模型的转化

```
python create_u2net_pretrain_model.py
```

其中，读取的是 u2netp，位于 save_models/u2netp.pth

转化预训练权重的时候，其动作为，将 u2netp 的除最后一层 sigmoid 进行移除，并保存为新的权重文件

但要注意转化时的输出通道   out_ch 为分类类别(包含背景)

其中，u2net5p 是最小的模型，u2net是最大的模型

```
u_net = U2NETP(in_ch=3, out_ch=2)
```

保存为：saved_models/pretrain_model/segm_u2net.pth，并使用 u2net_train.py 中 137行的代码，进行预训练权重的加载。

## 训练模型

```shell
python u2net_train.py
```

### 相关脚本

u2net_train.py、u2net_val.py、utils_metrics.py

​								u2net_val.py 可直接执行   (  python u2net_val.py  )

u2net_train.py 中需要设置 weights---类别的加权系数

训练模型时，数据预处理的部分、模型类别数通道都要保持一致：

​		u2net_train.py 的第 137、138、144~156 行、第 75 行设置模型类别

​		u2net_val.py     的第 49、68、69 行

注意：如果训练图片的 mask 图并未将像素值转化为 0~n，那要加上 data_loader.py 的292~298行的代码，做一个标签值的映射，但这个映射结果是随机的，可能。

### 训练参数

```python
# 损失函数：CrossEntropyLoss
device = torch.device("cuda:0")
# background 的权重是 1 ，其他类别的为  1.5，总共的数量和 U2NETP(3, 6)  的第二个参数对应
weights = np.array([1.00, 1.50, 1.50, 1.50, 1.50, 1.50], dtype=np.float32)
weights = torch.from_numpy(weights).to(device)
loss_CE = nn.CrossEntropyLoss(weight=weights).to(device)

#  图片文件类型
image_ext = '.jpg'
label_ext = '.png'

# 模型类型：u2netp 大概4.7MB，是最小的那个
model_name = 'u2netp'  # 'u2net'

# 模型通道数：3是 图片颜色的通道数，这个不变，5是类别数，算上背景总共5个类别
net = U2NETP(3, 6)

# 定义分割类别
num_classes = 6
name_classes = ["background", "first", "second", "third", "4th", "5th"]

#  训练集图片
data_dir = os.path.join(os.getcwd(), 'datasets', 'train_data' + os.sep)
tra_image_dir = os.path.join('images' + os.sep)
tra_label_dir = os.path.join('masks' + os.sep)
#  验证集图片
images_path = "datasets/test_data/images/"
gt_dir = "datasets/test_data/masks/"
# 		存放推理结果图片的路径
pred_dir = "datasets/test_data/predict_masks/"
# 		验证集的推理结果保存
# 		存放 miou 计算结果的 图片
miou_out_path = "miou_out"

# 权重文件的路径
model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

# 保存权重文件并测试的频率为 200 个 iters 一次
save_frq = 200
# 训练回合数
epoch_num = 120
# batch size
batch_size = 4
```



## torch转onnx

​		u2net_onnx.py  的所有参数应该与  u2net.py  保持一致。

​		转了 onnx 以后注意一下  onnx 权重的 输入名字 和 输出名字

```
python torch2onnx.py
```

## 模型微调

​		如果要在原有的训练好的基础上进行模型微调，就把预训练权重改成上一次训练好的模型，需要调整学习率、损失系数等等的参数，看情况使用 u2net_train.py 中 180~184 行的代码进行梯度裁剪。

## 测试脚本

```
python u2net_demo.py  # 调用  pth 权重的测试
python onnx_infer.py  # onnxruntime 调用 onnx 权重的测试
python onnx_opencv_infer.py  # opencv 调用 onnx 权重的测试
```

图片保存于   ./test_out/

配置类别对应颜色

注意一下，输出的推理结果输出的是12345的rgb值，这个推理结果的rgb和标签的rgb是对应的

```python
#  这里给的顺序是  rgb
cls = dict([(1,(0,0,255)),    # 蓝
            (2,(255,0,0)),    # 红
            (3,(255,0,255)),  # 粉
            (4,(0,255,0)),    # 绿
            (5,(0,255,255))]) # 青 
```
