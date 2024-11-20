import cv2 as cv
import numpy as np
import onnxruntime as rt
import time
import torch
from PIL import Image
from skimage import transform


# tensor变量转ndarray
def ToTensorLab(image):
    # 生成一个  全0的3维数组
    tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
    image = image / np.max(image)
    if image.shape[2] == 1:
        tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
        tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
    else:
        tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229  # r
        tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224  # g
        tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225  # b

    # 对 tmpImg 做一次 transpose 就把 tmpimg 给返回出去
    tmpImg = tmpImg.transpose((2, 0, 1))

    return torch.from_numpy(tmpImg.copy())

image_path = 'datasets/source_test_data/images/852.png'
# opencv 读取 路径得到 img,    bgr  形式
img_cv = cv.imdecode(np.fromfile(image_path, dtype=np.uint8), cv.IMREAD_COLOR)

image = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)
#                         resize 为 模型输入的  高、宽
img = transform.resize(image, (1783, 2534), mode='constant')  # 3 通道   (320, 320, 3)
# 做了一个值的调整，并转成 torch.Size([3, 320, 320]) 的尺寸
sample = ToTensorLab(img)  # torch.Size([3, 320, 320])

inputs_test = sample.unsqueeze(0)  # torch.Size([1, 3, 320, 320])
inputs_test = inputs_test.type(torch.FloatTensor)
# 要注意一下  img  是个啥格式的数据
img = inputs_test.numpy()
# sess = rt.InferenceSession("saved_models/u2netp/u2netp.onnx", providers=["CUDAExecutionProvider"])
sess = rt.InferenceSession("saved_models/u2netp/u2netp.onnx", providers=["CPUExecutionProvider"])

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
print(input_name)

time_start = time.time()

result = sess.run([output_name], {input_name: img})
time_end = time.time()
print("ONNX_pytorch")
print('totally cost', time_end - time_start)

pred = result[0]

# predict_np = torch.argmax(torch.tensor(pred), dim=1, keepdim=True)
# predict_np = predict_np.cpu().detach().numpy().squeeze()
predict_np = pred.squeeze()
predict_np = cv.resize(predict_np, (img_cv.shape[1], img_cv.shape[0]), interpolation=cv.INTER_NEAREST)
cls = dict([(1, (0, 255, 0)),  # 绿色
            (2, (255, 0, 0)),
            (3, (255, 0, 255)),
            (4, (255, 255, 0)),
            (5, (0, 0, 255))])
r = predict_np.copy()
b = predict_np.copy()
g = predict_np.copy()
for c in cls:
    r[r == c] = cls[c][0]
    g[g == c] = cls[c][1]
    b[b == c] = cls[c][2]
rgb = np.zeros((img_cv.shape[0], img_cv.shape[1], 3))
print('类别', np.unique(predict_np))
rgb[:, :, 0] = r
rgb[:, :, 1] = g
rgb[:, :, 2] = b
Image.fromarray(rgb.astype(np.uint8)).save('results_onnx.png')
