import os
import numpy as np
from PIL import Image


def convert_png_to_npy(png_dir, npy_dir):
    """
    将 png 标签文件批量转换为 npy 文件
    :param png_dir: 保存 PNG 文件的文件夹路径
    :param npy_dir: 保存 npy 文件的文件夹路径
    """
    # 如果目标文件夹不存在，则创建
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)

    # 遍历 png 文件夹中的所有文件
    for filename in os.listdir(png_dir):
        if filename.endswith(".png"):
            # 生成 PNG 文件的完整路径
            png_path = os.path.join(png_dir, filename)

            # 读取 PNG 图像并转换为 NumPy 数组
            image = Image.open(png_path)
            npy_array = np.array(image)

            # 获取文件名（不含扩展名）
            base_filename = os.path.splitext(filename)[0]

            # 保存为 .npy 文件
            npy_path = os.path.join(npy_dir, base_filename + '.npy')
            np.save(npy_path, npy_array)

            print(f"转换完成: {filename} -> {base_filename}.npy")


# 示例：指定 PNG 和 npy 的文件夹路径
png_dir = r'D:\RNC14X\multi_U2NET\png\000000000000000'  # 输入的 PNG 标签图片文件夹路径
npy_dir = r'D:\RNC14X\multi_U2NET\npy\000000000000000'  # 输出的 npy 文件保存路径

# 调用函数进行批量转换
convert_png_to_npy(png_dir, npy_dir)

