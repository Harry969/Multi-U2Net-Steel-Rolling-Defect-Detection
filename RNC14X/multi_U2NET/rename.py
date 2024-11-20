import os


def rename_files(directory, prefix, start_index=1):
    """
    统一重命名文件，格式为 prefix_*.npy, 其中 * 是递增的编号。
    :param directory: 需要处理的文件夹路径
    :param prefix: 文件名前缀 ('ground_truth' 或 'prediction')
    :param start_index: 起始编号
    """
    # 获取文件夹中的所有 .npy 文件
    files = [f for f in os.listdir(directory) if f.endswith('.npy')]

    # 对每个文件进行重命名
    for i, file in enumerate(sorted(files), start=start_index):
        # 生成新的文件名，格式为 'prefix_001.npy'
        new_name = f"{prefix}_{i:06}.npy"
        old_file_path = os.path.join(directory, file)
        new_file_path = os.path.join(directory, new_name)

        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {old_file_path} -> {new_file_path}")


if __name__ == "__main__":
    # 真实标签文件夹
    # gt_dir = r"D:\RNC14X\multi_U2NET\npy\0"  # 修改为你的真实标签文件夹路径
    # # 基线预测文件夹
    # base_dir = r"D:\unet-pytorch\unet-pytorch\9876543211111111_npy"  # 修改为你的基线预测文件夹路径
    # 你模型的预测结果文件夹
    pred_dir = r"D:\RNC14X\multi_U2NET\npy\000000000000000"  # 修改为你的预测结果文件夹路径

    # 重命名 gt_dir 中的文件为 'ground_truth_*.npy'
    # rename_files(gt_dir, prefix="ground_truth", start_index=1)
    #
    # # 重命名 base_dir 中的文件为 'prediction_*.npy'
    # rename_files(base_dir, prefix="prediction", start_index=1)

    # 重命名 pred_dir 中的文件为 'prediction_*.npy'
    rename_files(pred_dir, prefix="prediction", start_index=1)


