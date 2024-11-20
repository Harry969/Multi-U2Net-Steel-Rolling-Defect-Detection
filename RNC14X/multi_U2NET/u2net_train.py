import torch
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

import numpy as np
import glob
import os

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from torchvision.transforms import Compose
import random

from model.u2net import U2NETP
from u2net_val import eval_print_miou


# ------- 0. set random seed --------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(1000)
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# ------- 1. define loss function --------
device = torch.device("cuda:0")

# 权重设置
weights = np.array([1.00, 1.50, 1.50, 1.50], dtype=np.float32)
weights = torch.from_numpy(weights).to(device)
loss_CE = nn.CrossEntropyLoss(weight=weights).to(device)


def muti_CrossEntropyLoss_loss(d0, d1, d2, d3, d4, d5, d6, labels_v):
    labels_v = labels_v.squeeze(1).long()
    loss0 = loss_CE(d0, labels_v)
    loss1 = loss_CE(d1, labels_v)
    loss2 = loss_CE(d2, labels_v)
    loss3 = loss_CE(d3, labels_v)
    loss4 = loss_CE(d4, labels_v)
    loss5 = loss_CE(d5, labels_v)
    loss6 = loss_CE(d6, labels_v)

    loss = loss0 * 1.5 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss0, loss


min_train_loss = float('inf')  # 初始化为正无穷大


# ------- 2. set the directory of training process --------
if __name__ == '__main__':
    model_name = 'u2netp'
    model_dir = os.path.join('saved_models', model_name + os.sep)

    # 数据集路径
    data_dir = os.path.join("datasets/train_data" + os.sep)
    tra_image_dir = os.path.join('image' + os.sep)
    tra_label_dir = os.path.join('mask' + os.sep)

    # val阶段
    num_classes = 4
    name_classes = ["_background_", "Inclusions", "Patches", "Scratches"]
    images_path = r"E:\pk\ca1\RNC14X\multi_U2NET\datasets\test_data\Image"
    gt_dir = r"E:\pk\ca1\RNC14X\multi_U2NET\datasets\test_data\mask"
    pred_dir = r"E:\pk\ca1\RNC14X\multi_U2NET\datasets\test_data\predict_masks"
    predict_label = r"E:\pk\ca1\RNC14X\multi_U2NET\datasets\test_data\predict_labels"
    miou_out_path = "miou_out_tab"
    seg_pretrain_u2netp_path = r'E:\pk\ca1\RNC14X\multi_U2NET\saved_models\pretrain_model\power_seg_u2netp.pth'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    epoch_num = 150
    batch_size = 2
    train_num = 0
    val_num = 0

    tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + '.jpg')

    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split(os.sep)[-1]
        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]
        tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + '.png')

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")

    train_num = len(tra_img_name_list)

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=Compose([
            RescaleT(512),
            RandomCrop(488),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    # ------- 3. define model --------
    if model_name == 'u2netp':
        net = U2NETP(3, num_classes)

    if torch.cuda.is_available():
        net.cuda()

    import torch.optim.lr_scheduler as lr_scheduler

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=0)

    # ------- 4.1 define lr_scheduler (ReduceLROnPlateau based on loss) --------
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 5445

    for epoch in range(0, epoch_num):
        net.train()

        # 使用 tqdm 包装 dataloader，显示训练进度条
        with tqdm(salobj_dataloader, desc=f"Epoch {epoch + 1}/{epoch_num}", unit="batch") as tepoch:
            for i, data in enumerate(tepoch):
                ite_num += 1
                ite_num4val += 1

                inputs, labels = data['image'], data['label']
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)

                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                                requires_grad=False)
                else:
                    inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
                loss2, loss = muti_CrossEntropyLoss_loss(d0, d1, d2, d3, d4, d5, d6, labels_v)

                loss.backward()
                optimizer.step()

                running_loss += loss.data.item()
                running_tar_loss += loss2.data.item()

                # 更新进度条的描述信息，显示当前 loss
                tepoch.set_postfix(loss=running_loss / ite_num4val, loss2=running_tar_loss / ite_num4val)

                # del temporary outputs and loss
                del d0, d1, d2, d3, d4, d5, d6, loss2, loss

                # 更新最小损失并保存模型
                if running_loss / ite_num4val < min_train_loss:
                    min_train_loss = running_loss / ite_num4val
                    torch.save(net.state_dict(), model_dir + model_name + "_min_loss.pth")

                if ite_num % save_frq == 0:
                    torch.save(net.state_dict(), model_dir + model_name + "_bce_itr_%d.pth" % ite_num)
                    model_path_save = model_dir + model_name + "_bce_itr_" + str(ite_num) + ".pth"

                    # 计算 IoU
                    current_miou = eval_print_miou(num_classes, name_classes, images_path, gt_dir, pred_dir,
                                                   predict_label, miou_out_path, model_path_save)

                    running_loss = 0.0
                    running_tar_loss = 0.0
                    net.train()
                    ite_num4val = 0

            # 在每个 epoch 结束时调用 lr_scheduler 更新学习率
            scheduler.step(min_train_loss)



