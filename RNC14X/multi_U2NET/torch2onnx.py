import onnx
import torch.onnx
from model.u2net_onnx import U2NET, U2NETP


u2net = U2NETP(3, 3)
print(u2net)
model_dir = "saved_models/u2netp/u2netp.pth"

if torch.cuda.is_available():
    u2net.load_state_dict(torch.load(model_dir))
    u2net.cuda()
    u2net.eval()
    # dummy_input = torch.randn(1, 3, 320, 320).cuda()  # 你模型的输入   NCHW
    dummy_input = torch.randn(1, 3, 1774, 2534).cuda()	  # 你模型的输入   NCHW
else:
    u2net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    u2net.eval()
    # dummy_input = torch.randn(1, 3, 320, 320)  # 你模型的输入   NCHW
    dummy_input = torch.randn(1, 3, 1774, 2534)  # 你模型的输入   NCHW


# batch_size:1,model_image_size:3*512*4096
# 关键参数 verbose=True 会使导出过程中打印出该网络的可读表示
# opset_version=14 用于 opencv
# opset_version=11 用于 onnxruntime
torch.onnx.export(u2net, dummy_input, 'saved_models/u2netp/u2net.onnx', verbose=True, opset_version=14)
onnx_model = onnx.load('saved_models/u2netp/u2netp.onnx')  # load onnx model
# torch.onnx.export(u2net, dummy_input, 'saved_models/mojiaoji/u2netp_bce_itr_15960.onnx', verbose=True, opset_version=11)
# onnx_model = onnx.load('saved_models/mojiaoji/u2netp_bce_itr_15960.onnx')  # load onnx model
onnx.checker.check_model(onnx_model)  # check onnx model
print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
