# ParNet-Pytorch
English:\
Based on pytorch reproduction, the parnet [1] network suitable for yolo target detection backbone, retaining the convolution part. \
The down_sampling of the first layer is improved to Focus, and the SPP module in yolo is introduced in the final stage \
Lead out three outputs and adjust the number of channels to 256, 512, 1024, which can be directly connected with YOLO's PanFPN \
enviroments:
pytorch>=1.7.1\
tensorboard (If you don’t need visualization, you don’t need to install it, just clear the call in main) \
Experimental results: \
YOLOX； ParNet=s ；NVIDIA 3080ti ；forward time 253.24ms\
                    darknetnet53 ；forward time 5.68ms\
Refer to part of the logic of https://github.com/Pritam-N/ParNet, but the overall code is still its own style\
Note: Don’t think that the amount of calculation is small just because the number of layers is small. This amount of calculation is too high to use in one GPU. In the midnight of December 6, 2021, it broke my defenses.\
Official code: https://github.com/imankgoyal/NonDeepNetworks 
 
 

中文： \
基于pytorch复现的适用于yolo目标检测backbone的parnet[1]网络，保留了卷积部分\
并且将第一层的down_sampling改进为Focus,在最后阶段引入yolo中的SPP模块\
引出三路输出并将其通道数调整为256,512,1024，可直接与YOLO的PanFPN连接\
环境：\
pytorch==1.8\
tensorboard（如果不需要可视化，不需要安装，清除main中调用即可）\
实验结果：\
YOLOX； ParNet=s ；NVIDIA 3080ti ；forward time 253.24ms\
                    darknetnet53 ；forward time 5.68ms\
注：不要因为看着层数少就认为计算量少，这计算量 完全就是大力出奇迹，2021年12月6号的凌晨，大半夜的给我搞破防了\
有参考https://github.com/Pritam-N/ParNet 的部分逻辑，但代码整体还是自己的风格\
官方代码：https://github.com/imankgoyal/NonDeepNetworks \
[1]Goyal, Ankit, Alexey Bochkovskiy, Jia Deng and Vladlen Koltun. “Non-deep Networks.” ArXiv abs/2110.07641 (2021): n. pag.
