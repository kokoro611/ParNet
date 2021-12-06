# ParNet-Pytorch
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
官方代码：https://github.com/imankgoyal/NonDeepNetworks \
[1]Goyal, Ankit, Alexey Bochkovskiy, Jia Deng and Vladlen Koltun. “Non-deep Networks.” ArXiv abs/2110.07641 (2021): n. pag.
