# ParNet
基于pytorch复现的适用于yolo目标检测backbone的parnet[1]网络，保留了卷积部分，并且引入yolo中的SPP模块\
引出三路输出并将其通道数调整为256,512,1024，可直接与YOLO的PanFPN连接\
环境：\
pytorch==1.8\
tensorboard（如果不需要可视化，不需要安装，清除main中调用即可）\
官方代码：https://github.com/imankgoyal/NonDeepNetworks \
[1]Goyal, Ankit, Alexey Bochkovskiy, Jia Deng and Vladlen Koltun. “Non-deep Networks.” ArXiv abs/2110.07641 (2021): n. pag.\
