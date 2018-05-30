class configRes():
    train_dir = 'D:/data/5/train'  # 训练集数据
    val_dir = 'D:/data/5/test'  # 验证集数据
    resnet_model_save_dir = 'D:/data/resnet.h5'
    nb_classes = 5
    nb_epoch = 3
    batch_size = 16
    IM_WIDTH, IM_HEIGHT = 224, 224
    FC_SIZE = 1024  # 全连接层的节点个数
    NB_IV3_LAYERS_TO_FREEZE = 170  # 冻结层的数量
    lr = 0.0001
    momentum = 0.9
    draw = True
class configInc():
    train_dir = 'D:/data/5/train'  # 训练集数据
    val_dir = 'D:/data/5/test'  # 验证集数据
    inception_model_save_dir = 'D:/data/inception.h5'
    nb_classes = 5
    nb_epoch = 3
    batch_size = 16
    IM_WIDTH, IM_HEIGHT = 229, 229  # InceptionV3指定的图片尺寸
    FC_SIZE = 1024  # 全连接层的节点个数
    NB_IV3_LAYERS_TO_FREEZE = 170  # 冻结层的数量
    lr = 0.0001
    momentum = 0.9
    draw = False
