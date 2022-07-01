
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import YOLOLoss, weights_init
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, get_classes
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    Cuda            = True

    classes_path    = 'model_data/voc_classes.txt'

    anchors_path    = 'model_data/yolo_anchors.txt'
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    model_path      = ''
    input_shape     = [640, 640]

    num_workers         = 8

    train_annotation_path   = '2012_train.txt'
    val_annotation_path     = '2007_val.txt'

    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)

    #------------------------------------------------------#
    #   创建yolo模型
    #------------------------------------------------------#
    model = YoloBody(anchors_mask, num_classes)
    weights_init(model)

    if model_path !='':
    #------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    yolo_loss    = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask)
    loss_history = LossHistory("logs/")

    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    batch_size  = 20
    lr          = 1e-4

    end_epoch   = 200
        
    optimizer       = optim.SGD(model_train.parameters(), lr, weight_decay = 5e-4)
    lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

    train_dataset   = YoloDataset(train_lines, input_shape, num_classes, train = True)
    val_dataset     = YoloDataset(val_lines, input_shape, num_classes, train = False)
    gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
    gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
                        
    epoch_step      = num_train // batch_size
    epoch_step_val  = num_val // batch_size
        
    if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")


    for epoch in range(0, end_epoch):
        fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch,
                epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
        lr_scheduler.step()
