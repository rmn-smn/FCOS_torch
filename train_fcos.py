import os
import torch
from datasets.dataset_voc import MyVOCDetection
from model.trainer import Trainer
from model.fcos import FCOS
import multiprocessing
from model.utils import seed_everything

if __name__ == '__main__':

    NUM_CLASSES = 20
    BATCH_SIZE = 16
    IMAGE_SHAPE = (768, 768)
    MAXITER = 60000
    NUM_WORKERS = multiprocessing.cpu_count()
    dataset_path = os.path.join(os.sep,'Volumes','Storage','Datasets','voc')
    weights_path = os.path.join('.', 'weights','test.pt')

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    seed_everything()

    fpn_strides = {
        'p3': 8,
        'p4': 16,
        'p5': 32,
    }

    train_dataset_07 = MyVOCDetection(
        dataset_path ,year = "2007", image_set = 'trainval', download = False, image_size=IMAGE_SHAPE
    )
    train_dataset_12 = MyVOCDetection(
        dataset_path ,year = "2012", image_set = 'trainval', download = False, image_size=IMAGE_SHAPE
    )
    train_dataset_0712 = torch.utils.data.ConcatDataset([train_dataset_07,train_dataset_12])

    micro_dataset = torch.utils.data.Subset(
        train_dataset_0712,
        torch.linspace(0, 5, steps=5).long()
    )

    train_loader = torch.utils.data.DataLoader(train_dataset_0712, batch_size=BATCH_SIZE,
                    pin_memory=True)

    micro_train_loader = torch.utils.data.DataLoader(
        micro_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=NUM_WORKERS
    )

    fcos = FCOS(NUM_CLASSES, fpn_strides,fpn_channels=64,stem_channels=[64, 64],device=DEVICE)

    trainer = Trainer(
        fcos, 
        train_loader, 
        max_iter=MAXITER,
        max_lr=8e-3,
        weight_decay = 1e-4,
        checkpoint_path=weights_path,
        checkpoint_interval=10,
        print_interval= 1
    )
    #trainer.load_checkpoint(weights_path)
    trainer.fit()
