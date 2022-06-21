
from model.fcos import FCOS
import os
import torch
from datasets.dataset_voc import MyVOCDetection
from datasets.dataset_voc_tiny import VOC2007DetectionTiny
import multiprocessing
from model.inference import inference_for_validation

if __name__ == '__main__':  

    # # for mAP evaluation
    os.system('rm -rf mAP')
    os.system('git clone https://github.com/Cartucho/mAP.git')
    os.system('rm -rf mAP/input/*')

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    # Set a few constants related to data loading.
    NUM_CLASSES = 20
    BATCH_SIZE = 1#16
    IMAGE_SHAPE = (224, 224)
    NUM_WORKERS = multiprocessing.cpu_count()
    GOOGLE_DRIVE_PATH = '.'

    weights_path = os.path.join('.', "fcos_weights_c128_l4_it40k_builtin_fpn_regnet_x_3_2gf.pt")

    fpn_strides = {
        'p3': 8,
        'p4': 16,
        'p5': 32,
    }

    val_dataset = MyVOCDetection(
        GOOGLE_DRIVE_PATH, "val", image_size=IMAGE_SHAPE[0],
        download=False  # True (for the first time)
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=NUM_WORKERS
    )

    detector = FCOS(NUM_CLASSES, fpn_strides,fpn_channels=128,stem_channels=[128, 128, 128, 128],device='cpu')

    detector.to(device=DEVICE)
    detector.load_state_dict(torch.load(weights_path, map_location="cpu")['model_state_dict'])
    #detector.load_state_dict(torch.load(weights_path, map_location="cpu"))

    inference_for_validation(
        detector,
        val_loader,
        val_dataset.idx_to_class,
        score_thresh=0.5,
        nms_thresh=0.5,
        device=DEVICE,
        output_dir="mAP/input",
    )
    
    os.system('cd mAP && python main.py')