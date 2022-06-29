
from model.fcos import FCOS
import os
import torch
from datasets.dataset_voc import MyVOCDetection
from datasets.dataset_voc_tiny_padded import VOC2007DetectionTiny
import multiprocessing
from model.inference import inference_for_validation

if __name__ == '__main__':  



    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    # Set a few constants related to data loading.
    NUM_CLASSES = 20
    BATCH_SIZE = 1#16
    IMAGE_SHAPE = (224, 224)
    NUM_WORKERS = multiprocessing.cpu_count()
    dataset_path = os.path.join(os.sep,'Volumes','Storage','Datasets')
    mAP_path = os.path.join(os.sep,'Volumes','Storage','Datasets','mAP')

    weights_path = os.path.join('weights', "fcos_weights_c256_l4_builtin_fpn_regnet_x_3_2gf.pt")

    # # for mAP evaluation
    print('rm -rf ' + mAP_path)
    os.system('rm -rf ' + mAP_path)
    os.system('git clone https://github.com/Cartucho/mAP.git ' + mAP_path)
    os.system('rm -rf ' + mAP_path +'/input/*')

    fpn_strides = {
        'p3': 8,
        'p4': 16,
        'p5': 32,
    }

    val_dataset = MyVOCDetection(
        os.path.join(dataset_path,'voc'), year="2007", image_set="test", image_size=IMAGE_SHAPE[0],
        download=True  # True (for the first time)
    )
    # val_dataset = VOC2007DetectionTiny(
    #     os.path.join(dataset_path,'voc_tiny'), "val", image_size=IMAGE_SHAPE[0],
    #     download=False  # True (for the first time)
    # )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=NUM_WORKERS
    )

    detector = FCOS(NUM_CLASSES, fpn_strides,fpn_channels=256,stem_channels=[256,256,256,256],device='cpu')

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
        output_dir=os.path.join(mAP_path,'input'),
    )
    
    os.system('cd ' + mAP_path +' && python main.py')