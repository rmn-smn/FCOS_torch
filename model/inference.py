import os
import torch
import shutil
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def inverse_image_transform(image):
    image = transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            )(image)
    image = transforms.Normalize(
                mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
            )(image)
    return image

def show_image(image,targets = None, predictions = None, idx_to_class = None):
    # if channels is first dimension permute to last dimension
    if image.shape[0] == 3:
        image = image.permute(1,2,0)
    # display image
    fig, ax = plt.subplots()
    ax.imshow(image)
    # add bounding boxes if provided
    if targets is not None:
        for target in targets:
            # voc box format is 
            # [x-top-left, y-top-left, x-bottom-right, y-bottom-right]
            # (x,y) lower
            x0, y0, x1, y1 = target[:4]
            if idx_to_class is not None and target[4].item() in idx_to_class.keys():
                label = idx_to_class[target[4].item()]
            else:
                label = target[4]
            # box width, height
            h,w = abs(x1-x0),abs(y1-y0)
            # add rectangle and center point to plot
            box = Rectangle((x0,y0),h,w,fill=False,
                edgecolor='blue')
            ax.add_patch(box) 
            ax.text(x0,y0+10,label,backgroundcolor='blue')

    if predictions is not None:
        for pred in predictions:
            # voc box format is 
            # [x-top-left, y-top-left, x-bottom-right, y-bottom-right]
            # (x,y) lower
            x0, y0, x1, y1 = pred[:4]
            if idx_to_class is not None and pred[4].item() in idx_to_class.keys():
                label = idx_to_class[pred[4].item()]
            else:
                label = pred[4]
            score = pred[5]
            # box width, height
            h,w = abs(x1-x0),abs(y1-y0)
            # add rectangle and center point to plot
            box = Rectangle((x0,y0),h,w,fill=False,
                edgecolor='green')
            ax.add_patch(box) 
            ax.text(
                x0,y0+10,'{}: {:.2}'.format(int(label),float(score)),
                backgroundcolor='green')
    plt.show()
            

def inference_for_validation(
    model,
    dataloader,
    idx_to_class,
    score_thresh = 0.5,
    nms_thresh = 0.5,
    output_dir = None,    
    device = "cpu"
):
    # move model to device
    model.to(device = device)

    # set model to inference mode
    model.eval()

    # if output_dir is provided create directories to store predictions 
    # and target values to calculate mAP
    if output_dir is not None:
        det_dir = os.path.join(output_dir,'detection-results')
        gt_dir = os.path.join(output_dir,'ground-truth')
        if os.path.exists(det_dir):
            shutil.rmtree(det_dir)
        os.mkdir(det_dir)
        if os.path.exists(gt_dir):
            shutil.rmtree(gt_dir)
        os.mkdir(gt_dir)

    # inference as implemented in the detector only works for single images
    # so we need to itarate over all input batches as well as all the 
    # batch dimension.
    for batch in dataloader:        
        for i,(im_path,image,labels) in enumerate(zip(*batch)):
            # move images and labels to device
            image = image.to(device)
            labels = labels.to(device)
            # disable gradient calc for performance
            with torch.no_grad():
                    # dims: pred_boxes(num_preds, 4) pred_classes(num_preds,) 
                    #       pred_scores(num_preds,)
                    pred_boxes, pred_classes, pred_scores = model(
                        image.unsqueeze(0),
                        test_score_thresh=score_thresh,
                        test_nms_thresh=nms_thresh,
                    )
            # skip current image if no predictions are returned
            if pred_classes.shape[0] == 0:
                continue

            # filter out background (negative) labels
            valid_labels = labels[:, 4] != -1
            labels = labels[valid_labels]

            # filter out background (negative) predictions
            valid_pred = pred_classes != -1
            pred_boxes = pred_boxes[valid_pred]
            pred_classes = pred_classes[valid_pred]
            pred_scores = pred_scores[valid_pred]

            # unnormalize image for display
            image = inverse_image_transform(image)

            # concat predictions in to single tensor
            predictions = torch.cat(
                [pred_boxes, pred_classes.unsqueeze(1), 
                pred_scores.unsqueeze(1)], dim=1
            )     

            # write results to specified directory for mAP  calculation
            # use mAP code https://github.com/Cartucho/mAP
            if output_dir is not None:
                file_name = os.path.basename(im_path).replace(".jpg", ".txt")
                with open(os.path.join(det_dir, file_name), "w") as f_det, open(
                    os.path.join(gt_dir, file_name), "w"
                ) as f_gt:
                    for b in labels:
                        f_gt.write(
                            f"{idx_to_class[b[4].item()]} {b[0]:.2f} {b[1]:.2f} {b[2]:.2f} {b[3]:.2f}\n"
                        )
                    for b in predictions:
                        f_det.write(
                            f"{idx_to_class[b[4].item()]} {b[5]:.6f} {b[0]:.2f} {b[1]:.2f} {b[2]:.2f} {b[3]:.2f}\n"
                        )
            else:
                show_image(image,labels,predictions,idx_to_class)      
