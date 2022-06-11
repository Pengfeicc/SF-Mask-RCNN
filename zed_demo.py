import torch
import argparse
from models import maskrcnn
import pyzed.sl as sl
import numpy as np
import torchvision.transforms as transforms
import threading
import yaml
import time
import cv2
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
  
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default='rgb_noisydepth_confidencefusion', help=".yaml")
    parser.add_argument("--thresh", type=float, default=0.7, help="thresh")
    parser.add_argument("--weight_path", type=str, default='weights/SFMask_RGBD_ConfidenceFusion.tar',
                        help="if it is given, evaluate this")
    args = parser.parse_args()

    with open('cfgs/' + args.cfg + '.yaml') as config_file:
        cfg = yaml.safe_load(config_file)

    model = maskrcnn.get_model_instance_segmentation(cfg=cfg)
    model.load_state_dict(torch.load(args.weight_path))
    model.cuda()
    model.eval()
    
    init_cap_params = sl.InitParameters()
    init_cap_params.camera_resolution = sl.RESOLUTION.HD720
    init_cap_params.depth_mode = sl.DEPTH_MODE.QUALITY
    init_cap_params.coordinate_units = sl.UNIT.MILLIMETER
    init_cap_params.depth_minimum_distance = 350
    init_cap_params.depth_maximum_distance = 800
    init_cap_params.depth_stabilization = True
    init_cap_params.camera_image_flip = sl.FLIP_MODE.AUTO
    init_cap_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    model = maskrcnn.get_model_instance_segmentation(cfg=cfg)
    model.load_state_dict(torch.load(args.weight_path))
    model.cuda()
    model.eval()

    cap = sl.Camera()
    if not cap.is_opened():
        print("Opening ZED Camera...")
    status = cap.open(init_cap_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()
        
    display = True
    #runtime_daw = sl.RuntimeParameters()
    runtime_fill = sl.RuntimeParameters(sensing_mode=sl.SENSING_MODE.FILL) # add depth filtering when raw depth image as input
    left = sl.Mat()
    depth_img = sl.Mat()
    res = sl.Resolution(640, 360)

    running = True
    while running:
        start_time = time.time()
        err_code = cap.grab(runtime_fill)
        if err_code != sl.ERROR_CODE.SUCCESS:
            break
        cap.retrieve_image(left, sl.VIEW.LEFT, resolution=res)
        cap.retrieve_measure(depth_img, sl.MEASURE.DEPTH, resolution=res)
        rgb = left.get_data()
        rgb = rgb[:, :, :3] # slip BGRA to BGR
        depth = depth_img.get_data()
        depth = torch.from_numpy(depth)
        depth = depth.repeat(3, 1, 1)

        depth = (depth - init_cap_params.depth_minimum_distance) / (init_cap_params.depth_maximum_distance - init_cap_params.depth_minimum_distance)

        val_mask = torch.ones([depth.shape[1], depth.shape[2]])
        val_mask[np.where(depth[0] == 0.0)] = 0
        val_mask = val_mask.unsqueeze(0)

        rgb_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img = torch.cat([rgb_transform(rgb), depth, val_mask], dim=0)
        img = img.unsqueeze(0)
        images = list(image.to(device) for image in img)

        pred_result = model(images)[0]
        pred_mask = pred_result['masks'].cpu().detach().numpy()
        pred_labels = pred_result['labels']
        pred_scores = pred_result['scores']
        pred_boxes = pred_result['boxes']
        img_arr = []

        for num in range(len(pred_result['labels'])):
            mask_arr = pred_mask[num][0]
            img_arr.append(mask_arr)

        object_idx = [i for i in range(len(pred_labels)) if pred_labels[i] == 1]
        result_image = draw_prediction(rgb, object_idx, img_arr, pred_boxes, pred_scores, args.thresh)
        cv2.imshow("demo", result_image)
        k = cv2.waitKey(100)
        if k == 27:
            cv2.destroyWindow()
            cap.close()
            break

def draw_prediction(rgb_img, object_idx, img_arr, boxes, score, thresh):

    if len(object_idx) == 0:
        return False

    rgb_img = np.uint8(rgb_img.transpose(0, 1, 2))

    for i in object_idx:
        if score[i] > thresh:
            c = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))
            color = np.uint8(c)
            obj_img = img_arr[i]

            obj_img[obj_img >= 0.5] = 1
            obj_img[obj_img < 0.5] = 0

            r = obj_img * color[0]
            g = obj_img * color[1]
            b = obj_img * color[2]

            stacked_img = np.stack((r, g, b), axis=0)
            stacked_img = stacked_img.transpose(1, 2, 0)

            rgb_img = cv2.addWeighted(rgb_img, 1, stacked_img.astype(np.uint8), 0.5, 0) # draw the mask on the rgb images

    rgb_img = np.uint8(rgb_img)
    return rgb_img



if __name__ == "__main__":
    main()
