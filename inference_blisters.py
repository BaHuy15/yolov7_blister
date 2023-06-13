

import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import time
import cv2
import glob
import torch


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# from models.common import DetectMultiBackend
from utils.datasets import LoadStreams, LoadImages
from models.experimental import attempt_load

# from yolo_utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
# from yolo_utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
#                            increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh, yaml_load)
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# from yolo_utils.plots import Annotator, colors, save_one_box
from utils.general import scale_coords as scale_boxes
from utils.plots import plot_one_box
# from yolo_utils.torch_utils import select_device, smart_inference_mode
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

class Prediction():
    """
    Prediction class used for running the predictions yolov5 model
    """
    def __init__(self, weights,class_name, img_size = 1024, device = 0, fp16 = False, batch_size = 1, conf_thresh = 0.5,
                data = 'yolov5_inference/data/coco128.yaml', save_dir = None):
        """
        Initializes the prediction with the specified weights, image size, device, precision type, batch size,
        confidence threshold and data
        :param weights: Model weights used for inference
        :param img_size: (int) Image size used for inference
        :param device: GPU device index used for running inference
        :param fp16: Whether or not to use a 16 bit precision (fp16)
        :param batch_size: Batch size used for running inference
        :param conf_thresh: Confidence threshold used when generating detections
        :param data: Data used when running the inference
        :param save_dir: Save directory used when saving results
        """
        self.img_size = (img_size, img_size)
        self.conf_threshold = conf_thresh
        self.weights = weights
        self.device = select_device(device)
        self.data = data
        self.fp16 = fp16
        #================Modify=================#
        self.trace=True
        self.gray_scale=True # False ---> train RGB
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.class_names = class_name #None #yaml_load(data)['names'] if data else {i: f'class{i}' for i in range(999)}
        if self.save_dir is not None:
            os.makedirs(save_dir, exist_ok = True)
            self.save_result = True
        else:
            self.save_result = False

        self.init_model()
    def init_model(self):
        self.model = attempt_load(self.weights, map_location=self.device)
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.img_size[0], s=stride)  # check img_size
        half = self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.trace:# ===========> error
            self.model = TracedModel(self.model, self.device, imgsz,self.gray_scale)
        if half:
            self.model.half() # to FP16
            if self.gray_scale:
                self.model(torch.zeros(1, 1, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
            else:
                self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters()))) 

    def apply_constrast(self,Image, MaskWidth, MaskHeight, Factor):
        imageGray = cv2.cvtColor(Image, cv2.COLOR_RGB2GRAY)
        imageGray = cv2.bilateralFilter(imageGray, 9, 75, 75)
        _, region = cv2.threshold(imageGray, 1, 255, cv2.THRESH_BINARY)
        region3 = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
        imageMean = cv2.GaussianBlur(Image, (MaskWidth, MaskHeight),0)
        imageMean16 = np.int16(imageMean)
        image16 = np.int16(Image)
        emphasize16 = ((image16 - imageMean16) * Factor) + image16
        emphasize16[emphasize16>255] = 255
        emphasize16[emphasize16<0] = 0
        emphasize = np.uint8(emphasize16)
        emphasize = cv2.bitwise_and(emphasize, region3)
        return emphasize
    
    def preprocess_images(self, images, factor_cont = None , use_contrast = False):
        assert isinstance(images, list)
        x = []
        for image in images:
            assert image is not None
            if self.gray_scale:
                image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, self.img_size, interpolation = cv2.INTER_AREA).reshape(self.img_size[0],self.img_size[1],-1)
            if use_contrast:
                image = self.apply_constrast(image,15,15,factor_cont)

            x.append((image * (1/255)).transpose((2, 0, 1)))
        return np.asarray(x).astype(np.float32)
        # assert isinstance(images, list)
        # x = []
        # for image in images:
        #     assert image is not None
        #     image = cv2.resize(src=image, dsize=self.img_size, interpolation = cv2.INTER_AREA)
        #     if use_contrast:
        #         image = self.apply_constrast(image,15,15,factor_cont)
        #     image = np.float32(image)
        #     image = image*(1/255)
        #     image = image.transpose((2, 0, 1))
        #     x.append(image)
        # x = np.asarray(x).astype(np.float32)
        # return x
    def tuple_size(self,image_size):
        if (isinstance(image_size,tuple)):
            return image_size[0]
        else:
            return image_size
        
    def predict(self, images, crop_coordinate = None, use_contrast = False):
        start_time = time.time()
        self.check_input_images(images)
        im = self.preprocess_images(images, factor_cont = 2.5, use_contrast = use_contrast) # preprocessing all images

        # with self.dt[0]:
        im = torch.from_numpy(im).to(self.device)
        #==================================modify==============================#
        half = self.device.type != 'cpu' 
        im = im.half() if half else im.float()  # uint8 to fp16/32l

        # im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if half:
            size=self.tuple_size(self.img_size)
            if self.gray_scale:
                self.model(torch.zeros(1, 1,size,size).to(self.device).type_as(next(self.model.parameters())))  # run once
            else:
                self.model(torch.zeros(1, 3,size,size).to(self.device).type_as(next(self.model.parameters())))  # run once

        preprocessing_time = time.time()

        # Inference
        with torch.no_grad():
            # with self.dt[1]:
            preds = self.model(im, augment= False)[0]

        inference_time = time.time()

        # NMS
        # with self.dt[2]:
        preds = non_max_suppression(preds, conf_thres = self.conf_threshold, iou_thres = 0.45, classes=None ,agnostic=False)
        non_max_suppression_time = time.time()
        self.print_time(start_time, preprocessing_time, inference_time, non_max_suppression_time)

        for i, det in enumerate(preds):  # per image
            origin_image = images[i]
            if self.gray_scale:
                origin_image=cv2.cvtColor(origin_image,cv2.COLOR_BGR2GRAY).reshape(origin_image.shape[0],origin_image.shape[1],-1)
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], origin_image.shape).round()

            # Save the result image
            if self.save_result:
                save_image = origin_image
                for detection in det:
                    """Detection: (x1 ,y1 ,x2 ,y2 ,score ,class_id)"""
                    # Draw rectangle
                    save_image = cv2.rectangle(save_image, (int(detection[0]), int(detection[1])), (int(detection[2]),\
                        int(detection[3])), color = (255, 255, 255), thickness = 2)
                    class_id = self.class_names[int(detection[5])]
                    score = round(float(detection[4]), 2)

                    # Insert class and score
                    insert_position = (int(detection[0]) - 15 if int(detection[0]) > 15 else 0, int(detection[1]) - 15 if int(detection[1]) > 15 else 0)
                    save_image = cv2.putText(save_image, f'{class_id}_{score}', insert_position , \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 if save_image.shape[0] < 512 else 0.8, (255, 255, 255), 1 , cv2.LINE_AA)

                cv2.imwrite(os.path.join(self.save_dir,f"image_{i}.png"), origin_image)


        # Convert to original coordinate # Mean that this is cropped images, so we need to convert position to original coordinate
        if crop_coordinate is not None:
            for i, pred in enumerate(preds):
                for j ,detection in enumerate(pred):
                    preds[i][j][0] = preds[i][j][0] + crop_coordinate[0]
                    preds[i][j][1] = preds[i][j][1] + crop_coordinate[1]
                    preds[i][j][2] = preds[i][j][2] + crop_coordinate[0]
                    preds[i][j][3] = preds[i][j][3] + crop_coordinate[1]

        return preds

    def check_input_images(self, images):
        assert isinstance(images, list), "images param should be a list of images, [image1,..image_n]"
        assert len(images) == self.batch_size, "number of input images should be equal with batch_size"

    def print_time(self, start_time, preprocessing_time, inference_time, nms_time):# print_time
        """This function prints the time it took for the pre-process, inference and NMS
            stages when performing an object detection task."""

        preprocess = int((preprocessing_time - start_time) * 1000)
        inference = int((inference_time - preprocessing_time) * 1000)
        nms = int((nms_time - inference_time)* 1000)
        total = preprocess + inference + nms

        print('Speed: %.1fms ; pre-process: %.1fms, inference: %.1fms, NMS: %.1fms per %d images at shape %s'
              % (total, preprocess, inference, nms, self.batch_size, self.img_size))
def save_run(path):
    # Save directory
    if path.endswith('crop_img'):
        save_dir_1='/home/tonyhuy/yolov7_blister/blister_data/predict_crop_data'
        return save_dir_1
    if path.endswith('test'):
        save_dir_2='/home/tonyhuy/yolov7_blister/blister_data/predict_test_data'
        return save_dir_2
def main():
    # path to weights file this file train grayscale image [1,h,w]
    weights='/home/tonyhuy/yolov7/runs/train/yolov749/weights/last.pt'
    #path to folder contains image
    path='/home/tonyhuy/yolov7_blister/blister_data/crop_img' # '/home/tonyhuy/yolov7/blister_data/images/test'
    save_dir=save_run(path)
    device = '0'
    img_size = 640
    conf_thresh = 0.5
    class_name=['blister_on_hand','blister']
    images=[]
    img_path=glob.glob(f'{path}/*.png')
    for path in img_path:
        img=cv2.imread(path)
        images.append(img)
    batch_size=60 #in cropdata have 60 images,number of batch size = number of image in inference file
    data = None#'yolov5_inference/data/coco128.yaml'
    cfg=Prediction(weights,class_name,img_size = img_size, device = device, fp16 = False, batch_size = batch_size, conf_thresh=conf_thresh,
                data = data, save_dir = save_dir)
    
    pred=cfg.predict(images, crop_coordinate = None, use_contrast = False)


# demo
if __name__=='__main__':
    main()
