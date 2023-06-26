import argparse
import time
from pathlib import Path

import cv2
import pandas as pd
import torch
import numpy as np
from PIL import Image
import torch.backends.cudnn as cudnn
from numpy import random
import torch.utils.data
from utils.datasets import letterbox
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--gray_scale', action='store_true', help='train grayscale image') #True
    parser.add_argument('--save_csv', action='store_false', help='save inference csv file') #True 
    parser.add_argument('--classifier', action='store_true', help='Dont classify') #False
    parser.add_argument('--save_normalized_boxes', action='store_true', help='dont normalize boxes') #False
    opt = parser.parse_args()
    return opt


class Prediction():
    def __init__(self,arg):
        self.opt=arg()
        self.classifier=self.opt.classifier
        self.save_csv=self.opt.save_csv
        self.save_normalized_boxes=self.opt.save_normalized_boxes
        self.gray_scale=self.opt.gray_scale
        
        self.source=self.opt.source
        self.weights=self.opt.weights
        self.view_img=self.opt.view_img
        self.save_txt=self.opt.save_txt
        self.imgsz=self.opt.img_size
        self.trace=not self.opt.no_trace
        self.device_=self.opt.device
        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1 

    
    
    def start(self):
        with torch.no_grad():
            if self.opt.update:  # update all models (to fix SourceChangeWarning)
                for self.weights in ['yolov7.pt']:
                    self.pred()
                    strip_optimizer(self.weights)
            else:
                self.pred()

    def device(self):
        device = select_device(self.device_)
        return device

    def save_dir(self):
        # Directories
        save_dir = Path(increment_path(Path(self.opt.project) / self.opt.name, exist_ok=self.opt.exist_ok))  # increment run
        (save_dir / 'labels' if self.opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        #save image
        save_img = not self.opt.nosave and not self.source.endswith('.txt')  # save inference images
        return save_dir,save_img
    

    def load_model(self,device):
        model = attempt_load(self.weights, map_location=device)
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(self.imgsz, s=stride)  # check img_size

        half = device.type != 'cpu'  # half precision only supported on CUDA
        if self.trace:# ===========> error
            model = TracedModel(model, device, imgsz,self.gray_scale)
        if half:
            model.half() # to FP16
            if self.gray_scale:
                model(torch.zeros(1, 1, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            else:
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        # Second-stage classifier
        if self.classifier:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
            return model,modelc,stride,imgsz,half
        else:
            return model,stride,imgsz,half
    
    def create_dataset(self,stride,imgsz):
        webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(self.source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(self.source, img_size=imgsz, stride=stride)
        return dataset,webcam
    
    def convert_img(self,img,device):
        if self.gray_scale:
            # img=img.detach().cpu().numpy()
            img=img.transpose(1,2,0)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img=img.reshape(img.shape[0],img.shape[1],1)
            img=img.transpose(2, 0, 1) 

        half = device.type != 'cpu'  # half precision only supported on CUDA
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32l
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img
    def check_input(self,path):
        if (isinstance(path,str)):
            if(path.endswith('npy')):
                image=np.load(path)
            else:
                image=cv2.imread(path)
        return image
    def prepare_numpy(self,path):
        device=self.device()
        half = device.type != 'cpu'
        image=self.check_input(path)
        # Convert
        im1=letterbox(image,self.imgsz,32)[0]
        if self.gray_scale:
            img=im1[:, :, ::-1]
            img=self.convert_and_reshape(img)
            image=self.convert_and_reshape(image)
            img = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32l
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            return image,img
        else:
            img = im1[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32l
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            return image,img
    def predict_numpy(self):
        save_dir,save_img=self.save_dir()
        device=self.device()
        if self.classifier:
            #Load model parameters
            model,modelc,stride,imgsz,half=self.load_model(device)
        else:
            model,stride,imgsz,half=self.load_model(device)
        dataset,webcam=self.create_dataset(stride,imgsz)
        
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        if self.save_csv:
            data={'file_name':[],'time_inference':[],'image_size':[]}
        # Init time
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img=self.convert_img(img,device)
            # Warmup
            if half:
                if self.gray_scale:
                    model(torch.zeros(1, 1, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                else:
                    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once


            if half and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
                self.old_img_b = img.shape[0]
                self.old_img_h = img.shape[2]
                self.old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=self.opt.augment)[0]
            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=self.opt.augment)[0]
            t2 = time_synchronized()
            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if self.classifier:
                pred = apply_classifier(pred, modelc, img, im0s)
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                    if self.gray_scale:
                        im0=cv2.cvtColor(im0,cv2.COLOR_BGR2GRAY).reshape(im0.shape[0],im0.shape[1],1)
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                    if self.gray_scale:
                        im0=cv2.cvtColor(im0,cv2.COLOR_BGR2GRAY).reshape(im0.shape[0],im0.shape[1],1)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg or runs/detect
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # runs/detect/exp18/labels/right_test_9
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per classs
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        #____________________________________coordinate____________________________________#
                        # print(f'coordinate {xyxy} \n check save txt:{save_txt}')
                        if self.save_txt:  # Write to file
                            if self.save_normalized_boxes:
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            #___________________________Optional save unormalize box________________________#
                            else:
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # unnormalized xywh
                            line = (cls, *xywh, conf) if self.opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        if save_img or self.view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            if self.save_csv:
                inference_time=f'{(1E3 * (t2 - t1)):.1f}'
                data['file_name'].append(p.stem)
                data['time_inference'].append(inference_time)
                data['image_size'].append(imgsz)                
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            if self.view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
        if self.save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if self.save_txt else ''
        if self.save_csv:
            # Create a DataFrame from the data
            df = pd.DataFrame(data)
            #Save csv file
            file_save=f'inference_{imgsz}.csv'
            # Save data to CSV file
            df.to_csv(file_save, index=False)
        print(f'Done. ({time.time() - t0:.3f}s)')

        

        

    def convert_and_reshape(self,img):
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY).reshape(img.shape[0],img.shape[1],1)
        return img



    def pred(self):
        save_dir,save_img=self.save_dir()
        device=self.device()
        if self.classifier:
            #Load model parameters
            model,modelc,stride,imgsz,half=self.load_model(device)
        else:
            model,stride,imgsz,half=self.load_model(device)
        dataset,webcam=self.create_dataset(stride,imgsz)
        
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        if self.save_csv:
            data={'file_name':[],'time_inference':[],'image_size':[]}
        # Init time
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img=self.convert_img(img,device)
            # Warmup
            if half:
                if self.gray_scale:
                    model(torch.zeros(1, 1, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                else:
                    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

            if half and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
                self.old_img_b = img.shape[0]
                self.old_img_h = img.shape[2]
                self.old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=self.opt.augment)[0]
            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=self.opt.augment)[0]
            t2 = time_synchronized()
            # Apply NMS
            print(self.opt)
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if self.classifier:
                pred = apply_classifier(pred, modelc, img, im0s)
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                    if self.gray_scale:
                        im0=cv2.cvtColor(im0,cv2.COLOR_BGR2GRAY).reshape(im0.shape[0],im0.shape[1],1)
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                    if self.gray_scale:
                        im0=cv2.cvtColor(im0,cv2.COLOR_BGR2GRAY).reshape(im0.shape[0],im0.shape[1],1)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg or runs/detect
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # runs/detect/exp18/labels/right_test_9
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per classs
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        #____________________________________coordinate____________________________________#
                        # print(f'coordinate {xyxy} \n check save txt:{save_txt}')
                        if self.save_txt:  # Write to file
                            if self.save_normalized_boxes:
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            #___________________________Optional save unormalize box________________________#
                            else:
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # unnormalized xywh
                            line = (cls, *xywh, conf) if self.opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        if save_img or self.view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            if self.save_csv:
                inference_time=f'{(1E3 * (t2 - t1)):.1f}'
                data['file_name'].append(p.stem)
                data['time_inference'].append(inference_time)
                data['image_size'].append(imgsz)                
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            if self.view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
        if self.save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if self.save_txt else ''
        if self.save_csv:
            # Create a DataFrame from the data
            df = pd.DataFrame(data)
            #Save csv file
            file_save=f'inference_{imgsz}.csv'
            # Save data to CSV file
            df.to_csv(file_save, index=False)
        print(f'Done. ({time.time() - t0:.3f}s)')

def main():
    config=Prediction(arg)

    path='/home/tonyhuy/yolov7/blister_data/crop_img/left_test_10.png'

    image,img=config.prepare_numpy(path)
    print(image.shape,img.shape)
    config.start()
    

# demo
if __name__=='__main__':
    main()
