import albumentations as A
import cv2
from enum import IntEnum
import glob
import argparse
import yaml


class ImageCompressionType(IntEnum):
    JPEG = 0
    WEBP = 1
class Auto_augment():
    def __init__(self,path,idx,save_dir,augment_name,save):
        self.path=path
        self.save_dir=save_dir
        self.save=save
        self.idx=idx
        self.augment_name=augment_name[self.idx]
        self.name=self.path.split('_')[-1].split('.')[0]
    def __len__(self):
        return len(self.path)

    def read_images(self):
        image = cv2.imread(self.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    def choose_augment(self):
        transform = A.Compose([
                    A.RandomCrop(width=256, height=256),
                    A.HorizontalFlip(p=0.5),
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5),
                    A.HueSaturationValue (hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
                    A.ChannelShuffle(),
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
                    A.RandomContrast(limit=0.2, always_apply=False, p=0.5),
                    A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5),
                    A.RandomBrightness (limit=0.2, always_apply=False, p=0.5),
                    A.Blur(blur_limit=7, always_apply=False, p=0.5),
                    A.MedianBlur(blur_limit=7, always_apply=False, p=0.5),
                    A.ToGray(),
                    A.ImageCompression(quality_lower=99, quality_upper=100, compression_type=ImageCompressionType.JPEG, always_apply=False, p=0.5)
                    ])
        return transform[self.idx]

    def transform(self):
        image=self.read_images()
        transform=self.choose_augment()
        # Augment an image
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        if self.save:
            cv2.imwrite(f'{self.save_dir}/test_{self.name}_{self.augment_name}.png',transformed_image)
        return transformed_image
    
def main(opt):
    with open(opt.augment_yaml, 'r') as file:
        augment_name= yaml.safe_load(file)['augment_name']
    img_dir=sorted(glob.glob(f'{opt.path}/*.png'))
    for path in img_dir:
        for idx in range(len(augment_name)):
            aug=Auto_augment(path,idx,opt.save_dir,augment_name,opt.save)
            image=aug.transform()

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/tonyhuy/yolov7_blister/blister_data/images/test/', help='this is path to image')
    parser.add_argument('--save', action='store_true', help='save augmented images')
    parser.add_argument('--save_dir', type=str, default='/home/tonyhuy/yolov7_blister/augmented_data', help='this is where aug images are saved')
    parser.add_argument('--augment_yaml', type=str, default='/home/tonyhuy/yolov7_blister/data/augment_list.yaml', help='this is where aug images are saved')
    opt = parser.parse_args()
    return opt

if __name__=="__main__":
    opt=args()
    main(opt)
