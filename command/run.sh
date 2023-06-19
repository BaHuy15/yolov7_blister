#_______________________________ Run training file____________________________________#
# Image size 320
python train.py --workers 8 --device 3  --batch-size 8 --data data/blister.yaml --img 320  --cfg cfg/training/yolov7.yaml --weights 'yolov7.pt' --name yolov7 --hyp data/hyp.scratch.p5.yaml

# Image size 640
python train.py --workers 8 --device 1  --batch-size 8 --data data/blister.yaml --img 640  --cfg cfg/training/yolov7.yaml --weights 'yolov7.pt' --name yolov7 --hyp data/hyp.scratch.p5.yaml

#_______________________________ Run inference file____________________________________#
#If inference grayscale
python inference_blisters.py --gray_scale --weights /home/tonyhuy/yolov7/runs/train/yolov749/weights/last.pt --image_size 640  --device '2' --path /home/tonyhuy/yolov7_blister/blister_data/crop_img --conf_thresh 0.5 --batch_size 60  --class_name /home/tonyhuy/yolov7_blister/data/blister.yaml
#If not
python inference_blisters.py  --weights /home/tonyhuy/yolov7/runs/train/yolov749/weights/last.pt --image_size 640  --device '2' --path /home/tonyhuy/yolov7_blister/blister_data/crop_img --conf_thresh 0.5 --batch_size 60  --class_name /home/tonyhuy/yolov7_blister/data/blister.yaml

#_______________________________ Generate augment data____________________________________#
python hand_augment.py  --path /home/tonyhuy/yolov7_blister/blister_data/images/test/ --save --save_dir /home/tonyhuy/yolov7_blister/augmented_data --augment_yaml data/augment_list.yaml 



