#_______________________________ Run training file____________________________________#
# Image size 320
python train.py --workers 8 --device 3  --batch-size 8 --data data/blister.yaml --img 320  --cfg cfg/training/yolov7.yaml --weights 'yolov7.pt' --name yolov7 --hyp data/hyp.scratch.p5.yaml

# Image size 640
python train.py --workers 8 --device 1  --batch-size 8 --data data/blister.yaml --img 640  --cfg cfg/training/yolov7.yaml --weights 'yolov7.pt' --name yolov7 --hyp data/hyp.scratch.p5.yaml

#_______________________________ Run inference file____________________________________#
python inference_blisters.py

#_______________________________ Generate augment data____________________________________#
python hand_augment.py  --path /home/tonyhuy/yolov7_blister/blister_data/images/test/ --save --save_dir /home/tonyhuy/yolov7_blister/augmented_data --augment_yaml data/augment_list.yaml 



