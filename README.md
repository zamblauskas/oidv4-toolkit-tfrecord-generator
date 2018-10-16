[EscVM/OIDv4_ToolKit](https://github.com/EscVM/OIDv4_ToolKit) downloads images of classes of interest from [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html) instead of downloading the full set (many many GBs of data).  
This script generates TFRecord to be used for training a Tensorflow object detection model.

Usage:
```bash
python generate-tfrecord.py \
--classes_file=./OIDv4_ToolKit/classes.txt \
--class_descriptions_file=./OIDv4_ToolKit/OID/csv_folder/class-descriptions-boxable.csv \
--annotations_file=./OIDv4_ToolKit/OID/csv_folder/train-annotations-bbox.csv \
--images_dir=./OIDv4_ToolKit/OID/Dataset/train \
--output_file=./train.tfrecord
```
