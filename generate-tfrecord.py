import pandas as pd
import tensorflow as tf
from PIL import Image
import os

tf.app.flags.DEFINE_string('classes_file', None, 'Path to the text file containing downloaded classes, name per line')
tf.app.flags.DEFINE_string('class_descriptions_file', None, 'Path to the CSV file with class id mapping to human readable label name')
tf.app.flags.DEFINE_string('annotations_file', None, 'Path to the CSV file with bbox annotations')
tf.app.flags.DEFINE_string('images_dir', None, 'Path to the directory with downloaded images')
tf.app.flags.DEFINE_string('output_file', None, 'Path to the resulting TFRecord file')
tf.app.flags.mark_flags_as_required([
    'classes_file',
    'class_descriptions_file',
    'annotations_file',
    'images_dir',
    'output_file'
])
FLAGS = tf.app.flags.FLAGS


def main(_):
    classes = list(filter(None, open(FLAGS.classes_file).read().split('\n')))
    classes = {name: idx + 1 for idx, name in enumerate(classes)}
    print(f'Classes: {classes}')

    class_descriptions = {row[0]: row[1] for _, row in pd.read_csv(FLAGS.class_descriptions_file, header=None).iterrows()}

    annotations = pd.read_csv(FLAGS.annotations_file)
    annotations['LabelName'] = annotations['LabelName'].map(lambda n: class_descriptions[n])
    annotations = annotations.groupby('ImageID')

    images = tf.gfile.Glob(FLAGS.images_dir + '/*/*.jpg')
    images = map(lambda i: (os.path.basename(i).split('.jpg')[0], i), images)
    images = dict(images)
    print(f'{len(images)} images')

    writer = tf.python_io.TFRecordWriter(FLAGS.output_file)
    for image_id, path in images.items():
        img_width, img_height = Image.open(path).size
        img_data = tf.gfile.GFile(path, 'rb').read()

        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes_int = []

        image_annotations = annotations.get_group(image_id)
        for _, row in image_annotations.loc[image_annotations['LabelName'].isin(classes.keys())].iterrows():
            xmins.append(row['XMin'])
            xmaxs.append(row['XMax'])
            ymins.append(row['YMin'])
            ymaxs.append(row['YMax'])
            classes_text.append(row['LabelName'].encode('utf8'))
            classes_int.append(classes[row['LabelName']])

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_height])),
            'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_width])),
            'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_id.encode('utf8')])),
            'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_id.encode('utf8')])),
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_data])),
            'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpg'])),
            'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
            'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
            'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
            'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
            'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
            'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes_int))
        }))

        writer.write(tf_example.SerializeToString())
        print('.', end='')
    writer.close()
    print(" done")

if __name__ == '__main__':
    tf.app.run()