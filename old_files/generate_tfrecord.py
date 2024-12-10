import os
import tensorflow as tf
from lxml import etree
import pandas as pd
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

def create_example(image_path, annotations, label_map_dict):
    # Read the image data
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()
    image_format = b'jpeg'  # You can change this depending on your image type (e.g. 'png')

    # Create tf.train.Example
    xmins = [annotation['xmin'] for annotation in annotations]
    xmaxs = [annotation['xmax'] for annotation in annotations]
    ymins = [annotation['ymin'] for annotation in annotations]
    ymaxs = [annotation['ymax'] for annotation in annotations]
    classes_text = [annotation['class'] for annotation in annotations]
    classes = [label_map_dict[class_text] for class_text in classes_text]

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/height': dataset_util.int64_feature(annotations[0]['height']),
        'image/width': dataset_util.int64_feature(annotations[0]['width']),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return example

def parse_annotations(label_file, label_map_dict):
    # Assuming the label file is in XML format
    tree = etree.parse(label_file)
    root = tree.getroot()

    annotations = []
    for obj in root.iter('object'):
        class_name = obj.find('name').text
        class_id = label_map_dict[class_name]
        bndbox = obj.find('bndbox')

        # Normalize bounding boxes to [0, 1] range based on image size
        xmin = float(bndbox.find('xmin').text) / float(root.find('size').find('width').text)
        xmax = float(bndbox.find('xmax').text) / float(root.find('size').find('width').text)
        ymin = float(bndbox.find('ymin').text) / float(root.find('size').find('height').text)
        ymax = float(bndbox.find('ymax').text) / float(root.find('size').find('height').text)

        annotations.append({
            'class': class_name,
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax,
            'width': root.find('size').find('width').text,
            'height': root.find('size').find('height').text
        })
    
    return annotations

def create_tf_example(row, label_map_dict):
    # Access the image filename from 'external id'
    image_path = os.path.join('kili_data/images', row['external id'] + '.jpg')  # Ensure your image has a .jpg extension if needed

    # Parse the label file (assuming it's a corresponding XML or similar)
    annotation_file = row['label file']
    annotations = parse_annotations(annotation_file, label_map_dict)

    # Create tf.train.Example from annotations
    return create_example(image_path, annotations, label_map_dict)

def generate_tfrecord(csv_file, label_map_dict, output_path):
    # Create a writer for the output TFRecord file
    writer = tf.io.TFRecordWriter(output_path)

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Iterate through each row in the DataFrame and create TFRecord entries
    for index, row in df.iterrows():
        tf_example = create_tf_example(row, label_map_dict)
        writer.write(tf_example.SerializeToString())

    writer.close()

def create_label_map(label_list, label_map_file):
    # Create a label map file
    with open(label_map_file, 'w') as f:
        for i, label in enumerate(label_list):
            f.write(f"item {{\n  id: {i+1}\n  name: '{label}'\n}}\n")

def main():
    # Configuration
    csv_file = './kili_data/images/remote_assets.csv'  # Path to your CSV file
    label_map_file = './kili_data/label_map.pbtxt'  # Path to the label map file
    output_tfrecord = 'train.record'  # Path for output TFRecord file

    # Define the classes in your dataset
    label_list = ['block', 'specimen']  # Replace with your actual class names
    create_label_map(label_list, label_map_file)

    # Load the label map and create the dictionary
    label_map_dict = label_map_util.get_label_map_dict(label_map_file)

    # Generate TFRecord file
    generate_tfrecord(csv_file, label_map_dict, output_tfrecord)

    print("TFRecord file generated!")

if __name__ == '__main__':
    main()
