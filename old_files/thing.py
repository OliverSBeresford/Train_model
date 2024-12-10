import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection import model_lib_v2

# Paths configuration
base_dir = "kili_data"  # Your dataset folder
pipeline_config_path = "pipeline.config"  # Path to your pipeline config (you can create or modify it)
model_dir = "model_output"  # For saving checkpoints and exported models
train_record_path = "train.record"  # Generated TFRecord file for training data
val_record_path = "val.record"  # Generated TFRecord file for validation data
label_map_path = "label_map.pbtxt"  # Path to your custom label map

# Define the function to convert Pascal VOC to TFRecord
def voc_to_tfrecord():
    os.system(f"python generate_tfrecord.py -x {base_dir}/images -a {base_dir}/labels -o {train_record_path}")
    os.system(f"python generate_tfrecord.py -x {base_dir}/images/val -a {base_dir}/labels/val -o {val_record_path}")

# Ensure that the label map file exists
if not os.path.exists(label_map_path):
    print(f"Label map file not found at {label_map_path}. Please create it.")
    exit()

# Load and modify pipeline config
configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

with tf.io.gfile.GFile(pipeline_config_path, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

# Modify configuration
pipeline_config.model.ssd.num_classes = 3  # Set this to the number of your classes
pipeline_config.train_config.batch_size = 8  # Adjust as needed
pipeline_config.train_config.fine_tune_checkpoint = "efficientdet-lite0-checkpoint-path"  # Path to EfficientDet checkpoint
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_config.optimizer.adam_optimizer.learning_rate.constant_learning_rate.learning_rate = 0.01

# Set input paths for TFRecord files
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [train_record_path]
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [val_record_path]

# Set the label map path
pipeline_config.train_input_reader.label_map_path = label_map_path
pipeline_config.eval_input_reader[0].label_map_path = label_map_path

# Save updated pipeline config
config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(pipeline_config_path, "wb") as f:
    f.write(config_text)

# Train the model
model_lib_v2.train_loop(
    pipeline_config_path=pipeline_config_path,
    model_dir=model_dir,
    train_steps=10000  # Adjust based on your needs
)

# Export to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
tflite_model = converter.convert()

# Save the TFLite model
with open("efficientdet_lite0_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model training and export completed!")
