import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2 #Use OpenCV instead of Matplot

from collections import defaultdict
from io import StringIO

#sys.path.append("..")
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'inference_graph'#'faster_rcnn_resnet101_coco_2018_01_28'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

NUM_CLASSES = 11


global detection_graph
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def run_inference_for_single_image(image, graph):
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
      return output_dict

from tensorflow import keras
import tensorflow as tf
import numpy as np
#import log

config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)

keras.backend.set_session(session)
#model_ = keras.models.load_model('crime_model.model')

INPUT_SHAPE = (-1, 224, 224, 3)

def predict_(image_arr):
    try:
        with session.as_default():
            with session.graph.as_default():
                image_arr = np.array(image_arr).reshape(INPUT_SHAPE)
                predicted_labels = model_.predict(image_arr, verbose=1)
                return predicted_labels
    except Exception as ex:
        pass
        #log.log('Prediction Error', ex, ex.__traceback__.tb_lineno)
#cats_ = [i for i in os.listdir('Dataset/')]
video_file = '1'
cap = cv2.VideoCapture(0)
video_type = cv2.VideoWriter_fourcc(*'MJPG')
video_overlay = cv2.VideoWriter("%s_overlay.avi" % (video_file), video_type, 20.0, (700, 600))
with detection_graph.as_default():
  with tf.Session() as sess:
    while True:
      ret, image_np = cap.read()
      image_np = image_np[::-1]
      image_np = cv2.flip(image_np,1)
      image = cv2.resize(image_np, (300, 300))
      img = image.copy()
      img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      img = cv2.resize(img,(224,224))
      #model_._make_predict_function()
      #actual = predict_(img)
      #print(model_.predict([img.reshape(-1,224,224,3)]))
      
      #actual = np.argmax(actual)
      output_dict = run_inference_for_single_image(image_np, detection_graph)
      vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],output_dict['detection_scores'],category_index,instance_masks=output_dict.get('detection_masks'),use_normalized_coordinates=True,line_thickness=8)

        
      #cv2.putText(image_np,str(cats_[actual]),(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255))
      video_overlay.write(image_np)
      cv2.imshow('object detection', cv2.resize(image_np, (700,700)))
      key = cv2.waitKey(33)
      if key==27:
        video_overlay.release()
        cv2.destroyWindow()
        break
cap.release()

