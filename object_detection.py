import cv2
import numpy as np
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from distutils.version import StrictVersion
from collections import defaultdict
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


title = "Obj Detection"
PATH_TO_FROZEN_GRAPH = 'frozen_inference_graph.pb'
PATH_TO_LABELS = 'labelmap.pbtxt'
NUM_CLASSES = 10


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')



def Captcha_detection(image, average_distance_error=3):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            
            image_np = cv2.imread(image)
            image_np = cv2.resize(image_np, (0,0), fx=3, fy=3) 
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=2)
            cv2.imshow(title, cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            print('Press enter to view next decoded captcha')
            
            cv2.waitKey(0)
            #


            captcha_array = []
            for i,b in enumerate(boxes[0]):
                for Symbol in range(37):
                    if classes[0][i] == Symbol: # check if detected class equal to our symbols
                        if scores[0][i] >= 0.65: # do something only if detected score more han 0.65
                                            # x-left        # x-right
                            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2 # find x coordinates center of letter
                            # to captcha_array array save detected Symbol, middle X coordinates and detection percentage
                            captcha_array.append([category_index[Symbol].get('name'), mid_x, scores[0][i]])

            for number in range(20):
                for captcha_number in range(len(captcha_array)-1):
                    if captcha_array[captcha_number][1] > captcha_array[captcha_number+1][1]:
                        temporary_captcha = captcha_array[captcha_number]
                        captcha_array[captcha_number] = captcha_array[captcha_number+1]
                        captcha_array[captcha_number+1] = temporary_captcha


            average = 0
            captcha_len = len(captcha_array)-1
            while captcha_len > 0:
                average += captcha_array[captcha_len][1]- captcha_array[captcha_len-1][1]
                captcha_len -= 1
            # Increase average distance error
            average = average/(len(captcha_array)+average_distance_error)

            
            captcha_array_filtered = list(captcha_array)
            captcha_len = len(captcha_array)-1
            while captcha_len > 0:
                if captcha_array[captcha_len][1]- captcha_array[captcha_len-1][1] < average:
                    # check which symbol has higher detection percentage
                    if captcha_array[captcha_len][2] > captcha_array[captcha_len-1][2]:
                        del captcha_array_filtered[captcha_len-1]
                    else:
                        del captcha_array_filtered[captcha_len]
                captcha_len -= 1
            captcha_string = ""
            for captcha_letter in range(len(captcha_array_filtered)):
                captcha_string += captcha_array_filtered[captcha_letter][0]
                
            print('CAPTCHA STRING = ',captcha_string)
            cv2.imwrite("output_/{}.jpg".format(captcha_string), cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

if os.path.exists('output_'):
    pass
else:
    os.makedirs('output_')


all_ = [i for i in os.listdir('test_data/')]
if len(all_)<1:
    print('No data found in test_data')
else:
    for i in all_:
        Captcha_detection('test_data/{}'.format(i))

