import argparse
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import os
import CNN_img
import motion
import MC_network
import load
import gc

import io
import os
import scipy.misc
import six
import time

from six import BytesIO

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def load_image_into_numpy_array(image):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path (this can be local or on colossus)

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Load the COCO Label Map
category_index = {
    1: {'id': 1, 'name': 'person'},
    2: {'id': 2, 'name': 'bicycle'},
    3: {'id': 3, 'name': 'car'},
    4: {'id': 4, 'name': 'motorcycle'},
    5: {'id': 5, 'name': 'airplane'},
    6: {'id': 6, 'name': 'bus'},
    7: {'id': 7, 'name': 'train'},
    8: {'id': 8, 'name': 'truck'},
    9: {'id': 9, 'name': 'boat'},
    10: {'id': 10, 'name': 'traffic light'},
    11: {'id': 11, 'name': 'fire hydrant'},
    13: {'id': 13, 'name': 'stop sign'},
    14: {'id': 14, 'name': 'parking meter'},
    15: {'id': 15, 'name': 'bench'},
    16: {'id': 16, 'name': 'bird'},
    17: {'id': 17, 'name': 'cat'},
    18: {'id': 18, 'name': 'dog'},
    19: {'id': 19, 'name': 'horse'},
    20: {'id': 20, 'name': 'sheep'},
    21: {'id': 21, 'name': 'cow'},
    22: {'id': 22, 'name': 'elephant'},
    23: {'id': 23, 'name': 'bear'},
    24: {'id': 24, 'name': 'zebra'},
    25: {'id': 25, 'name': 'giraffe'},
    27: {'id': 27, 'name': 'backpack'},
    28: {'id': 28, 'name': 'umbrella'},
    31: {'id': 31, 'name': 'handbag'},
    32: {'id': 32, 'name': 'tie'},
    33: {'id': 33, 'name': 'suitcase'},
    34: {'id': 34, 'name': 'frisbee'},
    35: {'id': 35, 'name': 'skis'},
    36: {'id': 36, 'name': 'snowboard'},
    37: {'id': 37, 'name': 'sports ball'},
    38: {'id': 38, 'name': 'kite'},
    39: {'id': 39, 'name': 'baseball bat'},
    40: {'id': 40, 'name': 'baseball glove'},
    41: {'id': 41, 'name': 'skateboard'},
    42: {'id': 42, 'name': 'surfboard'},
    43: {'id': 43, 'name': 'tennis racket'},
    44: {'id': 44, 'name': 'bottle'},
    46: {'id': 46, 'name': 'wine glass'},
    47: {'id': 47, 'name': 'cup'},
    48: {'id': 48, 'name': 'fork'},
    49: {'id': 49, 'name': 'knife'},
    50: {'id': 50, 'name': 'spoon'},
    51: {'id': 51, 'name': 'bowl'},
    52: {'id': 52, 'name': 'banana'},
    53: {'id': 53, 'name': 'apple'},
    54: {'id': 54, 'name': 'sandwich'},
    55: {'id': 55, 'name': 'orange'},
    56: {'id': 56, 'name': 'broccoli'},
    57: {'id': 57, 'name': 'carrot'},
    58: {'id': 58, 'name': 'hot dog'},
    59: {'id': 59, 'name': 'pizza'},
    60: {'id': 60, 'name': 'donut'},
    61: {'id': 61, 'name': 'cake'},
    62: {'id': 62, 'name': 'chair'},
    63: {'id': 63, 'name': 'couch'},
    64: {'id': 64, 'name': 'potted plant'},
    65: {'id': 65, 'name': 'bed'},
    67: {'id': 67, 'name': 'dining table'},
    70: {'id': 70, 'name': 'toilet'},
    72: {'id': 72, 'name': 'tv'},
    73: {'id': 73, 'name': 'laptop'},
    74: {'id': 74, 'name': 'mouse'},
    75: {'id': 75, 'name': 'remote'},
    76: {'id': 76, 'name': 'keyboard'},
    77: {'id': 77, 'name': 'cell phone'},
    78: {'id': 78, 'name': 'microwave'},
    79: {'id': 79, 'name': 'oven'},
    80: {'id': 80, 'name': 'toaster'},
    81: {'id': 81, 'name': 'sink'},
    82: {'id': 82, 'name': 'refrigerator'},
    84: {'id': 84, 'name': 'book'},
    85: {'id': 85, 'name': 'clock'},
    86: {'id': 86, 'name': 'vase'},
    87: {'id': 87, 'name': 'scissors'},
    88: {'id': 88, 'name': 'teddy bear'},
    89: {'id': 89, 'name': 'hair drier'},
    90: {'id': 90, 'name': 'toothbrush'},
}

#tensorflow2
#def preDetection():
#    start_time = time.time()
#    tf.keras.backend.clear_session()
#    with tf.Session() as sess:
    #detect_fn = tf.saved_model.load('models/research/object_detection/test_data/efficientdet_d5_coco17_tpu-32/saved_model/', tags=None, export_dir=".")
#        tf.saved_model.loader.load(sess, tags = ['serve'], export_dir = 'models/research/object_detection/test_data/efficientdet_d5_coco17_tpu-32/saved_model/')
#    model = tf.saved_model.load(export_dir = 'models/research/object_detection/test_data/efficientdet_d5_coco17_tpu-32/saved_model/', tags=["serve"])
#    end_time = time.time()
#    elapsed_time = end_time - start_time
#   print('Elapsed time: ' + str(elapsed_time) + 's')

def detectImage(img):

    elapsed = []

    image_np = load_image_into_numpy_array(img)
    input_tensor = np.expand_dims(image_np, 0)
    start_time = time.time()
    detections = detect_fn(input_tensor)
    end_time = time.time()
    elapsed.append(end_time - start_time)

    plt.rcParams['figure.figsize'] = [42, 21]
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    print(detections['detection_boxes'][0].numpy())
    print("second")
    print(detections['detection_classes'][0].numpy().astype(np.int32))
    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'][0].numpy(),
            detections['detection_classes'][0].numpy().astype(np.int32),
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.40,
            agnostic_mode=False)
    plt.subplot(2, 1, i+1)
    plt.imshow(image_np_with_detections)

    mean_elapsed = sum(elapsed) / float(len(elapsed))
    print('Elapsed time: ' + str(mean_elapsed) + ' second per image')

#preDetection()


config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
sess = tf.compat.v1.Session(config=config)

parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--l", type=int, default=1024, choices=[256, 512, 1024, 2048])
parser.add_argument("--N", type=int, default=128, choices=[128])
parser.add_argument("--M", type=int, default=128, choices=[128])
args = parser.parse_args()

if args.l == 256:
    I_QP = 37
elif args.l == 512:
    I_QP = 32
elif args.l == 1024:
    I_QP = 27
elif args.l == 2048:
    I_QP = 22

batch_size = 4
Height = 256
Width = 256
Channel = 3
lr_init = 1e-4

folder = np.load('folder.npy')

Y0_com = tf.compat.v1.placeholder(tf.float32, [batch_size, Height, Width, Channel])
Y1_raw = tf.compat.v1.placeholder(tf.float32, [batch_size, Height, Width, Channel])
learning_rate = tf.compat.v1.placeholder(tf.float32, [])

with tf.compat.v1.variable_scope("flow_motion"):

    flow_tensor, _, _, _, _, _ = motion.optical_flow(Y0_com, Y1_raw, batch_size, Height, Width)
    # Y1_warp_0 = tf.contrib.image.dense_image_warp(Y0_com, flow_tensor)

# Encode flow
flow_latent = CNN_img.MV_analysis(flow_tensor, args.N, args.M)

entropy_bottleneck_mv = tfc.EntropyBottleneck()
string_mv = entropy_bottleneck_mv.compress(flow_latent)
# string_mv = tf.squeeze(string_mv, axis=0)

flow_latent_hat, MV_likelihoods = entropy_bottleneck_mv(flow_latent, training=True)

flow_hat = CNN_img.MV_synthesis(flow_latent_hat, args.N)

# Motion Compensation
Y1_warp = tfa.image.dense_image_warp(Y0_com, flow_hat)

MC_input = tf.concat([flow_hat, Y0_com, Y1_warp], axis=-1)
Y1_MC = MC_network.MC(MC_input)

# Encode residual
Res = Y1_raw - Y1_MC

res_latent = CNN_img.Res_analysis(Res, num_filters=args.N, M=args.M)

entropy_bottleneck_res = tfc.EntropyBottleneck()
string_res = entropy_bottleneck_res.compress(res_latent)
# string_res = tf.squeeze(string_res, axis=0)

res_latent_hat, Res_likelihoods = entropy_bottleneck_res(res_latent, training=True)

Res_hat = CNN_img.Res_synthesis(res_latent_hat, num_filters=args.N)

# Reconstructed frame
Y1_com = Res_hat + Y1_MC


# Total number of bits divided by number of pixels.
train_bpp_MV = tf.reduce_sum(input_tensor=tf.math.log(MV_likelihoods)) / (-np.log(2) * Height * Width * batch_size)
train_bpp_Res = tf.reduce_sum(input_tensor=tf.math.log(Res_likelihoods)) / (-np.log(2) * Height * Width * batch_size)

# Mean squared error across pixels.
total_mse = tf.reduce_mean(input_tensor=tf.math.squared_difference(Y1_com, Y1_raw))
warp_mse = tf.reduce_mean(input_tensor=tf.math.squared_difference(Y1_warp, Y1_raw))
MC_mse = tf.reduce_mean(input_tensor=tf.math.squared_difference(Y1_raw, Y1_MC))

psnr = 10.0*tf.math.log(1.0/total_mse)/tf.math.log(10.0)

# The rate-distortion cost.
l = args.l

train_loss_total = l * total_mse + (train_bpp_MV + train_bpp_Res)
train_loss_MV = l * warp_mse + train_bpp_MV
train_loss_MC = l * MC_mse + train_bpp_MV

# Minimize loss and auxiliary loss, and execute update op.
step = tf.compat.v1.train.create_global_step()

train_MV = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss_MV, global_step=step)
train_MC = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss_MC, global_step=step)
train_total = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss_total, global_step=step)

aux_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate*10.0)
aux_step = aux_optimizer.minimize(entropy_bottleneck_mv.losses[0])

aux_optimizer2 = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate*10.0)
aux_step2 = aux_optimizer2.minimize(entropy_bottleneck_res.losses[0])

train_op_MV = tf.group(train_MV, aux_step, entropy_bottleneck_mv.updates[0])
train_op_MC = tf.group(train_MC, aux_step, entropy_bottleneck_mv.updates[0])
train_op_all = tf.group(train_total, aux_step, aux_step2,
                        entropy_bottleneck_mv.updates[0], entropy_bottleneck_res.updates[0])

tf.compat.v1.summary.scalar('psnr', psnr)
tf.compat.v1.summary.scalar('bits_total', train_bpp_MV + train_bpp_Res)
save_path = './OpenDVC_PSNR_' + str(l)
summary_writer = tf.compat.v1.summary.FileWriter(save_path, sess.graph)
saver = tf.compat.v1.train.Saver(max_to_keep=None)

sess.run(tf.compat.v1.global_variables_initializer())
var_motion = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='flow_motion')
saver_motion = tf.compat.v1.train.Saver(var_list=var_motion, max_to_keep=None)
#saver_motion.restore(sess, save_path='motion_flow/model.ckpt-200000')

# Train
iter = 0

while(True):

    if iter <= 100000:
        frames = 2

        if iter <= 20000:
            train_op = train_op_MV
        elif iter <= 40000:
            train_op = train_op_MC
        else:
            train_op = train_op_all
    else:
        frames = 7
        train_op = train_op_all

    if iter <= 300000:
        lr = lr_init
    elif iter <= 600000:
        lr = lr_init / 10.0
    else:
        lr = lr_init / 100.0

    data = np.zeros([frames, batch_size, Height, Width, Channel])
    data = load.load_data(data, frames, batch_size, Height, Width, Channel, folder, I_QP)

    for ff in range(frames-1):

        if ff == 0:

            F0_com = data[0]
            F1_raw = data[1]

            _, F1_decoded = sess.run([train_op, Y1_com],
                                     feed_dict={Y0_com: F0_com / 255.0,
                                                Y1_raw: F1_raw / 255.0,
                                                learning_rate: lr})

        else:

            F0_com = F1_decoded * 255.0
            F1_raw = data[ff+1]

            _, F1_decoded = sess.run([train_op, Y1_com],
                                     feed_dict={Y0_com: F0_com / 255.0,
                                                Y1_raw: F1_raw / 255.0,
                                                learning_rate: lr})

        print('Training_OpenDVC Iteration:', iter)

        iter = iter + 1

        if iter % 500 == 0:

             merged_summary_op = tf.compat.v1.summary.merge_all()
             summary_str = sess.run(merged_summary_op, feed_dict={Y0_com: F0_com/255.0,
                                                                  Y1_raw: F1_raw/255.0})

             summary_writer.add_summary(summary_str, iter)

        if iter % 20000 == 0:

             checkpoint_path = os.path.join(save_path, 'model.ckpt')
             saver.save(sess, checkpoint_path, global_step=iter)

    if iter > 700000:
        break

    del data
    del F0_com
    del F1_raw
    del F1_decoded

    gc.collect()
