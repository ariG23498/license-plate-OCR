"""
Use shadow net to recognize the scene text of a single image
"""
import argparse
import os.path as ops

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glog as logger
import wordninja

from config import global_config
from crnn_model import crnn_net
from data_provider import tf_io_pipline_fast_tools

CFG = global_config.cfg




def recognize(image, weights_path, char_dict_path, ord_map_dict_path, is_vis=False, is_english=True):
    """

    :param image_path:
    :param weights_path:
    :param char_dict_path:
    :param ord_map_dict_path:
    :param is_vis:
    :return:
    """
    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    new_heigth = 32
    scale_rate = new_heigth / image.shape[0]
    new_width = int(scale_rate * image.shape[1])
    new_width = new_width if new_width > CFG.ARCH.INPUT_SIZE[0] else CFG.ARCH.INPUT_SIZE[0]
    image = cv2.resize(image, (new_width, new_heigth), interpolation=cv2.INTER_LINEAR)
    image_vis = image
    image = np.array(image, np.float32) / 127.5 - 1.0

    inputdata = tf.placeholder(
        dtype=tf.float32,
        shape=[1, new_heigth, new_width, CFG.ARCH.INPUT_CHANNELS],
        name='input'
    )

    codec = tf_io_pipline_fast_tools.CrnnFeatureReader(
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path
    )

    net = crnn_net.ShadowNet(
        phase='test',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=CFG.ARCH.NUM_CLASSES
    )

    inference_ret = net.inference(
        inputdata=inputdata,
        name='shadow_net',
        reuse=False
    )

    decodes, _ = tf.nn.ctc_beam_search_decoder(
        inputs=inference_ret,
        sequence_length=int(new_width / 4) * np.ones(1),
        merge_repeated=False,
        beam_width=10
    )

    # config tf saver
    saver = tf.train.Saver()

    # config tf session
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        preds = sess.run(decodes, feed_dict={inputdata: [image]})

        preds = codec.sparse_tensor_to_str(preds[0])[0]
        if is_english:
            preds = ' '.join(wordninja.split(preds))

        # logger.info('Predict image {:s} result: {:s}'.format(
        #     ops.split(image_path)[1], preds)
        # )

        if is_vis:
            plt.figure('CRNN Model Demo')
            plt.imshow(image_vis[:, :, (2, 1, 0)])
            plt.show()

    sess.close()

    return preds