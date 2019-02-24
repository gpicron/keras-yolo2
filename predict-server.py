#! /usr/bin/env python

import argparse
import asyncio
import os
import cv2
import numpy as np
from tqdm import tqdm
from werkzeug.utils import secure_filename

from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json
from flask import Flask, request, Response
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

argparser.add_argument(
    '-l',
    '--lite',
    action='store_true',
    help='use tf lite')


def _main_(args):
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input
    use_lite = args.lite

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    ###############################
    #   Make the model
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'],
                labels              = config['model']['labels'],
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################

    yolo.load_weights(weights_path)

    global graph
    graph = tf.get_default_graph()

    # Initialize the Flask application
    app = Flask(__name__)


    # route http posts to this method
    @app.route('/detect', methods=['POST'])
    def detect():

        file = request.files['data']

        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        # Move the file form the temporal folder to
        # the upload folder we setup
        tmp_filename = os.path.join('/tmp', filename)

        file.save(tmp_filename)
        img = cv2.imread(tmp_filename)

        with graph.as_default():
            boxes = yolo.predict(img)


        # build a response dict to send back to client
        image_h, image_w, _ = img.shape
        response = {
            'size': '{}x{}'.format(image_w, image_h),
            'detections': [
                {
                    'label' : config['model']['labels'][bb.get_label()],
                    'score' : float(bb.get_score()),
                    'xmin' : int(bb.xmin * image_w),
                    'xmax' : int(bb.xmax * image_w),
                    'ymin' : int(bb.ymin * image_h),
                    'ymax' : int(bb.ymax * image_h)
                } for bb in boxes
            ]
        }

        response_json = json.dumps(response)

        return Response(response=response_json, status=200, mimetype="application/json")


    # start flask app
    app.run(host="0.0.0.0", port=5000)

if __name__ == '__main__':
    args = argparser.parse_args()


    _main_(args)
