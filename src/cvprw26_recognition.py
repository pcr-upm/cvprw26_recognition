#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import numpy as np
from images_framework.src.annotations import GenericCategory
from images_framework.src.utils import load_geoimage
from images_framework.src.recognition import Recognition
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)


class CVPRW26Recognition(Recognition):
    """
    Facial expression recognition using ResNet
    """
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.model = None
        self.gpu = None
        self.batch_size = None
        self.epoch = None
        self.patience = None
        self.width = 224
        self.height = 224
        self.classes = None
        self.depth = None
        self.channels = None

    def parse_options(self, params):
        super().parse_options(params)
        import argparse
        parser = argparse.ArgumentParser(prog='CVPRW26Recognition', add_help=False)
        parser.add_argument('--gpu', dest='gpu', type=int, action='append',
                            help='GPU ID (negative value indicates CPU).')
        parser.add_argument('--batch-size', dest='batch_size', type=int, default=16,
                            help='Number of images in each mini-batch.')
        parser.add_argument('--epoch', dest='epoch', type=int, default=1000,
                            help='Number of sweeps over the dataset to train.')
        parser.add_argument('--patience', dest='patience', type=int, default=40,
                            help='Number of epochs with no improvement after which training will be stopped.')
        args, unknown = parser.parse_known_args(params)
        print(parser.format_usage())
        self.gpu = args.gpu
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.patience = args.patience

    def train(self, anns_train, anns_valid):
        print('Training model')

    def load(self, mode):
        from images_framework.src.constants import Modes
        # Set up a neural network to train
        print('Load model')
        if mode is Modes.TEST:
            sv_path = self.path + 'data/' + self.database + '/'
        print('CPU mode' if -1 in self.gpu else 'GPU mode with devices ' + str(self.gpu))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu[0])

    def process(self, ann, pred):
        for img_pred in pred.images:
            # Load image
            image, _ = load_geoimage(img_pred.filename, self.depth, self.channels)
            for obj_pred in img_pred.objects:
                warped_image = self.preprocess(image, obj_pred.bb)
                # Generate prediction
                warped_image = np.expand_dims(warped_image, 0)
                predictions = self.model.predict(warped_image, verbose=0)
                # Save prediction
                obj_pred.add_category(GenericCategory(self.classes[np.argmax(predictions)], np.max(predictions)))
