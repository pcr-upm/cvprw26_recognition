#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.getcwd())
import cv2
import copy
import numpy as np
import importlib.util
from tqdm import tqdm
from pathlib import Path
from images_framework.src.constants import Modes
from images_framework.src.composite import Composite
from images_framework.src.viewer import Viewer
from src.cvprw26_recognition import CVPRW26Recognition


def parse_options():
    """
    Parse options from command line.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--anns-file', '-a', dest='anns_file', required=True,
                        help='Ground truth annotations file.')
    parser.add_argument('--input-data', '-d', dest='input_data', required=False, default='',
                        help='Input as image, directory, camera or video file.')
    parser.add_argument('--show-viewer', '-v', dest='show_viewer', action="store_true",
                        help='Show results visually.')
    parser.add_argument('--save-file', '-f', dest='save_file', action="store_true",
                        help='Save experiments in a text file.')
    parser.add_argument('--save-image', '-i', dest='save_image', action="store_true",
                        help='Save processed images.')
    args, unknown = parser.parse_known_args()
    print(parser.format_usage())
    anns_file = args.anns_file
    show_viewer = args.show_viewer
    save_file = args.save_file
    save_image = args.save_image
    return unknown, anns_file, show_viewer, save_file, save_image


def load_annotations(anns_file):
    """
    Load ground truth annotations according to each database.
    """
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from images_framework.src.annotations import GenericGroup, GenericImage, PersonObject, GenericCategory
    from images_framework.src.categories import Name
    from images_framework.categories.emotions import Emotion as Oe
    from src.load_csv import parse_rafdb, parse_affectnet
    print('Open annotations file: ' + str(anns_file))
    if os.path.isfile(anns_file) and anns_file == 'csv/rafdb_test_pose_bboxq_illum.csv':
        dbpath = '/home/database/classification/faces/expressions/raf/basic/'
        gender = {0: Name('Male'), 1: Name('Female')}
        race = {0: Name('Caucasian'), 1: Name('African-American'), 2: Name('Asian')}
        category = {0: Oe.FACE.SURPRISE, 1: Oe.FACE.FEAR, 2: Oe.FACE.DISGUST, 3: Oe.FACE.HAPPINESS, 4: Oe.FACE.SADNESS, 5: Oe.FACE.ANGER, 6: Oe.FACE.NEUTRAL}
        samples = parse_rafdb(Path(anns_file))
    elif os.path.isfile(anns_file) and anns_file == 'csv/affectnetplus_test_annotations_quality_illum.csv':
        dbpath = '/home/database/classification/faces/expressions/affectnet/'
        gender = {0: Name('Female'), 1: Name('Male')}
        race = {0: Name('Asian'), 1: Name('Indian'), 2: Name('Black'), 3: Name('White'), 4: Name('Middle-Eastern'), 5: Name('Latino-Hispanic')}
        category = {0: Oe.FACE.NEUTRAL, 1: Oe.FACE.HAPPINESS, 2: Oe.FACE.SADNESS, 3: Oe.FACE.SURPRISE, 4: Oe.FACE.FEAR, 5: Oe.FACE.DISGUST, 6: Oe.FACE.ANGER, 7: Oe.FACE.CONTEMPT}
        samples = parse_affectnet(Path(anns_file))
    else:
        raise ValueError('Annotations file does not exist')
    anns = []
    for sample in tqdm(samples, file=sys.stdout):
        seq = GenericGroup()
        image = GenericImage(dbpath + sample['path'])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        obj = PersonObject()
        obj.headpose = Rotation.from_euler('YXZ', [float(sample['yaw']), float(sample['pitch']), float(sample['roll'])], degrees=True).as_matrix()
        obj.add_attribute(GenericCategory(label=gender[int(sample['gender'])]))
        obj.add_attribute(GenericCategory(label=race[int(sample['race'])]))
        obj.add_attribute(GenericCategory(label=Name('illumination'), score=float(sample['illumination'])))
        obj.add_category(GenericCategory(category[int(sample['expression'])]))
        image.add_object(obj)
        seq.add_image(image)
        anns.append(seq)
    return anns


def main():
    """
    Facial expression recognition database script.
    """
    unknown, anns_file, show_viewer, save_file, save_image = parse_options()
    # Load computer vision components
    composite = Composite()
    sr = CVPRW26Recognition('')
    composite.add(sr)
    composite.parse_options(unknown)
    anns = load_annotations(anns_file)
    composite.load(Modes.TEST)
    spec = importlib.util.find_spec('images_framework')
    output_path = os.path.join('images_framework' if spec is None else os.path.dirname(spec.origin), 'output')
    if show_viewer:
        viewer = Viewer('images_viewer')
    if save_file:
        ofs = open(output_path+'/results.txt', 'w', encoding='utf-8')
    if save_image:
        viewer = Viewer('images_save')
        dirname = os.path.join(output_path, 'images/')
        Path(dirname).mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(len(anns)), file=sys.stdout):
        pred = copy.deepcopy(anns[i])
        for img_pred in pred.images:
            for obj_pred in img_pred.objects:
                obj_pred.clear()
                if obj_pred.bb == (-1, -1, -1, -1):
                    if all(np.array_equal(contour, np.array([[[-1, -1]], [[-1, -1]], [[-1, -1]]])) for contour in obj_pred.multipolygon):
                        obj_pred.bb = cv2.boundingRect(np.array([pt for contour in obj_pred.multipolygon for pt in contour]))
                        obj_pred.bb = (obj_pred.bb[0], obj_pred.bb[1], obj_pred.bb[0] + obj_pred.bb[2], obj_pred.bb[1] + obj_pred.bb[3])
                    else:
                        raise ValueError('Cannot perform alignment due to undefined object location')
        composite.process(anns[i], pred)
        if show_viewer:
            for img_pred in pred.images:
                viewer.set_image(img_pred)
            composite.show(viewer, anns[i], pred)
            viewer.show(0)
        if save_file:
            composite.evaluate(ofs, anns[i], pred)
        if save_image:
            for img_pred in pred.images:
                viewer.set_image(img_pred)
            composite.show(viewer, anns[i], pred)
            viewer.save(dirname)
            composite.save(dirname, pred)
    if save_file:
        ofs.close()


if __name__ == '__main__':
    main()
