#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.insert(0, os.getcwd())
import copy
import numpy as np
import pandas as pd
import importlib.util
from tqdm import tqdm
from pathlib import Path
from scipy.spatial.transform import Rotation
from pcr_framework.src.constants import Modes
from pcr_framework.src.composite import Composite
from pcr_framework.src.viewer import Viewer
from src.cvprw26_recognition import CVPRW26Recognition


def parse_options():
    """
    Parse options from command line.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-path', '-d', dest='db_path', required=True,
                        help='Ground truth annotations path.')
    parser.add_argument('--show-viewer', '-v', dest='show_viewer', action="store_true",
                        help='Show results visually.')
    parser.add_argument('--save-file', '-f', dest='save_file', action="store_true",
                        help='Save experiments in a text file.')
    parser.add_argument('--save-image', '-i', dest='save_image', action="store_true",
                        help='Save processed images.')
    parser.add_argument('--demographic-factor', dest='demographic_factor', required=True, choices=['Gender', 'Race'],
                        help='Demographic factor.')
    parser.add_argument('--confounding-factor', dest='confounding_factor', required=True, choices=['Pose', 'Illumination'],
                        help='Confounding factor.')
    args, unknown = parser.parse_known_args()
    print(parser.format_usage())
    db_path = args.db_path
    show_viewer = args.show_viewer
    save_file = args.save_file
    save_image = args.save_image
    demo_factor = args.demographic_factor
    conf_factor = args.confounding_factor
    return unknown, db_path, show_viewer, save_file, save_image, demo_factor, conf_factor


def load_annotations(db_path, database):
    """
    Load ground truth annotations according to each database.
    """
    from PIL import Image
    from pcr_framework.src.datasets import Database
    from pcr_framework.src.annotations import GenericVideo, GenericImage, PersonObject, GenericCategory
    from pcr_framework.categories.emotions import Emotion as Oe
    from src.load_csv import parse_rafdb, parse_affectnet, parse_affwild2, parse_multipie
    datasets = [subclass().get_names() for subclass in Database.__subclasses__()]
    idx = next((idx for idx, subset in enumerate(datasets) if database in subset), None)
    if idx is None:
        raise ValueError(f'Database is not implemented: {database}')
    db = Database.__subclasses__()[idx]()
    categories = db.get_categories()
    if 'raf' in db.get_names():
        gender = {0: 'Male', 1: 'Female'}
        race = {0: 'Caucasian', 1: 'African-American', 2: 'Asian'}
        samples = parse_rafdb(Path('csv/rafdb_test_pose_bboxq_illum.csv'))
    elif 'affectnet' in db.get_names():
        gender = {0: 'Female', 1: 'Male'}
        race = {0: 'Asian', 1: 'Indian', 2: 'Black', 3: 'White', 4: 'Middle-Eastern', 5: 'Latino-Hispanic'}
        samples = parse_affectnet(Path('csv/affectnetplus_test_annotations_quality_illum.csv'))
    elif 'affwild2' in db.get_names():
        gender = {0: 'Female', 1: 'Male'}
        race = {0: 'Asian', 1: 'Indian', 2: 'Black', 3: 'White'}
        samples = parse_affwild2(Path('csv/dataframe_val_pose_demographic_illum.csv'))
    elif 'multipie' in db.get_names():
        gender = {0: 'Female', 1: 'Male'}
        race = {0: 'Asian', 1: 'Indian', 2: 'Black', 3: 'White'}
        samples = parse_multipie(Path('csv/test_expr_gender_pose_balanced.csv'))
        # samples = parse_multipie(Path('csv/balanced_subset.csv'))
        # samples = parse_multipie(Path('csv/biased_subset.csv'))
    anns = []
    for sample in tqdm(samples, file=sys.stdout):
        seq = GenericVideo()
        image = GenericImage(db_path + sample['path'])
        width, height = Image.open(image.filename).size
        image.tile = np.array([0, 0, width, height])
        obj = PersonObject()
        obj.headpose = Rotation.from_euler('YXZ', [-float(sample['yaw']*90), 0, 0], degrees=True).as_matrix()
        obj.add_category(GenericCategory(gender[int(sample['gender'])]))
        obj.add_category(GenericCategory(race[int(sample['race'])]))
        obj.add_category(GenericCategory(str(float(sample['illumination']))))
        obj.add_category(GenericCategory(categories[int(sample['expression'])]))
        image.add_object(obj)
        seq.add_image(image)
        anns.append(seq)
    return anns, [cat for cat in categories.values()]


def expression_counts_by_demo_and_conf(df, categories, demo_factor, conf_factor):
    """
    Create a table of expression counts by demographic factor and confounding factor bins.
    """
    # Create bins for confounding factor
    if demo_factor == 'Gender' or demo_factor == 'Race':
        df['DemoBin'] = df[demo_factor]
    else:
        raise ValueError(f"Unknown demographic factor: {demo_factor}")
    if conf_factor == 'Pose':
        df['ConfBin'] = np.where(np.abs(df[conf_factor]) <= 15, '0_15', '15_90')
    elif conf_factor == 'Illumination':
        df['ConfBin'] = np.where(df[conf_factor] <= 125, '0_125', '125_255')
    else:
        raise ValueError(f"Unknown confounding factor: {conf_factor}")
    df['EmotionGt'] = pd.Categorical(df['EmotionGt'], categories=categories, ordered=True)
    df['DemoBin'] = pd.Categorical(df['DemoBin'], categories=sorted(df['DemoBin'].unique()), ordered=True)
    df['ConfBin'] = pd.Categorical(df['ConfBin'], categories=sorted(df['ConfBin'].unique()), ordered=True)
    # Create crosstab
    print(pd.crosstab(index=df['EmotionGt'], columns=[df['DemoBin'], df['ConfBin']], dropna=False))
    return df


def fairness_gaps_table(df, categories):
    """
    Compute macro-TPR by demographic and confounding factor bins.
    """
    def _macro_tpr_from_confusion(y, pred, num_classes):
        # Compute confusion matrix
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        if y.size > 0:
            np.add.at(cm, (y, pred), 1)
        # Compute TPR per class
        den = cm.sum(axis=1).astype(np.float64)
        tpr = np.full(cm.shape[0], np.nan, dtype=np.float64)
        valid = den > 0
        tpr[valid] = cm.diagonal()[valid] / den[valid]
        # Compute macro-TPR
        valid_tpr = np.isfinite(tpr)
        if not np.any(valid_tpr):
            return float("nan")
        return float(np.mean(tpr[valid_tpr])) * 100
    
    emotion_map = {name: idx for idx, name in enumerate(categories)}
    anno = df['EmotionGt'].map(emotion_map).values
    pred = df['EmotionPred'].map(emotion_map).values
    # Create results table
    rows = []
    for d in df["DemoBin"].unique():
        # Aggregate over ALL confounding factors
        mask = df["DemoBin"] == d
        score = _macro_tpr_from_confusion(anno[mask], pred[mask], len(categories))
        rows.append({"demo": d, "conf": "All", "macro_tpr": score})
        # For each confounding factor individually
        for c in df["ConfBin"].unique():
            mask = (df["DemoBin"] == d) & (df["ConfBin"] == c)
            score = _macro_tpr_from_confusion(anno[mask], pred[mask], len(categories))
            rows.append({"demo": d, "conf": c, "macro_tpr": score})
    print("\n=== Macro-TPR (%) per demographic factor using a visual confounder ===")
    df_results = pd.DataFrame(rows)
    table = df_results.pivot(index="demo", columns="conf", values="macro_tpr").sort_index()
    print(table.to_string(float_format=lambda x: f"{x:.2f}"))
    print("\n=== Empirical and standardized fairness gaps (%) ===")
    M_g = table["All"]
    conf_cols = table.columns.drop("All")
    M_g_std = table[conf_cols].mean(axis=1)
    gap = M_g.max() - M_g.min()
    gap_std = M_g_std.max() - M_g_std.min()
    print(f"GAP: {gap:.6f}")
    print(f"GAPstd: {gap_std:.6f}")
    print(f"∆: {gap-gap_std:.6f}")


def main():
    """
    Facial expression recognition database script.
    """
    unknown, db_path, show_viewer, save_file, save_image, demo_factor, conf_factor = parse_options()
    # Load computer vision components
    composite = Composite()
    sr = CVPRW26Recognition('')
    composite.add(sr)
    composite.parse_options(unknown)
    composite.load(Modes.TEST)
    spec = importlib.util.find_spec('pcr_framework')
    output_path = os.path.join('pcr_framework' if spec is None else os.path.dirname(spec.origin), 'output')
    # Analyze annotation data
    anns, categories = load_annotations(db_path, sr.database)
    print("\n=== Expression counts within each demomographic-confounding factor bin ===")
    records = [{'Pose': Rotation.from_matrix(obj.headpose).as_euler('YXZ', degrees=True)[0], 'EmotionGt': obj.categories[3].label.name, 'Gender': obj.categories[0].label.name, 'Race': obj.categories[1].label.name, 'Illumination': float(obj.categories[2].label.name)} for ann in anns for img in ann.images for obj in img.objects]
    df = pd.DataFrame.from_records(records)
    df = expression_counts_by_demo_and_conf(df, categories, demo_factor, conf_factor)
    if show_viewer:
        viewer = Viewer('cvprw26_recognition_database')
    if save_file:
        ofs = open(output_path+'/results.txt', 'w', encoding='utf-8')
    if save_image:
        viewer = Viewer('cvprw26_recognition_database')
        dirname = os.path.join(output_path, 'images/')
        Path(dirname).mkdir(parents=True, exist_ok=True)
    preds = []
    for i in tqdm(range(len(anns)), file=sys.stdout):
        pred = copy.deepcopy(anns[i])
        for img_pred in pred.images:
            for obj_pred in img_pred.objects:
                obj_pred.clear()
        composite.process(anns[i], pred)
        preds.extend(obj_pred.categories[0].label.name for img_pred in pred.images for obj_pred in img_pred.objects)
        if show_viewer:
            for img_pred in pred.images:
                viewer.set_image(img_pred)
            composite.show(viewer, anns[i], pred)
            viewer.show(0, as_video=False)
        if save_file:
            composite.evaluate(ofs, anns[i], pred)
        if save_image:
            for img_pred in pred.images:
                viewer.set_image(img_pred)
            composite.show(viewer, anns[i], pred)
            viewer.save(dirname, as_video=False, format='tif')
            composite.save(dirname, pred)
    if save_file:
        ofs.close()
    # Compute unbiased metrics
    df['EmotionPred'] = preds
    fairness_gaps_table(df, categories)

if __name__ == '__main__':
    main()
