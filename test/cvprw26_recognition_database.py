#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.getcwd())
import copy
import numpy as np
import pandas as pd
import importlib.util
from tqdm import tqdm
from pathlib import Path
from scipy.spatial.transform import Rotation
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
    anns_file = args.anns_file
    show_viewer = args.show_viewer
    save_file = args.save_file
    save_image = args.save_image
    demo_factor = args.demographic_factor
    conf_factor = args.confounding_factor
    return unknown, anns_file, show_viewer, save_file, save_image, demo_factor, conf_factor


def load_annotations(anns_file):
    """
    Load ground truth annotations according to each database.
    """
    from PIL import Image
    from images_framework.src.annotations import GenericGroup, GenericImage, PersonObject, GenericCategory
    from images_framework.src.categories import Name
    from images_framework.categories.emotions import Emotion as Oe
    from src.load_csv import parse_rafdb, parse_affectnet
    print('Open annotations file: ' + str(anns_file))
    if os.path.isfile(anns_file) and anns_file == 'csv/rafdb_test_pose_bboxq_illum.csv':
        dbpath = '/home/database/classification/faces/expressions/raf/basic/'
        gender = {0: Name('Male'), 1: Name('Female')}
        race = {0: Name('Caucasian'), 1: Name('African-American'), 2: Name('Asian')}
        categories = {0: Oe.FACE.SURPRISE, 1: Oe.FACE.FEAR, 2: Oe.FACE.DISGUST, 3: Oe.FACE.HAPPINESS, 4: Oe.FACE.SADNESS, 5: Oe.FACE.ANGER, 6: Oe.FACE.NEUTRAL}
        samples = parse_rafdb(Path(anns_file))
    elif os.path.isfile(anns_file) and anns_file == 'csv/affectnetplus_test_annotations_quality_illum.csv':
        dbpath = '/home/database/classification/faces/expressions/affectnet/'
        gender = {0: Name('Female'), 1: Name('Male')}
        race = {0: Name('Asian'), 1: Name('Indian'), 2: Name('Black'), 3: Name('White'), 4: Name('Middle-Eastern'), 5: Name('Latino-Hispanic')}
        categories = {0: Oe.FACE.NEUTRAL, 1: Oe.FACE.HAPPINESS, 2: Oe.FACE.SADNESS, 3: Oe.FACE.SURPRISE, 4: Oe.FACE.FEAR, 5: Oe.FACE.DISGUST, 6: Oe.FACE.ANGER, 7: Oe.FACE.CONTEMPT}
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
        obj.headpose = Rotation.from_euler('YXZ', [-float(sample['yaw']*90), float(sample['pitch']*90), float(sample['roll']*90)], degrees=True).as_matrix()
        obj.add_attribute(GenericCategory(label=gender[int(sample['gender'])]))
        obj.add_attribute(GenericCategory(label=race[int(sample['race'])]))
        obj.add_attribute(GenericCategory(label=Name('illumination'), score=float(sample['illumination'])))
        obj.add_category(GenericCategory(categories[int(sample['expression'])]))
        image.add_object(obj)
        seq.add_image(image)
        anns.append(seq)
    return anns, [cat.name for cat in categories.values()]


def expression_counts_by_demo_and_conf(df, categories, demo_factor, conf_factor):
    """
    Create a table of expression counts by demographic factor and confounding factor bins.
    """
    # Create bins for confounding factor
    if conf_factor == 'Pose':
        df['ConfBin'] = np.where(np.abs(df['Yaw']) <= 15, '0_15', '15_90')
    elif conf_factor == 'Illumination':
        df['ConfBin'] = np.where(df['Illumination'] <= 125, '0_125', '125_255')
    else:
        raise ValueError(f"Unknown confounding factor: {conf_factor}")
    df['EmotionGt'] = pd.Categorical(df['EmotionGt'], categories=categories, ordered=True)
    df[demo_factor] = pd.Categorical(df[demo_factor], categories=sorted(df[demo_factor].unique()), ordered=True)
    df['ConfBin'] = pd.Categorical(df['ConfBin'], categories=sorted(df['ConfBin'].unique()), ordered=True)
    # Create crosstab
    print(pd.crosstab(index=df['EmotionGt'], columns=[df[demo_factor], df['ConfBin']], dropna=False))
    return df


def macro_tpr_table(df, categories):
    """
    Compute macro-TPR by demographic and confounding factor bins.
    """
    # Convert emotion names to indices
    emotion_map = {name: idx for idx, name in enumerate(categories)}
    y = df['EmotionGt'].map(emotion_map).values
    pred = df['EmotionPred'].map(emotion_map).values
    # Map gender names to indices
    gender_map = {'Female': 0, 'Male': 1}
    gender = df['Gender'].map(gender_map).values
    # Create results table
    rows = []
    for g in [0, 1]:
        for illum_bin in ['0_125', '125_255']:
            mask = (gender == g) & (df['ConfBin'] == illum_bin)
            y_filtered = y[mask]
            pred_filtered = pred[mask]
            # Compute confusion matrix and TPRs
            cm = np.zeros((len(categories), len(categories)), dtype=np.int64)
            if y_filtered.size == 0:
                return cm
            np.add.at(cm, (y_filtered, pred_filtered), 1)
            den = cm.sum(axis=1).astype(np.float64)
            tpr = np.full(cm.shape[0], np.nan, dtype=np.float64)
            valid = den > 0
            tpr[valid] = cm.diagonal()[valid] / den[valid]
            macro_tpr = np.nanmean(tpr)*100
            gender_name = 'Female' if g == 0 else 'Male'
            rows.append({
                'gender': gender_name,
                'illumination': illum_bin,
                'macro_tpr': macro_tpr
            })
    # Create pivot table
    df_results = pd.DataFrame(rows)
    table = df_results.pivot(index='gender', columns='illumination', values='macro_tpr')
    table = table[['0_125', '125_255']]  # Ensure column order
    table.columns = ['0-125', '125-255']
    # Calculate gap (M-F) per illumination bin, then average
    gap_per_bin = table.loc['Male'].values - table.loc['Female'].values
    gap_avg = np.mean(gap_per_bin)
    table['Gap (M-F)'] = gap_avg
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))


def _confusion_matrix_expr(y, pred, num_classes):
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    if y.size == 0:
        return cm
    np.add.at(cm, (y, pred), 1)
    return cm

def _class_tpr_from_confusion(cm):
    """Compute TPR per class from confusion matrix."""
    den = cm.sum(axis=1).astype(np.float64)
    tpr = np.full(cm.shape[0], np.nan, dtype=np.float64)
    valid = den > 0
    tpr[valid] = cm.diagonal()[valid] / den[valid]
    return tpr

def _macro_tpr_from_confusion(cm):
    """Compute macro-TPR from confusion matrix."""
    tpr = _class_tpr_from_confusion(cm)
    valid = np.isfinite(tpr)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(tpr[valid]))*100

def fairness_gaps_table(df, categories):
    """
    Compute fairness gaps by demographic and confounding factor bins.
    """
    emotion_map = {name: idx for idx, name in enumerate(categories)}
    # Convert emotion names to indices
    y = df['EmotionGt'].map(emotion_map).values
    pred = df['EmotionPred'].map(emotion_map).values
    
    # Compute confusion matrices per gender
    cm_f = _confusion_matrix_expr(y[df['Gender'] == 'Female'], pred[df['Gender'] == 'Female'], len(categories))
    cm_m = _confusion_matrix_expr(y[df['Gender'] == 'Male'], pred[df['Gender'] == 'Male'], len(categories))
    
    m_f = _macro_tpr_from_confusion(cm_f)
    m_m = _macro_tpr_from_confusion(cm_m)
    gap_obs = float(m_m - m_f)
    
    print(f"M_female: {m_f:.6f}")
    print(f"M_male:   {m_m:.6f}")
    print(f"GAP: {gap_obs:.6f}")

    MIN_PER_BIN = 20
    
    # Convert emotion names to indices
    y = df['EmotionGt'].map(emotion_map).values
    pred = df['EmotionPred'].map(emotion_map).values
    illumination_bins = df['ConfBin'].values
    
    # Map illumination bins
    illum_map = {'0_125': 0, '125_255': 1}
    illumination_id = pd.Series(illumination_bins).map(illum_map).values
    
    num_illum_bins = 2
    
    # Count samples per gender and illumination bin
    counts_f = np.bincount(illumination_id[df['Gender'] == 'Female'], minlength=num_illum_bins).astype(np.int64)
    counts_m = np.bincount(illumination_id[df['Gender'] == 'Male'], minlength=num_illum_bins).astype(np.int64)
    
    # Determine supported bins
    supported = (counts_f >= MIN_PER_BIN) & (counts_m >= MIN_PER_BIN)
    supported_idx = np.where(supported)[0]
    
    if supported_idx.size == 0:
        print("G_std (M-F): nan")
        return
    
    # Compute weights
    w_ref = np.zeros(num_illum_bins, dtype=np.float64)
    w_ref[supported] = 1.0 / float(supported.sum())
    
    # Compute macro-TPR per gender and illumination bin
    m_gb = np.full((2, num_illum_bins), np.nan, dtype=np.float64)
    
    for iid in supported_idx:
        # Female
        mask_f = (df['Gender'] == 'Female') & (illumination_id == iid)
        cm_f = _confusion_matrix_expr(y[mask_f], pred[mask_f], len(categories))
        m_gb[0, iid] = _macro_tpr_from_confusion(cm_f)
        
        # Male
        mask_m = (df['Gender'] == 'Male') & (illumination_id == iid)
        cm_m = _confusion_matrix_expr(y[mask_m], pred[mask_m], len(categories))
        m_gb[1, iid] = _macro_tpr_from_confusion(cm_m)
    
    # Compute standardized gap
    m_std_f = float(np.sum(w_ref[supported] * m_gb[0, supported]))
    m_std_m = float(np.sum(w_ref[supported] * m_gb[1, supported]))
    gap_std = float(m_std_m - m_std_f)
    
    print(f"GAPstd: {gap_std:.6f}")
    print(f"∆: {gap_obs-gap_std:.6f}")


def main():
    """
    Facial expression recognition database script.
    """
    unknown, anns_file, show_viewer, save_file, save_image, demo_factor, conf_factor = parse_options()
    anns, categories = load_annotations(anns_file)
    # Analyze annotation data
    print("\n=== Expression counts within each demomographic-confounding factor bin ===")
    records = [{'Yaw': Rotation.from_matrix(obj.headpose).as_euler('YXZ', degrees=True)[0], 'EmotionGt': obj.categories[0].label.name, 'Gender': obj.attributes[0].label.name, 'Race': obj.attributes[1].label.name, 'Illumination': obj.attributes[2].score} for seq in anns for img in seq.images for obj in img.objects]
    df = pd.DataFrame.from_records(records)
    df = expression_counts_by_demo_and_conf(df, categories, demo_factor, conf_factor)
    # Load computer vision components
    composite = Composite()
    sr = CVPRW26Recognition('')
    composite.add(sr)
    composite.parse_options(unknown)
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
    # Compute unbiased metrics
    df['EmotionPred'] = preds
    print("\n=== Macro-TPR (%) per demographic factor using a visual confounder ===")
    macro_tpr_table(df, categories)
    print("\n=== Empirical and standardized fairness gaps (%) per demographic factor using a visual confounder ===")
    fairness_gaps_table(df, categories)

if __name__ == '__main__':
    main()
