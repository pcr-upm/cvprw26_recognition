# Deconfounding demographic bias estimation in facial expression recognition

If you use this code for your own research, you must reference our conference paper:

```
Deconfounding demographic bias estimation in facial expression recognition
Iván Ferre, Roberto Valle, José M. Buenaposada, Luis Baumela.
IEEE Conference on Computer Vision and Pattern Recognition Workshops, CVPRW 2026.
```

#### Requisites
- images-framework
- tqdm
- matplotlib
- pandas
- pytorch
- torchinfo

#### Usage
```
usage: cvprw26_recognition_test.py [-h] [--input-data INPUT_DATA] [--show-viewer] [--save-image]
```

* Use the --input-data option to set an image, directory, camera or video file as input.

* Use the --show-viewer option to show results visually.

* Use the --save-image option to save the processed images.
```
usage: Alignment --database DATABASE
```

* Use the --database option to select the database model.
```
usage: CVPRW26Recognition [--gpu GPU] [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--patience PATIENCE]
```

* Use the --gpu option to set the GPU identifier (negative value indicates CPU mode).
```
> python test/cvprw26_recognition_test.py --input-data test/example.jpg --database affectnet --gpu 0 --save-image
```

```
> python test/cvprw26_recognition_database.py --anns-file csv/affectnetplus_test_annotations_quality_illum.csv --database affectnet --gpu 0 --demographic-factor Gender --confounding-factor Illumination
```

```
Confusion matrix:
[[292   0   8   5  16 140  24  15]
 [ 24  43   9   1 275 139   4   5]
 [132   1 154  16  55  88  40  14]
 [ 40   0  10 207  23  77  52  91]
 [  3   2   0   0 467  21   3   4]
 [ 18   3   0   1  52 385  26  14]
 [ 42   0   6   8  23 155 256  10]
 [ 14   2   4  31  90 131  19 209]]
mAccuracy: 50.338%
mRecall: 50.344%
mPrecision: 61.529%
> Anger: Recall: 58.400% Precision: 51.681% Specificity: 92.198% Dice: 54.836%
> Contempt: Recall: 8.600% Precision: 84.314% Specificity: 99.771% Dice: 15.608%
> Disgust: Recall: 30.800% Precision: 80.628% Specificity: 98.943% Dice: 44.573%
> Fear: Recall: 41.400% Precision: 76.952% Specificity: 98.228% Dice: 53.836%
> Happiness: Recall: 93.400% Precision: 46.653% Specificity: 84.738% Dice: 62.225%
> Neutral: Recall: 77.154% Precision: 33.891% Specificity: 78.543% Dice: 47.095%
> Sadness: Recall: 51.200% Precision: 60.377% Specificity: 95.199% Dice: 55.411%
> Surprise: Recall: 41.800% Precision: 57.735% Specificity: 95.627% Dice: 48.492%

=== Macro-TPR (%) per demographic factor using a visual confounder ===
conf    0_125  125_255   All
demo                        
Female  47.67    47.49 47.63
Male    48.78    51.24 50.40

=== Empirical and standardized fairness gaps (%) ===
GAP: 2.772236
GAPstd: 2.430690
∆: 0.341546
```
