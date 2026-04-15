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
> python test/cvprw26_recognition_test.py --input-data test/example.tif --database affwild2 --gpu 0 --save-image
```
