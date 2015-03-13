# ufl_scene_text_recognition
C++ (OpenCV) implementation of the Unsupervised Feature Learning algorithm of Adam Coates and Andrew Ng for Scene Text Detection and Recognition

## Content

* **./** source code for scene text recognition. Depends on OpenCV >= 3.0-beta and liblinear >= 1.94 (included) and blas (part of liblinear but maybe potentially replaced by libblas system lib).

* **liblinear-1.94/** snapshot of liblinear in order to be able to reproduce exactly the same model.

* **train/** source code and data to build the filter bank, extract train/test features, and train the linear classifier. See train/README

## Compilation

```
 cmake .
 make
```

## Usage
```
 ./ufl_predict_char train/all_train_data.liblinear.model_s2 train/first_layer_centroids.xml train/data/sample/img_1.pgm
```
