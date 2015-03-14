
## Compile

```
cmake .
make
```

## Use:

You may need to extract the dataset from **scene_chars_dataset.zip** and **background_dataset.zip** in the **data/** folder.

1. Learn the filters and save them into first_layer_centroids.xml:
```
./extract_filters data/characters/icdar/img_ICDAR_train/* data/characters/chars74k/img/* data/characters/synthetic/img/*
```

2. Extract Features:
```
./extract_features first_layer_centroids.xml data/characters/icdar/img_ICDAR_train_labels.txt > data/all_train_data.svm
./extract_features first_layer_centroids.xml data/characters/chars74k/img_labels.txt >> data/all_train_data.svm
./extract_features first_layer_centroids.xml data/characters/synthetic/img_labels.txt >> data/all_train_data.svm

./extract_features first_layer_centroids.xml data/characters/icdar/img_ICDAR_test_labels.txt > data/test_data.svm
```

3. Scale data:
```
svm-scale -s data/all_train_data.svm.range data/all_train_data.svm > data/all_train_data.svm.scaled
svm-scale -r data/all_train_data.svm.range data/test_data.svm > data/test_data.svm.scaled2
```

4. Train the model :

```
 ../liblinear-1.94/train -s 2 -c 4 -e 0.001 data/all_train_data.svm.scaled all_train_data.liblinear.model_s2
```

5. Evaluate the test accuracy:
```
../liblinear-1.94/predict data/test_data.svm.scaled2 all_train_data.liblinear.model_s2 out
```

6. (Optionally) Plot the confusion matrix:
```
cp out confusion_matrix/
cd confusion_matrix/
python confusion_matrix.py 
```

IMPORTANT: if training data has changed (e.g. using a different filter bank, or changing the number of training examples) you must change the feature scale factors on ../ufl_predict_char.cpp in order to your model make correct predictions with new samples.
