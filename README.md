# jobcare-recommendation
잡케어 추천 알고리즘 경진대회


### Model Architecture
![competition-model](https://user-images.githubusercontent.com/46340424/151581921-16dd64fc-68d7-41aa-8cf6-ff3bf7a95178.png)

### Benchmark
|Score|CV|Public LB|Private LB|
|-----|--|------|-------|
|LightGBM(5fold-StratifiedKfold)-0.36|0.690|0.691|-|
|LightGBM(10fold-StratifiedKfold)-0.36|0.692|-|-|
|Catboost(5fold-StratifiedKfold)-0.4|0.712|0.698|-|
|Catboost(10fold-StratifiedKfold)-0.4|-|-|-|

### Requirements
+ numpy
+ pandas
+ scikit-learn
+ lightgbm
+ optuna
+ neptune.ai
+ hydra
