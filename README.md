# jobcare-recommendation
잡케어 추천 알고리즘 경진대회


### Model Architecture
![competition-model](https://user-images.githubusercontent.com/46340424/151582477-9c676890-2cff-45b9-902d-6e706b5eb5d1.png)


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
