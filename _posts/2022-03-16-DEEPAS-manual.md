---
layout: post
title:  "DEEPAS(v0.1) manual"
date:   2022-03-16 17:53:11 +0900
categories: deepas
---

<meta charset="utf-8">

# Contents

- [Introduction](#introduction)
- [Setup](#setup)
  - [Linux OS](#linux-os)
- [Information](#information)
  - [CUDA version](#cuda-version)
  - [GPU supported](#gpu-supported)
  - [Supported ML/DL model](#supported-ml/dl-model)
  - [Hyperparameter search space format](#hyperparameter-search-space-format)
    - [bayesian](#bayesian)
    - [grid](#grid)
  - [DNN hyperparameter tuning](#dnn-hyperparameter-tuning)
    - [DNN hyperparameter](#dnn-hyperparameter)
  - [result](#result)
- [Example](#example)
  - [import](#import)
  - [initial setting](#initial-setting)
  - [loading dataset](#loading-dataset)
  - [data preprocessing](#data-preprocessing)
  - [hyperparamter tuning](#hyperparamter-tuning)
  - [build model](#build-model)
  - [training](#training)
  - [evaluation](#evaluation)
  - [explainable ai](#explainable-ai)
- [User Guide](#user-guide)
  - [`get_base_path`](#get_base_path)
  - [`config`](#config)
  - [`read_data`](#read_data)
  - [`split_label`](#split_label)
  - [`check_na`](#check_na)
  - [`fill_na`](#fill_na)
  - [`split_data`](#split_data)
  - [`extract_column`](#extract_column)
  - [`drop_column`](#drop_column)
  - [`merge_data`](#merge_data)
  - [`convert_feature_to_num`](#convert_feature_to_num)
  - [`convert_df_to_np`](#convert_df_to_np)
  - [`save`](#save)
  - [`save_model_object`](#save_model_object)
  - [`tune_model`](#tune_model)
  - [`create_model`](#create_model)
    - [`predict`](#predict)
    - [`predict_proba`](#predict_proba)
  - [`train_model`](#train_model)
  - [`get_metric`](#get_metric)
  - [`evaluate_model`](#evaluate_model)
  - [`FeatureImportance`](#FeatureImportance)
    - [`explain`](#explain)


<h1>Introduction</h1>
deepas is an automated machine learning tool for classification. It supports CatBoost, LightGBM, XGBoost and DNN(PyTorch-based). 


# Setup

## Linux OS 

```shell
$ conda create -n [env_name] python=3.7 -y
$ conda activate [env_name]
$ conda install jupyter -y
$ pip install deepas-1.0-py3-none-any.whl
$ pip install torch==1.10.0+cu111 -f torch-1.10.0+cu111-cp37-cp37m-linux_x86_64.whl
[optional] $ sh install.sh
```


# information

## CUDA version 
- The appropriate version of CUDA is `11.1`. So, Please check it out.

## GPU supported 
- Usage GPU of LightGBM is not supported.
- GPU is available in CatBoost, XGBoost and DNN.
## Supported ML/DL model 
- CatBoost
- LightGBM
- XGBoost
- DNN (PyTorch-based)

## Hyperparameter search space format 
- 지원하는 모델 중 ML 모델은 각 모델별로 해당 모델에 적합한 하이퍼파라미터를 입력해야 합니다. 각 모델별 Documentation을 참고하시기 바랍니다.
- [CatBoost Documentation](https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/parameter.html)

### bayesian

```python
{
    'n_estimators': [200, 300],
    'learning_rate': [0.1, 0.001],
    'max_depth': (3, 4)
}
```

- bayesian 방식의 하이퍼파라미터 search space는 위와 같이 최소값과 최대값을 리스트 또는 튜플의 형태로 입력해야 합니다.

### grid

```python
{
    'n_estimators': [100, 200, 300, 1000],
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': (3, 4, 5, 6, 7)
}
```

grid 방식의 하이퍼파라미터 search space는 위와 같이 일련의 값들을 리스트 또는 튜플의 형태로 입력해야 합니다.

## DNN hyperparameter tuning

DNN에서 하이퍼파라미터를 입력하기 위해서는 `n_layers`의 값에 따라 `dropout` 값과 `activation_function` 값 값의 개수를 `n_layers`의 값과 일치시켜야 합니다.

```python
{
    'optimizer': 'Adam',
    'lr': 0.01,
    'n_layers': 3,
    'n_unit': [3, 5, 7], # 3개
    'dropout': [0.0, 0.0, 0.1], # 3개
    'activation_function': ['ReLU', 'ReLU', 'ReLU'] # 3개
}
```

### DNN hyperparameter
- DNN의 경우 다음의 하이퍼파라미터를 조정할 수 있습니다. **아래 명시된 이름 이외의 다른 명칭은 쓸 수 없습니다.**
  - `optimizer`
    - 딥러닝의 최적화 알고리즘으로서, 가능한 값은 다음과 같습니다.
      - [`Adam`, `RMSprop`, `SGD`, `Adadelta`, `AdamW`, `Adamax`, `ASGD`, `LBFGS`, `NAdam`, `RAdam`, `Rprop`]
        - [pytorch: TORCH.OPTIM] 참고
        - LBFGS 알고리즘의 경우, 학습 속도가 느립니다.
  - `lr`
    - 최적화 알고리즘의 학습 속도와 모델의 성능을 결정짓는 하이퍼파라미터입니다. 
    - 일반적으로 `0.1` 이하의 값을 줍니다.
  - `n_layers`
    - 모델의 layer 개수를 지정하는 하이퍼파라미터입니다. 
    - 해당 값과 `n_unit`, `dropout`, `activation_function값의` 개수를 맞춰야 합니다.
  - `n_unit`
    - 각 layer별 unit의 개수를 지정하는 하이퍼파라미터입니다. 
    - 해당 값은 `n_layers` 값과 동일한 개수를 가져야 합니다.
  - `dropout`
    - 각 layer별 dropout의 비율을 지정하는 하이퍼파라미터입니다. 
    - dropout 비율은 `0.0`~`1.0` 사이의 값을 갖습니다. 
    - 해당 값은 `n_layers` 값과 동일한 개수를 가져야 합니다. 
  - `activation_function`
    - 각 layer별 활성화 함수를 지정하는 하이퍼파라미터입니다. 
    - 가능한 활성화 함수는 다음과 같습니다.
      - [`ELU`, `Hardshrink`, `Hardsigmoid`, `Hardtanh`, `Hardswish`, `LeakyReLU`, `LogSigmoid`, `PReLU`, `ReLU`, `ReLU6`, `RReLU`, `SELU`, `CELU`, `GELU`, `SiLU`, `Sigmoid`, `Mish`, `Softplus`, `Softshrink`, `Softsign`, `Tanh`, `Tanhshrink`]
        - [pytorch: Non-linear Activations] 참고
    - 해당 값은 `n_layers` 값과 동일한 개수를 가져야 합니다. 

## result
- 예제 코드대로 실행할 시, deepas 분석 결과는 `get_base_path`메소드에 의해 지정된 경로에 저장되며, 프로그램 실행 중 저장되는 파일이 있을 경우 해당 파일의 경로가 뜹니다.
- binary classification, multi classification에 상관없이 모든 metric 파일은 class의 개수대로 생성됩니다.
- `catboost_info` 파일의 경우 catboost 패키지 자체에서 생성됩니다.

# Example

## import

```shell
$ python
>>> import deepas
```

## initial setting
config 선언 작업은 반드시 필요하며, 예제 코드에 포함된 형식대로 작성해주시기 바랍니다.

```python
config = Config(
    model=MODEL,
    device=DEVICE,
    gpu_id=GPU_ID,
    model_fn=RESULT_PATH,
    n_epochs=DNN_EPOCHS
)
```

## loading dataset

```python
DATASETPATH = 'your_dataset_path'
SEP = ','
df = read_data(DATASETPATH, seperator=SEP)
```

## data preprocessing
데이터를 전처리 하기 위한 기본적인 메소드들을 제공하고 있습니다.

```python
# label과 data의 분리
X, y = split_label(df, target=TARGET)

# 특정 단일 column 분리
y_train_df = extract_column(df=y_train_df, columns=TARGET, dtype='int')

# 특정 단일 column 제거
X_train_df = drop_column(df=merged_train_df, columns=TARGET)

# 두 pandas.DataFrame 타입의 데이터를 merge
merged_train_df = merge_data(X=X_train_df, y=y_train_df)

# NaN값 확인
check_na(df)

# NaN값 처리
fill_na(df)

# trainset과 testset의 분리: sklearn.model_selection.train_test_split 참고
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train_df, X_val_df, y_train_df, y_val_df = split_data(
    data=df,
    test_size=0.3,
    target=TARGET
)

# categorical 데이터를 숫자로 변환
# 예: setosa, virginica, versica -> 0, 1, 2
cat_dict_train, y_train_df = convert_feature_to_num(feature=y_train_df)

# pandas.DataFrame 타입의 데이터를 numpy.ndarray 타입의 데이터로 변환
X_train_np, X_train_features = convert_df_to_np(X_train_df)
```

## hyperparamter tuning
- 하이퍼파라미터 튜닝을 위해서는 deepas의 `tune` 내의 메소드를 호출합니다. `tune_model`을 호출하는 즉시 tuning을 수행하기 위한 머신러닝/딥러닝 모델을 생성하고 tuning 작업까지 완료되며, 그 결과를 파일로 저장함과 동시에 사용자에게 결과를 반환합니다.
- 지원하는 하이퍼파라미터 튜닝 방식은 `grid`와 `bayesian`입니다. 
- `grid`와 `bayesian` 모두 Optuna 패키지를 기반으로 합니다.
- search space와 `n_iter`에 따라 튜닝 시간이 길어질 수 있습니다.

```python
from deepas.tune import *

params = tune_model(
    model=MODEL,
    X_train=X_train,
    y_train=y_train,
    target=TARGET,
    test_size=0.2,
    device=config.device,
    gpu_id=config.gpu_id,
    params=search_params,
    method=TUNING_METHOD,
    n_iter=N_ITER,
    tuning_result=RESULT_PATH,
    dnn_config=config
)
```

## build model
- 하이퍼파라미터 튜닝 여부와 상관없이 deepas의 모든 ML/DL 모델은 초기 구성이 필요합니다. 
  - 하이퍼파라미터 튜닝 수행 시, 내부적으로 모델을 생성 후 search space 범위 내의 하이퍼파라미터를 선정하여 모델을 구성하게 됩니다.
  - 하이퍼파라미터 튜닝을 수행하지 않을 시, 유저가 직접 모델을 구성해야 합니다.
- 하이퍼파라미터 튜닝을 수행했다면 best parameter로 자동으로 model이 구성됩니다.
- 하이퍼파라미터 튜닝을 수행하지 않았다면 유저가 직접 지정한 custom parameter를 이용하여 모델을 구성하게 됩니다.

```python
from deepas.classifier import * 

params = {
    'n_layers': 2,
    'optimizer': 'RAdam',
    'learning_rate': 0.00011872217316340456,
    'n_unit': [3, 5],
    'dropout': [0.0004326686602487604, 1.0065587516094237e-05],
    'activation_function': ['Softsign', 'RReLU']
}

clf = create_model(
    X_train=X_train, 
    y_train=y_train, 
    model=MODEL, 
    params=params, 
    device=config.device, 
    gpu_id=config.gpu_id, 
    load_model=LOAD_MODEL,
    model_path=MODEL_PATH, 
    hyperparameter_flag=HYPERPARAMETER_TUNING
)
```

## training
- 모델의 학습을 위해 `train_model` 메소드를 호출합니다.
- 학습이 끝난 후에 반환되는 값은 예측값과 예측확률값, 학습이 완료된 모델입니다.
- 예측값과 예측확률값은 모델의 성능 평가에서 활용됩니다.
- 학습이 완료된 모델은 XAI 단계에서 활용됩니다.

```python
y_pred, y_probas, trained_model = train_model(
    clf, 
    X_train, 
    y_train, 
    X_val, 
    save_model=True,
    result_path=RESULT_PATH, 
    epochs=DNN_EPOCHS, 
    dnn_config=config
)
```

## evaluation
- 학습이 끝난 모델에 대한 성능 평가 지표를 확인할 수 있습니다.
- `y_pred`와 `y_proba` 값을 기반으로 주요 지표들을 정량화합니다. 주요 지표들은 다음과 같습니다.
  - True Positive(TP)의 개수
  - False Positive(FP)의 개수
  - True Negative(TN)의 개수
  - False Negative(FN)의 개수
  - ROC-AUC
  - Accuracy
  - Recall
  - Precision
  - F1_Score
  - Specificity
- 성능 평가 결과는 파일로 저장됩니다.

```python
report = evaluate_model(
    model=trained_model, 
    y_val=y_val, 
    result_path=RESULT_PATH,
    cat_dict_train=cat_dict_train
)
```

## explainable ai
- tabular data의 어떤 feature가 모델에 가장 큰 영향을 끼치는지를 파악할 수 있는, XAI 기법을 사용할 수 있습니다.
- 제공되는 기법은 SHAP, Permutation Importance이며, CatBoost, LightGBM, XGBoost에서는 tree-intrinsic feature importance를 추가적으로 사용할 수 있습니다.
- DNN에서 SHAP을 이용할 경우, activation function에 대한 warning이 뜰 수 있습니다.

```python
import deepas.xai.feature_importance as exp

explainer = exp.FeatureImportance(ai_method=MODEL, trained_model=clf, xai_method=XAI_METHOD, device=config.device, gpu_id=config.gpu_id)
exp_result = explainer.explain(X_train, y_train, RESULT_PATH, feature_names=list(X_train_features))
```



# User Guide

## get_base_path
ML/DL 결과를 저장할 기본 경로를 반환

```python
get_base_path(
    filepath: str,
    model: str,
    result_path: str = None
) -> (str, str)
```

#### Parameters
- `filepath`: 데이터셋 경로. 해당 경로를 통해 저장할 base path를 결정함
- `model`: 사용할 모델의 종류; `catboost`, `lightgbm`, `xgboost`, `dnn`
- `result_path`: default=`None`. 결과 파일들을 저장할 경로가 있다면 명시하여 base path로 지정함

## Config
ML/DL 모델을 돌리기 위한 기본 설정으로, GPU 할당 및 DNN 설정을 위해 반드시 필요함

```python
Config(
    model: str = None,
    device: str = 'cpu',
    model_fn: str = None,
    gpu_id: int = 0,
    train_ratio: float = .8,
    batch_size: int = 64,
    n_epochs: int = 20,
    verbose: int = 1,
    **kwargs
)
```

#### Parameters
- `model`: default=`None`. 사용할 모델의 종류; `catboost`, `lightgbm`, `xgboost`, `dnn`
- `device`: default=`cpu`. 사용할 장치의 종류; `gpu`, `cpu`
- `model_fn`: default=`None`. DNN 모델을 불러올 경로
- `gpu_id`: default=`0`. 사용할 장치가 `gpu`일 경우 gpu unit의 번호; `0`, `1`, ...
- `train_ratio`: default=`.8`. DNN 모델에서 사용할 train dataset의 비율
- `batch_size`: default=`64`. DNN 모델에서 사용할 batch의 크기
- `n_epochs`: default=`20`. DNN 모델에서 사용할 학습의 횟수
- `verbose`: default=`1`. DNN 모델에서 지정할 verbosity

## read_data
지정된 경로를 입력받아 데이터를 읽고 pandas.DataFrame 타입으로 반환

```python
read_data(
    filepath: str,
    seperator: str = '\t',
    low_memory: bool = False,
    engine: str = 'c',
    index_column: str = None,
    reset_index_flag: bool = False,
) -> DataFrame
```

#### Parameters
- `filepath`: 데이터셋 경로
- `seperator`: defulat=`\t`. 구분자
- `low_memory`: default=`False`. [pandas.read_csv] 참고
- `engine`: default=`c`. [pandas.read_csv] 참고
- `index_column`: pandas.DataFrame의 tabular data에서 index로 설정할 column의 이름
- `reset_index_flag`: default=`False`. pandas.DataFrame의 tabular data에서 index 초기화를 하고자 할 경우 `True`로 설정

## split_label
target 변수와 데이터를 분리

```python
split_label(
    data: DataFrame,
    target: str
) -> (DataFrame, DataFrame)
```

#### Parameters
- `data`: pandas.DataFrame 타입의 데이터
- `target`: 예측 대상이 되는 target 변수

## check_na
NaN값 확인

```python
check_na(
    data: DataFrame,
) -> (bool, int)
```

#### Parameters
- `data`: pandas.DataFrame 타입의 데이터

## fill_na
NaN값 처리

```python
fill_na(
    data: DataFrame,
    fill_na = nan,
    del_na_by_col: bool = False,
    del_na_by_row: bool = False
) -> DataFrame
```

#### Parameters
- `data`: pandas.DataFrame 타입의 데이터
- `fill_na`: default=`numpy.nan`. NaN값을 대체할 값
- `del_na_by_col`: default=`False`. NaN값이 있는 열 일괄 제거 옵션
- `del_na_by_row`: default=`False`. NaN값이 있는 행 일괄 제거 옵션

## split_data
train 데이터와 test 데이터를 분리

```python
split_data(
    data: DataFrame,
    target: str,
    test_size: float = 0.3,
    stratify: str = None,
    random_state: int = 42
) -> (DataFrame, DataFrame, DataFrame, DataFrame)
```

#### Parameters
- `data`: pandas.DataFrame 타입의 데이터
- `target`: 예측 대상이 되는 target 변수
- `test_size`: default=`0.3`. [sklearn.model_selection.train_test_split] 참고
- `stratify`: default=`None`. [sklearn.model_selection.train_test_split] 참고
- `random_state`: default=`42`. [sklearn.model_selection.train_test_split] 참고 

## extract_column
특정 column 추출

```python
def extract_column(
    df: DataFrame,
    columns,
    dtype: str = 'int'
) -> DataFrame
```

#### Parameters
- `df`: pandas.DataFrame 타입의 데이터
- `columns`: 추출할 column의 이름
- `dtype`: default=`int`. [pandas.DataFrame.astype] 참고

## drop_column
특정 column 제거

```python
drop_column(
    df: DataFrame,
    columns
) -> DataFrame
```

#### Parameters
- `df`: pandas.DataFrame 타입의 데이터
- `columns`: 제거할 column의 이름

## merge_data
pandas.DataFrame 타입의 두 tabular data를 결합

```python
merge_data(
    X: DataFrame,
    y: DataFrame,
    axis: int = 1,
    join: str = 'inner'
) -> DataFrame
```

#### Parameters
- `X`: pandas.DataFrame 타입의 데이터
- `y`: pandas.DataFrame 타입의 데이터
- `axis`: default=`1`. [pandas.concat] 참고
- `join`: default=`inner`. [pandas.concat] 참고

## convert_feature_to_num
category 타입의 데이터값을 숫자로 변환

```python
convert_feature_to_num(
    feature: DataFrame
) -> (DataFrame, DataFrame)
```

#### Parameters
- `feature`: category 또는 str 타입의 데이터가 있는 pandas.Series 또는 pandas.DataFrame 타입의 데이터

## convert_df_to_np
pandas.DataFrame 타입의 데이터를 numpy.ndarray 타입의 데이터로 변환

```python
convert_df_to_np(
    df: DataFrame = None
) -> (ndarray, list)
```

#### Parameters
- `df`: default=`None`. pandas.DataFrame 타입의 데이터

## save
dict 타입의 데이터를 파일 형식으로 저장 및 pandas.DataFrame 타입의 데이터로 반환
```python
save(
    path: str,
    filename: str,
    _dict: dict,
    format: str = 'csv',
    seperator: str = ',',
    index: bool = False,
    transpose_flag: bool = False,
    column_name: list = None,
    index_column: list = None,
    index_column_name: str = None,
    dtype: dict = None,
    force_drop_index_col: bool = False,
    force_drop_last_row: bool = False,
    force_prohibit_drop_index_col: bool = False,
    force_prohibit_save: bool = False
) -> DataFrame
```

#### Parameters
- `path`: 파일을 저장할 기본 경로
- `filename`: 저장할 파일의 이름
- `_dict`: dict 타입의 저장할 데이터
- `format`: default=`csv`. 저장할 파일의 포맷
- `seperator`: default=`,`. [pandas.DataFrame.to_csv] 참고
- `index`: default=`False`. [pandas.DataFrame.to_csv] 참고
- `transpose_flag`: default=`False`. 데이터는 내부적으로 pandas.DataFrame으로 변환됨. 변환된 pandas.DataFrame 타입의 데이터에 대하여 transpose 수행
- `column_name`: default=`None`. 저장할 데이터의 column 이름을 명시적으로 지정
- `index_column`: default=`None`. 저장할 데이터의 index를 명시적으로 지정
- `index_column_name`: default=`None`. 저장할 데이터의 index에 해당하는 column의 이름
- `dtype`: default=`None`. 저장할 데이터의 타입을 명시적으로 지정
- `force_drop_index_col`: default=`False`. pandas.DataFrame으로 변환된 데이터에서 index column 강제 제거
- `force_drop_last_row`: default=`False`. pandas.DataFrame으로 변환된 데이터에서 마지막 행 강제 제거
- `force_prohibit_drop_index_col`: default=`False`. pandas.DataFrame으로 변환된 데이터에서 index column 강제 제거 방지. 해당 옵션은 `force_drop_index_col`보다 높은 우선순위를 가지며, `True`일 경우 `force_drop_index_col=False`가 됨
- `force_prohibit_save`: default=`False`. 결과 파일 저장 방지 옵션

## save_model_object
학습이 완료된 모델을 저장. CatBoost, LightGBM, XGBoost에 해당

```python
save_model_object(
    model: str,
    model_object,
    result_path: str,
    save_model: bool = True,
) -> None
```

#### Parameters
- model: 모델 이름; `catboost`, `lightgbm`, `xgboost`, `dnn`
- model_object: CatBoost, LightGBM, XGBoost model object. 
- result_path: 모델 파일을 저장할 경로 지정
- save_model: default=`True`. `True`일 경우 모델을 저장

## tune_model
hyperparameter tuning 수행 및 결과 반환/파일 저장

```python
tune_model(
    X_train: DataFrame,
    y_train: DataFrame,
    target: str,
    params: dict,
    model: str,
    method: str,
    device: str = None,
    gpu_id: int = None,
    test_size = 0.3,
    metric: str = 'f1_score',
    n_iter: int = 100,
    tuning_result: str = None,
    dnn_config = None
)
```

#### Parameters
- `X_train`: pandas.DataFrame 타입의 데이터
- `y_train`: pandas.DataFrame 타입의 데이터
- `target`: 예측 대상이 되는 target 변수
- `params`: hyperparameter로 탐색할 search space
- `model`: 사용할 모델의 종류; `catboost`, `lightgbm`, `xgboost`, `dnn`
- `method`: hyperparameter 탐색 방법; `grid` 또는 `bayesian`
- `device`: default=`None`. 사용할 장치의 종류; `gpu`, `cpu`
- `gpu_id`: default=`None`. 사용할 장치가 `gpu`일 경우 gpu unit의 번호; `0`, `1`, ...
- `test_size`: default=`0.3`. [sklearn.model_selection.train_test_split] 참고
- `metric`: default=`f1_score`. hyperparameter tuning에서 사용할 기준이 되는 metric. 현재는 f1_score만 지원함
- `n_iter`: default=`100`. [study optimize] 참고
- `tuning_result`: default=`None`. 결과를 저장할 path
- `dnn_config`: default=`None`. DNN에 대한 hyperparameter tuning시의 DNN 모델의 학습 횟수 

## create_model
ML/DL 모델 생성

```python
create_model(
    model: str,
    params=None,
    device: str = None,
    gpu_id: int = None,
    load_model: bool = False,
    model_path: str = None,
    X_train=None,
    y_train=None,
    hyperparameter_flag: bool = False
)
```

#### Parameters
- `model`: 사용할 모델의 종류; `catboost`, `lightgbm`, `xgboost`, `dnn`
- `params`: default=`None`. 모델 하이퍼파라미터. 사용자 지정 custom hyperparameter 또는 hyperparameter tuning을 이용해 찾아낸 best hyperparameter가 이에 해당
- `device`: default=`None`. 사용할 장치의 종류; `gpu`, `cpu`
- `gpu_id`: default=`None`. 사용할 장치가 `gpu`일 경우 gpu unit의 번호; `0`, `1`, ...
- `load_model`: default=`False`. 저장된 모델을 사용하고자 할 경우 True로 설정
- `model_path`: default=`None`. 저장된 모델의 경로를 지정
- `X_train`: default=`None`. pandas.DataFrame 타입의 데이터
- `y_train`: default=`None`. pandas.DataFrame 타입의 데이터
- `hyperparameter_flag`: default=`False`. hyperparameter tuning 메소드 내부에서 사용하기 위한 파라미터

### predict
ML/DL 모델을 이용하여 예측 수행

```python
model.predict(
    X_val
)
```

#### Parameters
- `X_val`: pandas.DataFrame 타입의 데이터

### predict_proba
ML/DL 모델로부터 예측 확률 반환

```python
model.predict_proba(
    X_val
)
```

#### Parameters
- `X_val`: pandas.DataFrame 타입의 데이터

## train_model
ML/DL 모델 학습 수행

```python
train_model(
    model,
    X_train,
    y_train,
    X_val,
    save_model: bool = False,
    result_path: str = None,
    epochs: int = 100,
    dnn_config=None
)
```

#### Parameters
- `model`: `create_model` 메소드에 의해 생성된 모델
- `X_train`: pandas.DataFrame 타입의 데이터
- `y_train`: pandas.DataFrame 타입의 데이터
- `X_val`: pandas.DataFrame 타입의 데이터
- `save_model`: default=`False`. 학습된 모델을 저장하고자 할 경우 True로 설정
- `result_path`: default=`None`. 학습된 모델을 저장할 경로를 지정
- `epochs`: default=`100`. DNN에 해당하는 파라미터로서, DNN 모델의 학습 횟수를 지정
- `dnn_config`: default=`None`. DNN 모델 학습 시 필요한 설정 정보

## get_metric
```python
def get_metric(
        y_val,
        y_pred,
        metric: str = f1_score
)
```
#### Parameters
- `y_val`: model의 예측 값
- `y_pred`: model의 예측 확률값
- `metric`: default=`f1_score`. 평가할 지표. 현재 f1_score만 가능  

## evaluate_model
ML/DL 모델 평가 수행

```python
evaluate_model(
    model,
    X_val,
    y_val,
    result_path: str = None,
    cat_dict_train: dict = None
)
```

#### Parameters
- `model`: 학습이 완료된 모델
- `X_val`: pandas.DataFrame 타입의 데이터
- `y_val`: pandas.DataFrame 타입의 데이터
- `result_path`: default=`None`. 모델 평가 결과를 저장할 경로를 지정
- `cat_dict_train`: default=`None`. target 변수가 categorical 또는 str 타입일 경우, `convert_feature_to_num` 메소드를 통해 변환되기 이전의 원래의 데이터 

## FeatureImportance
XAI를 수행하기 위한 instance 생성

```python
FeatureImportance(
    ai_method: str = 'dnn',
    trained_model=None,
    xai_method: str = 'shap',
    device: str = None,
    gpu_id: int = None
)
```

#### Parameters
- `ai_method`: default=`dnn`. 사용할 모델의 종류; `catboost`, `lightgbm`, `xgboost`, `dnn`
- `trained_model`: default=`None`. 학습이 완료된 모델을 지정
- `xai_method`: default=`shap`. 사용할 XAI 기법의 종류; `shap`, `permutation`, `intrinsic`
- `device`: default=`None`. 사용할 장치의 종류; `gpu`, `cpu`
- `gpu_id`: default=`None`. 사용할 장치가 `gpu`일 경우 gpu unit의 번호; `0`, `1`, ...

### explain
XAI를 수행

```python
FeatureImportance.explain(
    X: DataFrame,
    y: DataFrame,
    result_path,
    category_name_dict=None,
    feature_names=None,
    permutation_importance_n_repeats: int = 150
)
```

#### Parameters
- `X`: pandas.DataFrame 타입의 데이터
- `y`: pandas.DataFrame 타입의 데이터
- `result_path`: XAI 결과를 저장할 경로를 지정
- `category_name_dict`: default=`None`. target 변수가 categorical 또는 str 타입일 경우, `convert_feature_to_num` 메소드를 통해 변환되기 이전의 원래의 데이터
- `feature_names`: default=`None`. 해석할 feature 리스트로서, `X` 파라미터의 feature의 리스트를 지정
- `permutation_importance_n_repeats`: default=`150`. `FeatureImportance`에서 XAI 인스턴스 생성 시, `xai_method=permutation`일 경우 사용되는 파라미터로서 Permutation Importance 계산 시, iteration의 횟수. [sklearn.inspection.permutation_importance] 참고


[CatBoost document]: https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier 
[LightGBM document]: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
[XGBoost document]: https://xgboost.readthedocs.io/en/stable/parameter.html
[pandas.read_csv]: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
[pandas.DataFrame.set_index]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.set_index.html
[sklearn.model_selection.train_test_split]: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
[pandas.DataFrame.astype]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html
[pandas.concat]: https://pandas.pydata.org/docs/reference/api/pandas.concat.html
[study optimize]: https://optuna.readthedocs.io/en/stable/reference/cli.html?highlight=optimize#study-optimize
[sklearn.inspection.permutation_importance]: https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance
[pytorch: Non-linear Activations]: https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
[pytorch: TORCH.OPTIM]: https://pytorch.org/docs/stable/optim.html#module-torch.optim
[pandas.DataFrame.to_csv]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html