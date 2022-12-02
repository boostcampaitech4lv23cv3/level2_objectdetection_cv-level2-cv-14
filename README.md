# BoostCamp AI Tech4 level-1-Mask Detection Project
***
## Member🔥
| [김지훈](https://github.com/kzh3010) | [원준식](https://github.com/JSJSWON) | [송영섭](https://github.com/gih0109) | [허건혁](https://github.com/GeonHyeock) | [홍주영](https://github.com/archemist-hong) |
| :-: | :-: | :-: | :-: | :-: |
| <img src="https://avatars.githubusercontent.com/kzh3010" width="100"> | <img src="https://avatars.githubusercontent.com/JSJSWON" width="100"> | <img src="https://avatars.githubusercontent.com/gih0109" width="100"> | <img src="https://avatars.githubusercontent.com/GeonHyeock" width="100"> | <img src="https://avatars.githubusercontent.com/archemist-hong" width="100"> |
***
## Index
* [Demo Video](#demo-video)
* [Project Summary](#project-summary)
* [Requirements](#requirements)
* [Procedures](#procedures)
* [Features](#features)
* [Result](#result)
* [Conclusion](#Conclusion)  
***
## sample image
<img width="120%" src="/opt/ml/baseline/image/sample.png"/>
  

***
## Project Summary

### 주제
- 재활용 품목 분류를 위한 Object Detection

### 개요 및 기대효과
- 바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.\
분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.\
따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다.\
우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다.

### 데이터 셋의 구조도
- train 이미지 개수 : 4883장
- test 이미지 개수 : 4871장
- 이미지 크기 : (1024, 1024)

### Data Classes

|Class|Mask|Gender|AGE|
|---|---|---|---|
|0|General trash|5|Plastic|
|1|Paper|6|Styrofoam|
|2|Paper pack|7|Plastic bag|
|3|Metal|8|Battery|
|4|Glass|9|Clothing|
***
## Procedures




***
## Result
#### EDA
- 데이터 클래스에 불균형이 있는 점을 파악(나이, 성별)
- 데이터 형식은 전체적으로 통일되어 있음(얼굴 위치)

#### 데이터 전처리
- 주어진 Train 데이터를 9:1 비율로 Train, Validation셋으로
모델 학습이 잘 되고 있는지에 대해 검증하기 위해서 사용
- 데이터가 사람 1명과 배경만 있는 단순한 구조이기 때문에
처음에는 Face Detection 모델을 이용해 얼굴을 Crop 후
학습을 진행하려 했으나 Mask를 쓴 얼굴은 모델이 잘 탐지하지 못할 거라 예상해 다른 모델을 사용.
- RGB-Salient Object Detection Task에서 TRACER 모델을 사용하기로 결정. 이 모델은 주어진 이미지에서 Salient(핵심적인) Object를 찾아주고 Box 형태가 아닌 Segmentation 형태로 output이 나온다. 해당 모델이 사람만 segmentation해줄 것으로 예상해 사용했고 결과가 괜찮아서 학습할 때 사용했다.
 
#### Data Augmentation
- Resize는 모델의 학습과 학습시간단축
- ShiftScaleRotate는 일반화, 데이터 Label의 특징을 유지하기 위해서 약하게 넣어줌(shift,scale 10%,rotation 5°)
- RandomBrightnessContrast는 Mask가 흰색 말고 다른 색도 있기 때문에 일반화를 위해서 밝기와 대조를 아주 약하게 넣어줬다.(각각 20%)
- HorizontalFlip은 좌우 반전이기 때문에 단순 데이터 증강을 위해 사용
- CoarseDropout은 cut-out기법으로 Overfitting 방지를 위해 사용
 
#### 모델 개요
- 기본 모델은 빠른 실험과 아이디어 구현을 위해서 Efficientnet 계열 모델을 사용하기로 결정.
- Backbone으로 Efficientnetv2_s를 이용하여 공통된 feature를 추출한 후, 3-Way fc를 통과하여 Mask, Gender, Age 각각의 Class를 분류.
- Multi-Label 문제를 해결하기 위하여 각각의 Class에 대하여 Cross-entropy Loss를 설정하여 더하는 방식으로 설정하였고 추가로 Data Imbalance를 고려하여 Loss에 Weight를 추가.

#### 시연 결과
|<img width="100%" src="/images/wandb_loss.png"/>|<img width="100%" src="/images/wandb_accuracy.png"/>|<img width="100%" src="/images/wandb_f1score.png"/>|
|----|----|----|

|Submit/F1 Score|Submit/Accuracy|
|----|----|
|<div style="text-align: center">0.7040</div>|<div style="text-align: center">75.0635</div>|

***
## Conclusion
#### 잘한 점들
- 팀 단위로 진행하기 전 각자 생각한 가설과 대응하는 전략에 따라 다양한 실험을 진행해 이전에 시도하지 못한 방법을 적용해 볼 수 있었다.
- 팀으로 진행 후에는 git flow를 이용한 협업을 경험 할 수 있었다.
- Template을 사용해 익힘으로서 다른 대회 및 플랫폼에서도 활용할 수 있게 되었다.

#### 아쉬웠던 점들:
- git 사용 시 Issue별로 구분해 사용 못했다.
- 시간 부족으로 Ensemble 시도 못했다.
- 개인적으로 실험을 할 때 문제가 생길 경우 혼자서 해결해야 해서 많은 노력과 시간이 소요되어 다양한 시도를 할 여유가 부족했다.

#### 프로젝트를 통해 배운점:
- pytorch-template 구조 파악, 기능 추가 구현으로 추후에 활용 가능.
- wandb를 통한 실험 피드백.
- Git Flow를 통한 협업 프로세스 체험 및 적용 가능.
- 주어진 데이터 분석을 철저히 진행 후 이에 대응하는 전략을 세워야 한다.
***
## Requirements
* Python >= 3.8 (3.8 recommended)
* PyTorch >= 1.2
* albumentations>=1.3.0
* tqdm
* wandb
* pytorch_metric_learning
* sklearn
* timm
* tensorboard >= 1.14
* Black == 22.10.0
***
## Folder Structure
  ```
  level1-imageclassification-cv-08/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  │
  ├── config.json - holds configuration for training
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── new_project.py - initialize new project with template files
  │
  ├── TRACER/ - face detection modules
  │
  ├── Base/ - abstract Base classes
  │   ├── Base_data_loader.py
  │   ├── Base_model.py
  │   └── Base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   ├── data_loaders.py
  │   └── transform.py
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  │── utils/ - small utility functions
  │   └── util.py
  │
  └── submit/ - submission are saved here 
  ```
  ***
  ### Config file format
Config files are in `.json` format:
```javascript
{
    "name": "Mask_Base",                    // training session name
    "n_gpu": 1,                             // number of GPUs to use for training.
    "arch": {
        "type": "TimmModelMulti",
        "args": {
            "model_name": "efficientnetv2_rw_s",  // name of model architecture to train
            "pretrained": true
        }
    },
    "data_loader": {
        "type": "setup",                // selecting data loader
        "args": {
            "stage": "train",           // selecting the stage between train and eval
            "input_size": 240,          // image resize size
            "batch_size": 16,           // batch size
            "num_workers": 4            // number of cpu processes to be used for data loading
        }
    },
    "optimizer": {
        "type": "Adam",                 // optimizer type
        "args": {
            "lr": 0.0001,               // learning rate
            "weight_decay": 0,          // weight decay
            "amsgrad": true
        }
    },
    "loss": "all_loss",                 // loss type
    "loss_name": [
        [
            "focal",
            [
                1.4,            // multi_loss CE weight
                7.0,
                7.0
            ],
            0.375
        ],
        [
            "focal",
            [
                1.6285,
                2.5912
            ],
            0.25
        ],
        [
            "focal",
            [
                2.8723,
                3.9589,
                4.5075,
                14.0625,
                14.0625,
                28.4211
            ],
            0.375
        ]
    ],
    "metrics": [
        "accuracy", "f1" // list of metrics to evaluate
    ],
    "lr_scheduler": {
        "type": "StepLR", // learning rate scheduler type
        "args": {
            "step_size": 5,
            "gamma": 0.1
        }
    },
    "trainer": {
        "epochs": 50,             // number of training epochs
        "save_dir": "saved/",     // checkpoints are saved in save_dir/models/name
        "save_period": 1,         // save checkpoints every save_freq epochs
        "verbosity": 2,           // 0: quiet, 1: per epoch, 2: full
        "monitor": "min val/loss",// mode and metric for model performance monitoring. set 'off' to disable.
        "early_stop": 3,          // number of epochs to wait before early stop. set 0 to disable.
        "tensorboard": false     // enable tensorboard visualization
    },
    "wandb": true,   // enable wandb logging
    "visualize": false, // enable visualization of wandb visualization
    "submit_dir": "/submit", // submission csv directory
    "info_dir": "/input/data/eval"// eval dataset directory
}
```

**Add addional configurations if you need.**

### Train, Test using config files
Modify `config.json` by your setting:

  ```
  python train.py --config config.json
  
  python test.py --config config.json --resume "Your Checkpoint Path"
  ```

### Streamlit Prediction
Run and Check your Mask Classification Prediction result:

```
streamlit run app.py --server.port="Your Port Number"
```

## License
[This project is licensed under the MIT License. See  LICENSE for more details](https://github.com/victoresque/pytorch-template)