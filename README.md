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

### class & box color imformation
|Class|Box|Class|Box|Class|Box|Class|Box|Class|Box|
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|General_trash|<span style="background-color:rgb(255,0,0)">　　　</span>|Paper|<span style="background-color:rgb(255, 128, 0)">　　　</span>|Paper_pack|<span style="background-color:rgb(255, 255, 0)">　　　</span>|Metal|<span style="background-color:rgb(128, 255, 0)">　　　</span>|Glass|<span style="background-color:rgb(0, 255, 255)">　　　</span>|
|Plastic|<span style="background-color:rgb(0, 128, 255)">　　　</span>|Styrofoam|<span style="background-color:rgb(0, 0, 255)">　　　</span>|Plastic_bag|<span style="background-color:rgb(127, 0, 255)">　　　</span>|Battery|<span style="background-color:rgb(255, 0, 255)">　　　</span>|Clothing|<span style="background-color:rgb(128, 128, 128)">　　　</span>|
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

#### 데이터 전처리
 
#### Data Augmentation
 
#### 모델 개요


***
## Conclusion
#### 잘한 점들

#### 아쉬웠던 점들:

#### 프로젝트를 통해 배운점:
***
## Requirements
***
## Folder Structure
  ### Config file format

**Add addional configurations if you need.**

### Train, Test using config files

### Streamlit Prediction


