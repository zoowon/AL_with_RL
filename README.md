<div align="center">

# 2025-2 강화학습 프로젝트 : Active Learning for Image Classification with Reinforcement Learning

<b>Yoon Seo Park</b>, <b>Joo Won Park</b>

<b>Dept. of Computer Science & Engineering, Sogang University</b>

<a>![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)</a>

<img src="/asset/result.png" alt="Logo" width="100%"></div>

## About Project
프로젝트에 대한 설명은 [프로젝트 보고서](https://www.google.com)를 통해 확인 부탁드립니다.

## Installation
1. 가상 환경 세팅
```
conda create -n RL python=3.8
conda activate RL
```
2. 환경 setup
```
setup.sh
```

## Train
데이터셋, sampling 기법, gradient step (DQN 활용 시), GPU 번호를 선택하여 모델 훈련 및 cycle 별 test 진행
```
python main.py --dataset {dataset name} --method {sampling method name} --step {step} --gpu {gpu num}
```
데이터셋은 cifar10, cifar100, fashionmnist 중에서 선택

Sampling method는 random, DQN 중에서 선택

Gradient step은 method가 DQN인 경우에만 int 형으로 기입 (이외에는 무효)

GPU의 default값은 0

## Test
Sampling 기법 별로 학습된 최종 cycle의 weight를 활용하여 Image classification 진행
```
python test.py
```

### Ready for Test
Pretrained된 model을 [링크](https://www.google.com)에서 다운로드 후, 아래와 같이 폴더를 구성 후 test 진행
```
${ROOT} 
|-- checkpoints  
|   |-- cifar10
|   |   |-- random.pth
|   |   |-- DQN_100step.pth
|   |   |-- DQN_200step.pth
|   |-- cifar100
|   |   |-- random.pth
|   |   |-- DQN_100step.pth
|   |   |-- DQN_200step.pth
|   |-- fashionmnist
|   |   |-- random.pth
|   |   |-- DQN_100step.pth
|   |   |-- DQN_200step.pth
```