<div align="center">

# 2025-2 강화학습 프로젝트 : Active Learning for Image Classification with Reinforcement Learning

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-EE4C2C.svg)](https://pytorch.org/)

**Yoonseo Park** · **Joo Won Park**

**Department of Computer Science & Engineering, Sogang University**

*2025-2 Reinforcement Learning Course Project*

**상세한 프로젝트 보고서 및 분석은 [프로젝트 보고서](https://github.com/zoowon/AL_with_RL/tree/main/Report)를 참조하세요**

<img src="/asset/active_learning_relative_all.png" alt="Results" width="100%">

</div>

---

## 📋 목차

- [개요](#-개요)
- [방법론](#-방법론)
  - [Active Learning 프레임워크](#active-learning-프레임워크)
  - [DQN 기반 샘플 선택](#dqn-기반-샘플-선택)
  - [State 표현](#state-표현)
  - [Reward 설계](#reward-설계)
- [프로젝트 구조](#-프로젝트-구조)
- [설치 방법](#-설치-방법)
- [사용법](#-사용법)
  - [학습](#학습)
  - [테스트](#테스트)
- [실험](#-실험)
- [결과](#-결과)
- [참고 문헌](#-참고-문헌)

---

## 🎯 개요

본 프로젝트는 **Deep Q-Network (DQN)** 을 활용하여 라벨링을 위한 가장 유익한 샘플을 지능적으로 선택하는 이미지 분류용 **Active Learning** 프레임워크를 구현합니다. 전통적인 랜덤 샘플링과 달리, DQN 기반 접근법은 강화학습을 통해 최적의 샘플 선택 정책을 학습하여 데이터 효율성과 모델 성능을 크게 향상시킵니다.

### 문제 정의

실제 환경에서 라벨링된 데이터를 얻는 것은 비용이 많이 들고 시간이 오래 걸립니다. Active Learning은 다음과 같은 방법으로 이 문제를 해결합니다:
- 가장 가치 있는 라벨링되지 않은 샘플을 반복적으로 선택
- 정보가 많은 인스턴스에 대해서만 라벨 요청
- 최소한의 라벨링된 데이터로 모델 성능 최대화

### 우리의 접근법

샘플 선택 문제를 **contextual bandit** 으로 공식화하고 DQN을 사용하여 모델 개선에 가장 기여하는 샘플을 학습합니다. 에이전트는 과거 선택 경험을 통해 active learning 사이클에 걸쳐 점점 더 나은 선택을 하도록 학습합니다.

---

## 🔬 방법론

### Active Learning 프레임워크

Active learning 파이프라인은 다음과 같은 반복 프로세스를 따릅니다:

```
[Active Learning 실험 절차]

1. 초기화
   - 라벨링된 샘플: N개
   - 미라벨링 샘플: (50,000-N)개

2. 각 사이클 t = 1 ... 10 에서
   2-1. 현재 라벨링된 데이터로 분류기(ResNet-18) 학습
   2-2. 테스트셋에서 성능 평가
   2-3. 미라벨링 샘플 중 정보가 많은 K 개 선택
        * Random: 균등 랜덤 샘플링
        * DQN: 학습된 선택 정책으로 샘플링
   2-4. 선택된 K개를 라벨링된 데이터셋에 추가

3. 마지막 사이클 이후
   - 최종 모델 체크포인트 저장
```

### DQN 기반 샘플 선택

#### Contextual Bandit으로의 공식화

샘플 선택을 **contextual bandit** 문제로 모델링합니다:
- **State (s)**: 미라벨링 샘플의 특징 표현
- **Action (a)**: 이진 결정 {0: 선택 안 함, 1: 선택}
- **Reward (r)**: 샘플의 유익성에 기반한 즉각적인 피드백

#### Q-Network 구조

```python
Input: State Vector (num_classes + 3 차원)
  ↓
Linear(state_dim → 128) + ReLU
  ↓
Linear(128 → 128) + ReLU
  ↓
Linear(128 → 2)  # 액션 {0, 1}에 대한 Q-value
  ↓
Output: 각 액션에 대한 Q(s, a)
```

#### 선택 프로세스

1. **State 계산**: 각 미라벨링 샘플에 대해 state 벡터 계산
2. **Q-value 추정**: Q-network를 통한 forward pass → Q(s, 0)과 Q(s, 1)
3. **점수 매기기**: Q(s, 1)을 정보성 점수로 사용
4. **Top-K 선택**: Q(s, 1) 점수가 가장 높은 K개 샘플 선택
5. **Transition 저장**: replay buffer에 (state, action, reward) 저장
6. **Network 업데이트**: experience replay를 사용하여 Q-network 학습

### State 표현

각 미라벨링 샘플에 대해, 예측 불확실성을 포착하는 포괄적인 state 벡터를 구성합니다:

```python
State = [
    # 1. Softmax 확률 (C 차원)
    prob_class_1, prob_class_2, ..., prob_class_C,
    
    # 2. 최대 확률 (1 차원)
    max_probability,
    
    # 3. 상위 2개 예측 간 마진 (1 차원)
    margin = prob_top1 - prob_top2,
    
    # 4. 예측 엔트로피 (1 차원)
    entropy = -Σ(p_i * log(p_i))
]
```

**총 차원**: C + 3
- CIFAR-10, FashionMNIST: 10 + 3 = 13 차원
- CIFAR-100: 100 + 3 = 103 차원

**의미**:
- **높은 엔트로피** → 더 불확실 → 잠재적으로 더 유익함
- **작은 마진** → 분류기가 상위 클래스 간 혼란 → 라벨링할 가치 있음
- **낮은 최대 확률** → 낮은 신뢰도 → 유익할 가능성 높음

### Reward 설계

간단하지만 효과적인 보상 신호를 사용합니다:

```python
reward = 1.0  if 분류기가 잘못된 라벨을 예측
reward = 0.0  if 분류기가 올바른 라벨을 예측
```

**의미**:
- 현재 분류기가 잘못 분류하는 샘플이 더 유익함
- "어려운" 샘플을 선택하도록 학습하면 모델이 더 빠르게 개선됨
- 단순한 이진 보상이 명확한 학습 신호 제공

### 학습 알고리즘

```python
각 Active Learning 사이클마다:
    1. 현재 라벨링된 데이터셋으로 ResNet-18 분류기 학습 (200 에폭)
    
    2. 모든 미라벨링 샘플에 대해 state 계산
    
    3. DQN 사용 시:
        a. 모든 미라벨링 샘플에 대해 Q(s, 1) 계산
        b. 가장 높은 Q(s, 1)을 가진 상위 K개 샘플 선택
        c. 선택된 샘플에 대한 reward 계산
        d. replay buffer에 transition 추가:
           - 선택된 샘플: (state, action=1, reward)
           - 일부 미선택 샘플: (state, action=0, reward=0)
        e. replay buffer의 미니배치로 Q-network 학습
           - Loss: MSE(Q(s,a), reward)  # contextual bandit에서 γ=0
           - Optimizer: Adam
           - Gradient steps: 100~500 (설정 가능)
    
    4. 라벨링/미라벨링 데이터셋 업데이트
```

### 주요 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|-----------|-------|-------------|
| **Active Learning** |
| Initial Labeled (N) | 1,000 또는 2,000 (CIFAR-100) | 시작 라벨링 샘플 수 |
| Addendum (K) | 1,000 또는 2,000 (CIFAR-100) | 사이클당 추가되는 샘플 수 |
| Cycles | 10 | 총 AL 반복 횟수 |
| **분류기 학습** |
| Epochs | 200 | 사이클당 학습 에폭 수 |
| Batch Size | 128 | 미니배치 크기 |
| Learning Rate | 0.1 | 초기 학습률 (SGD) |
| LR Milestones | 160 | 학습률 감소 에폭 |
| Gamma | 0.1 | 학습률 감소 계수 |
| **DQN 설정** |
| Hidden Dim | 128 | Q-network 은닉층 유닛 수 |
| Learning Rate | 1e-3 | DQN optimizer 학습률 (Adam) |
| Buffer Capacity | 10,000 | Replay buffer 크기 |
| Batch Size | 64 | DQN 학습 배치 크기 |
| Gradient Steps | 100~500 | 사이클당 학습 스텝 수 |

---

## 📁 프로젝트 구조

```
AL_with_RL/
├── main.py               # 메인 학습 스크립트
├── config.py             # 하이퍼파라미터 설정
├── requirements.txt      # Python 의존성
├── setup.sh              # 환경 설정 스크립트
├── README.md             # 본 파일
│
├── methods/              # 샘플링 전략
│   ├── random.py         # Random baseline 샘플러
│   └── DQN.py            # DQN 기반 샘플러
│       ├── ReplayBuffer  # Experience replay
│       ├── QNetwork      # Q-value 네트워크
│       ├── DQNAgent      # DQN 학습 로직
│       └── DQN_sampling  # 샘플 선택 함수
│
├── models/               # 신경망 모델
│   ├── resnet.py         # ResNet-18 분류기
│   └── sampler.py        # 커스텀 데이터 샘플러
│
├── data/                 # 데이터셋 저장소 (자동 다운로드)
│   ├── cifar10/
│   ├── cifar100/
│   └── fashionmnist/
│
├── checkpoints/          # 저장된 모델 가중치
│   ├── cifar10/
│   ├── cifar100/
│   └── fashionmnist/
│
├── logs/                 # 학습 로그
│   ├── cifar10/
│   ├── cifar100/
│   └── fashionmnist/
│
└── asset/                # README용 자료
    └── active_learning_accuracy_all.png
    └── active_learning_relative_all.png
```

---

## 🚀 설치 방법

### 사전 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (권장)
- Conda 또는 virtualenv

### Step 1: 가상환경 생성

```bash
conda create -n RL python=3.8 -y
conda activate RL
```

### Step 2: 의존성 설치

```bash
bash setup.sh
```

또는 수동으로 설치:

```bash
pip install torch torchvision
pip install numpy==1.21.6 tqdm==4.66.2
```

### 데이터셋 설정

데이터셋(CIFAR-10, CIFAR-100, FashionMNIST)은 처음 학습 스크립트를 실행할 때 torchvision에서 자동으로 다운로드됩니다.

### 사전 학습된 가중치 다운로드

실험에 사용된 사전 학습된 모델 체크포인트는 다음 Google Drive 링크에서 다운로드할 수 있습니다:

**📥 [다운로드 링크](https://drive.google.com/drive/folders/1XTuVJQ7raXbRQ53PB5zzwCeDJwUx--2x?usp=drive_link)**

다운로드 후 압축을 해제하고 프로젝트 루트의 `checkpoints/` 디렉토리에 배치하세요:

포함된 모델:
- ✅ CIFAR-10: DQN (200 steps)
- ✅ CIFAR-100: DQN (100 steps)
- ✅ FashionMNIST: DQN (100 steps)

---

## 💻 사용법

### 학습

지정된 데이터셋과 샘플링 방법으로 active learning 모델을 학습합니다:

```bash
python main.py --dataset <DATASET> --method <METHOD> [--step <STEPS>] [--gpu <GPU_ID>]
```

#### 인자

- `--dataset`: 데이터셋 이름
  - 선택: `cifar10`, `cifar100`, `fashionmnist`
  - **필수**
  
- `--method`: 샘플링 방법
  - 선택: `random`, `DQN`
  - **필수**
  
- `--step`: DQN 학습을 위한 gradient step 수
  - 타입: Integer
  - `--method DQN`일 때만 사용
  - 예시: `100`, `200`, `300`, `400`, `500`
  - **선택** (random 샘플링에서는 무시됨)
  
- `--gpu`: GPU 디바이스 번호
  - 타입: Integer
  - 기본값: `0`
  - **선택**

#### 예시

**1. CIFAR-10에서 Random 샘플링**
```bash
python main.py --dataset cifar10 --method random
```

**2. CIFAR-10에서 DQN 샘플링 (100 gradient steps)**
```bash
python main.py --dataset cifar10 --method DQN --step 100
```

**3. CIFAR-100에서 DQN 샘플링 (500 gradient steps), GPU 1 사용**
```bash
python main.py --dataset cifar100 --method DQN --step 500 --gpu 1
```

**4. FashionMNIST에서 Random 샘플링**
```bash
python main.py --dataset fashionmnist --method random --gpu 0
```

#### 학습 출력

로그는 자동으로 다음 위치에 저장됩니다:
```
logs/<dataset>/<method>_[<step>step_]<timestamp>.log
```

체크포인트는 다음 위치에 저장됩니다:
```
checkpoints/<dataset>/resnet18_<method>[_<step>].pth
```

## 🧪 실험

### 실험 설정

두 가지 샘플링 전략으로 세 가지 벤치마크 데이터셋에 대한 실험을 수행합니다:

| 데이터셋 | 클래스 수 | 학습 크기 | 테스트 크기 | 이미지 크기 |
|---------|---------|------------|-----------|------------|
| CIFAR-10 | 10 | 50,000 | 10,000 | 32×32×3 |
| CIFAR-100 | 100 | 50,000 | 10,000 | 32×32×3 |
| FashionMNIST | 10 | 60,000 | 10,000 | 28×28×1→32×32×3 |

### 샘플링 전략

1. **Random 샘플링 (Baseline)**
   - 미라벨링 풀에서 균등 랜덤 선택
   - 학습 없음
   - 성능의 하한선 역할

2. **DQN 기반 샘플링 (제안 방법)**
   - Q-network를 통한 학습된 선택 정책
   - 다양한 gradient step으로 테스트: {100, 200, 300, 400, 500}
   - random baseline을 능가하는 것이 목표

### 평가 지표

- **테스트 정확도**: 각 사이클 후 테스트셋에서의 분류 정확도
- **샘플 효율성**: 라벨링된 샘플당 성능 향상
- **학습 곡선**: 10 사이클에 걸친 정확도 진행 상황

---

## 📊 결과

결과는 다음 위치에 저장됩니다:

### 체크포인트

최종 학습된 모델은 다음에 저장됩니다:
```
checkpoints/
├── cifar10/
│   ├── resnet18_random.pth
│   ├── resnet18_DQN_100.pth
│   ├── resnet18_DQN_200.pth
│   └── ...
├── cifar100/
│   └── ...
└── fashionmnist/
    └── ...
```

### 학습 로그

에폭별 지표가 포함된 상세 로그:
```
logs/
├── cifar10/
│   ├── random_<timestamp>.log
│   ├── DQN_100step_<timestamp>.log
│   └── ...
├── cifar100/
│   └── ...
└── fashionmnist/
    └── ...
```

### 결과

<img src="/asset/active_learning_accuracy_all.png" alt="Accuracy Visualization" width="100%">

<img src="/asset/active_learning_relative_all.png" alt="Relative Improvement Visualization" width="100%">

*데이터셋 및 gradient step에 따른 Random vs. DQN 샘플링 비교*

---

## 📚 참고 문헌

### 핵심 논문

1. **Deep Q-Network**
   - Mnih, Volodymyr, et al. *"Human-level control through deep reinforcement learning."* nature 518.7540 (2015): 529-533.
2. **Active Learning**
   - Gal, Yarin, Riashat Islam, and Zoubin Ghahramani. *"Deep bayesian active learning with image data."* International conference on machine learning. PMLR, 2017.


### 프레임워크 및 도구

- **PyTorch**: 딥러닝 프레임워크
- **Torchvision**: 컴퓨터 비전 데이터셋 및 모델
- **NumPy**: 수치 연산 라이브러리

---

## 👥 SVM 팀원

- **박윤서** - 서강대학교 컴퓨터공학과
- **박주원** - 서강대학교 컴퓨터공학과

**과목**: 강화학습개론 (2025-2)

---

<div align="center">

**상세한 프로젝트 보고서 및 분석은 [프로젝트 보고서](https://github.com/zoowon/AL_with_RL/tree/main/Report)를 참조하세요**

</div>
