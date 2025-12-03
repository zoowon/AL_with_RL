import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List, Sequence, Tuple, Optional


class ReplayBuffer:
    """간단한 리플레이 버퍼 (state, action, reward만 저장: contextual bandit 설정)."""
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.states: List[torch.Tensor] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.pos = 0

    def __len__(self):
        return len(self.states)

    def push(self, state: torch.Tensor, action: int, reward: float):
        """state는 CPU tensor로 저장 (나중에 device로 옮겨서 사용)."""
        state = state.detach().cpu()
        if len(self.states) < self.capacity:
            self.states.append(state)
            self.actions.append(int(action))
            self.rewards.append(float(reward))
        else:
            self.states[self.pos] = state
            self.actions[self.pos] = int(action)
            self.rewards[self.pos] = float(reward)
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        idxs = np.random.choice(len(self.states), batch_size, replace=False)
        states = torch.stack([self.states[i] for i in idxs], dim=0)
        actions = torch.tensor([self.actions[i] for i in idxs], dtype=torch.long)
        rewards = torch.tensor([self.rewards[i] for i in idxs], dtype=torch.float32)

        return states, actions, rewards


class QNetwork(nn.Module):
    """state → Q(s,a) (a∈{0,1})를 출력하는 간단한 MLP."""

    def __init__(self, state_dim: int, hidden_dim: int = 128, n_actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class DQNAgent:
    """
    Q-network를 '스코어 함수'처럼 사용하는 에이전트.
    - Q(s,1)을 score로 보고, unlabeled에서 top-K를 선택.
    - 학습은 contextual bandit 형태로: target = reward (γ=0).
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        device: torch.device,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        buffer_capacity: int = 10000,
        batch_size: int = 64
    ):
        self.device = device
        self.n_actions = n_actions
        self.batch_size = batch_size

        self.q_net = QNetwork(state_dim, hidden_dim=hidden_dim, n_actions=n_actions).to(device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

    def q_values(self, states: torch.Tensor):
        """states: [N, state_dim] (device로 옮긴 뒤 호출)."""
        return self.q_net(states)

    def add_transition(self, state: torch.Tensor, action: int, reward: float):
        """단일 transition 저장 (state는 CPU tensor로 저장)."""
        self.buffer.push(state, action, reward)

    def train_step(self, grad_steps: int = 1):
        """
        replay buffer에서 미니배치를 여러 번 뽑아서 학습.
        target = reward (γ=0인 contextual bandit 형태).
        """
        if len(self.buffer) < self.batch_size:
            return

        self.q_net.train()

        for _ in tqdm(range(grad_steps), desc="DQN train steps", leave=False):
            states, actions, rewards = self.buffer.sample(self.batch_size)
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)

            q_all = self.q_net(states)                      # [B, 2]
            q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

            target = rewards  # γ=0 → Q(s,a) ≈ E[r | s,a]

            loss = F.mse_loss(q_sa, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def _build_state_from_logits(logits: torch.Tensor, num_classes: int):
    """
    분류기 logits → state 벡터 생성.
    - softmax 확률 C개
    - max_prob, margin(top1-top2), entropy 3개
    → 총 (C + 3) 차원.
    """
    # logits: [C]
    probs = torch.softmax(logits, dim=0)  # [C]
    max_prob, _ = probs.max(dim=0)

    if num_classes >= 2:
        top2, _ = probs.topk(2)
        margin = top2[0] - top2[1]
    else:
        margin = torch.tensor(0.0, device=probs.device)

    entropy = -(probs * (probs + 1e-8).log()).sum()

    state = torch.cat(
        [probs, max_prob.unsqueeze(0), margin.unsqueeze(0), entropy.unsqueeze(0)], dim=0
    )  # [C+3]

    return state


def _compute_states_for_pool(
    model: nn.Module,
    dataset: Dataset,
    indices: Sequence[int],
    device: torch.device,
    num_classes: int
):
    """
    unlabeled pool 전체에 대해 state tensor 계산.
    - 반환: states [N, state_dim] (CPU tensor)
    """
    model.eval()
    states: List[torch.Tensor] = []

    with torch.no_grad():
        for idx in tqdm(indices, desc="DQN state 계산", leave=False):
            img, _ = dataset[int(idx)]
            # img: Tensor [C,H,W] 라고 가정 (transform에서 ToTensor 적용).
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            img = img.to(device)
            if img.dim() == 3:
                img = img.unsqueeze(0)  # [1,C,H,W]

            out = model(img)
            # ResNet이 (logits, features) 형태를 반환할 수도 있으므로 첫 번째만 사용
            if isinstance(out, (tuple, list)):
                logits = out[0]
            else:
                logits = out
            logits = logits.squeeze(0)  # [C]

            state = _build_state_from_logits(logits, num_classes=num_classes)
            states.append(state.cpu())

    if len(states) == 0:
        # 빈 pool인 경우 shape만 맞춰서 반환
        return torch.empty(0, num_classes + 3)

    return torch.stack(states, dim=0)  # [N, state_dim]


def DQN_sampling(
    labeled_set: Sequence[int],
    unlabeled_indices: Sequence[int],
    addendum: int,
    model: nn.Module,
    train_set: Dataset,
    device: torch.device,
    agent: Optional[DQNAgent] = None,
    num_classes: int = 10,
    train_steps: int = 200
):
    """
    DQN 기반 Active Learning 샘플러 (top-K 버전).

    - 입력:
        labeled_set: 현재 라벨이 있는 인덱스 리스트/배열
        unlabeled_indices: 현재 unlabeled pool 인덱스 배열
        addendum: 이번 cycle에서 새로 뽑을 샘플 수 K
        model: 현재까지 학습된 classifier (ResNet18)
        train_set: 전체 train Dataset (CIFAR10/100/FashionMNIST)
        device: torch.device (cuda / cpu)
        agent: 이전 cycle까지 학습된 DQNAgent (없으면 새로 생성)
        num_classes: dataset의 클래스 수
        train_steps: 한 cycle에서 DQN 학습 step 수

    - 동작:
        1) unlabeled 전체에 대해 state 계산
        2) 현재 Q-network로 Q(s,1)을 score로 계산
        3) score가 큰 순서로 K개 선택
        4) 선택된 샘플에 대해 reward(오분류 여부) 계산 후 transition 저장
        5) 일부 미선택 샘플에 대해서는 (action=0, reward=0) transition 저장
        6) replay buffer 기반으로 DQN 학습
        7) labeled_set / unlabeled_indices 업데이트

    - 반환:
        new_labeled_set (list), new_unlabeled_indices (np.ndarray), updated_agent
    """
    # 타입 정리
    if isinstance(labeled_set, np.ndarray):
        labeled_list = labeled_set.tolist()
    else:
        labeled_list = list(labeled_set)

    unlabeled_array = np.array(list(unlabeled_indices), dtype=int)

    if addendum <= 0 or len(unlabeled_array) == 0:
        # 선택할 게 없으면 그대로 반환
        if agent is None:
            # state_dim을 모르는 상태라 Q-network를 만들 수 없으므로 그냥 None 유지
            return labeled_list, unlabeled_array, agent  # type: ignore[return-value]
        return labeled_list, unlabeled_array, agent

    # 1) unlabeled pool 전체에 대한 state 계산 (CPU tensor)
    states = _compute_states_for_pool(
        model=model,
        dataset=train_set,
        indices=unlabeled_array,
        device=device,
        num_classes=num_classes
    )  # [N, state_dim]

    if states.shape[0] == 0:
        if agent is None:
            return labeled_list, unlabeled_array, agent  # type: ignore[return-value]
        return labeled_list, unlabeled_array, agent

    state_dim = states.shape[1]

    # 2) DQNAgent 초기화 (첫 cycle에서만)
    if agent is None:
        agent = DQNAgent(
            state_dim=state_dim,
            n_actions=2,
            device=device,
            hidden_dim=128,
            lr=1e-3,
            buffer_capacity=10000,
            batch_size=64
            )

    # 3) 현재 Q-network로 Q(s,1) score 계산
    agent.q_net.eval()
    with torch.no_grad():
        q_all = agent.q_values(states.to(device))  # [N, 2]
        scores = q_all[:, 1].detach().cpu()        # Q(s,1)만 사용

    # 4) score 상위 K개 선택
    k = min(addendum, len(unlabeled_array))
    topk_scores, topk_pos = torch.topk(scores, k=k)
    selected_positions = topk_pos.numpy()                  # pool 내 위치
    selected_indices = unlabeled_array[selected_positions]  # 실제 데이터 인덱스

    # 남은 pool 인덱스 계산
    mask = np.ones(len(unlabeled_array), dtype=bool)
    mask[selected_positions] = False
    remaining_indices = unlabeled_array[mask]

    # 5) transition 생성 및 버퍼에 저장
    #   (1) 선택된 샘플: action=1, reward = 현재 모델이 틀렸는지 여부 (misclassification)
    model.eval()
    for pool_pos, data_index in zip(selected_positions, selected_indices):
        state = states[int(pool_pos)]  # CPU tensor [state_dim]

        img, label = train_set[int(data_index)]
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        img = img.to(device)
        if img.dim() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            out = model(img)
            if isinstance(out, (tuple, list)):
                logits = out[0]
            else:
                logits = out
            pred_label = logits.argmax(dim=1).item()

        # label은 Dataset에서 int 또는 tensor로 나올 수 있음
        if isinstance(label, torch.Tensor):
            true_label = int(label.item())
        else:
            true_label = int(label)

        reward = 1.0 if pred_label != true_label else 0.0

        agent.add_transition(state, action=1, reward=reward)

    #   (2) 일부 미선택 샘플: action=0, reward=0 (label은 보지 않음)
    non_selected_positions = np.where(mask)[0]
    if len(non_selected_positions) > 0 and len(selected_positions) > 0:
        num_neg = min(len(non_selected_positions), len(selected_positions))
        neg_positions = np.random.choice(non_selected_positions, size=num_neg, replace=False)
        for pool_pos in neg_positions:
            state = states[int(pool_pos)]
            agent.add_transition(state, action=0, reward=0.0)

    # 6) DQN 학습
    agent.train_step(grad_steps=train_steps)

    # 7) labeled / unlabeled 세트 업데이트
    labeled_list.extend(selected_indices.tolist())
    # 중복 방지 및 정렬 (선택 사항)
    labeled_list = sorted(set(int(i) for i in labeled_list))
    remaining_indices = np.array(sorted(int(i) for i in remaining_indices), dtype=int)

    return labeled_list, remaining_indices, agent
