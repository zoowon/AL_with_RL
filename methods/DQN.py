import math
import random
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# -----------------------------
# 1. DQN 네트워크 & Replay Buffer
# -----------------------------

class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(
            lambda x: torch.stack(x),
            zip(*batch)
        )
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Q(s,a)를 근사하는 DQN 에이전트
    - state_dim: C + 4  (클래스 확률 C + [entropy, margin, max_prob, budget_used])
    - action_dim: 2 (0: skip, 1: query)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_capacity: int = 10000,
        eps_start: float = 1.0,
        eps_end: float = 0.1,
        eps_decay: int = 1000,
        target_update: int = 100,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.gamma = gamma
        self.batch_size = batch_size

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0
        self.target_update = target_update
        self.action_dim = action_dim

    # ε-greedy exploration 스케줄
    def epsilon(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1.0 * self.steps_done / self.eps_decay)

    def select_action(self, state_vec: np.ndarray) -> int:
        """
        state_vec: numpy 1D (state_dim,)
        return: action (0 or 1)
        """
        self.steps_done += 1
        eps = self.epsilon()
        if random.random() < eps:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state)  # (1, 2)
            action = int(q_values.argmax(dim=1).item())
        return action

    def push_transition(self, state_vec, action: int, reward: float, next_state_vec, done: bool):
        state = torch.as_tensor(state_vec, dtype=torch.float32)
        if next_state_vec is not None:
            next_state = torch.as_tensor(next_state_vec, dtype=torch.float32)
        else:
            next_state = torch.zeros_like(state)
        reward_t = torch.tensor([reward], dtype=torch.float32)
        done_t = torch.tensor([done], dtype=torch.float32)
        action_t = torch.tensor([action], dtype=torch.long)
        self.buffer.push(state, action_t, reward_t, next_state, done_t)

    def optimize(self):
        if len(self.buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        # Q(s,a)
        q_values = self.policy_net(state).gather(1, action)

        # target = r + γ max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_state).max(dim=1, keepdim=True)[0]
            target = reward + (1 - done) * self.gamma * next_q_values

        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target 네트워크 동기화
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


# -----------------------------
# 2. state 구성 함수
# -----------------------------

def build_state(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    num_classes: int,
    budget_used_frac: float,
) -> torch.Tensor:
    """
    state = [p (C), entropy, margin, max_prob, budget_used_frac]
    """
    model.eval()
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))  # (1, C)
        probs = torch.softmax(logits, dim=1)[0]        # (C,)

    p = probs.cpu()
    # 엔트로피 (정규화)
    entropy = -(p * (p + 1e-12).log()).sum().item()
    entropy /= math.log(num_classes)
    # margin
    top2 = torch.topk(p, k=2).values
    margin = (top2[0] - top2[1]).item()
    max_prob = top2[0].item()

    state = torch.cat([
        p,
        torch.tensor([entropy, margin, max_prob, budget_used_frac], dtype=torch.float32),
    ])
    return state  # (C+4,)


# -----------------------------
# 3. Active Learning용 DQN 샘플러
# -----------------------------

def DQN_sampling(
    labeled_set: List[int],
    unlabeled_indices,
    addendum: int,
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    agent: Optional[DQNAgent] = None,
    num_classes: int = 10,
) -> Tuple[List[int], List[int], DQNAgent]:
    """
    DQN 기반 샘플링 함수.
    - labeled_set: 현재 라벨된 샘플 인덱스 리스트
    - unlabeled_indices: 풀에 남아있는 인덱스들 (list or np.ndarray)
    - addendum: 이번 사이클에 새로 라벨링할 샘플 수
    - model: 현재 학습된 분류 모델
    - dataset: torch Dataset (getitem -> (image, label))
    - device: torch.device
    - agent: DQNAgent (처음 호출 시 None이면 내부에서 생성)
    - num_classes: 데이터셋 클래스 개수
    """
    unlabeled_indices = list(unlabeled_indices)
    random.shuffle(unlabeled_indices)

    # state_dim = num_classes + 4 (p(C) + entropy + margin + max_prob + budget_used)
    state_dim = num_classes + 4
    if agent is None:
        agent = DQNAgent(state_dim=state_dim, action_dim=2)

    newly_selected: List[int] = []
    budget_used = 0

    for i, idx in enumerate(unlabeled_indices):
        if budget_used >= addendum:
            break

        # 라벨은 보지 않고, 이미지만으로 state 구성
        image, _ = dataset[idx]
        state = build_state(model, image, device, num_classes, budget_used / addendum)

        # next_state는 단순히 다음 샘플 기준으로 구성 (모델은 episode 동안 고정)
        next_state_vec = None
        if i + 1 < len(unlabeled_indices):
            image_next, _ = dataset[unlabeled_indices[i + 1]]
            next_state_vec = build_state(model, image_next, device, num_classes, budget_used / addendum)

        action = agent.select_action(state.cpu().numpy())

        reward = 0.0
        done = False

        if action == 1:
            # 이때만 label을 쿼리 → AL 시나리오 유지
            _, label = dataset[idx]
            model.eval()
            with torch.no_grad():
                logits = model(image.unsqueeze(0).to(device))
                pred = logits.argmax(dim=1).item()
            reward = 1.0 if pred != int(label) else 0.0

            newly_selected.append(idx)
            budget_used += 1
            if budget_used >= addendum:
                done = True

        agent.push_transition(
            state.cpu().numpy(),
            action,
            reward,
            next_state_vec.cpu().numpy() if next_state_vec is not None else None,
            done,
        )
        agent.optimize()

        if done:
            break

    # labeled / unlabeled 세트 갱신
    new_labeled_set = list(labeled_set) + newly_selected
    new_unlabeled_indices = [idx for idx in unlabeled_indices if idx not in newly_selected]

    return new_labeled_set, new_unlabeled_indices, agent
