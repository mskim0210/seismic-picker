import torch
import torch.nn as nn


class WeightedCrossEntropyLoss(nn.Module):
    """가중 크로스엔트로피 손실함수.

    class 불균형 처리를 위해 P파/S파에 높은 가중치 부여.

    loss = -sum_c(w_c * y_c * log(p_c + eps))

    Args:
        class_weights: [noise, P, S] 가중치 (기본: [1.0, 30.0, 30.0])
        epsilon: log(0) 방지
    """

    def __init__(self, class_weights=None, epsilon=1e-7):
        super().__init__()
        if class_weights is None:
            class_weights = [1.0, 30.0, 30.0]
        self.register_buffer(
            "weights",
            torch.tensor(class_weights, dtype=torch.float32).view(1, -1, 1)
        )
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, 3, T) - softmax 출력
            targets:     (B, 3, T) - Gaussian 라벨
        Returns:
            scalar loss
        """
        log_pred = torch.log(predictions + self.epsilon)
        loss = -self.weights * targets * log_pred
        return loss.mean()


class FocalCrossEntropyLoss(nn.Module):
    """Focal Cross-Entropy 손실함수.

    쉬운 샘플(noise)의 가중치를 줄여 어려운 샘플(P/S)에 집중.

    FL = -alpha * (1 - p)^gamma * y * log(p + eps)

    Args:
        alpha: class별 가중치 [noise, P, S]
        gamma: focusing parameter (높을수록 쉬운 샘플 무시)
        epsilon: log(0) 방지
    """

    def __init__(self, alpha=None, gamma=2.0, epsilon=1e-7):
        super().__init__()
        if alpha is None:
            alpha = [0.1, 0.45, 0.45]
        self.register_buffer(
            "alpha",
            torch.tensor(alpha, dtype=torch.float32).view(1, -1, 1)
        )
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, 3, T)
            targets:     (B, 3, T)
        Returns:
            scalar loss
        """
        p = predictions + self.epsilon
        focal_weight = (1.0 - p) ** self.gamma
        loss = -self.alpha * focal_weight * targets * torch.log(p)
        return loss.mean()
