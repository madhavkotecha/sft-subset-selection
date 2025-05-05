from typing import List, Optional
import numpy as np
import math

class DynamicWeights:
    """
    A class that implements dynamic weight adjustment based on rewards and exploration-exploitation trade-off.
    Uses exponential moving average for reward estimation and epsilon-based exploration.
    """
    
    def __init__(
            self,
            weights: List[float],
            smoothing_factor: float = 0.9
    ):
        self.num_domains = len(weights)
        self.weights = weights
        self._estimated_reward = [0] * self.num_domains
        self._probabilities = self._normalize_weights(weights)
        self.eps = 1 / self.num_domains
        self.prev_eps = None
        self.smoothing_factor = smoothing_factor
        self.vars_to_log = ["_probabilities", "_estimated_reward"]

    def _normalize_weights(self, weights: List[float]) -> List[float]:
        total = np.sum(weights)
        return [w / total for w in weights]

    def _update_epsilon(self, iteration: int) -> None:
        self.prev_eps = self.eps
        self.eps = min(
            1/self.num_domains,
            math.sqrt(math.log(self.num_domains)/(self.num_domains*iteration))
        )

    def _update_estimated_reward(self, index: int, reward: float) -> None:
        self._estimated_reward[index] = (
            self.smoothing_factor * self._estimated_reward[index] + 
            (1 - self.smoothing_factor) * math.exp(reward)
        )

    def _update_weights_and_probabilities(self, use_exp: bool = True) -> None:
        if use_exp:
            total_estimated_rewards = sum(math.exp(r * self.prev_eps) for r in self._estimated_reward)
            scaling_factor = (1 - self.num_domains * self.eps) / total_estimated_rewards
            self.weights = [
                math.exp(r * self.prev_eps) * scaling_factor + self.eps
                for r in self._estimated_reward
            ]
        else:
            total_estimated_rewards = sum(r * self.prev_eps for r in self._estimated_reward)
            scaling_factor = (1 - self.num_domains * self.eps) / total_estimated_rewards
            self.weights = [
                r * self.prev_eps * scaling_factor + self.eps
                for r in self._estimated_reward
            ]
        
        self._probabilities = self._normalize_weights(self.weights)

    def update(self, index: int, reward: float, iteration: int) -> List[float]:
        self._update_estimated_reward(index, reward)
        self._update_epsilon(iteration)
        self._update_weights_and_probabilities(use_exp=True)
        return self._probabilities

    def group_update(self, idx: List[int], rewards: List[float], iteration: int) -> List[float]:
        self._update_epsilon(iteration)

        for index, reward in zip(idx, rewards):
            self._update_estimated_reward(index, reward)
            
        self._update_weights_and_probabilities(use_exp=False)
        return self._probabilities