import gymnasium as gym
import numpy as np

class Discretizer(gym.ObservationWrapper):
    def __init__(self, env, n_bins=10):
        super().__init__(env)
        self.bins = n_bins
        self.is_tuple = False
        
        # handling BatteryGridWorld environment
        if isinstance(env.observation_space, gym.spaces.Tuple):
            self.is_tuple = True
            # 5x5 grid, 16 battery levels = 400 states
            self.n_states = 5 * 5 * 16
            self.observation_space = gym.spaces.Discrete(self.n_states)
            return

        # handling continuous environments
        if "CartPole" in env.spec.id:
            self.buckets = [
                np.linspace(-2.4, 2.4, n_bins),
                np.linspace(-3.0, 3.0, n_bins),
                np.linspace(-0.2, 0.2, n_bins),
                np.linspace(-3.0, 3.0, n_bins)
            ]
        elif "MountainCar" in env.spec.id:
            self.buckets = [
                np.linspace(-1.2, 0.6, n_bins),
                np.linspace(-0.07, 0.07, n_bins)
            ]
        else:
            self.buckets = [np.linspace(-1, 1, n_bins)] * env.observation_space.shape[0]

        self.n_states = (n_bins + 1) ** len(self.buckets)
        self.observation_space = gym.spaces.Discrete(self.n_states)

    def observation(self, o):
        if self.is_tuple:
            x, y, b = o
            return int(x + (y * 5) + (b * 25))

        idx = 0
        mult = 1
        for i, val in enumerate(o):
            b_idx = np.digitize(val, self.buckets[i])
            idx += b_idx * mult
            mult *= (self.bins + 1)
            
        return int(idx)