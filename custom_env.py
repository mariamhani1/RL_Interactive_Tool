import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class BatteryEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.size = 5
        self.max_bat = 15
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.size),
            spaces.Discrete(self.size),
            spaces.Discrete(self.max_bat + 1)
        ))
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        self.pos = np.array([0, 0])
        self.bat = self.max_bat
        self.goal = np.array([4, 4])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.array([0, 0])
        self.bat = self.max_bat
        return (self.pos[0], self.pos[1], self.bat), {}

    def step(self, action):
        # 0:Up, 1:Right, 2:Down, 3:Left
        vec = {
            0: np.array([-1, 0]), 1: np.array([0, 1]),
            2: np.array([1, 0]),  3: np.array([0, -1])
        }
        
        nxt = self.pos + vec[action]
        
        # Check bounds
        if 0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size:
            self.pos = nxt
            
        self.bat -= 1
        
        done = False
        rew = -1 
        
        if np.array_equal(self.pos, self.goal):
            rew = 50
            done = True
        elif self.bat <= 0:
            rew = -10
            done = True
            
        return (self.pos[0], self.pos[1], self.bat), rew, done, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.Surface((300, 300))
            
        cvs = pygame.Surface((300, 300))
        cvs.fill((255, 255, 255)) 
        
        sz = 300 // self.size
        
        for r in range(self.size):
            for c in range(self.size):
                rect = pygame.Rect(c*sz, r*sz, sz, sz)
                pygame.draw.rect(cvs, (200, 200, 200), rect, 1)
                
        # Goal
        g_rect = pygame.Rect(self.goal[1]*sz, self.goal[0]*sz, sz, sz)
        pygame.draw.rect(cvs, (0, 255, 0), g_rect)
        
        # Agent
        a_rect = pygame.Rect(self.pos[1]*sz + 10, self.pos[0]*sz + 10, sz-20, sz-20)
        col = (0, 0, 255) if self.bat > 5 else (255, 0, 0)
        pygame.draw.circle(cvs, col, a_rect.center, (sz-20)//2)
        
        return np.transpose(np.array(pygame.surfarray.pixels3d(cvs)), (1, 0, 2))