import numpy as np
import random
from utils import Discretizer

# Dynamic Programming Methods
def policy_iter(env, g=1.0, tol=1e-9):
    # start with random policy
    pol = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
    
    while True:
        # policy evaluation
        V = eval_pol(env, pol, g, tol)
        
        # policy improvement
        stable = True
        for s in range(env.observation_space.n):
            old_a = np.argmax(pol[s])
            
            # one-step lookahead to find the best action 
            q_vals = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                for prob, next_s, r, _ in env.P[s][a]:
                    q_vals[a] += prob * (r + g * V[next_s])
            
            best_a = np.argmax(q_vals)
            pol[s] = np.eye(env.action_space.n)[best_a]
            
            if old_a != best_a:
                stable = False
        
        if stable:
            return pol, V

def eval_pol(env, pol, g=1.0, tol=1e-9):
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = 0
            for a, prob_a in enumerate(pol[s]):
                for prob, next_s, r, _ in env.P[s][a]:
                    v += prob_a * prob * (r + g * V[next_s])
            
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    return V

def value_iter(env, g=1.0, tol=1e-9):
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            # the lookahead to find the best action value
            q_vals = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                for prob, next_s, r, _ in env.P[s][a]:
                    q_vals[a] += prob * (r + g * V[next_s])
            
            best_v = np.max(q_vals)
            delta = max(delta, np.abs(best_v - V[s]))
            V[s] = best_v
            
        if delta < tol:
            break
            
    # output policy
    pol = np.zeros([env.observation_space.n, env.action_space.n])
    for s in range(env.observation_space.n):
        q_vals = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_s, r, _ in env.P[s][a]:
                q_vals[a] += prob * (r + g * V[next_s])
        
        best_a = np.argmax(q_vals)
        pol[s, best_a] = 1.0
        
    return pol, V

# Monte Carlo Prediction
def mc_pred(env, pol, episodes=500, g=0.99, lr=0.1):
    V = np.zeros(env.observation_space.n)
    hist = []

    for _ in range(episodes):
        s, _ = env.reset()
        traj = []
        done = False
        while not done:
            a = np.random.choice(env.action_space.n, p=pol[s])
            s_next, r, term, trunc, _ = env.step(a)
            done = term or trunc
            traj.append((s, r))
            s = s_next

        G = 0
        seen = set()
        for t in range(len(traj)-1, -1, -1):
            st, rew = traj[t]
            G = g * G + rew
            if st not in seen:
                seen.add(st)
                V[st] += lr * (G - V[st])
        
        hist.append(V[0])
    return V, hist

def td_pred(env, pol, episodes=500, g=0.99, lr=0.1):
    V = np.zeros(env.observation_space.n)
    hist = []

    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        while not done:
            a = np.random.choice(env.action_space.n, p=pol[s])
            s_next, r, term, trunc, _ = env.step(a)
            done = term or trunc
            
            target = r + g * V[s_next]
            V[s] += lr * (target - V[s])
            s = s_next
            
        hist.append(V[0])
    return V, hist

# Control Part
class Agent:
    def __init__(self, env, mode="q_learning", lr=0.1, g=0.99, eps=0.1, n=1):
        self.env = env
        self.mode = mode
        self.lr = lr
        self.g = g
        self.eps = eps
        self.n = n
        
        if hasattr(env.observation_space, 'n'):
            self.n_states = env.observation_space.n
        else:
            self.n_states = 15000 
            
        self.n_actions = env.action_space.n
        self.Q = np.zeros((self.n_states, self.n_actions))

    def act(self, s):
        if random.uniform(0, 1) < self.eps:
            return self.env.action_space.sample()
        return np.argmax(self.Q[s])

    def train(self, episodes, sleep=0):
        import time
        scores = []
        
        for e in range(episodes):
            s, _ = self.env.reset()
            a = self.act(s)
            
            # n-step buffers
            s_buf = [s]
            a_buf = [a]
            r_buf = [0.0]
            
            score = 0
            t = 0
            T = float('inf')
            
            while True:
                if sleep > 0: time.sleep(sleep)

                if t < T:
                    s_next, r, term, trunc, _ = self.env.step(a)
                    done = term or trunc
                    s_buf.append(s_next)
                    r_buf.append(r)
                    score += r
                    
                    if done:
                        T = t + 1
                    else:
                        a_next = self.act(s_next)
                        a_buf.append(a_next)
                        a = a_next 
                
                tau = t - self.n + 1
                if tau >= 0:
                    limit = min(tau + self.n, int(T)) if T != float('inf') else tau + self.n
                    G = 0
                    for k in range(tau + 1, limit + 1):
                        G += (self.g ** (k - tau - 1)) * r_buf[k]
                    
                    if tau + self.n < T:
                        s_n = s_buf[tau + self.n]
                        # Q-learning vs SARSA logic where Q-learning bootstraps from the max Q-value
                        if self.mode == "q_learning":
                            boot = np.max(self.Q[s_n])
                        else:
                            a_n = a_buf[tau + self.n]
                            boot = self.Q[s_n][a_n]
                        G += (self.g ** self.n) * boot
                        
                    s_curr = s_buf[tau]
                    a_curr = a_buf[tau]
                    self.Q[s_curr][a_curr] += self.lr * (G - self.Q[s_curr][a_curr])
                
                if tau == T - 1: break
                t += 1
            
            scores.append(score)
            if self.eps > 0.01: self.eps *= 0.995
                
        return scores