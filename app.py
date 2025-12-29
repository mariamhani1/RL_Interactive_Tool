import streamlit as st
import gymnasium as gym
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1.title
st.set_page_config(page_title="RL Interactive Tool", layout="wide")

# 2.sidebar 
st.sidebar.title("RL Learning Tool")
mode = st.sidebar.radio(
    "Select Module:",
    ["1. Foundation (DP & Prediction)", "2. Control (Q-Learning)", "3. Custom Environment"]
)

# 3.main content 
st.title("Reinforcement Learning Interactive Tool")


if mode == "1. Foundation (DP & Prediction)":
    st.header("Dynamic Programming & Prediction")
    
    # 1. environment controls
    col1, col2 = st.columns(2)
    with col1:
        env_name = st.selectbox("Select Environment", ["FrozenLake-v1", "CliffWalking-v1"])
    with col2:
        hole_penalty = st.slider("Hole Penalty (Reward)", min_value=-50, max_value=0, value=0, step=1,
                               help="Adjusts the reward for falling into a hole. Lower values make the agent safer.")
        
    # 2. hyperparameters
    gamma = st.slider("Discount Factor (Gamma)", 0.0, 1.0, 0.99, 
                     help="Determines how much future rewards are valued. 0 = short-sighted, 1 = far-sighted.")

    # 3. setup environment
    env = gym.make(env_name, is_slippery=False, render_mode="rgb_array").unwrapped
    
    # HACK: Manually modify rewards for FrozenLake to satisfy "Reward Adjustment" requirement
    if env_name == "FrozenLake-v1":
        if hole_penalty != 0:
            new_P = env.unwrapped.P.copy()
            for state in new_P:
                for action in new_P[state]:
                    transitions = new_P[state][action]
                    new_transitions = []
                    for prob, next_state, reward, done in transitions:
                        if done and reward == 0 and next_state != 15: 
                             reward = hole_penalty
                        new_transitions.append((prob, next_state, reward, done))
                    new_P[state][action] = new_transitions
            env.unwrapped.P = new_P

    dp_method = st.radio("Select Method", ["Value Iteration", "Policy Iteration"], horizontal=True)

    if st.button(f"Run {dp_method}"):
        if dp_method == "Value Iteration":
            from algorithms import value_iter
            policy, V = value_iter(env, g=gamma)
        else:
            from algorithms import policy_iter
            policy, V = policy_iter(env, g=gamma)
        
        st.subheader("Optimal Value Function (Heatmap)")
        st.write("The heatmap shows the 'value' of being in each state. Darker/Higher = Better.")
        
        if env_name == "FrozenLake-v1":
            grid_shape = (4, 4)
        else:
            grid_shape = (4, 12)
            
        V_grid = V.reshape(grid_shape)
        
        if env_name == "CliffWalking-v1":
             fig, ax = plt.subplots(figsize=(12, 5))
             annot_kws = {"size": 8}
        else:
             fig, ax = plt.subplots(figsize=(6, 5))
             annot_kws = {"size": 10}
             
        sns.heatmap(V_grid, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, annot_kws=annot_kws)
        st.pyplot(fig)
        
        st.success("Converged! Notice how the values change when you adjust Gamma or the Hole Penalty.")
    st.markdown("---")
    st.subheader("Prediction: TD(0) vs. Monte Carlo")
    st.write("Compare how the two algorithms estimate the value of the Start State (0) over time.")

    col3, col4 = st.columns(2)
    with col3:
        n_episodes = st.slider("Episodes", 100, 2000, 500)
    with col4:
        alpha_lr = st.slider("Learning Rate (Alpha)", 0.01, 0.5, 0.1)

    if st.button("Run Comparison"):
        random_policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
        
        from algorithms import mc_pred, td_pred
        
        V_mc, hist_mc = mc_pred(env, random_policy, episodes=n_episodes, lr=alpha_lr)
        V_td, hist_td = td_pred(env, random_policy, episodes=n_episodes, lr=alpha_lr)
        
        st.line_chart({
            "Monte Carlo": hist_mc,
            "TD(0)": hist_td
        })
        
        st.info("Notice: TD usually learns faster (curve rises earlier) but might bias initially. MC has high variance.")


elif mode == "2. Control (Q-Learning)":
    st.header("Control: Q-Learning & SARSA")
    st.markdown("Train an agent to solve both simple Grids and complex Physics environments.")
    
    col1, col2 = st.columns(2)
    with col1:
        env_choice = st.selectbox("Environment", 
                                ["FrozenLake-v1", "CliffWalking-v0", "CartPole-v1", "MountainCar-v0"])
        algo_choice = st.selectbox("Algorithm", ["q_learning", "sarsa"])
        
    with col2:
        n_steps_val = st.slider("n-steps (for SARSA)", 1, 5, 1, help="Lookahead steps. n=1 is standard. n>1 is n-step TD.")
        n_episodes = st.slider("Episodes", 100, 3000, 500)
        lr_alpha = st.slider("Learning Rate", 0.01, 0.5, 0.15)
        epsilon_exp = st.slider("Exploration Rate (Epsilon)", 0.0, 1.0, 0.3)
        
    speed_mode = st.checkbox("Watch Agent Learn (Slow Motion)", value=False, 
                           help="Slows down the simulation so you can see the decision making.")
    sleep_delay = 0.01 if speed_mode else 0
    
    if st.button("Start Training"):
        status_text = st.empty()
        status_text.write("Initializing Environment...")
        
        if "FrozenLake" in env_choice:
            base_env = gym.make(env_choice, is_slippery=False, render_mode="rgb_array")
        else:
            base_env = gym.make(env_choice, render_mode="rgb_array")
        
        if env_choice in ["CartPole-v1", "MountainCar-v0"]:
            from utils import Discretizer
            env = Discretizer(base_env, n_bins=10)
            status_text.write(f"Applied Discretization (State Space size: {env.n_states})")
        else:
            env = base_env
            
        from algorithms import Agent
        agent = Agent(env, mode=algo_choice, lr=lr_alpha, 
                    eps=epsilon_exp, n=n_steps_val)
        
        status_text.write(f"Training {algo_choice} for {n_episodes} episodes...")
        
        reward_history = agent.train(n_episodes, sleep=sleep_delay)
        
        st.success("Training Complete!")
        
        import pandas as pd
        window_size = 50
        series = pd.Series(reward_history)
        smoothed = series.rolling(window=window_size).mean()
        
        st.subheader("Training Performance (Smoothed)")
        st.line_chart(smoothed)
        
        st.info(f"Final 50-episode Average Reward: {smoothed.iloc[-1]:.2f}")
        
        if env_choice == "CartPole-v1" and smoothed.iloc[-1] > 20:
             st.balloons()
             st.markdown("**Note:** Solving CartPole with a Table is hard! We used 'Discretization' to turn the physics into bins.")



elif mode == "3. Custom Environment":
    st.header("Custom Environment: Battery GridWorld")
    st.markdown("""
    **The Challenge:** The agent must reach the Green Goal before its battery dies.
    * **Blue Agent:** High Battery
    * **Red Agent:** Low Battery
    * **State:** (X, Y, Battery Level)
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        n_episodes = st.slider("Training Episodes", 100, 1000, 300)
    with col2:
        speed_custom = st.checkbox("View Simulation", value=True)
    
    if st.button("Train Custom Agent"):
        from custom_env import BatteryEnv
        from utils import Discretizer
        from algorithms import Agent
        
        raw_env = BatteryEnv("rgb_array")
        env = Discretizer(raw_env) 
        
        agent = Agent(env, mode="q_learning", lr=0.1, g=0.95)
        
        status = st.empty()
        status.write("Training...")
        
        rewards = agent.train(n_episodes, sleep=0)
        
        status.empty()
        
        obs, _ = env.reset()
        state = obs
        done = False
        
        image_spot = st.empty()
        
        while not done:
            action = agent.act(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            frame = env.env.render() 
            image_spot.image(frame, caption=f"Battery Level", width=300)
            
            state = next_state
            
            import time
            if speed_custom:
                time.sleep(0.2)  
                
        st.success("Simulation Finished. Did it reach the green square?")