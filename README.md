# Reinforcement Learning Interactive Tool

An interactive web application built with Streamlit and Gymnasium to demonstrate core Reinforcement Learning concepts, algorithms, and applications through real-time visualization.

## Overview
This tool allows users to experiment with RL agents in various environments, bridging the gap between theory and practice. It covers Dynamic Programming, Prediction methods, Control algorithms, and handling complex state spaces.

## Key Features

### 1. Foundation (DP & Prediction)
*   **Algorithms**: Value Iteration, Policy Iteration, Monte Carlo (MC), and Temporal Difference (TD).
*   **Visuals**: Heatmaps of the Value Function showing state desirability.

### 2. Control (Q-Learning & SARSA)
*   **Algorithms**: Q-Learning (Off-Policy) and n-step SARSA (On-Policy).
*   **Environments**: Discrete (FrozenLake, CliffWalking) and Continuous (CartPole, MountainCar using Discretization).
*   **Interactive**: Real-time hyperparameter adjustment.

### 3. Custom Environment: Battery GridWorld
A custom Gymnasium environment where an agent navigates a 5x5 grid to reach a target while managing finite battery.
*   **State Space**: Tuple (x, y, battery_level).
*   **Rewards**: +50 for goal, -1 per step, -10 if battery dies.

## Installation & Usage

### Prerequisites
*   Python 3.8+

### Setup
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the application:
    ```bash
    streamlit run app.py
    ```

3.  Open the link provided in the terminal (usually strict `http://localhost:8501`) to interact with the tool.

## Project Structure
*   `app.py`: Main Streamlit application and UI logic.
*   `algorithms.py`: Implementation of RL algorithms and Discretization wrapper.
*   `custom_env.py`: Custom BatteryGridWorld environment.
*   `requirements.txt`: Project dependencies.
