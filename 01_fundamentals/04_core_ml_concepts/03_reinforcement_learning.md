# Reinforcement Learning

## What is Reinforcement Learning?

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative rewards. Unlike supervised learning, where the model learns from labeled data, in RL, the agent learns from the consequences of its actions.

## Key Concepts

### 1. Agent
The learner or decision-maker that interacts with the environment.

### 2. Environment
The external system that the agent interacts with. It provides feedback in the form of rewards or penalties based on the agent's actions.

### 3. Actions
The choices available to the agent that affect the state of the environment.

### 4. States
The current situation of the agent in the environment. The state can change based on the actions taken by the agent.

### 5. Rewards
Feedback from the environment in response to the agent's actions. The goal of the agent is to maximize the total reward over time.

### 6. Policy
A strategy that the agent employs to determine the next action based on the current state. It can be deterministic or stochastic.

### 7. Value Function
A function that estimates the expected return (cumulative reward) from a given state, helping the agent to evaluate the long-term benefit of its actions.

## Why Does This Matter?

Reinforcement Learning is crucial for developing intelligent systems that can learn from their experiences and improve over time. It has applications in various fields, including robotics, gaming, finance, and healthcare. For example, RL algorithms are used in training autonomous vehicles to navigate complex environments and in developing AI agents that can play games like Chess or Go at superhuman levels.

## Practical Example

Imagine teaching a dog to fetch a ball. The dog (agent) learns to perform the action of fetching the ball (action) when you throw it (environment). If the dog brings the ball back and receives a treat (reward), it is more likely to repeat that action in the future. Over time, the dog learns the best way to fetch the ball to maximize its treats.

## Thought Experiment

Consider a scenario where you are training a robot to navigate a maze. The robot receives a positive reward for reaching the exit and a negative reward for hitting walls. How would you design the robot's learning process to ensure it learns the most efficient path to the exit? What factors would you consider in defining the rewards?

## Conclusion

Reinforcement Learning is a powerful paradigm that enables agents to learn optimal behaviors through trial and error. By understanding the key concepts and their applications, you can leverage RL techniques to solve complex decision-making problems in various domains.

## Mathematical Foundation

### Key Formulas

**Markov Decision Process (MDP):**
Defined by tuple $(S, A, P, R, \gamma)$ where:
- $S$ = state space
- $A$ = action space  
- $P(s'|s,a)$ = transition probability
- $R(s,a,s')$ = reward function
- $\gamma$ = discount factor $(0 \leq \gamma \leq 1)$

**Value Functions:**

**State Value Function:**
$$V^\pi(s) = E_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_{t+1} | s_0 = s\right]$$

**Action Value Function (Q-function):**
$$Q^\pi(s,a) = E_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_{t+1} | s_0 = s, a_0 = a\right]$$

**Bellman Equations:**

**Bellman Equation for $V^\pi$:**
$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

**Bellman Equation for $Q^\pi$:**
$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')]$$

**Temporal Difference Learning:**
$$V(s) \leftarrow V(s) + \alpha[r + \gamma V(s') - V(s)]$$

### Solved Examples

#### Example 1: Simple Grid World Value Iteration

Given: 3×3 grid world with:
- Start: (0,0), Goal: (2,2) with reward +10
- Walls at (1,1), Actions: up, down, left, right
- Discount factor $\gamma = 0.9$

Find: Optimal value function using value iteration

Solution:
Step 1: Initialize values
$V_0(s) = 0$ for all states except goal: $V_0(2,2) = 10$

Step 2: Value iteration update
$$V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V_k(s')]$$

Step 3: First iteration calculations
For state (0,1):
- Up: blocked → stay at (0,1), reward = -1
- Down: go to (0,0), reward = -1  
- Right: blocked by wall, reward = -1
- Left: go to (0,1), reward = -1

$$V_1(0,1) = \max\{-1 + 0.9(0), -1 + 0.9(0), -1 + 0.9(0), -1 + 0.9(0)\} = -1$$

For state (1,2):
- Right: go to (2,2), reward = -1 + 10 = 9
- Other actions lead to lower rewards

$$V_1(1,2) = -1 + 0.9(10) = 8$$

After convergence: Optimal path emerges with highest values leading to goal.

#### Example 2: Q-Learning Update

Given: Agent in state $s = 1$, takes action $a = \text{right}$, receives reward $r = 5$, ends in state $s' = 3$
Current Q-values: $Q(1, \text{right}) = 2$, $Q(3, \text{up}) = 8$, $Q(3, \text{down}) = 6$
Learning parameters: $\alpha = 0.1$, $\gamma = 0.9$

Find: Updated Q-value

Solution:
Step 1: Find maximum Q-value for next state
$$\max_{a'} Q(s', a') = \max\{Q(3, \text{up}), Q(3, \text{down})\} = \max\{8, 6\} = 8$$

Step 2: Apply Q-learning update rule
$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$
$$Q(1, \text{right}) \leftarrow 2 + 0.1[5 + 0.9(8) - 2]$$
$$Q(1, \text{right}) \leftarrow 2 + 0.1[5 + 7.2 - 2] = 2 + 0.1(10.2) = 3.02$$

Result: Updated Q-value is 3.02, showing positive learning from the experience.

#### Example 3: Policy Evaluation

Given: 2-state MDP with policy $\pi(\text{stay}|s_1) = 0.7$, $\pi(\text{move}|s_1) = 0.3$
Transition probabilities and rewards:
- Stay in $s_1$: reward = 1, stay with probability 1
- Move from $s_1$ to $s_2$: reward = 5, transition probability 1
- From $s_2$: deterministic return to $s_1$ with reward = 0

Find: State values under this policy with $\gamma = 0.8$

Solution:
Step 1: Set up Bellman equations
$$V^\pi(s_1) = 0.7[1 + 0.8 V^\pi(s_1)] + 0.3[5 + 0.8 V^\pi(s_2)]$$
$$V^\pi(s_2) = 0 + 0.8 V^\pi(s_1)$$

Step 2: Substitute and solve
From equation 2: $V^\pi(s_2) = 0.8 V^\pi(s_1)$

Substituting into equation 1:
$$V^\pi(s_1) = 0.7[1 + 0.8 V^\pi(s_1)] + 0.3[5 + 0.8(0.8 V^\pi(s_1))]$$
$$V^\pi(s_1) = 0.7 + 0.56 V^\pi(s_1) + 1.5 + 0.192 V^\pi(s_1)$$
$$V^\pi(s_1) = 2.2 + 0.752 V^\pi(s_1)$$
$$0.248 V^\pi(s_1) = 2.2$$
$$V^\pi(s_1) = \frac{2.2}{0.248} \approx 8.87$$

$$V^\pi(s_2) = 0.8 \times 8.87 = 7.10$$

Result: $V^\pi(s_1) = 8.87$, $V^\pi(s_2) = 7.10$