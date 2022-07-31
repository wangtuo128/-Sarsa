import numpy as np
import gym


# 定义Agent类
class SarsaAgent:
    """
     实例构造器“__init__”，首先设置Sarsa算法超参数：
     1. dim_of_action: 动作的维度
     2. dim_of_state: 状态的维度
     3. learning_rate: 学习率
     4. gamma: 衰减率

     接着，初始化Q表格: Q_table，Q_table的行数为状态的维度，列数为动作的维度
    """
    def __init__(self, dim_of_action, dim_of_state, learning_rate, gamma):
        self.dim_of_action = dim_of_action
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_table = np.zeros((dim_of_state, dim_of_action))

    """
    SelectAction: Agent基于当前状态，采用贪婪算法选择动作（选择当前状态下Q值最高的动作）
    """
    def SelectAction(self, state):
        Q_row = self.Q_table[state, :]
        max_in_row = np.max(Q_row)
        alternative_action = np.where(Q_row == max_in_row)[0]
        selected_action = np.random.choice(alternative_action)
        return selected_action

    """
    Learn: Agent采用Sarsa算法更新Q_table
    """
    def Learn(self, state, action, reward, state_of_next_step, action_of_next_step, done):
        if done:
            TD_target = reward
        else:
            TD_target = reward + self.gamma * self.Q_table[state_of_next_step, action_of_next_step]
        self.Q_table[state, action] += self.learning_rate * (TD_target - self.Q_table[state, action])
