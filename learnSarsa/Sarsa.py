import numpy as np
import gym


def main():
    environment = gym.make("CliffWalking-v0")
    sarsaAgent = SarsaAgent(
        dim_of_action=environment.action_space.n,
        dim_of_state=environment.observation_space.n,
        learning_rate=0.1,
        gamma=0.9
    )

    is_render = False
    for episode in range(500):
        total_reward, total_steps = run_episode(environment, sarsaAgent, is_render)
        print('Episode %s: steps = %s, reward = %.lf' % (episode, total_steps, total_reward))

        if episode % 20 == 0:
            is_render = True
        else:
            is_render = False
    print(sarsaAgent.Q_table)


# 定义Agent类
class SarsaAgent:
    """
     __init__：实例构造器，首先设置Sarsa算法超参数：
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


def run_episode(environment, sarsaAgent, is_render=False):
    total_steps = 0
    total_reward = 0

    state = environment.reset()
    action = sarsaAgent.SelectAction(state)

    while True:
        state_of_next_step, reward, done, _ = environment.step(action)
        action_of_next_step = sarsaAgent.SelectAction(state_of_next_step)
        sarsaAgent.Learn(state, action, reward, state_of_next_step, action_of_next_step, done)
        action = action_of_next_step
        state = state_of_next_step
        total_reward += reward
        total_steps += 1
        if is_render:
            environment.render()
        if done:
            break
    return total_reward, total_steps


if __name__ == "__main__":
    main()
