import numpy as np

num_states = 7
# {"0": "C1", "1":"C2", "2":"C3", "3":"Pass", "4":"Pub", "5":"FB", "6":"Sleep"}
i_to_n = {'0': 'C1', '1': 'C2', '2': 'C3', '3': 'Pass', '4': 'Pub', '5': 'FB', '6': 'Sleep'}  # 索引到状态名的字典

n_to_i = {}  # 状态名到索引的字典
for i, name in zip(i_to_n.keys(), i_to_n.values()):
    n_to_i[name] = int(i)

# C1 C2 C3 Pass Pub FB Sleep
Pss = [  # 状态转移概率矩阵
    [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
    [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.2],
    [0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0],
    [0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
]

# 定义以C1为起始状态的马尔科夫链，并使用上面的方法来验证最后一条马尔科夫链起始状态的收获值
chains = [
    ["C1", "C2", "C3", "Pass", "Sleep"],
    ["C1", "FB", "FB", "C1", "C2", "Sleep"],
    ["C1", "C2", "C3", "Pub", "C2", "C3", "Pass", "Sleep"],
    ["C1", "FB", "FB", "C1", "C2", "C3", "Pub", "C1", "FB",
     "FB", "FB", "C1", "C2", "C3", "Pub", "C2", "Sleep"]
]

Pss = np.array(Pss)
# 奖励函数，分别于状态对应
rewards = [-2, -2, -2, 10, 1, -1, 0]
gamma = 0.5


# 计算收获值 G
def compute_return(start_index=0, chain=None, gamma=0.5) -> float:
    """计算一个马尔科夫奖励过程中某状态的收获值
    Args:
        start_index 要计算的状态在链中的位置
        chain 要计算的马尔科夫过程
        gamma 衰减系数
    Returns:
        retrn: 收获值
    """
    retrn, power, gamma = 0.0, 0, gamma
    for i in range(start_index, len(chain)):
        retrn += np.power(gamma, power) * rewards[n_to_i[chain[i]]]
        power += 1

    return retrn


# 使用矩阵运算求解状态的价值
def compute_value(Pss, rewards, gamma=0.05):
    """通过求解矩阵方程的形式直接计算状态的价值
    Args:
        Pss 状态转移概率矩阵 shape(7,7)
        rewards 即时奖励 list
        gamma 衰减系数
    return
        values 各个状态的价值
    """
    assert (0 <= gamma <= 1.0)  # 排除gamma异常情况
    # 将rewards转为numpy数组并修改为列向量的形式
    rewards = np.array(rewards).reshape((-1, 1))
    # np.eye(7,7) 为单位矩阵，inv方法为求矩阵的逆
    values = np.dot(np.linalg.inv(np.eye(7, 7) - gamma * Pss), rewards)

    return values


if __name__ == '__main__':
    # 计算状态的收获值
    print(compute_return(0, chains[3], gamma=0.5))
    # 计算各个状态的价值
    values = compute_value(Pss, rewards, gamma=0.99999)
    print(values)