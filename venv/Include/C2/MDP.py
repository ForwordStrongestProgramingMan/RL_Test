# 导入工具函数：根据状态和行为生成操作相关字典的建，显示字典内容
from utils import str_key, display_dict
# 设置转移概率、奖励值以及读取它们的方法
from utils import set_prob, set_reward, get_prob, get_reward
# 设置状态价值、策略概率以及读取它们的方法
from utils import set_value, set_pi, get_value, get_pi
# 构建学生马尔科夫决策过程
S = ['浏览手机中', '第一节课', '第二节课', '第三节课', '休息中']
A = ['浏览手机', '学习', '离开浏览', '泡吧', '退出学习']
R = {}  # 奖励Rsa字典
P = {}  # 状态转移概率Pss'a字典
gamma = 1.0  # 衰减因子
# 根据学生马尔科夫决策过程示例的数据设置状态转移概率和奖励，默认概率为1
set_prob(P, S[0], A[0], S[0])  # 浏览手机中 -> 浏览手机 -> 浏览手机中
set_prob(P, S[0], A[2], S[1])  # 浏览手机中 -> 学习 -> 第一节课
set_prob(P, S[1], A[0], S[0])  # 第一节课 -> 浏览手机 -> 浏览手机中
set_prob(P, S[1], A[1], S[2])  # 第一节课 -> 学习 -> 第二节课
set_prob(P, S[2], A[1], S[3])  # 第二节课 -> 学习 -> 第三节课
set_prob(P, S[2], A[4], S[4])  # 第二节课 -> 退出学习 -> 退出
set_prob(P, S[3], A[1], S[4])  # 第三节课 -> 学习 -> 退出
set_prob(P, S[3], A[3], S[1], p=0.2)  # 第三节课 -> 泡吧 -> 第一节课
set_prob(P, S[3], A[3], S[2], p=0.4)  # 第三节课 -> 泡吧 -> 第二节课
set_prob(P, S[3], A[3], S[3], p=0.4)  # 第三节课 -> 泡吧 -> 第三节课

set_reward(R, S[0], A[0], -1)  # 浏览手机中 -> 浏览手机 : -1
set_reward(R, S[0], A[2], 0)  # 浏览手机中 -> 离开浏览 : 0
set_reward(R, S[1], A[0], -1)  # 第一节课 -> 浏览手机 : -1
set_reward(R, S[1], A[1], -2)  # 第一节课 -> 学习 : -2
set_reward(R, S[2], A[1], -2)  # 第二节课 -> 学习 : -2
set_reward(R, S[2], A[4], 0)  # 第二节课 -> 退出学习 : 0
set_reward(R, S[3], A[1], 10)  # 第三节课 -> 学习 : 10
set_reward(R, S[3], A[3], +1)  # 第三节课 -> 泡吧 : -1

MDP = (S, A, R, P, gamma)

# S = ['浏览手机中','第一节课','第二节课','第三节课','休息中']
# A = ['继续浏览','学习','离开浏览','泡吧','退出学习']
# 设置行为策略：pi(a|.) = 0.5
Pi = {}
set_pi(Pi, S[0], A[0], 0.5)  # 浏览手机中 - 浏览手机
set_pi(Pi, S[0], A[2], 0.5)  # 浏览手机中 - 离开浏览
set_pi(Pi, S[1], A[0], 0.5)  # 第一节课 - 浏览手机
set_pi(Pi, S[1], A[1], 0.5)  # 第一节课 - 学习
set_pi(Pi, S[2], A[1], 0.5)  # 第二节课 - 学习
set_pi(Pi, S[2], A[4], 0.5)  # 第二节课 - 退出学习
set_pi(Pi, S[3], A[1], 0.5)  # 第三节课 - 学习
set_pi(Pi, S[3], A[3], 0.5)  # 第三节课 - 泡吧

print("----策略概率:----")
display_dict(Pi)
# 初始时价值为空，访问时会返回0
print("----状态价值:----")
V = {}
display_dict(V)

if __name__ == '__main__':
    print("----状态行为概率转移矩阵-----")
    display_dict(P)
    print("-----奖励字典 (函数) 信息:-----")
    display_dict(R)


def compute_q(mdp, V, s, a):
    """根据给定的MDP，价值函数V，计算状态行为对s,a的价值qsa"""

    S, A, R, P, gamma = mdp
    q_sa = 0
    for s_prim in S:
        q_sa += get_prob(P, s, a, s_prim) * get_value(V, s_prim)
    q_sa = get_reward(R, s, a) + gamma * q_sa
    return q_sa


def compute_v(mdp, V, PI, s):
    """给定MDP下依据某一策略pi和当前状态价值函数V计算某状态s的价值"""

    S, A, R, P, gamma = mdp
    v_s = 0
    for a in A:
        v_s += get_pi(PI, s, a) * compute_q(mdp, V, s, a)
    return v_s


# 根据当前策略使用回朔法来更新状态价值
def update_V(MDP, V, Pi):
    """给定一个MDP和一个策略，更新该策略下的价值函数
    """
    S, _, _, _, _ = MDP
    V_prime = V.copy()
    for s in S:
        V_prime[str_key(s)] = compute_v(MDP, V_prime, Pi, s)
    return V_prime


# 策略评估，得到该策略下的最终状态价值
def policy_evaluate(MDP, V, Pi, n):
    """使用n次迭代计算来评估一个MDP在给定策略Pi下的状态价值，初始价值为V
    """
    print("################################")
    display_dict(V)
    for i in range(n):
        V = update_V(MDP, V, Pi)
        display_dict(V)
    return V


V = policy_evaluate(MDP, V, Pi, 100)
display_dict(V)
# 验证状态在某策略下的价值
v = compute_v(MDP, V, Pi, "第三节课")
print("第三节课在当前策略下的价值为:{:.2f}".format(v))


# 价值迭代得到最优状态价值过程
def compute_v_from_max_q(MDP, V, s):
    """根据一个状态下的所有可能的行为价值中最大的一个来确定当前状态价值
    """
    S, A, R, P, gamma = MDP
    v_s = -float('inf')
    for a in A:
        qsa = compute_q(MDP, V, s, a)
        if qsa > v_s:
            v_s = qsa
    return v_s


def update_V_without_pi(MDP, V):
    """在不依赖策略的情况下直接透过后续状态的价值来更新状态价值
    """
    S, _, _, _, _ = MDP
    V_prime = V.copy()
    for s in S:
        set_value(V_prime, s, compute_v_from_max_q(MDP, V_prime, s))
    return V_prime


# 价值迭代
def value_iterate(MDP, V, n):
    """价值迭代
    """
    for i in range(n):
        V = update_V_without_pi(MDP, V)
        display_dict(V)
    return V


V = {}
# 通过价值迭代得到最优状态价值
V_star = value_iterate(MDP, V, 4)
display_dict(V_star)

# 验证最优行为价值
s, a = "第三节课", "泡吧"
q = compute_q(MDP, V_star, "第三节课", "泡吧")
print("在状态{}选择行为{}的最优价值为:{:.2f}".format(s, a, q))


def display_q_star(MDP, V_star):
    S, A, _, _, _ = MDP
    for s in S:
        for a in A:
            print("q*({},{},{})".format(s, a, compute_q(MDP, V_star, s, a)))


display_q_star(MDP, V_star)