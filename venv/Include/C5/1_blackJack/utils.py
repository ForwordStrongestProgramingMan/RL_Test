import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random  # 随即策略时用到


def str_key(*args):
    new_arg = []
    for arg in args:
        if type(arg) in [tuple, list]:
            new_arg += [str(i) for i in arg]
        else:
            new_arg.append(str(arg))
    return "_".join(new_arg)


def set_dict(target_dict, value, *args):
    target_dict[str_key(*args)] = value


def get_dict(target_dict, *args):
    return target_dict.get(str_key(*args), 0)


def greedy_pi(A, s, Q, a):
    """依据贪婪策略，计算在行为空间A中， 状态s下， a行为被贪婪选中得几率
    考虑多个行为得价值相同的情况"""
    max_q, a_max_q = -float('inf'), []
    for a_opt in A:
        q = get_dict(Q, s, a_opt)
        if q > max_q:
            max_q = q
            a_max_q = [a_opt]
        elif q == max_q:
            a_max_q.append(a_opt)
    n = len(a_max_q)
    if n == 0: return 0.0
    return 1.0 / n if a in a_max_q else 0.0


def greedy_policy(A, s, Q):
    """在给定一个状态下，从行为空间中选择一个行为a,使得Q(s,a) = max(Q(s,))
    考虑多个行为价值相同的情况"""
    max_q, a_max_q = -float('inf'), []
    for a_opt in A:
        q = get_dict(Q, s, a_opt)
        if q > max_q:
            max_q = q
            a_max_q = [a_opt]
        elif q == max_q:
            a_max_q.append(a_opt)
    return random.choice(a_max_q)


def epsilon_greedy_pi(A, s, Q, a, epsilon=0.1):
    m = len(A)
    greedy_p = greedy_pi(A, s, Q, a)
    if greedy_p == 0:
        return epsilon / m
    return (1 - epsilon) * greedy_p + epsilon / m


def epsilon_greedy_policy(A, s, Q, epsilon, show_random_num=False):
    pis = []
    m = len(A)
    for i in range(2):
        pis.append(epsilon_greedy_pi(A, s, Q, A[i], epsilon))
    rand_value = random.random()  # 产生一个0,1的随机数
    # if show_random_num:
    #    print("产生的随机数概率为:{:.2f}".format(rand_value))
    # print(rand_value)
    for i in range(2):
        if show_random_num:
            print("随机数:{:.2f}, 拟减去概率{}".format(rand_value, pis[i]))
        rand_value -= pis[i]
        if rand_value < 0:
            return A[i]


#
# x = np.arange(1, 11, 1)
# y = np.arange(12, 22, 1)
# X, Y = np.meshgrid(x, y)
# print(X, Y)
# print(X[1, 2])


def draw_value(value_dict, useable_ace=True, is_q_dict=False, A=None):
    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.arange(1, 11, 1)
    y = np.arange(12, 22, 1)
    X, Y = np.meshgrid(x, y)
    row, col = X.shape
    Z = np.zeros((row, col))
    if is_q_dict:
        n = len(A)
    for i in range(row):
        for j in range(col):
            state_name = str(X[i, j]) + "_" + str(Y[i, j]) + "_" + str(useable_ace)
            if not is_q_dict:
                Z[i, j] = get_dict(value_dict, state_name)
            else:
                assert (A is not None)
                for a in A:
                    new_state_name = state_name + "_" + str(a)
                    q = get_dict(value_dict, new_state_name)
                    if q >= Z[i, j]:
                        Z[i, j] = q
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color="lightgray")
    plt.show()


def draw_policy(policy, A, Q, epsilon, useable_ace=False):
    def value_of(a):
        if a == A[0]:
            return 0
        else:
            return 1

    rows, cols = 11, 10
    useable_ace = bool(useable_ace)
    Z = np.zeros((rows, cols))
    for i in range(11, 22):  # 玩家总牌点
        for j in range(1, 11):  # 庄家第一张牌
            s = j, i, useable_ace
            s = str_key(s)
            a = policy(A, s, Q, epsilon)
            Z[i - 11, j - 1] = value_of(a)
    plt.imshow(Z, cmap=plt.cm.cool, interpolation=None, origin="lower", extent=[0.5, 11.5, 10.5, 21.5])  # extent = [horizontal_min，horizontal_max，vertical_min，vertical_max]
