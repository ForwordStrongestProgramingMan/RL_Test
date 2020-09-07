from random import shuffle
from queue import Queue
from tqdm import tqdm

from utils import str_key


class Gamer():
    """游戏者"""

    def __init__(self, name="", A=None, display=False):
        self.name = name
        self.cards = []  # 手中的牌
        self.display = display  # 是否显示对局文字信息
        self.policy = None  # 策略
        self.learning_method = None  # 学习方法
        self.A = A  # 行为空间

    def __str__(self):
        return self.name

    def value_of(self, card):
        """根据牌的字符判断牌的数值大小，A被输出为1， JQK均为10， 其余牌字符按照对应数值取值"""
        try:
            v = int(card)
        except:
            if card == 'A':
                v = 1
            elif card in ['J', 'Q', 'K']:
                v = 10
            else:
                v = 0
        finally:
            return v

    def get_points(self):
        """
        统计一手牌分值，如果使用了A的1点，同时返回True
        :return: (返回牌总点数，是否使用了可复用Ace)
        ['A','10'] 返回 （21, True)
        """
        num_of_useable_ace = 0  # 默认没有拿到Ace
        total_point = 0
        cards = self.cards
        if cards is None:
            return 0, False
        for card in cards:
            v = self.value_of(card)
            if v == 1:
                num_of_useable_ace += 1
                v = 11
            total_point += v
        while total_point > 21 and num_of_useable_ace > 0:
            total_point -= 10
            num_of_useable_ace -= 1
        return total_point, bool(num_of_useable_ace)

    def receive(self, cards=[]):
        cards = list(cards)
        for card in cards:
            self.cards.append(card)

    def discharge_cards(self):  # 玩家把手中的牌清空，扔牌
        """扔牌"""
        self.cards.clear()

    def cards_info(self):  # 玩家手中牌的信息
        """显示牌面具体信息"""
        self._info("{}{}现在的牌:{}\n".format(self.role, self, self.cards))

    def _info(self, msg):
        if self.display:
            print(msg, end="")


class Dealer(Gamer):
    def __init__(self, name="", A=None, display=False):
        super(Dealer, self).__init__(name, A, display)
        self.role = "庄家"
        self.policy = self.dealer_policy

    def first_card_value(self):
        if self.cards is None or len(self.cards) == 0:
            return 0
        return self.value_of(self.cards[0])

    def dealer_policy(self, Dealer=None):
        dealer_points, _ = self.get_points()
        if dealer_points > 17:
            action = self.A[1]  # 停止要牌
        else:
            action = self.A[0]
        return action


class Player(Gamer):
    def __init__(self, name="", A=None, display=False):
        super(Player, self).__init__(name, A, display)
        self.policy = self.naive_policy
        self.role = "玩家"  # "庄家"还是"玩家"

    def get_state(self, dealer):
        dealer_first_card_value = dealer.first_card_value()
        player_points, useable_ace = self.get_points()
        return dealer_first_card_value, player_points, useable_ace

    def get_state_name(self, dealer):
        return str_key(self.get_state(dealer))

    def naive_policy(self, dealer=None):
        player_points, _ = self.get_points()
        if player_points < 20:
            action = self.A[0]
        else:
            action = self.A[1]
        return action


class Arena():
    """负责游戏管理"""
    def __init__(self, display=None, A=None):
        self.cards = ['A','2','3','4','5','6','7','8','9','10','J','Q',"K"]*4
        self.card_q = Queue(maxsize=52)  # 洗好的牌放入其中
        self.cards_in_pool = []  # 已经用过的公开的牌
        self.display = display
        self.episodes = []  # 产生的对局信息列表
        self.load_cards(self.cards)  # 把初始状态的52张牌装入发牌器
        self.A = A  # 获得行为空间

    def load_cards(self, cards):
        """
        把收集的牌洗一洗，重新装入发牌器中
        :param cards: 要装入发牌器的多张牌 list
        :return: None
        """
        shuffle(cards)
        for card in cards:
            self.card_q.put(card)
        cards.clear()

    def reward_of(self, dealer, player):
        """判断玩家奖励值，附带庄家，玩家的牌点信息"""
        dealer_points, _ = dealer.get_points()
        player_points, useable_ace = player.get_points()
        if player_points > 21:
            reward = -1
        else:
            if player_points > dealer_points or dealer_points > 21:
                reward = 1
            elif player_points == dealer_points:
                reward = 0
            else:
                reward = -1
        return reward, dealer_points, player_points, useable_ace

    def serve_card_to(self, player, n=1):
        """
        给庄家和玩家发牌，如果牌不够则将公开牌池的牌洗一洗重新发牌
        :param player: 一个庄家或玩家
        :param n: 一次连续发牌的数量
        :return: None
        """
        cards = []  # 将要发出的牌
        for _ in range(n):
            if self.card_q.empty():
                self._info("\n发牌器没有牌了，整理废牌，重新洗牌;")
                shuffle(self.cards_in_pool)
                self._info("一共整理了{}张已用的牌，重新放入发牌器\n".format(len(self.cards_in_pool)))
                assert (len(self.cards_in_pool) > 20)  # 确保有足够的牌
                self.load_cards(self.cards_in_pool)
            cards.append(self.card_q.get())  # 从发牌器发出一张牌
        self._info("发了{}张牌（{}）给{}{}".format(n, cards, player.role, player))
        player.receive(cards)
        player.cards_info()

    def _info(self, message):
        if self.display:
            print(message, end="")

    def recycle_cards(self, *players):
        """回收玩家手中的牌到公开使用过的牌池中"""
        if len(players) == 0:
            return
        for player in players:
            for card in player.cards:
                self.cards_in_pool.append(card)
            player.discharge_cards()  # 玩家手中的牌清空

    def play_game(self, dealer, player):
        """
        玩一局21点，生成一个状态序列以及最终奖励 (中间奖励为0)
        :param dealer: 庄家手中的牌 list
        :param player: 玩家手中的牌 list
        :return: episode reward
        """

        self._info("=================开始新一局===============\n")
        self.serve_card_to(dealer, n=2)
        self.serve_card_to(player, n=2)
        episode = []  # 记录一个对局信息
        if player.policy is None:
            self._info("玩家需要一个策略")
            return
        if dealer.policy is None:
            self._info("庄家需要一个策略")
            return

        while True:
            action = player.policy(dealer)  # epsilon的贪婪策略
            # 玩家的策略产生一个行为
            self._info("{}{}选择:{};".format(player.role, player, action))
            episode.append((player.get_state_name(dealer), action))  # 记录一个(s, a)
            if action == self.A[0]:
                self.serve_card_to(player)
            else:
                break

        reward, player_points, dealer_points, useable_ace = self.reward_of(dealer, player)

        if player_points > 21:
            self._info("玩家爆点{}输了，得分:{}\n".format(player_points, reward))
            self.reward_of(dealer, player)
            self.episodes.append((episode, reward))
            self._info("=============本局结束===============")
            return episode, reward

        # 玩家没有超过21点
        self._info("\n")
        while True:
            action = dealer.policy()
            self._info("{}{}选择:{};".format(dealer.role, dealer, action))
            if action == self.A[0]:
                self.serve_card_to(dealer)
            else:
                break

        self._info("\n双方均停止了叫牌;\n")
        reward, player_points, dealer_points, useable_ace = self.reward_of(dealer, player)
        player.cards_info()
        dealer.cards_info()
        if reward == +1:
            self._info("玩家赢了!")
        elif reward == -1:
            self._info("玩家输了!")
        else:
            self._info("双发和局!")
        self._info("玩家{}点，庄家{}点\n".format(player_points, dealer_points))

        self._info("====================本局结束===============\n")
        self.recycle_cards(player, dealer)
        self.episodes.append((episode, reward))
        return episode, reward

    def play_games(self, dealer, player, num=2, show_statistic=True):
        """一次性玩多局游戏"""
        results = [0, 0, 0]  # 玩家负 和 胜局数
        self.episodes.clear()
        for i in tqdm(range(num)):
            episode, reward = self.play_game(dealer, player)
            results[1+reward] += 1
            if player.learning_method is not None:
                player.learning_method(episode, reward)
        if show_statistic:
            print("共玩了{}局，玩家赢{}局，和{}局，输{}局，胜率：{:.2f},不输率:{:.2f}" \
                  .format(num, results[2], results[1], results[0], results[2] / num, (results[2] + results[1]) / num))
        pass


