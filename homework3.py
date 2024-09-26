import numpy as np
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Params:
    def __init__(self):
        # Số lượng xe tối đa cho mỗi địa điểm
        self.max_car = 20

        # Số lượng xe tối đa được chuyển mỗi đêm
        self.max_move = 5

        # Phần thưởng được cung cấp khi cho thuê một chiếc xe
        self.reward_per_car = 10

        # chi phí phát sinh nếu có hơn max_car / 2 chiếc xe
        self.cost_per_slot_night = 4

        # Chi phí di chuyển một chiếc xe
        self.cost_per_car = 2

        # Small number determining the accuracy of policy evaluation's estimation
        # theta
        self.theta = 0.01

        # Discount value
        self.gamma = 0.9

        # Kỳ vọng cho yêu cầu thuê ở vị trí đầu tiên
        self.lambda_request_first = 3

        # Kỳ vọng cho yêu cầu thuê ở vị trí thứ hai
        self.lambda_request_second = 4

        # Kỳ vọng cho yêu cầu trả ở vị trí đầu tiên
        self.lambda_return_first = 3

        # Kỳ vọng cho yêu cầu trả ở vị trí thứ hai
        self.lambda_return_second = 2

        # Loại phiên bản: phiên bản gốc và phiên bản sửa đổi
        self.problem_types = ['original_problem', 'modified_problem']


# Lớp thuật toán lặp chính sách
class PolicyIteration:
    def __init__(self, params, problem_type):
        # Thiết lập các tham số
        self.params = params

        # Thiết lập loại vấn đề
        self.problem_type = problem_type

        # Tất cả các trạng thái có thể
        # Các trạng thái là 1 cặp số chỉ số lượng xe ở vị trí 1 và 2
        self.S = [(x, y) for x in range(self.params.max_car + 1) for y in range(self.params.max_car + 1)]

        # giá trị
        self.V = np.zeros((self.params.max_car + 1, self.params.max_car + 1))

        # Chính sách
        self.pi = np.zeros((self.params.max_car + 1, self.params.max_car + 1))

        # Danh sách các chính sách
        self.pis = []

    def solve_problem(self):
        """
        Resolve Jack's Car Rental problem using Policy Iteration

        """
        i = 0
        while True:
            print('Iteration', i + 1)

            # Policy evaluation
            self.pis.append(self.pi.copy())
            while True:
                delta = 0
                for s in self.S:
                    v = self.V[s]
                    self.V[s] = self.V_eval(s, self.pi[s])
                    delta = np.maximum(delta, abs(v - self.V[s]))
                if delta < self.params.theta:
                    break
                print('Delta', delta)

            # Policy improvement
            policy_stable = True
            for s in self.S:
                old_action = self.pi[s]
                values = {a: self.V_eval(s, a) for a in self.A(s)}
                self.pi[s] = np.random.choice([a for a, value in values.items()
                                               if value == np.max(list(values.values()))])
                if old_action != self.pi[s]:
                    policy_stable = False

            if policy_stable:
                break

            i += 1

    def A(self, s):
        """
        Lấy tất cả các hành động có thể với một trạng thái
        Thỏa mãn các điều kiện sau:

        a: số lượng xe được chuyển từ vị trí 1 đến vị trí 2

        -max_move <= a <= max_move
        0 <= s1 - a <= max_car
        0 <= s2 + a <= max_car
        """

        return list(range(max(-self.params.max_move, s[0] - self.params.max_car, -s[1]), 
                          min(self.params.max_move, s[0], self.params.max_car - s[1]) + 1))

    def V_eval(self, s, a):
        """
        Compute value given a state and an action for the state following the formula:
        sum over all s',r of p(s',r|s, a)[r + gamma*V(s')]
        :param s: state
        :param a: action
        :return: value
        """

        value = 0
        s_first, s_second = s

        # Chuyển xe
        s_first -= int(a)
        s_second += int(a)

        # Tổng chi phí chuyển xe
        if self.problem_type == 'original_problem':
            cost = self.params.cost_per_car * abs(a)
        else:
            if a > 0:
                a -= 1
            cost = self.params.cost_per_car * abs(a)
            if s_first > self.params.max_car / 2:
                cost += self.params.cost_per_slot_night
            if s_second > self.params.max_car / 2:
                cost += self.params.cost_per_slot_night

        # Tính toán cho mỗi trạng thái mới có thể: xác suất, phần thưởng và giá trị của trạng thái mới, sau đó áp dụng công thức
        sum_prob_i = 0
        for i in range(s_first + 1):
            if i == s_first:
                p_i = 1 - sum_prob_i
            else:
                p_i = PolicyIteration.poisson(self.params.lambda_request_first, i)
                sum_prob_i += p_i
            r_i = i * self.params.reward_per_car
            sum_prob_j = 0
            for j in range(s_second + 1):
                if j == s_second:
                    p_j = 1 - sum_prob_j
                else:
                    p_j = PolicyIteration.poisson(self.params.lambda_request_second, j)
                    sum_prob_j += p_j
                r_j = j * self.params.reward_per_car
                sum_prob_k = 0
                for k in range(self.params.max_car + i - s_first + 1):
                    if k == self.params.max_car + i - s_first:
                        p_k = 1 - sum_prob_k
                    else:
                        p_k = PolicyIteration.poisson(self.params.lambda_return_first, k)
                        sum_prob_k += p_k
                    sum_prob_l = 0
                    for l in range(self.params.max_car + j - s_second + 1):
                        if l == self.params.max_car + j - s_second:
                            p_l = 1 - sum_prob_l
                        else:
                            p_l = PolicyIteration.poisson(self.params.lambda_return_second, l)
                            sum_prob_l += p_l

                        value += p_i * p_j * p_k * p_l * (
                                 r_i + r_j - cost + self.params.gamma * self.V[s_first - i + k, s_second - j + l])
        return value

    def print_pis(self):
        """
        Print policies
        """
        for idx, pi in enumerate(self.pis):
            plt.figure()
            plt.imshow(pi, origin='lower', interpolation='none', vmin=-self.params.max_move, vmax=self.params.max_move)
            plt.xlabel('#Cars at second location')
            plt.ylabel('#Cars at first location')
            plt.title('pi{:d} {:s}'.format(idx, self.problem_type))
            plt.colorbar()

    def print_V(self):
        """
        Print values
        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = np.arange(0, self.params.max_car + 1)
        Y = np.arange(0, self.params.max_car + 1)
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, self.V)
        plt.title('V {:s}'.format(self.problem_type))

    @staticmethod
    def poisson(lamb, n):
        """
        Tính xác suất của phân phối Poisson
        """
        return (lamb ** n) * math.exp(-lamb) / math.factorial(n)


def exercise4_7():
    print('Exercise 4.7')

    # Set up parameters
    params = Params()

    for problem_type in params.problem_types:
        print('Problem type:', problem_type)

        # Set up the algorithm
        policy_iteration = PolicyIteration(params, problem_type)

        # Solve the problem
        policy_iteration.solve_problem()

        # Hiển thị kết quả
        policy_iteration.print_pis()
        policy_iteration.print_V()

exercise4_7()
plt.show()