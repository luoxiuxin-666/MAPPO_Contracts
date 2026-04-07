#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/22 0022 16:38
# @Author  : ZhengHao
# @File    : picture_other.py
from ortools.algorithms.python import knapsack_solver
import numpy as np


def dp(candidate_contracts, total_R):
    # 2. 求解0/1背包问题
    n = len(candidate_contracts)
    capacity = int(total_R)
    # print(f" capacity is {capacity}")

    # dp[i][j] 表示考虑前i个物品，容量为j时的最大价值
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        item_idx = i - 1
        weight = candidate_contracts[item_idx][0]
        value = candidate_contracts[item_idx][1]
        for j in range(1, capacity + 1):
            if j < weight:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight] + value)

    # 3. 回溯找到哪些合同被选中
    j = capacity
    select_id = []
    for i in range(n, 0, -1):
        if dp[i][j] > dp[i - 1][j]:
            item_idx = i - 1
            select_id.append(item_idx)
            j -= candidate_contracts[item_idx][0]

    utility = 0
    for i, id in enumerate(select_id):
        utility += candidate_contracts[id][1]
    return select_id, utility

def radio(candidate_contracts, total_R):
    radios = []
    for i, contract in enumerate(candidate_contracts):
        radio = contract[1]/contract[0]
        radios.append([radio, contract[0], i])

    sorted_by_radio = sorted(radios, key=lambda x: x[0],reverse=True)
    print(sorted_by_radio)

    select_idx = []
    for i, radio in enumerate(sorted_by_radio):
        if total_R - radio[1] >= 0:
            select_idx.append(radio[2])
            total_R = total_R - radio[1]

    utility = 0
    for i, id in enumerate(select_idx):
        utility += candidate_contracts[id][1]
    return select_idx, utility

def orther_dp(candidate_contracts, total_R):

    # 背包的容量 (因为只有一个背包，所以只有一个元素的列表)
    capacities = [total_R]

    # 2. 创建求解器
    # 参数：求解器类型，求解器名称
    # 正确的代码行
    # 最新、最正确的代码行
    # 针对 v9.1.4.6206 的正确代码行
    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "KnapsackExample"
    )
    contract_R = [[]]
    contract_U = []
    for i, contract in enumerate(candidate_contracts):
        contract_R[0].append(contract[0])
        contract_U.append(contract[1])

    # 3. 初始化求解器
    # 参数：价值列表, 重量列表, 容量列表
    solver.init(contract_U, contract_R, capacities)

    # 4. 求解
    computed_value = solver.solve()

    # 5. 打印结果
    packed_items = []
    packed_weight = 0
    total_value = 0
    print(f"总价值 = {computed_value}")

    for i in range(len(contract_U)):
        if solver.best_solution_contains(i):
            packed_items.append(i)
            packed_weight += contract_R[0][i]
            total_value += contract_U[i]

    print(f"总重量 = {packed_weight}")
    print(f"被选择的物品索引: {packed_items}")
    # 验证一下总价值
    print(f"被选择物品的价值之和: {total_value}")

    return packed_items, total_value

if __name__ == "__main__":
    candidate_contracts = [[10, 60], [20, 100], [30, 120]]
    total_R = 50

    select_1, utility_1 = dp(candidate_contracts, total_R)
    select_2, utility_2 = radio(candidate_contracts,total_R)
    select_3, utility_3 = orther_dp(candidate_contracts,total_R)

    print(f"the select_by_dp is {select_1} and the utility is {utility_1}")
    print(f"the select_by_radio is {select_2} and the utility is {utility_2}")
    print(f"the select_by_orther_dp is {select_3} and the utility is {utility_3}")