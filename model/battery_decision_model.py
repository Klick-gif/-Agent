import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List
import numpy as np


def tem_eta(Tem):
    """温度效率函数"""
    return (-0.012 * Tem ** 2 + 0.8036 * Tem + 67.145) * 0.01


class BatteryOptimizationSolver:
    def __init__(self, config: Dict):
        """初始化液流电池储能电站优化求解器"""
        self.config = config
        self.model = None
        self.results = {}

        # 时间参数
        self.T = config['time_horizon']
        self.dt = config['time_step']
        self.T_n = int(self.T / self.dt)
        self.dt_minutes = self.dt * 60

        # 温度
        self.Tem_init: float = config['tem_init']

        # 电池参数
        self.C_nom = config['battery']['capacity']  # kWh
        self.P_max = config['battery']['max_power']  # kW
        self.eta_init = tem_eta(self.Tem_init)
        self.SOC_min = config['battery']['soc_min']
        self.SOC_max = config['battery']['soc_max']
        self.SOC_init: float = config['battery']['soc_init']
        self.H_init = config['battery']['soh_init']

        # DoD分段参数
        self.N_segments = config['degradation']['n_segments']
        self.DoD_max = 1 - self.SOC_min

        # 退化模型参数
        self.C_gh = 2.9 * self.C_nom * 1000  # 2.9Wh * cap元
        self.k_cal = config['degradation']['calendar_aging']

        # 碳排放参数
        self.alpha_grid = config['carbon']['grid_emission_factor']
        self.alpha_avoid = config['carbon']['avoided_emission_factor']
        self.C_carbon_max = config['carbon']['max_emission']
        self.C_init = config['carbon']['carbon_init']

        # 权重系数
        self.weights = config['weights']

        # 价格数据
        self.price_data = {
            'electricity': config['price_data']['electricity'][:self.T_n],
            'carbon': config['price_data']['carbon'][:self.T_n]
        }

        # 归一化因子
        self._calculate_normalization_factors()

    def _calculate_normalization_factors(self):
        """计算归一化因子"""
        self.P_norm = self.P_max
        self.E_norm = self.C_nom
        self.T_norm = 1.0
        self.price_norm = max(self.price_data['electricity'])
        self.cost_norm = self.C_gh
        self.carbon_norm = self.C_carbon_max

    def calculate_degradation_costs(self) -> List[float]:
        """计算DoD分段线性化的退化成本"""
        A, B, C, D = 0.1, 9.35, 0.1, 1.32
        costs = []
        for n in range(1, self.N_segments + 1):
            dod_n = n / self.N_segments
            dod_n_1 = (n - 1) / self.N_segments
            B_d_n = A * (dod_n ** B) + C * (dod_n ** D)
            B_d_n_1 = A * (dod_n_1 ** B) + C * (dod_n_1 ** D)
            c_n = (self.C_gh / self.C_nom) * (1 / self.N_segments) * (B_d_n - B_d_n_1)
            c_n_norm = c_n / 1e9
            costs.append(c_n_norm)
        return costs

    def build_model(self):
        """构建优化模型"""
        self.model = gp.Model("BatteryOptimization")
        self.model.setParam('OutputFlag', 0)

        # 决策变量
        self.P_charge = self.model.addVars(self.T_n, lb=0, ub=1.0, name="P_charge")
        self.P_discharge = self.model.addVars(self.T_n, lb=0, ub=1.0, name="P_discharge")
        self.t_op = self.model.addVars(self.T_n, lb=0, ub=0.5, name="t_op")

        # 状态变量
        self.SOC = self.model.addVars(self.T_n + 1, lb=self.SOC_min, ub=self.SOC_max, name="SOC")
        self.H = self.model.addVars(self.T_n + 1, lb=0.7, ub=1.0, name="H")
        self.C_carbon = self.model.addVars(self.T_n + 1, name="C_carbon")
        self.delta = self.model.addVars(self.T_n, self.N_segments, vtype=GRB.BINARY, name="delta")

        # 辅助控制变量
        self.mo = self.model.addVars(self.T_n, vtype=GRB.BINARY, name="mode")  # 0-1操作状态

        # 初始状态
        self.model.addConstr(self.SOC[0] == self.SOC_init)
        self.model.addConstr(self.H[0] == self.H_init)
        self.model.addConstr(self.C_carbon[0] == self.C_init)

        self._add_simplified_constraints()
        self._set_simplified_objective()

    def _add_simplified_constraints(self):
        """添加约束条件"""
        degradation_costs = self.calculate_degradation_costs()

        for t in range(self.T_n):
            # 充放电互斥 && 电力市场和辅助服务分离
            self.model.addConstr(self.P_charge[t] * self.P_discharge[t] == 0)

            # SOC动态
            net_energy = (self.eta_init * self.P_charge[t] - self.P_discharge[t] / self.eta_init) * self.P_norm * \
                         self.t_op[
                             t] / (self.C_nom * self.H[t])
            self.model.addConstr(self.SOC[t + 1] == self.SOC[t] + net_energy)

            # DoD分段
            self.model.addConstr(gp.quicksum(self.delta[t, n] for n in range(self.N_segments)) == 1)
            for n in range(self.N_segments):
                dod_lower = n / self.N_segments * self.DoD_max
                dod_upper = (n + 1) / self.N_segments * self.DoD_max
                self.model.addConstr(1 - self.SOC[t] >= dod_lower * self.delta[t, n])
                self.model.addConstr(1 - self.SOC[t] <= dod_upper + (1 - self.delta[t, n]) * self.DoD_max)

            # 健康状态
            cycle_degradation = gp.quicksum(degradation_costs[n] * self.delta[t, n] for n in range(self.N_segments)) * (
                    self.P_charge[t] + self.P_discharge[t])
            calendar_degradation = self.k_cal
            self.model.addConstr(
                self.H[t + 1] == self.H[t] - cycle_degradation * self.t_op[t] - calendar_degradation * self.dt)

            # 碳排放
            carbon_change = (self.alpha_grid * self.P_charge[t] - self.alpha_avoid * self.P_discharge[
                t]) * self.P_norm * self.t_op[t]
            self.model.addConstr(self.C_carbon[t + 1] == self.C_carbon[t] + carbon_change / self.carbon_norm)

            # 操作时间约束
            self.model.addConstr(self.P_charge[t] + self.P_discharge[t] <= self.P_max * self.mo[t])  # 非运行状态功率为 0
            self.model.addConstr(self.t_op[t] <= self.mo[t])  # t_op 只有在运行状态下才可能非零
            self.model.addConstr(self.t_op[t] >= 0.1 * self.mo[t])  # 下限约束

    def _set_simplified_objective(self):
        """设置完整目标函数"""
        # 归一化价格数据
        price_elec_norm = np.array(self.price_data['electricity']) / self.price_norm
        price_carbon_norm = [p / self.price_norm for p in self.price_data['carbon']]

        # 经济收益
        economic_obj = gp.quicksum(
            price_elec_norm[t] * (self.P_discharge[t] - self.P_charge[t]) * self.P_norm * self.t_op[t] +
            price_carbon_norm[t] * self.alpha_avoid * self.P_discharge[t] * self.P_norm * self.t_op[t]
            for t in range(self.T_n)
        )

        degradation_costs = self.calculate_degradation_costs()
        degradation_cost = gp.quicksum(
            gp.quicksum(degradation_costs[n] * self.delta[t, n] for n in range(self.N_segments)) *
            (self.P_discharge[t] + self.P_charge[t]) / self.P_norm * self.cost_norm
            for t in range(self.T_n)
        ) / self.cost_norm

        operation_cost = gp.quicksum(
            self.P_charge[t] * self.P_norm + self.P_discharge[t] * self.P_norm * 0.008
            for t in range(self.T_n)
        ) / self.cost_norm

        J_econ = economic_obj - operation_cost - degradation_cost

        J_life = gp.quicksum(
            50 * (1 - self.SOC[t]) ** 2 + 100 * (1 - self.H[t])
            for t in range(self.T_n)
        )

        J_carbon = gp.quicksum(
            (self.alpha_grid * self.P_charge[t] - self.alpha_avoid * self.P_discharge[
                t]) * 0.5 * self.P_norm * 0.01
            for t in range(self.T_n)
        ) * 0.01

        w_sum = sum(self.weights.values())
        w_econ = self.weights['economic'] / w_sum
        w_life = self.weights['life'] / w_sum
        w_carbon = self.weights['carbon'] / w_sum

        total_obj = (-w_econ * J_econ + w_life * J_life + w_carbon * J_carbon)
        self.model.setObjective(total_obj, GRB.MINIMIZE)

    def solve(self, time_limit: int = 700):
        """求解优化问题"""
        if self.model is None:
            self.build_model()

        self.model.setParam('TimeLimit', time_limit)
        self.model.setParam('MIPGap', 0.02)
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('Threads', 0)

        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            self._extract_results()
            return True
        elif self.model.status == GRB.TIME_LIMIT:
            if self.model.SolCount > 0:
                self._extract_results()
                return True
        return False

    def _extract_results(self):
        """提取求解结果"""
        self.results = {
            'P_charge': [self.P_charge[t].x * self.P_norm for t in range(self.T_n)],
            'P_discharge': [self.P_discharge[t].x * self.P_norm for t in range(self.T_n)],
            'P': [self.P_charge[t].x * self.P_norm + self.P_discharge[t].x * self.P_norm for t in range(self.T_n)],
            't_op': [self.t_op[t].x for t in range(self.T_n)],
            'SOC': [self.SOC[t].x * 100 for t in range(self.T_n + 1)],
            'H': [self.H[t].x for t in range(self.T_n + 1)],
            'C_carbon': [self.C_carbon[t].x * self.carbon_norm for t in range(self.T_n + 1)],
            'mode': [],
        }

        # 推断模式
        for t in range(self.T_n):
            if self.results['P_charge'][t] > 0.1:
                self.results['mode'].append(1)
            elif self.results['P_discharge'][t] > 0.1:
                self.results['mode'].append(-1)
            else:
                self.results['t_op'][t] = 0
                self.results['mode'].append(0)

        # 计算收益
        revenue = sum(
            self.price_data['electricity'][t] * (self.results['P_discharge'][t] - self.results['P_charge'][t]) *
            self.results['t_op'][t]
            for t in range(self.T_n)
        )
        carbon_revenue = sum(
            self.price_data['carbon'][t] * self.alpha_avoid * self.results['P_discharge'][t] * self.results['t_op'][t]
            for t in range(self.T_n)
        )

        self.results['electricity_revenue'] = revenue  # 电力交易
        self.results['carbon_revenue'] = carbon_revenue  # 碳交易
        self.results['total_revenue'] = revenue + carbon_revenue  # 总收益
        self.results['result_str'] = self.statistical_results()  # 结果

    def statistical_results(self) -> str:
        """生成结果字符串"""
        if not self.results:
            return "没有求解结果"

        result_str = ""
        result_str += f"{'=' * 50}\n"
        result_str += f"       液流电池储能电站初始化信息\n"
        result_str += f"{'=' * 50}\n"

        # 初始化信息
        result_str += f"\n=== **初始化信息** ===\n"
        result_str += f'荷电状态SOC: {round(self.SOC_init * 100, 2)}%\n'
        result_str += f"电解液温度T： {round(self.Tem_init, 1)}°C\n"
        result_str += f"系统效率eta(充，放)： {round(self.eta_init * 100, 2)}%\n"
        result_str += f"健康因子H: {self.H_init}\n"
        result_str += f"放电深度DoD: {round((1 - self.SOC_init) * 100, 2)}%\n"
        result_str += f"累计碳排放量C_carbon: {self.C_init}kg\n"

        # 外部输入：
        result_str += f"\n=== **外部输入** ===\n"
        result_str += f"电价预测，对后面{self.T}小时的电价预测，时间间隔是30分钟，单位是元/kWh:\n"
        result_str += ', '.join([str(round(x, 2)) for x in self.price_data['electricity']]) + '\n'
        result_str += f"碳交易平均价格： {int(np.mean(self.price_data['carbon']) * 1000)}元/吨\n"

        result_str += f"\n{'=' * 50}\n"
        result_str += f"       液流电池储能电站优化结果\n"
        result_str += f"{'=' * 50}\n"

        # 系统状态
        result_str += f"\n=== **系统状态变化** ===\n"
        result_str += f"初始SOC: {self.results['SOC'][0]:.2f}%\n"
        result_str += f"最终SOC: {self.results['SOC'][-1]:.2f}%\n"
        result_str += f"最终健康状态: {self.results['H'][-1]:.6f}\n"

        # 环境影响
        result_str += f"\n=== **环境影响** ===\n"
        initial_carbon = self.results['C_carbon'][0]
        final_carbon = self.results['C_carbon'][-1]
        net_carbon_change = final_carbon - initial_carbon

        if net_carbon_change < 0:
            result_str += f"实现碳减排: {-net_carbon_change / 1000:.2f} 吨 CO2\n"
        elif net_carbon_change > 0:
            result_str += f"产生碳排放: {net_carbon_change / 1000:.2f} 吨 CO2\n"
        else:
            result_str += "碳排放中性\n"

        # 经济效益
        result_str += f"\n=== **经济效益分析** ===\n"
        result_str += f"电力交易收益: {self.results['electricity_revenue']:.2f} 元\n"
        result_str += f"碳交易收益: {self.results['carbon_revenue']:.2f} 元\n"

        # 成本估算
        operation_cost = 0.008 * sum(p * t for p, t in zip(self.results['P'], self.results['t_op']))
        degradation_cost = (self.results['H'][0] - self.results['H'][-1]) * self.C_gh
        net_profit = self.results['total_revenue'] - operation_cost - degradation_cost

        result_str += f"运行成本: {operation_cost:.2f} 元\n"
        result_str += f"退化成本: {degradation_cost:.2f} 元\n"
        result_str += f"净利润: {net_profit:.2f} 元\n"

        # 模式统计
        mode_count = {-1: 0, 0: 0, 1: 0}
        for mode in self.results['mode']:
            mode_count[mode] += 1

        # 调度计划
        result_str += f"\n=== **调度计划** ===\n"
        result_str += "时间\t\t功率(MW) \t\t模式\t\t操作时间(h)\t\tSOC(%)\t\tprice(元/kWh)\n"
        result_str += "-" * 70 + "\n"
        from datetime import datetime, timedelta
        time_now = datetime.now()
        dt_minutes = getattr(self, 'dt_minutes', int(30*self.dt))
        for t in range(self.T_n):
            mode_str = {-1: "放电", 0: "待机", 1: "充电"}.get(self.results['mode'][t], "未知")
            time_str = time_now.strftime("%H:%M")

            # 根据模式选择显示充电或放电功率
            if self.results['mode'][t] == 1:  # 充电模式
                power = round(self.results['P_charge'][t] / 1000, 2)
            elif self.results['mode'][t] == -1:  # 放电模式
                power = round(self.results['P_discharge'][t] / 1000, 2)
            else:  # 待机模式
                power = 0.0

            result_str += (
                f"{time_str}\t{power:6.1f}\t\t{mode_str}\t\t{self.results['t_op'][t]:5.2f}\t\t"
                f"{self.results['SOC'][t]:5.2f}\t\t{self.price_data['electricity'][t]:.2f}\n")
            time_now += timedelta(minutes=dt_minutes)

        return result_str
