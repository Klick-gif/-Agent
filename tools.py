from baseclass import BatteryOptimizationInput
from langchain_core.tools import tool
from random import uniform
from typing import List
from dashscope import Application
import pickle
from config.config import llm, embeddings


# 导入你的优化求解器
# 注意：需要安装gurobipy: pip install gurobipy



def battery_optimization_algorithm(params: BatteryOptimizationInput) -> str:
    """
    使用Gurobi求解器进行电池优化调度
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
        import numpy as np

        GUROBI_AVAILABLE = True
    except ImportError:
        print("警告: Gurobi未安装，将使用简化算法")
        GUROBI_AVAILABLE = False

    try:
        if not GUROBI_AVAILABLE:
            print("error!")

        # 构建配置字典
        config = {
            'time_horizon': params.time_horizon_hours,
            'time_step': params.time_step_hours,
            'tem_init': params.tem_init,
            'battery': {
                'capacity': params.capacity_kwh,
                'max_power': params.max_power_kw,
                'soc_min': params.soc_min,
                'soc_max': params.soc_max,
                'soc_init': params.soc_init,
                'soh_init': params.soh_init
            },
            'degradation': {
                'n_segments': params.n_segments,
                'calendar_aging': params.calendar_aging
            },
            'carbon': {
                'grid_emission_factor': params.grid_emission_factor,
                'avoided_emission_factor': params.avoided_emission_factor,
                'max_emission': params.max_emission,
                'carbon_init': params.carbon_init
            },
            'weights': {
                'economic': params.weight_economic,
                'life': params.weight_life,
                'carbon': params.weight_carbon
            },
            'price_data': {
                'electricity': params.electricity_prices,
                'carbon': params.carbon_prices
            }
        }
        print(f"\nconfig: \n{config}")
        # 创建求解器并求解
        from model.battery_decision_model import BatteryOptimizationSolver
        solver = BatteryOptimizationSolver(config)
        success = solver.solve()

        if not success:
            return "优化求解失败，请检查参数设置，采用其他方法。"

        return solver.results['result_str']

    except Exception as e:
        return f"优化调度过程中出现错误: {str(e)}"
    

def core_scheduling_decision(
        time_horizon_hours: float = 24.0,
        time_step_hours: float = 0.5,
        capacity_kwh: float = 400000.0,
        max_power_kw: float = 100000.0,
        soc_init: float = 0.75,
        soc_min: float = 0.25,
        soc_max: float = 0.9,
        soh_init: float = 0.95,
        tem_init: float = 35.0,
        n_segments: int = 4,
        calendar_aging: float = 1e-7,
        grid_emission_factor: float = 0.5968,
        avoided_emission_factor: float = 0.5568,
        max_emission: float = 200000.0,
        carbon_init: float = 0,
        weight_economic: float = 0.8,
        weight_life: float = 0.1,
        weight_carbon: float = 0.1,
        electricity_prices: List[float] = None,
        carbon_prices: List[float] = None
) -> str:
    try:
        print(f"\n电价序列长度: {len(electricity_prices)}\n碳价序列长度: {len(carbon_prices)}")
        # 处理价格数据
        if electricity_prices is not None and abs(len(electricity_prices) - (time_horizon_hours / time_step_hours)) < 0.1:
            print(123)
            try:
                elec_prices = electricity_prices
            except:
                elec_prices = [uniform(-0.1, 1.3) for _ in range(int(time_horizon_hours / time_step_hours))]
        else:
            elec_prices = [uniform(-0.1, 1.3) for _ in range(int(time_horizon_hours / time_step_hours))]

        if carbon_prices is not None and abs(len(carbon_prices) - (time_horizon_hours / time_step_hours)) < 0.1:
            try:
                carb_prices = carbon_prices
            except:
                carb_prices = [round(uniform(0.05, 0.1), 2) for _ in range(int(time_horizon_hours / time_step_hours))]
        else:
            carb_prices = [round(uniform(0.05, 0.1), 2) for _ in range(int(time_horizon_hours / time_step_hours))]
            print("\n碳价格：", carb_prices, "\n")

        # 创建输入参数对象
        params = BatteryOptimizationInput(
            time_horizon_hours=time_horizon_hours,
            time_step_hours=time_step_hours,
            tem_init=tem_init,
            capacity_kwh=capacity_kwh,
            max_power_kw=max_power_kw,
            soc_init=soc_init,
            soc_min=soc_min,
            soc_max=soc_max,
            soh_init=soh_init,
            n_segments=n_segments,
            calendar_aging=calendar_aging,
            grid_emission_factor=grid_emission_factor,
            avoided_emission_factor=avoided_emission_factor,
            max_emission=max_emission,
            carbon_init=carbon_init,
            weight_economic=weight_economic,
            weight_life=weight_life,
            weight_carbon=weight_carbon,
            electricity_prices=elec_prices,
            carbon_prices=carb_prices
        )

        # 调用优化算法
        scheduling_result = battery_optimization_algorithm(params)

        # 返回结果
        return scheduling_result

    except Exception as e:
        return f"调度决策失败: {str(e)}"


@tool
def implement_scheduling_decision(
        time_horizon_hours: float = 24.0,
        time_step_hours: float = 0.5,
        capacity_kwh: float = 400000.0,
        max_power_kw: float = 100000.0,
        soc_init: float = 0.75,
        soc_min: float = 0.25,
        soc_max: float = 0.9,
        soh_init: float = 0.95,
        tem_init: float = 35.0,
        n_segments: int = 4,
        calendar_aging: float = 1e-7,
        grid_emission_factor: float = 0.5968,
        avoided_emission_factor: float = 0.5568,
        max_emission: float = 200000.0,
        carbon_init: float = 0,
        weight_economic: float = 0.8,
        weight_life: float = 0.1,
        weight_carbon: float = 0.1,
        electricity_prices: List[float] = None,
        carbon_prices: List[float] = None
) -> str:
    """
    执行储能电站优化调度决策

    Args:
        time_horizon_hours: 调度时间范围(小时)
        time_step_hours: 时间步长(小时)
        capacity_kwh: 额定容量(kWh)
        max_power_kw: 最大功率(kW)
        soc_init: SOC系统当前荷电状态值
        soc_min: SOC系统荷电最小值
        soc_max: SOC系统荷电最大值
        soh_init: SOH电池健康状态初始值
        tem_init: 系统初始温度(°C)
        n_segments: DoD分段数量
        calendar_aging: 日历老化系数
        grid_emission_factor: 电网碳排放因子(kg CO2/kWh)
        avoided_emission_factor: 避免碳排放因子(kg CO2/kWh)
        max_emission: 最大碳排放量(kg)
        carbon_init: 碳排放初始值(kg)
        weight_economic: 经济性权重
        weight_life: 寿命权重
        weight_carbon: 碳排放权重
        electricity_prices: 电价序列JSON字符串
        carbon_prices: 碳价序列JSON字符串

    Returns:
        调度结果的字符串，包括初始化信息，调度信息，经济效益等信息。
    """
    return core_scheduling_decision(
        time_horizon_hours=time_horizon_hours,
        time_step_hours=time_step_hours,
        capacity_kwh=capacity_kwh,
        max_power_kw=max_power_kw,
        soc_init=soc_init,
        soc_min=soc_min,
        soc_max=soc_max,
        soh_init=soh_init,
        tem_init=tem_init,
        n_segments=n_segments,
        calendar_aging=calendar_aging,
        grid_emission_factor=grid_emission_factor,
        avoided_emission_factor=avoided_emission_factor,
        max_emission=max_emission,
        weight_economic=weight_economic,
        carbon_init=carbon_init,
        weight_life=weight_life,
        weight_carbon=weight_carbon,
        electricity_prices=electricity_prices,
        carbon_prices=carbon_prices
    )


@tool
def search_weather(query):
    """
    根据时间和地点查询天气
    """
    response = Application.call(api_key="sk-eb8efc4f031c464c8db80722d7c7dbd1",
                                app_id='d7702d2629f049b99faccbeb312f919d', prompt=query)
    content = response.output.text
    print(type(content))
    return content


with open('data\\battery_db', 'rb') as f:
        db = pickle.load(f)
retriever = db.as_retriever(search_kwargs={'k': 3})

@tool
def get_knowledge(query):
    """
    对询问相关知识概念时，做知识查询
    """
    prompt = "这是用户提出的问题：" + query + "主要针对这个问题作出知识回答"
    "下面我会给你提供一些相关资料召回率前三的内容仅作为参考，如果内容不是很符合问题就自己回答"
    context = retriever.invoke(query)
    for i in range(3):
        prompt += context[i].page_content
    return prompt 



if __name__ == "__main__":

    query = "请问储能电站是什么"

    """print(query + ":\n")
    res1 = llm.invoke(query)
    print(res1 + "\n\n") """

    new_query = get_knowledge(query)
    print(new_query + ":\n")
    res2 = llm.invoke(new_query)
    print(res2)
