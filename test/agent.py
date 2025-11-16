from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from random import uniform
from dashscope import Application
import json

# 假设你有这个配置，如果没有需要替换为你的LLM配置
from config.config import llm


class BatteryOptimizationInput(BaseModel):
    """储能电站优化调度的输入参数"""
    # 时间参数
    time_horizon_hours: float = Field(default=24.0, description="调度时间范围(小时)")
    time_step_hours: float = Field(default=0.5, description="时间步长(小时)")

    # 温度参数
    tem_init: float = Field(default=35.0, description="系统初始温度(°C)")

    # 电池参数
    capacity_kwh: float = Field(default=400000.0, description="额定容量(kWh)")
    max_power_kw: float = Field(default=100000.0, description="最大功率(kW)")
    soc_init: float = Field(default=0.75, description="SOC荷电状态系统当前值")
    soc_min: float = Field(default=0.25, description="SOC荷电状态最小值")
    soc_max: float = Field(default=0.9, description="SOC荷电状态最大值")
    soh_init: float = Field(default=0.95, description="SOH电池健康状态值")

    # 退化模型参数
    n_segments: int = Field(default=4, description="DoD分段数量")
    calendar_aging: float = Field(default=1e-7, description="日历老化系数")

    # 碳排放参数
    grid_emission_factor: float = Field(default=0.5968, description="电网碳排放因子(kg CO2/kWh)")
    avoided_emission_factor: float = Field(default=0.5568, description="避免碳排放因子(kg CO2/kWh)")
    max_emission: float = Field(default=200000.0, description="最大碳排放量(kg)")
    carbon_init: float = Field(default=0, description="初始碳排放量(kg CO2)")

    # 权重系数
    weight_economic: float = Field(default=0.8, description="经济性权重")
    weight_life: float = Field(default=0.1, description="寿命权重")
    weight_carbon: float = Field(default=0.1, description="碳排放权重")

    # 价格数据
    electricity_prices: List[float] = Field(description="电价序列(元/kWh)")
    carbon_prices: List[float] = Field(description="碳价序列(元/kg)")

# 导入你的优化求解器
# 注意：需要安装gurobipy: pip install gurobipy
try:
    import gurobipy as gp
    from gurobipy import GRB
    import numpy as np

    GUROBI_AVAILABLE = True
except ImportError:
    print("警告: Gurobi未安装，将使用简化算法")
    GUROBI_AVAILABLE = False


def battery_optimization_algorithm(params: BatteryOptimizationInput) -> str:
    """
    使用Gurobi求解器进行电池优化调度
    """
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


class BatterySchedulingAgent:
    """电池储能调度智能体"""

    def __init__(self, llm):
        self.llm = llm
        self.available_tools = {"implement_scheduling_decision": implement_scheduling_decision, "search_weather": search_weather}
        self.llm_with_tools = llm.bind_tools([implement_scheduling_decision, search_weather])
        parser = JsonOutputParser(pydantic_object=BatteryOptimizationInput)
        # 参数提取的提示模板
        self.extract_prompt = PromptTemplate(
            template=(
                "你是一个液流电池储能电站智慧调度辅助决策智能体。\n"
                "用户咨询：{query}\n\n"
                "请分析用户的需求：\n"
                "1. 如果用户只是询问相关知识或概念，没有提及任何的系统参数，电价信息的话，请直接回答用户的问题\n"
                "2. 如果用户明确要求进行调度决策、优化计算或制定调度计划，请调用 implement_scheduling_decision 工具，"
                "但是如果一个可用信息都没提及就不需要调用工具，请直接回答用户，储能电站容量以及最大功率必须由用户指定。\n"
                "3. 调用工具时，请根据用户提供的信息设置相应参数，未提及的参数使用默认值{format_instructions}\n\n"
                "常见的调度需求关键词：调度、优化、决策、计划、充放电策略等\n"
                "4. 如果用户的价格序列不标准与他的在后台修改了，需要和用户说明，得到工具的答案，需要格式化输出，有具体的时间，"
                "模式，操作时间，功率和电价，按调度计划完整时间表格的格式输出给用户，并给出一个对调度计划的总结"
                "参数映射提示：\n"
                "- 容量400MWh = 400000 kWh\n"
                "- 功率100MW = 100000 kW\n"
                "- 12小时 = time_horizon_hours: 12.0\n"
                "- 间隔0.5小时 = time_step_hours: 0.5\n"
                "- SOC当前荷电状态为68% = soc_init: 0.68"
            ),
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

    def process_query(self, query: str) -> str:
        """处理用户查询"""
        try:
            # 构建消息
            prompt_msg = self.extract_prompt.format(query=query)
            messages = [HumanMessage(content=prompt_msg)]

            # 循环处理工具调用
            while True:
                output = self.llm_with_tools.invoke(messages)
                messages.append(output)
                print(output.content)
                # 如果没有工具调用，返回最终结果
                if not output.tool_calls:
                    break

                # 处理工具调用
                for tool_call in output.tool_calls:
                    tool_name = tool_call["name"]
                    if tool_name in self.available_tools:
                        selected_tool = self.available_tools[tool_name]
                        # print(selected_tool)
                        tool_result = selected_tool.invoke(tool_call["args"])
                        messages.append(ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call["id"]
                        ))
            return output.content

        except Exception as e:
            return f"处理查询时出错: {str(e)}"
        
    async def process_query_stream(self, query: str):
            """流式处理用户查询 - 改进版本"""
            try:
                prompt_msg = self.extract_prompt.format(query=query)
                messages = [HumanMessage(content=prompt_msg)]

                while True:
                    full_content = ""
                    tool_calls_data = []# 保存最后一个chunk
                    
                    # 检查是否支持异步流式
                    if hasattr(self.llm_with_tools, 'astream') and None:
                        print("异步流式")
                        try:
                            async for chunk in self.llm_with_tools.astream(messages):
                                print(chunk, "\n")
                                if hasattr(chunk, 'content') and chunk.content:
                                    full_content += chunk.content
                                    yield chunk.content  # 立即返回内容
                                
                                if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                                    tool_calls_data.extend(chunk.tool_calls)
                            
                            output = AIMessage(content=full_content, tool_calls=tool_calls_data)
                        except Exception as e:
                            print(f"异步流式调用失败: {e}")
                            # 降级到普通调用
                            output = self.llm_with_tools.invoke(messages)
                            yield output.content
                
                    
                    # 不支持流式，使用普通调用
                    else:
                        print("LLM不支持流式输出，使用普通调用")
                        output = self.llm_with_tools.invoke(messages)
                        # 模拟流式输出，每次返回几个字符
                        content = output.content
                        chunk_size = 10
                        for i in range(0, len(content), chunk_size):
                            yield content[i:i+chunk_size]
                            import asyncio
                            await asyncio.sleep(0.05)
                    
                    messages.append(output)

                    # 如果没有工具调用，结束循环
                    if not output.tool_calls:
                        break

                    # 处理工具调用
                    yield "\n\n[正在调用工具进行计算...]\n\n"
                    
                    for tool_call in output.tool_calls:
                        tool_name = tool_call["name"]
                        if tool_name in self.available_tools:
                            selected_tool = self.available_tools[tool_name]
                            tool_result = selected_tool.invoke(tool_call["args"])
                            messages.append(ToolMessage(
                                content=str(tool_result),
                                tool_call_id=tool_call["id"]
                            ))
                            
                            # 工具调用结果也流式返回
                            result_preview = str(tool_result)[:200] + "..." if len(str(tool_result)) > 200 else str(tool_result)
                            yield result_preview
                            yield f"\n[工具执行完成]\n"

            except Exception as e:
                error_msg = f"\n\n处理查询时出错: {str(e)}"
                print(error_msg)
                yield error_msg