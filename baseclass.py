from pydantic import BaseModel, Field
from typing import List


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