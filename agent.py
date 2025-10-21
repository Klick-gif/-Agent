from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from tools import implement_scheduling_decision, search_weather, get_knowledge
from baseclass import BatteryOptimizationInput


# 假设你有这个配置，如果没有需要替换为你的LLM配置
from config.config import llm




class BatterySchedulingAgent:
    """电池储能调度智能体"""

    def __init__(self, llm):
        self.llm = llm
        self.available_tools = {
            "implement_scheduling_decision": implement_scheduling_decision,
            "search_weather": search_weather,
            "get_knowledge": get_knowledge  
            }
        self.llm_with_tools = llm.bind_tools([implement_scheduling_decision, search_weather, get_knowledge])
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
                        # print("LLM不支持流式输出，使用普通调用")
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