import json
from zhipuai import ZhipuAI
from openai import OpenAI
import datetime
import tools
import api
import os
import re
import pandas as pd
from loguru import logger

folders = ["database_in_use", "data"]

if any(not os.path.exists(folder) for folder in folders):
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    import data_process # for data process using
else:
    print("所有文件夹均已存在。不再重新预处理数据。")
    print("需要预处理数据，请删除文件夹后重新运行。")


import re
from numbers import Number
from typing import Union


class MathValidator:
    """数学表达式验证工具（增强运算符版）"""
    
    OPERATORS = {

        '//': (3, lambda a, b: a // b),
        '*': (2, lambda a, b: a * b),
        '/': (2, lambda a, b: a / b),
        '+': (1, lambda a, b: a + b),
        '-': (1, lambda a, b: a - b),
    }

    def __init__(self):
        self._step_details = []
        self._current_step = 0

    # 添加缺失的方法（关键修复）
    def _sanitize_expression(self, expr: str) -> str:
        """表达式预处理（增强负数处理）"""
        expr = expr.replace(' ', '').replace('^', '**')
        # 处理开头负数的情况
        if expr.startswith('-'):
            expr = '0' + expr
        # 处理括号内的负数
        expr = re.sub(r'(?<=\()\s*-\s*', '0-', expr)
        return expr

    def _validate_syntax(self, expr: str) -> bool:
        """语法校验（增强正则）"""
        return re.match(r'^[\d+\-*/()\.]+$', expr) is not None

    def _tokenize(self, expr: str) -> list:
        """增强版分词处理"""
        # 优先匹配多字符运算符
        token_re = re.compile(r'(\d+\.?\d*|//|\*\*|[+\-*/()])')
        tokens = []
        for m in token_re.finditer(expr):
            token = m.group(1)
            # 数字转换
            if token.replace('.', '', 1).isdigit():
                tokens.append(float(token) if '.' in token else int(token))
            else:
                tokens.append(token)
        return tokens

    def _shunting_yard(self, tokens: list) -> list:
        """增强优先级处理"""
        output, stack = [], []
        for token in tokens:
            if isinstance(token, (int, float)):
                output.append(token)
            elif token == '(':
                stack.append(token)
            elif token == ')':
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                stack.pop()
            else:
                while stack and stack[-1] != '(' and \
                      self.OPERATORS[token][0] <= self.OPERATORS.get(stack[-1], (0,))[0]:
                    output.append(stack.pop())
                stack.append(token)
        return output + stack[::-1]

    def _evaluate_postfix(self, postfix: list) -> float:
        """增强步骤记录"""
        stack = []
        step = 1
        for token in postfix:
            if isinstance(token, (int, float)):
                stack.append(token)
            else:
                if len(stack) < 2:
                    raise ValueError("操作数不足")
                b = stack.pop()
                a = stack.pop()
                op_func = self.OPERATORS[token][1]
                result = op_func(a, b)
                self._record_step(step, a, token, b, result)
                step += 1
                stack.append(result)
        return stack[0]

    def _record_step(self, step: int, a: float, op: str, b: float, result: float):
        """详细记录步骤"""
        self._step_details.append({
            'step': step,
            'operation': f"{a} {op} {b}",
            'result': round(result, 4)
        })


################更改API###########################

os.environ["ZHIPUAI_API_KEY"] = "7915b06988a447c0b0bf1ef153becc67.MoW2lUeGeypqZcIR"
# os.environ["ZHIPUAI_API_KEY"] = "b4809e81f037dc42c602083eeadb0d17.3qmPCbyKt1V1Ftjk"
# ZHIPUAI_API_KEY="7915b06988a447c0b0bf1ef153becc67.MoW2lUeGeypqZcIR"

def check_api_key(api_key_env: str = "ZHIPUAI_API_KEY") -> str:
    """
    检查 API Key 是否设定，若未设置则警告
    """
    api_key = os.getenv(api_key_env)
    if not api_key:
        logger.warning(f"{api_key_env} is not set. Please set the environment variable.")
    return api_key

def check_base_url(base_url_env: str = "BASE_HOST") -> str:
    """
    检查BASE_HOST是否设定
    """
    base_url = os.getenv(base_url_env)
    if not base_url:
        logger.warning(
            f"{base_url_env} is not set. Please set the environment variable."
        )
        return None
    else:
        return f"{base_url}/api/paas/v4/"



def create_chat_completion(messages, model="glm-4-plus"):
# def create_chat_completion(messages, model=llm_config["model"]):
    client = ZhipuAI(
        base_url=check_base_url(),
        api_key=check_api_key(),
    )
    response = client.chat.completions.create(
        model=model, stream=False, messages=messages
    )
    return response


# In[4]:



def choose_table(question):
    # 加载数据表并结构化
    with open("dict.json", "r", encoding="utf-8") as file:
        tables = json.load(file)
    
    formatted_tables = []
    for table in tables:
        fields_desc = "\n".join([f"- {desc}" for desc in table["字段含义"]])
        formatted_tables.append(f"数据表名：{table['数据表名']}\字段含义:\n{fields_desc}")
    context_text = "\n\n".join(formatted_tables)

    # 预处理问题
    question = remove_parentheses(question)
    question = question.replace("侧推","艏推")
    # question = expand_keywords(question)  # 若实现关键词扩展


# 请仅返回表名，不要解释。"""
    prompt = f"""请根据以下数据表信息，严格按步骤分析问题「{question}」：

{context_text}

分析步骤：
1. ​**提取设备编号**：识别所有设备名称，如二号柴油发电机、一号舵桨等。
2. ​**关联数据表**：检查每个表的“相关设备”和“字段含义”是否包含这些设备。
3. ​**选择所有匹配表**：列出所有相关表名，用逗号分隔。

示例：
问题：“二号和三号柴油发电机的转速”
正确表名：Port1_ksbg_1, Port2_ksbg_1（因为Port1_ksbg_1包含二号，Port2_ksbg_1包含三号）

请仅返回表名，不要解释。"""

    # 调用模型并验证结果
    messages = [{"role": "user", "content": prompt}]
    response = create_chat_completion(messages)
    # selected_tables = response.choices[0].message.content.strip()
    # valid_tables = validate_tables(selected_tables)  # 校验表名有效性
    print("choose_table",str(response.choices[0].message.content))
    
    return str(response.choices[0].message.content)

# In[5]:
#防止python代码块出现的情况
def glm4_create(max_attempts, messages, tools, model="glm-4-plus"):
# def glm4_create(max_attempts, messages, tools, model=llm_config["model"]):
    # client = ZhipuAI()
    client = ZhipuAI(
        base_url=check_base_url(),
        api_key=check_api_key(),
    )
    for attempt in range(max_attempts):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=0
        )
        # print(attempt)
        print(response.choices[0].message.content)
        if (
                response.choices
                and response.choices[0].message
                and response.choices[0].message.content
        ):
            if "```python" in response.choices[0].message.content or "```json" in response.choices[0].message.content: #    or "示例" in response.choices[0].message.content:
                # 如果结果包含字符串'python'或'json'，则继续下一次循环
                continue
            else:
                # 一旦结果不包含字符串'python'，则停止尝试
                break
        else:
            return response
    return response

def glm4_create_nopython(max_attempts, messages, tools, model="glm-4-plus"):
# def glm4_create(max_attempts, messages, tools, model=llm_config["model"]):
    # client = ZhipuAI()
    client = ZhipuAI(
        base_url=check_base_url(),
        api_key=check_api_key(),
    )
    for attempt in range(max_attempts):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=0
        )
        # print(attempt)
        print(response.choices[0].message.content)
        if (
                response.choices
                and response.choices[0].message
                and response.choices[0].message.content
        ):
            if "```python" in response.choices[0].message.content: #    or "示例" in response.choices[0].message.content:
                # 如果结果包含字符串'python'，则继续下一次循环
                continue
            else:
                # 一旦结果不包含字符串'python'，则停止尝试
                break
        else:
            return response
    return response

#防止json代码块出现
def glm4_create_nojson(max_attempts, messages, tools, model="glm-4-plus"):
# def glm4_create(max_attempts, messages, tools, model=llm_config["model"]):
    # client = ZhipuAI()
    client = ZhipuAI(
        base_url=check_base_url(),
        api_key=check_api_key(),
    )
    # context=""
    err_prompt = {
                    "role": "user",
                    "content": f"\n禁止使用代码块格式（如 ```json），直接生成符合JSON规范的参数，并使用create_plan工具。请生成计划"
                }
    for attempt in range(max_attempts):
        response = client.chat.completions.create(
            model=model,
            messages=messages+[err_prompt],
            tools=tools,
            temperature=0
        )
        # print(attempt)
        print(response.choices[0].message.content)
        if (
                response.choices
                and response.choices[0].message
                and response.choices[0].message.content
        ):
            if "```json" in response.choices[0].message.content or "```python" in response.choices[0].message.content: #    or "示例" in response.choices[0].message.content:
                
                # 如果结果包含字符串'json'，则继续下一次循环
                continue
            else:
                # 一旦结果不包含字符串'json'，则停止尝试
                break
        else:
            return response
    return response


function_map = {
    "calculate_uptime": api.calculate_uptime,
    "calculate_start_end": api.calculate_start_end,

    "compute_actual_operational_duration": api.compute_actual_operational_duration,
    "get_table_data": api.get_table_data,
    "load_and_filter_data": api.load_and_filter_data,
    "calculate_total_energy": api.calculate_total_energy,
    "calculate_total_deck_machinery_energy": api.calculate_total_deck_machinery_energy,
    "calculate_total_propulsion_system_energy": api.calculate_total_propulsion_system_energy,
    "calculate_total_electricity_generation": api.calculate_total_electricity_generation,
    "calculate_ajia_energy": api.calculate_ajia_energy,
    "calculate_tuojiang_energy": api.calculate_tuojiang_energy,

    "query_device_parameter": api.query_device_parameter,
    "get_device_status_by_time_range": api.get_device_status_by_time_range,

    "calculate_total_fuel_volume": api.calculate_total_fuel_volume,
    "calculate_total_4_fuel_volume": api.calculate_total_4_fuel_volume,
    "calculate_electricity_generation": api.calculate_electricity_generation,

    "calculate_max_or_average": api.calculate_max_or_average,
    "calculate_sum_data": api.calculate_sum_data,

    "count_swings_10": api.count_swings_10,
    "count_swings": api.count_swings,

    # "calculate_alarm": api.calculate_alarm,
    "rt_temp_or_press_parameter": api.rt_temp_or_press_parameter,

    "action_judgment": api.action_judgment,

    "natural_processing": api.natural_processing,
    "get_value_of_time": api.get_value_of_time,
    "get_all_device_status_by_time_range":api.get_all_device_status_by_time_range,

    "get_zuoye_stage_by_time_range": api.get_zuoye_stage_by_time_range,
    "calculate_new4_stage_by_time_range": api.calculate_new4_stage_by_time_range,

    "review_calculate": api.review_calculate,

    "is_open": api.is_open,
    "count_shoufanglan": api.count_shoufanglan,
    # "calculate_energy_consumption": api.calculate_energy_consumption,
    # "calculate_total_energy_consumption": api.calculate_total_energy_consumption,
    
}


def process_natural_step(instruction: str, context: dict, messages: list):
    """处理纯自然语言步骤"""
    # 上下文格式化
    history = "\n".join([f"步骤{k}: {v}" for k,v in context.items()])
    
    # 安全提示词模板
    safe_prompt = {
        "role": "user",
        "content": f"根据指令和上下文生成自然语言响应：\n指令：{instruction}\n历史结果：{history}\n要求："
                  f"1. 禁止使用代码\n2. 用中文完整句子\n3. 保留精确数值"
    }
    
    # 带安全检查的调用
    response = glm4_create(
        messages=messages + [safe_prompt],
        max_attempts=3,
        tools=[]
   

    )
    
    # 结果安全过滤
    clean_result = safety_filter(response.choices[0].message.content)
    return clean_result

def safety_filter(text: str) -> str:
    """安全过滤器"""
    # 代码模式检测
    code_patterns = [r"```\w*", r"\b(def|for|import)\b", r":\s*$"]
    for pattern in code_patterns:
        if re.search(pattern, text):
            raise Exception("检测到代码生成")
    
    # 敏感字符过滤
    text = text.replace("`", "").strip()
    return text

def get_answer_2_test(question, tools, api_look: bool = True):
    try:
        # 确保工具列表包含create_plan
        if not any(tool['function']['name'] == 'create_plan' for tool in tools):
            tools.insert(0, {
                "type": "function",
                "function": {
                    "name": "create_plan",
                    "description": "生成自然语言执行计划，必须包含步骤编号、工具名称和执行指令",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "step_number": {"type": "integer"},
                                        "tool_name": {"type": "string"},
                                        "instruction": {"type": "string"}
                                    },
                                    "required": ["step_number", "tool_name", "instruction"]
                                }
                            }
                        },
                        "required": ["steps"]
                    }
                }
            })

            # 添加自然语言处理虚拟工具
        

        tool_descs = "\n".join([f"{tool['function']['name']}: {tool['function']['description']}" 
                              for tool in tools])
        
        system_msg = {
            "role": "system",
            "content": f"""请严格按两阶段处理：

第一阶段：生成自然语言计划
1. 分析需求生成包含步骤编号、工具名称和执行指令的计划
2. 指令需明确如何利用前序步骤结果
3. 确保步骤间逻辑连贯性

第二阶段：执行计划
1. 严格按步骤顺序执行
2. 将前序步骤结果作为自然语言上下文传递
3. 遇到错误立即停止
4. 禁止自我理解，将一个词理解为另一个，如未查询的动作，就返回N，而不是将一个动作理解为另一个动作

可用工具：
{tool_descs}

输出要求：
• 首先生成执行计划（必须调用create_plan）
• 最终回答必须基于实际执行结果
"""
        }

        messages = [system_msg, {"role": "user", "content": question}]
        
        # 第一阶段：生成计划
        print("\n=== 计划制定 ===")
        plan_response = glm4_create(
            max_attempts=6,
            messages=messages,
            tools=tools
        )
        plan_msg = plan_response.choices[0].message
        messages.append(plan_msg.model_dump())
        
        # 验证计划格式
        if not plan_msg.tool_calls or plan_msg.tool_calls[0].function.name != 'create_plan':
            return "错误：未生成有效计划", messages
            
        # 解析计划步骤
        try:
            plan_args = json.loads(plan_msg.tool_calls[0].function.arguments)
            plan_steps = sorted(plan_args['steps'], key=lambda x: x['step_number'])
            print(f"解析到{len(plan_steps)}个步骤")
        except Exception as e:
            return f"计划解析失败：{str(e)}", messages

        # 第二阶段：执行计划
        print("\n=== 计划执行 ===")
        context = {}
        
        for step in plan_steps:
            step_num = step['step_number']
            tool_name = step['tool_name']
            instruction = step['instruction']
            
            print(f"\n▶ 步骤 {step_num}: {tool_name}（指令：{instruction}）")
            
            # 动态生成参数
            resolved_params = {}
            if tool_name != "create_plan":
                print(f"生成步骤{step_num}参数...")
                history_ctx = "\n".join([f"步骤{k}: {v}" for k,v in context.items()])
                param_prompt = {
                    "role": "user",
                    "content": f"根据指令和上下文生成参数：\n指令：{instruction}\n上下文：{history_ctx}\n需要调用工具：{tool_name}"
                }
                
                param_response = glm4_create(
                    max_attempts=6,
                    messages=messages + [param_prompt],
                    tools=[t for t in tools if t['function']['name'] == tool_name]
                )
                param_msg = param_response.choices[0].message
                messages.append(param_msg.model_dump())
                
                if param_msg.tool_calls:
                    resolved_params = json.loads(param_msg.tool_calls[0].function.arguments)
                    print(f"生成参数：{resolved_params}")

            # 执行工具调用
            try:
                
                
                 # 工具存在性校验
                # if tool_name not in [t['function']['name'] for t in tools]:
                #     raise ValueError(f"无效工具：{tool_name}")

                # 自然语言处理专用流程
                if tool_name == "natural_processing":
                    print(f"\n▶ 自然语言处理步骤 {step_num}")
                    result = process_natural_step(instruction, context, messages)
                    context[step_num] = result
                    continue
                
                print(f"执行 {tool_name}({resolved_params})")
                result = function_map[tool_name](**resolved_params)
                context[step_num] = str(result)  # 强制转换为自然语言
                print(f"结果：{str(result)[:80]}...")

                # 更新消息记录
                messages.append({
                    "role": "tool",
                    "content": context[step_num],
                    "tool_call_id": param_msg.tool_calls[0].id if param_msg.tool_calls else f"step_{step_num}"
                })

            except Exception as e:
                error_msg = f"步骤{step_num}失败：{str(e)}"
                messages.append({"role": "tool", "content": error_msg})
                return error_msg, messages

        # return context[len(plan_steps)], messages
        if len(context) > 0:
            # 生成总结提示
            summary_prompt = {
                "role": "user",
                "content": f"请综合以下所有步骤结果生成最终回答：\n" + 
                        "\n".join([f"步骤{num}: {result}" for num, result in context.items()])
            }
            
            # 调用大模型生成总结
            summary_response = glm4_create(
                messages=messages + [summary_prompt],
                max_attempts=3,
                tools=[]
            )
            final_answer = summary_response.choices[0].message.content
            messages.append(summary_response.choices[0].message.model_dump())
        else:
            final_answer = "未获得有效执行结果"

        return final_answer, messages

    except Exception as e:
        return f"处理失败：{str(e)}", messages

def get_answer_2(question, tools, api_look: bool = True):
    try:
        # 强制包含计划生成工具
        create_plan_tool = {
            "type": "function",
            "function": {
                "name": "create_plan",
                "description": "生成可包含工具调用和自然语言步骤的执行计划",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "step_number": {"type": "integer"},
                                    "tool_name": {"type": ["string", "null"]},  # 允许空值
                                    "instruction": {"type": "string"}
                                },
                                "required": ["step_number", "instruction"]
                            }
                        }
                    },
                    "required": ["steps"]
                }
            }
        }
        if not any(t['function']['name'] == 'create_plan' for t in tools):
            tools.insert(0, create_plan_tool)

        # 系统提示强化
        tool_descs = "\n".join([f"{t['function']['name']}: {t['function']['description']}" for t in tools])
        system_msg = {
            "role": "system",
            "content": f"""执行规则：
第一阶段：生成自然语言计划
1. 分析需求生成包含步骤编号、工具名称和执行指令的计划
2. 指令需明确如何利用前序步骤结果
3. 确保步骤间逻辑连贯性



第二阶段：执行计划
1. 严格按步骤顺序执行
2. 将前序步骤结果作为自然语言上下文传递
3. 遇到错误立即停止



可用工具：
{tool_descs}

输出要求：
• 首先生成执行计划，必须使用工具create_plan，其他过程中禁止使用该工具
• 最终回答必须基于实际执行结果,tool_calls为None指的是仅调用自然语言处理而不使用工具

"""
        }

        messages = [system_msg, {"role": "user", "content":"生成计划：\n"+ question+f"""
                                -计划的格式规则:
                                1. `steps` 必须是对象数组，且步骤按 `step_number` 顺序排列，每个对象必须包含 `step_number` 、 `tool_name` 和'instruction'三个字段。
                                2. `tool_name` tool_calls为None指的是仅使用自然语言处理而不使用工具
                                3. `instruction` 需清晰描述具体操作（例如："使用工具A调整参数"，"人工检查设备状态"）。
                                                         

                                -计划制定总要求：
                                1. 总计划数量不能超过8个，且必须至少包含一个有效步骤。
                                2. 多次调用一个函数请在一个步骤中进行。
                                3. 步骤列表不能为空，至少要包含一个具体操作步骤。
                                 
                                -制定细节
                                1. 示例：若题目中有类似“从征服者起吊到入水”，则应该先查“征服者起吊”，再查“征服者入水”
                                2. 示例：若题目查询2024-05-23到2024-05-24最早出水日期，每日出水时间原始数据：["2024-05-23 18:30", "2024-05-23 19:45", "2024-05-24 08:00"],则应该选择05-24这天（8:00），而不是5-23（18:30），因为虽然5-23早于05-24，但是8:00早于18:30
                                3. 示例：若题目中需要查询连续多天的多个状态，如对于题目“统计2024/10/03  00:00:00-2024/10/09  00:00:00(包含)每一天的的停泊/航渡/动力定位的时长”，可以先统计2024/10/03  00:00:00-2024/10/09  00:00:00的停泊状态，也就是calculate_new4_stage_by_time_range("2024-06-03 00:00:00","2024-06-09 00:00:00","停泊")....因为日期的起始点没有时间间隔
                                    备注：不可以将有时间间隔的时间作为起始点查询，如若查询2024-05-23和2024-05-24上午的平均时长，由于23日上午和34日上午不连续，所以应该分别计算时长，再取平均


                                -严厉禁止：
                                禁止使用代码块格式（如 ```json），直接生成符合JSON规范的参数
                                 
                                 """}]

        # 第一阶段：生成计划
        # print("\n=== 计划生成 ===")
        # plan_response = glm4_create(
        #     messages=messages,
        #     tools=tools,
        #     max_attempts=5
        # )
        # plan_msg = plan_response.choices[0].message
        # messages.append(plan_msg.model_dump())
        # print(plan_msg.model_dump())  # 输出完整消息结构

        # # 解析执行计划
        # plan_args = json.loads(plan_msg.tool_calls[0].function.arguments)

        # 第一阶段：生成计划（带有效性验证）
        max_plan_attempts = 3  # 最大生成尝试次数
        valid_plan = False
        plan_context={}

        for attempt in range(max_plan_attempts):
            print(f"\n=== 计划生成（尝试 {attempt+1}/{max_plan_attempts}）===")
            plan_response = glm4_create_nojson(
                messages=messages,
                tools=tools,
                max_attempts=3
            )
            plan_msg = plan_response.choices[0].message
            messages.append(plan_msg.model_dump())
            print(plan_msg.model_dump())  # 输出完整消息结构
            
            # 有效性验证
            try:
                # 检查工具调用结构
                if not plan_msg.tool_calls or len(plan_msg.tool_calls) == 0:
                    plan_context[attempt]=f"第{attempt}步错误：未生成工具调用"

                    print("错误：未生成工具调用")
                    continue
                
                tool_call = plan_msg.tool_calls[0]
                if not hasattr(tool_call.function, 'arguments') or not tool_call.function.arguments.strip():
                    print("错误：工具调用参数为空")
                    plan_context[attempt]=f"第{attempt}步错误：工具调用参数为空"
                    continue
                    
                # 解析参数
                plan_args = json.loads(tool_call.function.arguments)
                
                # 检查步骤有效性
                if 'steps' not in plan_args or not isinstance(plan_args['steps'], list):
                    print("错误：缺少有效步骤列表")
                    plan_context[attempt]=f"第{attempt}步错误：工具调用参数为空"
                    continue
                    
                plan_steps = plan_args['steps']
                if len(plan_steps) == 0:
                    print("错误：步骤列表为空")
                    continue
                    
                # 验证步骤格式
                valid_steps = True
                for step in plan_steps:
                    if not all(key in step for key in ['step_number', 'tool_name', 'instruction']):
                        valid_steps = False
                        break
                if not valid_steps:
                    print("错误：步骤缺少必要字段")
                    continue
                    
                valid_plan = True
                break
                
            except json.JSONDecodeError:
                print("错误：参数解析失败")
            except Exception as e:
                print(f"未知错误：{str(e)}")

        if not valid_plan:
            raise ValueError("无法生成有效计划，请检查模型提示或输入参数")
       
        plan_steps = sorted(plan_args['steps'], key=lambda x: x['step_number'])
        step_map = {s['step_number']: s for s in plan_steps}  # 新增映射表
        print(f"解析到{len(plan_steps)}个步骤")

        # 第二阶段：执行计划
        print("\n=== 计划执行 ===")
        context = {}
        successful_results = {}  # 新增：存储成功结果
        
        for step in plan_steps:
            step_num = step['step_number']
            tool_name = step['tool_name']
            instruction = step['instruction']

            print(f"\n▶ 步骤 {step_num} [{'工具:'+tool_name if tool_name else '自然语言处理'}]")
            print(f"指令：{instruction}")

            # 工具调用分支（新增多调用逻辑）
            step_results = []
            attempt = 1
            max_attempts = 4  # 最大尝试次数
            flag_tools=False

            # print(tools)
            # 正确获取所有工具名称
            available_tools = [tool['function']['name'] for tool in tools]

            # 自然语言处理分支
            if tool_name not in available_tools:
                # 生成自然语言结果
                print("\n▶ 自然语言处理")
                history = "\n".join([f"步骤{k}: {v}" for k, v in context.items()])
                prompt = {
                    "role": "user",
                    "content": f"根据历史结果生成步骤{step_num}的响应：\n{format_history(step_results)}\n\n当前指令：{instruction}"
                }
                
                response = glm4_create(
                    messages=messages + [prompt],
                    max_attempts=3,
                    tools=[]
                )
                result = response.choices[0].message.content
                context[step_num] = result

                step_results.append({
                        "attempt": attempt,
                        "params": None,
                        "result": str(result),
                        "is_success": True
                    })
                successful_results.setdefault(step_num, []).append({  # 使用列表存储多个结果
                                "tool": tool_name,
                                "params": None,
                                "result": str(result),
                                "attempt": attempt,
                                # "timestamp": datetime.now().isoformat()  # 添加时间戳区分调用顺序
                            })

                messages.append({"role": "assistant", "content": result})
                continue

            
            
            while attempt <= max_attempts:
                print(f"└ 尝试 #{attempt}")
                
                # 动态参数生成
                # param_prompt = {
                #     "role": "user",
                #     "content": f"生成步骤{step_num}参数（第{attempt}次调用）:\n工具：{tool_name}\n历史结果：{context}\n指令：{instruction}"
                # }

                # suggestions = []  # 新增：步骤级建议
                # # 添加历史建议（关键修改点）
                # suggestion_context = "\n".join([
                #     f"【建议{idx+1}】{s}" 
                #     for idx, s in enumerate(suggestions) 
                #     if s.strip()
                # ])
                
                
                # 动态参数生成（包含历史错误信息）
                # param_prompt = {
                #     "role": "user",
                #     "content": f"生成步骤{step_num}参数（第{attempt}次调用）:\n工具：{tool_name}\n\n历史结果：{format_history(step_results)}\n指令：{instruction}"  # 修改1：添加格式化历史
                # }
                param_prompt = {
                                "role": "user",
                                "content": f"""生成步骤{step_num}参数（第{attempt}次调用）
                        工具：{tool_name}
                        历史结果：{format_history(step_results)}
                        
                        指令：{instruction}"""
                            }
                
                param_response = glm4_create(
                    messages=messages + [param_prompt],
                    tools=[t for t in tools if t['function']['name'] == tool_name],
                    max_attempts=4
                )
                param_msg = param_response.choices[0].message
                messages.append(param_msg.model_dump())
                # print(f"第{step_num}步 第{attempt} 次的message",messages)

                has_valid_result = False  # 修改2：添加执行成功标志

                # 处理多个工具调用
                if param_msg.tool_calls:
                    for call in param_msg.tool_calls:
                        try:
                            params = json.loads(call.function.arguments)
                            print(f"调用参数：{params}")
                            
                            # 执行工具并记录结果
                            result = function_map[tool_name](**params)
                            # step_results.append({
                            #     "attempt": attempt,
                            #     "params": params,
                            #     "result": str(result)
                            # })
                            # 记录成功结果
                            step_results.append({
                                "attempt": attempt,
                                "params": params,
                                "result": str(result),
                                "error": None,
                                "is_success": True  # 修改3：添加成功标志
                            })

                            # successful_results[step_num] = {  # 修改：直接记录成功结果
                            #         "tool": tool_name,
                            #         "params": params,
                            #         "result": str(result)
                            #     }
                            successful_results.setdefault(step_num, []).append({  # 使用列表存储多个结果
                                    "tool": tool_name,
                                    "params": params,
                                    "result": str(result),
                                    "attempt": attempt,
                                    # "timestamp": datetime.now().isoformat()  # 添加时间戳区分调用顺序
                                })
                            
                            print(result)
                            
                            
                            # 更新消息记录
                            messages.append({
                                "role": "tool",
                                "content": str(result),
                                "tool_call_id": call.id
                            })

                            has_valid_result = True
                            
                        except Exception as e:
                            error_msg = f"调用失败：{str(e)}"
                            print(error_msg)
                            # 记录错误详情
                            step_results.append({
                                "attempt": attempt,
                                "params": params,
                                "result": error_msg,
                                "error": str(e),  # 修改4：记录原始错误
                                "is_success": False
                            })
                            # 将错误信息加入上下文
                            messages.append({
                                "role": "tool",
                                "content": error_msg,
                                "tool_call_id": call.id
                            })
                            # continue
                else:  # 无工具时进行二次尝试
                    # if flag_tools==True:
                    result=param_msg.content
                    print("第二次有工具但无工具调用的result:",result)
                    step_results.append({
                        "attempt": attempt,
                        "params": None,
                        "result": str(result),
                        "is_success": True
                    })
                    successful_results.setdefault(step_num, []).append({  # 使用列表存储多个结果
                                    "tool": tool_name,
                                    "params": None,
                                    "result": str(result),
                                    "attempt": attempt,
                                    # "timestamp": datetime.now().isoformat()  # 添加时间戳区分调用顺序
                                })
                    # else:
                    #     flag_tools=True
                    #     result=param_msg.content
                    #     print("第一次有工具但无工具调用的result:",result)
                    #     step_results.append({
                    #         "attempt": attempt,
                    #         "params": None,
                    #         "result": str(result),
                    #         "is_success": False
                    #     })
                        


                # 判断是否继续调用（示例：通过结果数量判断）
                if should_continue(instruction, step_results, attempt, max_attempts):  # 修改5：添加尝试次数参数
                    print("└ 需要继续调用")
                    attempt += 1
                else:
                    print("└ 完成调用")
                    break
                # # 获取是否继续和最新建议
                # continue_flag, new_suggestion = should_continue(instruction, step_results, attempt, max_attempts)
                # if new_suggestion:
                #     suggestions.append(new_suggestion)  # 记录所有建议
                    
                # if continue_flag:
                #     print(f"└ 继续调用，原因：{new_suggestion}")
                #     attempt += 1
                # else:
                #     print("└ 完成调用")
                #     break

            # 整合步骤结果
            # context[step_num] = integrate_results(step_results, instruction)

        # 最终汇总（保持不变）
        print("\n=== 结果汇总 ===")
        # print("context:",context)

        # 1. 直接使用context字典生成消息
        # summary_messages = [
        #     {
        #         "role": "user",
        #         # "content": f"步骤{step_num}，指令：{step_map[step_num]['instruction']}。结果：{result}"
        #         "content": f"步骤{step_num} - 工具:{details['tool']}\n参数：{details['params']}\n结果：{details['result']}"
        #     }
        #     # for step_num, result in context.items()
        #     for step_num, details in successful_results.items()

        # ]
        summary_messages = []
        for step_num, results in successful_results.items():
            for idx, result in enumerate(results, 1):
                summary_messages.append({
                    "role": "user",
                    "content": f"""步骤 {step_num} - 调用记录 #{idx}
                工具名称: {result['tool']}
                调用参数: {json.dumps(result['params'], indent=2)}
                执行结果: {result['result']}
                """
                })

        # print("summary_massage:",summary_messages)
        # 2. 添加汇总指令
        #去除查表查到的数据
        if "##" in question and "###" not in question:
            ques1=remove_parentheses(question)
        else:
            ques1=question
        summary_messages.append({
            "role": "user",
            "content": f"""对于问题：{ques1}，请总结以上所有步骤的结果生成对问题的回答，要求：
            1. 仅基于提供的内容
            2. 使用markdown列表格式，应有精简的注释，如：甲板机械能耗：45.98kW.
            3. 若有计算表达式，则应该保留，如：最终理论发电量（保留两位小数）：40.00kWh (2*2*10/1=40,能多步合并则合并。禁止分步骤，禁止输出中间结果，例如2*2=4，4*10=40，40/1=40)
            
            禁止内容：
            1. 禁止添加额外信息，如示例等
            2. 禁止自己联想或续写
            3. 禁止修改原文信息
            """
        })
        print("final_summary_massage:",summary_messages)

        # 3. 仅传递context内容
        final_response = glm4_create(
            messages=summary_messages,
            max_attempts=3,
            tools=[]
        )

        return final_response.choices[0].message.content, messages

    except Exception as e:
        return f"处理流程异常：{str(e)}", messages

def should_continue(instruction: str, results: list, current_attempt: int, max_attempts: int) -> bool:
    """增强版判断逻辑，包含错误处理和尝试次数限制"""
    if current_attempt >= max_attempts:
        return False
    
    if not results:
        return True
    
    last_result = results[-1]
    # 如果最后一次成功且满足条件
    if last_result.get('is_success'):
        return False
    if "进一步调用" in instruction or "继续调用" in instruction:
        return True
    
    # 如果最后一次失败，且还有剩余尝试次数
    if not last_result.get('is_success') and current_attempt < max_attempts:
        return True
    
    # 默认停止
    return False

def should_continue_test(instruction: str, results: list, current_attempt: int, max_attempts: int) -> tuple[bool, str]:
    """返回是否继续 和 修改建议"""
    if current_attempt >= max_attempts:
        return False, ""
    
    if not results:
        return True, ""
    
    last_result = results[-1]
    
    # 失败场景直接返回
    if not last_result.get('is_success'):
        return current_attempt < max_attempts, last_result.get('error', "")
    
    # 成功时调用大模型决策
    decision_prompt = {
        "role": "user",
        "content": f"""请分析是否需要继续尝试，若需要请给出具体修改建议：
        
        当前结果：{last_result['result']}
        用户需求：{instruction}
        
        决策规则：
        1. 需要继续的情况：工具是否需要再次调用，使用工具时参数是否有误
        2. 不需要继续的情况：需求完全满足、结果已精确
        
        请用JSON格式返回：{{"decision": "是/否", "suggestion": "具体修改建议"}}"""
    }
    
    try:
        response = glm4_create(messages=[decision_prompt], tools=[], max_attempts=1)
        decision_str = response.choices[0].message.content
        decision_data = json.loads(decision_str)
        
        print(f"决策建议：{decision_data.get('suggestion')}")
        return decision_data.get("decision", "否") == "是", decision_data.get("suggestion", "")
    
    except Exception as e:
        print(f"决策异常：{str(e)}")
        return True, ""

def format_history(results: list) -> str:
    """格式化历史记录包含错误信息"""
    history = []
    for i, r in enumerate(results, 1):
        entry = f"第{i}次尝试: "
        if r['is_success']:
            entry += f"成功获取{len(r['result'])}字符结果"
        else:
            entry += f"失败原因：{r['error']}"
        history.append(entry)
    return "\n".join(history)

def integrate_results(results: list, instruction: str) -> str:
    """整合多次调用结果"""
    if not results:
        return "未获取有效结果"
    
    # 简单合并示例
    if "获取数据" in instruction:
        merged_data = [res['result'] for res in results]
        return f"共获取{len(merged_data)}批数据"
    
    # 复杂整合可调用LLM
    integrate_prompt = {
        "role": "user",
        "content": f"""请总结以下多次调用结果：
        要求：
        1. 仅基于提供的内容
        2. 使用Markdown列表格式，应有精简的注释，如：甲板机械能耗：45.98kW.
        
        禁止内容：
        1. 禁止添加额外信息，如示例等
        2. 禁止自己联想或续写
        3. 禁止修改原文信息
        \n指令：{instruction}\n{json.dumps(results, indent=2)}"""
    }
    response = glm4_create(messages=[integrate_prompt], max_attempts=3,tools=[])
    return response.choices[0].message.content


def remove_parentheses(s):
    # 删除中文括号及其内容
    s = re.sub(r'##.*?##', '', s)
    # 删除英文括号及其内容
    return s


# In[6]:
def select_api_based_on_question(question, tools):
    # print('ok')
    # print(question)
    # 根据问题内容选择相应的 API

    # if "能耗" in question:
    action_list=["A架开机","ON DP","征服者起吊","征服者入水","缆绳解除","A架摆回","小艇落座","A架关机","OFF DP","折臂吊车开机","A架摆出","小艇检查完毕","小艇入水","缆绳挂妥","征服者出水","折臂吊车关机","征服者落座"]
    api_list_filter=["get_device_status_by_time_range","get_value_of_time"]
    question=question.strip()
    ques=question
    is_get_table_data=False

    if "动作" in question or "设备" in question:
        api_list_filter.append("get_all_device_status_by_time_range")

    if  any(value in question for value in action_list):
        api_list_filter.append("get_device_status_by_time_range")
        question = question + "##动作直接引用不要修改,如【A架摆回】。##"
        if ("征服者" in question and ("起吊" in question or "出水" in question)) and "布放" not in question:
            question= question+"在回收阶段，要查征服者起吊，需要按照征服者出水来查，因为在回收阶段这两个动作是一个意思。"

    if "舵桨" in question and "能耗" in question:
        print("舵桨jinl")
        api_list_filter.append("calculate_tuojiang_energy")
    
    if re.search(r"A架.*能耗", question):
        print("Ajiajinl=================================")
        # api_list_filter.append("calculate_ajia_energy")
        # print(api_list_filter)
    #推进系统改为推进器
    if ("推进系统" in question or "推进器" in question) and "能耗" in question:
        api_list_filter.append("calculate_total_propulsion_system_energy")

    if "甲板机械" in question and "能耗" in question:
        api_list_filter.append("calculate_total_deck_machinery_energy")
        question = question + "能耗为负值也要带入计算。"
    
    if "燃油消耗量" in question:
            # api_list_filter.append("get_table_data")
        api_list_filter.append("get_device_status_by_time_range")
        api_list_filter.append("calculate_total_fuel_volume")  
        api_list_filter.append("calculate_total_4_fuel_volume")
        if "号" not in question:
            api_list_filter.remove("calculate_total_fuel_volume")
        if "单位为kg" in question or "单位：kg" in question:
            question=question+"燃油消耗量的质量=体积*密度"

    if "能耗" in question and "平均能耗" not in question:
        # api_list_filter = ["calculate_total_deck_machinery_energy"]
        api_list_filter.append("calculate_total_energy")
        print("总发电量1111111111111111111111")
        api_list_filter.append("calculate_total_electricity_generation")
    # elif "理论发电量" in question:
    #     api_list_filter = ["calculate_theory_electricity_generation"]
    
    

    #时间类
    if any(key in question for key in ["实际运行时长", "实际运行时间", "实际开机时间", "实际开机时长"]):
        api_list_filter = ["compute_actual_operational_duration", "calculate_uptime"]
        question += "##实际运行时长为有电流的时长，效率计算示例：运行效率=实际运行时长/运行时长.##"

    # 第2优先级：包含设备名的平均时长（特殊复合条件）
    elif ("平均" in question) and \
        any(time_word in question for time_word in ["时长", "时间", "多久"]) and \
        any(device in question for device in ["折臂吊车", "DP", "A架"]):
        api_list_filter = ["calculate_uptime"]
        question = "##涉及多天的平均和比例问题，均以天数为分母计算。如：【2024/8/23 和 2024/8/25 上午A架运行的平均时间多少】理解为【(2024/8/23上午A架运行时间 + 2024/8/25上午A架运行时间)/2，注意不计算24日】##" + question

    # 第3优先级：明确包含"开机时长"的情况
    elif "开机时长" in question:
        api_list_filter.append("calculate_uptime")
        # 统一术语替换
        # question = question.replace("运行时长", "开机时长").replace("运行时间", "开机时间")

    # 第4优先级：普通运行时长/时间（需排除已处理情况）
    elif ("运行时长" in question or "运行时间" in question) and \
        "实际运行时长" not in question:  # 防止重复处理
        api_list_filter.append("calculate_uptime")
        # 标准化术语
        question = question.replace("运行时长", "开机时长").replace("运行时间", "开机时间")

    if "在运行" in question or "运行状态" in question:
        api_list_filter.append("is_open")


    if "收揽" in question or "放缆" in question:
        api_list_filter.append("count_shoufanglan")
    

    if "数值类型参数" in question:
        # api_list_filter.remove("query_device_parameter")
        api_list_filter.append("rt_temp_or_press_parameter")
        question= question+"至少触发报警数举例：当参数值（题目的假设值）低于200，而报警值（Parameter_Information_Alarm）是低于20时，则不一定会触发报警，因为低于200不一定是低于20"
 

    if "告警" in question or "预警" in question or "正常" in question or "故障" in question or "异常" in question or 'RPM' in question or "高于" in question or "低于" in question or "上下限" in question or "报警" in question:
        if is_get_table_data == False:
            print("get_table_data")
            is_get_table_data=True
            table_name_string = choose_table(question)
            # print(question)
            with open("dict.json", "r", encoding="utf-8") as file:
                table_data = json.load(file)

            table_name = [
                item for item in table_data if item["数据表名"] in table_name_string
            ]

            all_combined_fields = []
            for tmp in table_name:
                # 组合字段名和字段含义
                combined = []
                for name, meaning in zip(tmp["字段名"], tmp["字段含义"]):
                    combined.append(f"{name}: {meaning}")
                # 创建新字段
                tmp["字段名及字段"] = combined
                # 构建符合要求的字典结构
                all_combined_fields.append({
                    "数据表名": tmp["数据表名"],
                    "字段名及字段": combined
                })
                
            question = "##"+str(all_combined_fields)+"##"+"\n"+question
        api_list_filter.append("query_device_parameter")

    if not api_list_filter:
        # 如果问题不匹配上述条件，则根据表名选择 API
        
        table_name_string = choose_table(question)
        print(question)
        with open("dict.json", "r", encoding="utf-8") as file:
            table_data = json.load(file)

        table_name = [
            item for item in table_data if item["数据表名"] in table_name_string
        ]

        for tmp in table_name:
            # 组合字段名和字段含义
            combined = []
            for name, meaning in zip(tmp["字段名"], tmp["字段含义"]):
                combined.append(f"{name}: {meaning}")
            # 创建新字段
            tmp["字段名及字段"] = combined

        # print(table_name)
        print("进了大else")

        if "设备参数详情表" in [item["数据表名"] for item in table_name] or "范围" in question:
            api_list_filter.append("query_device_parameter")
            # question = question + ""
            content_p_1 = str(table_name) + question  # 补充 content_p_1
            # print(content_p_1)
        else:
            # print("ok")
            # api_list_filter=["get_table_data"]
            api_list_filter.append("get_device_status_by_time_range")
            content_p_1 = str(table_name) + question
            print("进入else中的else")
            # print(content_p_1)

    #移除##内的内容
    
    ques = remove_parentheses(ques)
    print(ques)
    if "作业" in ques and ("能耗" in ques or "发电" in ques):
        api_list_filter.append("get_device_status_by_time_range")
        api_list_filter.append("calculate_total_electricity_generation")
        # question = ques
        # question = ques + "若没有指明阶段，则默认计算一整天,例如计算从动作A到动作B期间的总能耗，则将A的时间点和B的时间点作为参数传入能耗计算。。"

    if "告警" in ques or "预警" in ques or "正常" in ques or "故障" in ques or "异常" in ques or 'RPM' in ques or "高于" in ques or "低于" in ques or "上下限" in ques or "报警" in ques:
        api_list_filter.append("query_device_parameter")
        # print(api_list_filter)
    if "能耗" in ques:
        # api_list_filter.append("calculate_start_end")
        api_list_filter.append("calculate_total_energy")
        api_list_filter.append("get_device_status_by_time_range")
        print("能耗问题-----------")
        
    if ("平均" in ques or "均值" in ques) and "平均值" not in ques:
        question = question + "##涉及多天的平均问题，均以天数为分母计算。如：【2024/8/23 和 2024/8/25 上午A架运行的平均时间多少】理解为【(2024/8/23上午A架运行时间 + 2024/8/25上午A架运行时间)/2，注意不计算24日】。##"
        # ques = remove_parentheses(question)
    if "作业" in ques and ("发电" in ques or "能耗" in ques):
        print("作业22222222222222222222222")
        api_list_filter.append("calculate_total_electricity_generation")
        # question = question + "##作业由下放阶段和回收阶段两个阶段组成.上午为下放阶段，阶段开始和结束默认为ON DP和OFF DP；下午为回收阶段，阶段开始和结束默认为A架开机和A架关机。若题目中有标志则优先使用标志。作业能耗=总发电量。##"
        # question = question + "##若题目中有标志则优先使用标志。作业能耗=总发电量。##"
        # ques = remove_parentheses(question)
    if ("时间" in ques or "时长" in ques or "多久" in ques) and ("折臂吊车" in ques or "DP" in ques or "A架开" in ques or "A架关" in ques) and ("时间点" not in ques) and ("相隔" not in ques):
        api_list_filter.append("calculate_start_end")
        print(ques)
        print("时间问题-----------")

    #作业问题、布放、回收问题
    if "作业" in question and "折臂吊车开机为开始" not in question:
        ###############下放、回收
        api_list_filter.append("get_zuoye_stage_by_time_range")
        print("下放、回收、作业问题-----------")

    #复赛新增四个阶段问题：停泊状态、航渡状态、动力定位状态、伴航状态
    if "停泊" in question or "航渡" in question or "动力定位" in question or "伴航" in question:
        print("四个阶段问题：停泊状态、航渡状态、动力定位状态、伴航状态-----------")
        api_list_filter.append("calculate_new4_stage_by_time_range")


    if "总发电量" in ques or "总发电" in ques:
        print("333333333333333333333333333")
        api_list_filter.append("calculate_electricity_generation")
        api_list_filter.append("calculate_total_electricity_generation")
    if "理论发电量" in ques:
        api_list_filter.append("calculate_total_4_fuel_volume")
        question = question + "##理论发电量=燃油消耗量*密度*燃油热值/3.6,##"
        ques = remove_parentheses(question)
    if "发电效率" in ques:
        print("444444444444444444444444444")
        api_list_filter.append("calculate_total_4_fuel_volume")
        api_list_filter.append("calculate_total_electricity_generation")
        question = question + "##理论发电量(kWh)=燃油消耗量*密度*燃油热值/3.6，发电效率=四个发电机总发电量/理论发电量。##"
    if "发电机组" in ques and "号" not in ques and "应急发电机组" not in ques:
        question = question + "##发电机组即为一、二、三、四号柴油发电机组。##" 
        ques = remove_parentheses(question)
        api_list_filter.append("calculate_total_4_fuel_volume")
    if "燃油消耗量" in ques:
        # api_list_filter.append("get_table_data")
        api_list_filter.append("calculate_total_fuel_volume")  
        api_list_filter.append("calculate_total_4_fuel_volume")
        if "号" not in ques:
            api_list_filter.remove("calculate_total_fuel_volume")
        # question = question + "##燃油消耗量=这段时间内燃油消耗率之和。##"
    # if "DP" in question:
    #     question = question + "ON DP和OFF DP的仅发生在下午，若查表有下午数据则忽略"
    

    #######################----------数据缺失逻辑
    if "数据缺失" in ques or "数据集缺失" in ques:
        api_list_filter.append("calculate_sum_data")
    

    #次数相关
    if "次数" in ques:
        if "摆动超过10" in ques:
            api_list_filter.append("count_swings_10")
        if "A架完整摆动" in ques:
            api_list_filter.append("count_swings")
        if  "深海作业" in ques:
            api_list_filter.append("calculate_start_end")

    #平均值、最大最小值相关
    if "平均值" in question or "最大值" in question or "最小值" in question :
        # table_name_string = choose_table(question)
        # # print(question)
        # with open("dict.json", "r", encoding="utf-8") as file:
        #     table_data = json.load(file)

        # table_name = [
        #     item for item in table_data if item["数据表名"] in table_name_string
        # ]
        # all_combined_fields = []
        # for tmp in table_name:
        #     # 组合字段名和字段含义
        #     combined = []
        #     for name, meaning in zip(tmp["字段名"], tmp["字段含义"]):
        #         combined.append(f"{name}: {meaning}")
        #     # 创建新字段
        #     tmp["字段名及字段"] = combined
        #     # 构建符合要求的字典结构
        #     all_combined_fields.append({
        #         "数据表名": tmp["数据表名"],
        #         "字段名及字段": combined
        #     })
        # question = str(all_combined_fields)+question
        if is_get_table_data == False:
            is_get_table_data=True
            table_name_string = choose_table(question)
            # print(question)
            with open("dict.json", "r", encoding="utf-8") as file:
                table_data = json.load(file)

            table_name = [
                item for item in table_data if item["数据表名"] in table_name_string
            ]

            all_combined_fields = []
            for tmp in table_name:
                # 组合字段名和字段含义
                combined = []
                for name, meaning in zip(tmp["字段名"], tmp["字段含义"]):
                    combined.append(f"{name}: {meaning}")
                # 创建新字段
                tmp["字段名及字段"] = combined
                # 构建符合要求的字典结构
                all_combined_fields.append({
                    "数据表名": tmp["数据表名"],
                    "字段名及字段": combined
                })
            question = "##"+str(all_combined_fields)+"##"+"\n"+question

        # question = str(table_name)+question
        api_list_filter.append("calculate_max_or_average")


    if "报告" in ques or "记录此时" in ques or "功率" in ques:
        # question = question.replace("功率","有功功率")
        # table_name_string = choose_table(question)
        # print(question)
        # with open("dict.json", "r", encoding="utf-8") as file:
        #     table_data = json.load(file)

        # table_name = [
        #     item for item in table_data if item["数据表名"] in table_name_string
        # ]

        # all_combined_fields = []
        # for tmp in table_name:
        #     # 组合字段名和字段含义
        #     combined = []
        #     for name, meaning in zip(tmp["字段名"], tmp["字段含义"]):
        #         combined.append(f"{name}: {meaning}")
        #     # 创建新字段
        #     tmp["字段名及字段"] = combined
        #     # 构建符合要求的字典结构
        #     all_combined_fields.append({
        #         "数据表名": tmp["数据表名"],
        #         "字段名及字段": combined
        #     })
        # question = str(all_combined_fields)+question+"查询功率值从【有功功率】【功率反馈】里找。"
        if is_get_table_data == False:
            is_get_table_data=True
            table_name_string = choose_table(question)
            # print(question)
            with open("dict.json", "r", encoding="utf-8") as file:
                table_data = json.load(file)

            table_name = [
                item for item in table_data if item["数据表名"] in table_name_string
            ]

            all_combined_fields = []
            for tmp in table_name:
                # 组合字段名和字段含义
                combined = []
                for name, meaning in zip(tmp["字段名"], tmp["字段含义"]):
                    combined.append(f"{name}: {meaning}")
                # 创建新字段
                tmp["字段名及字段"] = combined
                # 构建符合要求的字典结构
                all_combined_fields.append({
                    "数据表名": tmp["数据表名"],
                    "字段名及字段": combined
                })
            question = "##"+str(all_combined_fields)+"##"+"\n"+question+"查询功率值从【有功功率】【功率反馈】里找。"
        api_list_filter.append("get_value_of_time")

    if "A架" in ques and "能耗" in ques:
        # print("api_list_filter:", api_list_filter)
        if "calculate_total_energy" in api_list_filter:
            api_list_filter.remove("calculate_total_energy")
        # api_list_filter.remove("calculate_total_energy")
        # print("api_list_filter:", api_list_filter)

    if "假设2024/08/20" in ques or re.search(r".*进行.*分别是", ques):
        api_list_filter= ["action_judgment"]
        print("shhdhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
        # print("api_list_filter:", api_list_filter)
        question = question + "##若题目中没有指明下放还是回收阶段,则所要查询动作包括缆绳挂妥,A架摆出,征服者出水和征服者落座时,为回收阶段,否则为下放阶段.##"
    
    
    # api_list_filter.append["natural_processing"]
    api_list_filter = list(set(api_list_filter))
    print("question:", question)
    print("ques:",ques)
    print("api_list_filter:")
    print(api_list_filter)

    
    api_list_filter1=glm_filter_apilist(ques,api_list_filter)
    if api_list_filter1 != []:
        api_list_filter = api_list_filter1
    
    # print("after_glm4_api_list_filter:")
    # print(api_list_filter)

    #####
    # 根据问题内容选择相应的 API
    # if "过程" in question:
    #     api_list_filter.append("calculate_uptime")
    #     print(api_list_filter)

    # 过滤工具列表
    filtered_tools = [
        tool
        for tool in tools
        if tool.get("function", {}).get("name") in api_list_filter
    ]
    # 返回结果
    # print(filtered_tools)
    if "content_p_1" in locals():
        return content_p_1, filtered_tools
    else:
        return question, filtered_tools

import json
import re

def glm_filter_apilist(ques, apilist):
    """
    优化版API筛选函数，解决JSON解析问题
    """
    # 步骤1：生成工具描述
    tool_descriptions = "\n".join([
        f"{tool['function']['name']}: {tool['function']['description']}"
        for tool in tools.tools_all if tool['function']['name'] in apilist
    ])
    
    # 步骤2：强化系统提示
    system_prompt = """你是一个智能API选择器。请严格按以下要求操作：
    
    可用工具列表：
    {tool_descriptions}
    
    选择规则：
    1. 必须严格从候选列表中选择
    2. 返回格式必须是纯JSON数组,例如：
     ```json
[
    "api1",
    "api2",
    "api3",
    ...
]
```
    3. 必须去除任何额外字符或注释
    4. 优先包含get_device_status_by_time_range而不是calculate_start_end
    """
    
    # 步骤3：调用大模型
    response = glm4_create_nopython(
        max_attempts=3,
        messages=[
            {"role": "system", "content": system_prompt.format(tool_descriptions=tool_descriptions)},
            {"role": "user", "content": f"问题：{ques}"}
        ],
        tools=[]
    )
    
    # 步骤4：强化解析逻辑
    try:
        # 清理响应内容
        raw_content = response.choices[0].message.content
        # print("原始响应:", raw_content)  # 调试日志
        
        # 去除JSON标记和代码块
        cleaned_content = re.sub(r'```json|```', '', raw_content).strip()
        
        # 解析JSON
        filtered_list = json.loads(cleaned_content)
        
        # 有效性校验
        if not isinstance(filtered_list, list):
            print("非列表格式:", filtered_list)
            return []
            
        # 最终过滤
        valid_apis = [api for api in filtered_list if api in apilist]
        valid_apis = list(set(valid_apis))
        print("有效API:", valid_apis)  # 调试日志
        return valid_apis
        
    except Exception as e:
        print(f"解析异常: {str(e)}")
        return []
def enhanced(prompt1, context=None, instructions=None, modifiers=None):
    """
    增强提示词函数
    """
    prompt1 = str(prompt1)
    print("原始 prompt1:", prompt1)

    

    enhanced_prompt = prompt1  # 初始化为原始字符串
    enhanced_prompt = enhanced_prompt.replace("停泊发电机", "停泊/应急发电机")
    # enhanced_prompt = enhanced_prompt.replace("5月20日", "2024年5月20日")
    enhanced_prompt = enhanced_prompt.replace("XX小时XX分钟", "XX小时XX分钟，01小时01分钟格式,不需要秒")
    enhanced_prompt = enhanced_prompt.replace("请以XX:XX输出", "请以XX:XX输出，01:01的格式,不需要秒")
    enhanced_prompt = enhanced_prompt.replace("发生了什么", "什么设备在进行什么动作，动作直接引用不要修改,如【A架摆回】")
    # enhanced_prompt = enhanced_prompt.replace("深海作业A回收过程中", "最后一次")
    enhanced_prompt = enhanced_prompt.replace("深海作业A作业", "作业")
    # enhanced_prompt = enhanced_prompt.replace("深海作业A作业结束", "OFF DP")
    enhanced_prompt = enhanced_prompt.replace("作业开始", "作业开始（第一次'布放开始'）")
    enhanced_prompt = enhanced_prompt.replace("作业结束", "作业结束（最后一次'回收结束'）")
    enhanced_prompt = enhanced_prompt.replace("开始作业", "作业开始（第一次'布放开始'）")
    enhanced_prompt = enhanced_prompt.replace("结束作业", "作业结束（最后一次'回收结束'）")
    enhanced_prompt = enhanced_prompt.replace("小艇下水", "小艇入水")
    # enhanced_prompt = enhanced_prompt.replace("深海作业A的次数", "深海作业A的次数##一天至少两次A架开机和两次A架关机且存在其他动作才可视为作业进行##")
    # enhanced_prompt = enhanced_prompt.replace("作业", "作业##作业由下放阶段以及回收阶段两个阶段组成。一般来说第一次A架开关机为布放阶段，第二次A架开关机为回收阶段，但若题目中指明布放和回收的标志，则以题目为准。若题目中没有说明阶段则默认两个阶段。##")
    
    # enhanced_prompt = enhanced_prompt.replace("下放阶段以ON DP以及OFF DP为标志，回收阶段以A架开机以及关机为标志", "")
    enhanced_prompt = enhanced_prompt.replace("是否正确", "是否正确（回答正确或不正确）")
    # enhanced_prompt = enhanced_prompt.replace("期间", "期间（若为DP，仅计算第一次开关）")
    enhanced_prompt = enhanced_prompt.replace("（含）", "（含）这几天中")
    # enhanced_prompt = enhanced_prompt.replace("推进器", "推进器（推进器包括：一号推进变频器+二号推进变频器+侧推+可伸缩推）")
    enhanced_prompt = enhanced_prompt.replace("甲板机械", "甲板机械##甲板机械包括：折臂吊车、一号门架、二号门架(一号门架+二号门架统称为A架)、绞车##")
    enhanced_prompt = enhanced_prompt.replace("甲板机器", "甲板机械##甲板机械包括：折臂吊车、一号门架、二号门架(一号门架+二号门架统称为A架)、绞车##")
    enhanced_prompt = enhanced_prompt.replace("做功", "能耗")
    enhanced_prompt = enhanced_prompt.replace("时间差是多少", "时间差是多少分钟")
    enhanced_prompt = enhanced_prompt.replace("与温度相关", "与温度相关（单位为℃）")
    enhanced_prompt = enhanced_prompt.replace("与压力相关", "与压力相关（单位为kPa）")
    # enhanced_prompt = enhanced_prompt.replace("动力定位", "DP")
    enhanced_prompt = enhanced_prompt.replace("24:00","24:00(次日0:00)")
    enhanced_prompt = enhanced_prompt.replace("多个时间使用", "多个答案使用")
    enhanced_prompt = enhanced_prompt.replace("范围", "范围（上下限）")


    enhanced_prompt = enhanced_prompt.replace("1~4号","一至四号")
    enhanced_prompt = enhanced_prompt.replace("1至4号","一至四号")
    def replace_number(match):
        num_to_cn = {
            '1': '一', '2': '二', '3': '三', '4': '四'
        }
        num = match.group(1)
        return num_to_cn[num] + '号'
    enhanced_prompt = re.sub(r'(\d)号', replace_number, enhanced_prompt)

    if "期间" in enhanced_prompt and "DP" not in enhanced_prompt:
        enhanced_prompt=enhanced_prompt.replace("期间", "这段时间内")
    if "DP过程" in enhanced_prompt or "DP期间" in enhanced_prompt:
        enhanced_prompt=enhanced_prompt+"仅在第一次DP开关的时间段内计算"

    if "最晚的" in enhanced_prompt:
        enhanced_prompt=enhanced_prompt.replace("最晚的","(在当天中)最晚的")
    if "最早的" in enhanced_prompt:
        enhanced_prompt=enhanced_prompt.replace("最早的","(在当天中)最早的")
    

    if "主推进器" in enhanced_prompt and "号主推进器" not in enhanced_prompt:
        enhanced_prompt = enhanced_prompt.replace("主推进器","主推进器（主推进器包括一号推进变频器和二号推进变频器）")
    
    if "以折臂吊车开机为开始" in enhanced_prompt:
        enhanced_prompt = remove_parentheses(enhanced_prompt)
    if "2024" not in enhanced_prompt and re.search(r'月.*日', enhanced_prompt):
        enhanced_prompt= "2024年" + enhanced_prompt

    if re.search(r'(\d{1}:\d{2})时', enhanced_prompt):
        # 单次正则替换完成需求
        enhanced_prompt = re.sub(
            r'(\d{1}:\d{2})时',    # 模式：匹配"HH:MM时" 
            r'\1这一分钟内',        # 替换：保留时间并追加描述
            enhanced_prompt
        )
    
    action_list=["A架开机","ON DP","征服者起吊","征服者入水","缆绳解除","A架摆回","小艇落座","A架关机","OFF DP","折臂吊车开机","A架摆出","小艇检查完毕","小艇入水","缆绳挂妥","征服者出水","折臂吊车关机","征服者落座"]
    for value in action_list:
        if value in enhanced_prompt:
            enhanced_prompt=enhanced_prompt.replace(value,"【"+value+"】")

    
    #     enhanced_prompt = re.sub(r"深海.*之间", "", enhanced_prompt)
    #     # print(enhanced_prompt)

    # if re.search(r'A架.*阶段', enhanced_prompt):
    #     # enhanced_prompt = remove_parentheses(enhanced_prompt)
    #     enhanced_prompt = re.sub(r"A架.*阶段", "", enhanced_prompt)
    #     # print(enhanced_prompt)
    

    if "电流不平衡度" in enhanced_prompt:
        enhanced_prompt = enhanced_prompt + "###最终答案为所查到数值/100,例如：1. 若查到数值为78，则实际为0.78，返回整数则为1%。2. 若查到数据为4466%，则实际为44.66%,返回整数则为45%###"
    if "一号舵桨转舵A-Ua电压" in enhanced_prompt:
        enhanced_prompt = enhanced_prompt + "###最终答案为所查到数值/10,例如：若查到数值为8909，则实际为890.9，返回整数则为891.###"
    if "一号舵桨转舵A-频率" in enhanced_prompt:
        enhanced_prompt = enhanced_prompt + "###最终答案为所查到数值/100,例如：若查到数值为8909，则实际为89.09，返回整数则为89.###"
    # enhanced_prompt = enhanced_prompt.replace("电流不平衡度", "电流不平衡度(单位:%,实际值为数值/100,如391=3.91%)")
    # enhanced_prompt = enhanced_prompt.replace("一号舵桨转舵A-Ua电压", "一号舵桨转舵A-Ua电压(单位:V,实际值为数值/10，如3911=391.1V)")
    # enhanced_prompt = enhanced_prompt.replace("一号舵桨转舵A-频率", "一号舵桨转舵A-频率(单位:Hz,实际值为数值/100,如3911=39.11Hz)")

    # if "和" in enhanced_prompt:
    #     enhanced_prompt = enhanced_prompt.replace("和", "以及（若为时间，两者之间不要计算）")

    
    # enhanced_prompt += "请直接给出答案。"

    print("修改后 enhanced_prompt:", enhanced_prompt)
    
    
    return enhanced_prompt



def run_conversation_xietong(question):
    question = enhanced(question)
    # print(question)
    content_p_1, filtered_tool = select_api_based_on_question(
        question, tools.tools_all
    )  # 传入 question
    # print('---------------------')
    # print(content_p_1,filtered_tool)
    answer, select_result = get_answer_2(
        question=content_p_1, tools=filtered_tool, api_look=False
    )
    return answer


def get_answer(question):
    try:
        print(f"Attempting to answer the question: {question}")
        last_answer = run_conversation_xietong(question)
        print(f"Last answer: {last_answer}")
        last_answer = last_answer.replace(" ", "")
        return last_answer
    except Exception as e:
        print(f"Error occurred while executing get_answer: {e}")
        return "An error occurred while retrieving the answer."


def review_answer_test(answer):
    """
    验证并修正答案中的计算过程
    
    :param answer: 包含计算过程的原始答案文本
    :return: 修正后的答案文本（若无错误返回原答案）
    """
    # 构造验证提示词
    system_prompt = """您是一位严谨检查计算问题的专家，仅需要检查计算结果是否正确，请严格按以下要求工作：
1. 仔细检查用户提供的计算过程,如加减乘除极其混合运算,特别是计算时的进位与借位，时间点的比较
2. 发现计算错误时，指出错误位置并给出正确计算
3. 若没有错误，直接返回输入
4. 若没有计算步骤,则当作正确答案处理
5. 若出现AA:BB，视为AA小时BB分钟
6. 一个数字中不要加分隔符（如用逗号分隔千位等），保持数字的完整性

禁止内容：
1. 禁止检查数值是否合理，如禁止以下推理行为：认为方位角的范围应在0%到100%之间（或0°到360°），-1796%显然超出了这个范围，是一个不合理的结果。
2. 禁止添加额外信息，禁止自己联想或续写
"""

    user_prompt = f"""请检查以下计算过程，请逐步思考，按格式要求响应：
[原始答案]
{answer}

[检查要求]
1. 若发现计算错误，则将原答案中错误的结果，替换为正确的，保留原格式
2. 若无错误，返回{answer}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # 调用GLM4进行验证
    try:
        response = glm4_create(
            max_attempts=3,
            messages=messages,
            tools=[],
            model="glm-4-plus"
        )
        
        # feedback = response.choices[0].message.content.strip()

        feedback = response.choices[0].message.content
        # print(f"GLM4验证结果：{feedback}")
        print(f"GLM4验证结果：{feedback}")
        # print("ok")

        return feedback
    #     # 解析验证结果
    #     if "修正为：" in feedback:
    #         # 提取修正后的正确计算
    #         corrected_part = feedback.split("修正为：")[-1].strip()
    #         return f"[修正结果] {corrected_part}"
    #     elif "计算过程正确" in feedback:
    #         return answer
    #     else:
    #         return f"[验证未通过] 原始答案：{answer}"
            
    except Exception as e:
        print(f"验证过程中发生错误：{str(e)}")
        return answer  # 降级返回原始答案


#增强版时间解析
def parse_time(input_str: str, base_date: datetime = None) -> datetime:
    """增强版时间解析，支持多种格式"""
    # 按优先级尝试的格式列表
    formats = [
        ("%Y-%m-%d %H:%M:%S", 19),    # 完整日期时间带秒
        ("%Y-%m-%d %H:%M", 16),       # 完整日期时间
        ("%H:%M:%S", 8),              # 纯时间带秒
        ("%H:%M", 5)                  # 纯时间
    ]

    # 自动匹配最合适的格式
    for fmt, fmt_length in formats:
        if len(input_str) >= fmt_length:
            try:
                dt = datetime.strptime(input_str[:fmt_length], fmt)
                if base_date and fmt in ("%H:%M", "%H:%M:%S"):
                    return datetime.combine(base_date.date(), dt.time())
                return dt
            except ValueError:
                continue

    raise ValueError(f"无法解析的时间格式: {input_str}")

from datetime import datetime, timedelta
# 时间计算工具函数
def calculate_minutes(start_str: str, end_str: str) -> int:
    """计算两个时间点之间的分钟差，支持跨午夜和不同日期格式"""
    
    
    start_time = parse_time(start_str)
    end_time = parse_time(end_str, start_time)
    
    # if end_time < start_time:
    #     end_time += timedelta(days=1)
    if end_time < start_time:
        end_time ,start_time = start_time,end_time
        
    return int((end_time - start_time).total_seconds() // 60)

#数学验证工具调用入口
def validate_math_expression(expression: str) -> dict:
    """增强结果返回"""
    validator = MathValidator()
    try:
        expr = validator._sanitize_expression(expression)
        if not validator._validate_syntax(expr):
            return {"result": "非法字符", "steps": []}
            
        tokens = validator._tokenize(expr)
        postfix = validator._shunting_yard(tokens)
        result = validator._evaluate_postfix(postfix)
        return {
            "result": round(result, 4),
            "steps": validator._step_details
        }
    except Exception as e:
        return {
            "result": f"计算错误: {str(e)}",
            "steps": validator._step_details
        }

# def handle_tool_calls(tool_calls):
#     """处理工具调用（增加参数传递）"""
#     results = []
#     if tool_calls == None:
#         return []
#     for call in tool_calls:
#         try:
#             import json
#             args = json.loads(call.function.arguments)
#             tool_info = {
#                 "tool_call_id": call.id,
#                 "function": call.function.name,
#                 "args": args,  # 保留原始参数结构
#                 "output": None
#             }

#             if call.function.name == "calculate_minutes":
#                 # 时间计算工具
#                 result = calculate_minutes(args["start_str"], args["end_str"])
#                 tool_info["output"] = str(result)
#                 tool_info["args_display"] = f"{args['start_str']} 至 {args['end_str']}"  # 可读参数
                
#             elif call.function.name == "validate_math_expression":
#                 # 数学验证工具
#                 validation_result = validate_math_expression(args["expression"])
#                 tool_info["output"] = f"结果：{validation_result['result']}，步骤数：{len(validation_result['steps'])}"
#                 tool_info["args_display"] = f"表达式：{args['expression']}"  # 表达式参数
                
#             results.append(tool_info)
            
#         except Exception as e:
#             print(f"工具调用异常：{str(e)}")
#             continue
            
#     return results

def handle_tool_calls(tool_calls):
    """处理工具调用（增加参数传递）"""
    results = []
    if not tool_calls:
        return []
    for call in tool_calls:
        try:
            # 解析参数时增加容错
            import json
            args_str = call.function.arguments
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                print(f"参数解析失败，原始参数：{args_str}")
                continue

            tool_info = {
                "tool_call_id": call.id,
                "function": call.function.name,
                "args": args,  # 保留原始参数结构
                "output": None,
                "args_display": ""
            }

            if call.function.name == "calculate_minutes":
                # 验证必要参数是否存在
                if "start_str" not in args or "end_str" not in args:
                    print(f"缺少时间参数：{args}")
                    continue
                    
                result = calculate_minutes(args["start_str"], args["end_str"])
                tool_info["output"] = str(result)
                tool_info["args_display"] = f"{args['start_str']} 至 {args['end_str']}"

            # elif call.function.name == "validate_math_expression":
            #     # 验证必要参数是否存在
            #     if "expression" not in args:
            #         print(f"缺少表达式参数：{args}")
            #         continue
                    
            #     validation_result = validate_math_expression(args["expression"])
            #     tool_info["output"] = f"结果：{validation_result['result']}，步骤数：{len(validation_result['steps'])}"
            #     tool_info["args_display"] = f"表达式：{args['expression']}"
            elif call.function.name == "validate_math_expression":
                validation_result = validate_math_expression(args["expression"])
                
                # 构建详细步骤说明
                steps_str = "\n".join(
                    [f"步骤{step['step']}: {step['operation']} = {step['result']}"
                    for step in validation_result['steps']]
                ) if validation_result['steps'] else "无详细步骤"
                
                tool_info["output"] = f"""验证结果：{validation_result['result']}
        {steps_str}"""
                tool_info["args_display"] = f"表达式：{args['expression']}"

            results.append(tool_info)
            
        except Exception as e:
            print(f"工具调用异常：{str(e)}")
            continue
            
    return results

def review_answer_test2(ques,answer):
    """
    强化版计算检查函数，仅验证算术过程
    """
    # 构造更严格的提示词
    system_prompt = """您是一位计算步骤检查机器人，请严格按以下规则工作：
    
<任务要求>
1. 只检查加减乘除等基本运算是否正确（如：12 * 5 + 6 = 66）
2. 仅验证数学计算步骤，禁止检查数值合理性
3. 若出现AA:BB的格式，则将其视为AA小时BB分钟（例如：将08:30视为8小时30分钟，禁止视为8分钟30秒）
4. 保留原答案格式，仅修正计算错误
5. 能多步合并则合并。禁止分步骤，禁止输出中间结果，例如2*2=4，4*10=40，40/1=40
6. 一个数字中不要加分隔符（如用逗号分隔千位等），保持数字的完整性,例如“168,825.51kWh”应该改为“168825.51kWh”


<禁止行为>
- 禁止判断结果是否合理（如：-1796%是否可能）
- 禁止检查数值范围（如：百分比是否超过100%）
- 禁止任何逻辑推理（如："角度应该小于360°"）
- 禁止添加额外信息，禁止自己联想或续写

<优先行为>
1. 优先使用工具验证计算（时间计算工具、数学验证工具）
"""

    question=ques
    user_prompt = f"""对于问题{question}\n请检查以下计算步骤，若发现算术错误则修正结果，否则返回原文，禁止添加额外信息，禁止自己联想或续写：
{answer}

<响应格式>
[步骤检查]
若正确 -> "计算步骤正确"
若错误 -> "错误步骤：[原步骤] → [修正步骤]"

<工具使用>
若使用工具，则优先生成工具所需要的参数，如：若要检查时间差，则生成开始时间和结束时间；若要检查数学表达式则提取数学表达式;能多步合并则合并。禁止分步骤，禁止输出中间结果
-示例：禁止分步骤
    若题中出现：-总燃油质量：16806.88L*0.8448kg/L=14236.30kg，总能量：14236.30kg*42.6MJ/kg=606855.82MJ，理论发电量（保留两位小数）：606855.82MJ/3.6=168647.84kWh
    则表达式应直接写为：理论发电量（保留两位小数）16806.88L*0.8448kg/L*42.6MJ/kg/3.6
若无需工具 → 直接返回修正后的答案




[最终答案]
保留原格式输出修正后的完整答案
"""

    # 双工具定义
    tools_config = [
        {  # 时间计算工具
            "type": "function",
            "function": {
                "name": "calculate_minutes",
                "description": "计算时间差（分钟）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_str": {"type": "string", "description": "开始时间（HH:mm 或完整日期）"},
                        "end_str": {"type": "string", "description": "结束时间（HH:mm 或完整日期）"}
                    },
                    "required": ["start_str", "end_str"]
                }
            }
        },
        {  # 数学验证工具
            "type": "function",
            "function": {
                "name": "validate_math_expression",
                "description": "验证数学表达式计算步骤",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "数学表达式，如：3+5 * 2"}
                    },
                    "required": ["expression"]
                }
            }
        }
    ]

    try:
        response = glm4_create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tools=tools_config,
            max_attempts=3
        )

        if response is not None and response.choices:
            message = response.choices[0].message
        else:
            return answer

        # 安全获取 tool_calls
        tool_calls = getattr(message, 'tool_calls', [])
        
        # 关键修复：传递正确的 tool_calls 变量
        if tool_calls:
            tool_outputs = handle_tool_calls(tool_calls)
            
            corrected_parts = []
            for output in tool_outputs:
                if output['function'] == "calculate_minutes":
                    corrected_parts.append(f"时间差计算：{output['args_display']} → {output['output']}分钟")
                elif output['function'] == "validate_math_expression":
                    corrected_parts.append(f"表达式验证：{output['output']}")

            # 如果有有效工具结果则优先返回
            if corrected_parts:
                print("after",corrected_parts)
                return "\n".join(corrected_parts)

        # 无工具调用时处理模型响应
        if hasattr(message, 'content'):
            return message.content if "错误步骤" in message.content else answer
        else:
            return answer

    except Exception as e:
        print(f"处理异常：{str(e)}")
        return answer


def review_answer(answer):
    """
    强化版计算检查函数，仅验证算术过程
    """
    # 构造更严格的提示词
    system_prompt = """您是一位计算步骤检查机器人，请严格按以下规则工作：
    
<任务要求>
1. 只检查加减乘除等基本运算是否正确（如：12 * 5 + 6 = 66）
2. 仅验证数学计算步骤，禁止检查数值合理性
3. 若出现AA:BB的格式，则将其视为AA小时BB分钟（例如：将08:30视为8小时30分钟，禁止视为8分钟30秒）
4. 保留原答案格式，仅修正计算错误
5. 一个数字中不要加分隔符（如用逗号分隔千位等），保持数字的完整性,例如“168,825.51kWh”应该改为“168825.51kWh”



<禁止行为>
- 禁止判断结果是否合理（如：-1796%是否可能）
- 禁止检查数值范围（如：百分比是否超过100%）
- 禁止任何逻辑推理（如："角度应该小于360°"）
- 禁止添加额外信息，禁止自己联想或续写
"""

    user_prompt = f"""请检查以下计算步骤，若发现算术错误则修正结果，否则返回原文，禁止添加额外信息，禁止自己联想或续写：
{answer}

<响应格式>
[步骤检查]
若正确 -> "计算步骤正确"
若错误 -> "错误步骤：[原步骤] → [修正步骤]"

[最终答案]
保留原格式输出修正后的完整答案"""

    # 调用大模型（添加强制格式约束）
    response = glm4_create(
        max_attempts=3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        tools=[],
    )
    
    # 解析逻辑优化
    try:
        feedback = response.choices[0].message.content
        print(f"GLM4验证结果：{feedback}")
        
        if "错误步骤" not in feedback:
            return answer
        
        else:
            return feedback
    except:
        return answer

def remove_parentheses_2(s):
    # 删除中文括号及其内容
    s = re.sub(r'###.*?###', '', s)
    # 删除英文括号及其内容
    return s

def get_end_answer(question, answer):
    """
    根据问题总结答案内容，返回精简的回答
    
    :param question: 用户提出的问题
    :param answer: 需要总结的原始答案
    :return: 结构化整理后的最终答案
    """

    question=remove_parentheses_2(question)
    # print("get")
    # answer = answer + "\n"+ review_answer(answer)
    # answer = review_answer(question,answer)
    # answer_aftermath=review_answer(answer)
    answer_aftermath=review_answer_test2(question,answer)




    # print("start&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" + answer)
    # print("end&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    # answer = "首先计算两天的总能耗：总能耗 = 19923.6 + 10135.73 + 7795.5 + 5703.72 = 53558.55 kWh,然后计算平均能耗：平均能耗 = 总能耗 / 天数 = 53558.55 / 2 = 26779.275 kWh,保留两位小数，最终的平均作业能耗为：26779.28 kWh"
    # answer = answer + "\n"+ review_answer(answer)
    # print(f"Answer after math teacher: {answer}")
    # 构造对话消息
    action_list=["A架开机","ON DP","征服者起吊","征服者入水","缆绳解除","A架摆回","小艇落座","A架关机","OFF DP","折臂吊车开机","A架摆出","小艇检查完毕","小艇入水","缆绳挂妥","征服者出水","折臂吊车关机","征服者落座"]
    
    prompt = """【内容总结规范】
你是一个专业的内容总结助手，请根据用户的问题总结提供的答案内容以及数学老师提供的计算结果，直接返回能够准确回答问题和满足格式要求的最精简答案，且必须严格遵循以下规则：

1. 动作保留原则
- 若题目中要求回答动作，必须严格使用以下标准的动作名称：
["A架开机","ON DP","征服者起吊","征服者入水","缆绳解除","A架摆回","小艇落座","A架关机","OFF DP","折臂吊车开机","A架摆出","小艇检查完毕","小艇入水","缆绳挂妥","征服者出水","折臂吊车关机","征服者落座"]

❌ ​禁止行为：
不得替换近义词（如"开机"→"启动"）
不得改变格式（如"ONDP"→"ON DP"）
不得增减文字（如"A架关机"→"A架已关机"）
不得回答上述列表之外的动作（如："小船检查完毕"，"起吊"等）


2. 时间格式规范
- 时间戳统一去除秒级单位：
  示例：10:20:08 → 10:20

3. 按照题目要求保留单位,如40分钟，5342.67kWh，46Hz，698V等

4. 格式严格匹配
- 当题目注明格式要求时：
  ▫ 完全按照题目中注明的格式进行输出
  ▫ 示例：若要求"用空格分隔"，则输出"数据1 数据2"、若两个动作都没发生则输出"N N"等
  ▫ 示例：若要求"四舍五入"，则应该将数值四舍五入后输出，示例：697.53四舍五入后为698
- 若题目中无要求时：使用逗号分隔多个答案，示例："数据1,数据2"
- 回答关于是否类问题如果没有明确要求回答格式，使用Y(是)或N(否)
- 回答‘用了多长时间’问题时，统一使用整数分钟为单位
- 禁止使用代码块格式（如 ```json），若题目中有类似json的输出要求，则直接生成符合JSON规范的参数

5. 若参数相同，计算结果与数学老师给的结果不同时，优先考虑数学老师的结果，例如对于3：34到8：40的时长计算，若数学老师也计算了这段时长，则优先考虑数学老师给的结果

6. 严厉禁止行为
- 禁止添加任何假设性或推测性内容
- 一个数字中不要加分隔符（如用逗号分隔千位等），保持数字的完整性,例如“168,825.51kWh”应该改为“168825.51kWh”
- 禁止保留秒级时间数据
- 禁止单位与数值之间保留空格，如"435.65 kWh"
- 禁止将一个动作误认为是另一个，例如将【小艇】的动作视为【征服者】的动作，如需要查征服者落座，却查成了小艇落座，小艇动作和征服者动作没有关系
- 禁止添加额外信息，禁止自己联想或续写
""" 
    messages = [
        {
            "role": "system",
            "content": prompt,
            # "content": "你是一个专业的内容总结助手，请根据用户的问题总结提供的答案内容，直接返回能够准确回答问题和满足格式要求的精简答案，不要多余的内容,注意:1.动作直接引用不要修改,如【A架摆回】;2.需要带单位的必须带单位，如分钟，kWh等;3.若题目中有答案格式要求，请严格按照要求，多个答案以题目中所给格式隔开，若题中未注明则以逗号隔开；4.禁止返回假设内容；5.秒及数据直接舍弃，如10:20:08，直接替换为10:20"
        },
        {
            "role": "user",
            "content": f"请根据以下问题总结答案：{question}\n数学老师返回结果：{answer_aftermath}\n原始答案内容：{answer}"
        }
    ]
    # 调用GLM4接口（不启用工具）
    response = glm4_create(
        max_attempts=3,
        messages=messages,
        tools=[],
        model="glm-4-plus"
    )
    # 解析响应结果
    last_answer = response.choices[0].message.content.strip()
    print("总结前：",answer,"数学老师返回:",answer_aftermath)
    print("question:",question,"总结后：",last_answer)
    
    # last_answer = last_answer.replace(" ", "")
    try:
        if "空格" in question:
            return last_answer
        elif "details" in last_answer:
            return last_answer
        else:
            last_answer=last_answer.replace(" ",",")
            last_answer=last_answer.replace(",,",",")
            return last_answer
    except AttributeError:
        return "答案生成失败，请稍后再试"

# In[7]:


if __name__ == "__main__":
   
    # question = "假设2024/08/20 上午（0:00~12:00）征服者入水的时间是09:59，请指出小艇入水以及征服者起吊发生的时间（以XX:XX输出，如有多个请使用英文逗号隔开）。"
    # question = "1. 2024年5月23日23:00至24:00（包括这两个时间点）进行了一次特定类型的作业，请指出该次作业使用的甲板机器设备有哪些？（多个设备使用空格隔开，设备名称大小写敏感，且空格重要；每个设备仅写一次；如果没有使用任何设备，则输出N）"
    # question = "2. 请指出2024年5月25日深海作业A回收阶段中，A架摆出、征服者起吊和征服者落座的确切时间（以XX:XX格式输出，时间为24小时制，时间补零，如05:03；多个时间使用单个空格隔开，按照事件顺序输出；如果某事件未发生，则在该位置输出N）"


    # question = "14. 在2024年8月25日深海作业A过程中，从A架摆回至A架摆出之间相差的时长（根据设备记录的时间计算，四舍五入到整数，单位为分钟），以及在这段时间内四台发电机的总发电量（单位为kWh，保留两位小数）。如果遇到数据缺失或异常情况，请根据相邻数据点进行合理估算。"
    
    # question = "18. 在2024年6月15日深海作业A中，请输出布放和回收的开始和结束时间（以折臂吊车开机为开始，折臂吊车关机为结束，时间以24小时制HH:MM格式按发生顺序输出），并比较布放和回收阶段的燃油消耗量，输出较大的燃油消耗量（燃油消耗量单位为L，结果保留两位小数）。如果遇到数据缺失或异常情况，请根据相邻数据点进行合理估算。如果当天进行多次布放和回收，请比较所有布放和回收阶段的燃油消耗量，并输出消耗量最大的一次。"
    # question = "19. 在2024年8月24日深海作业A中，请输出布放和回收的开始和结束时间（以折臂吊车开机为开始，折臂吊车关机为结束，时间以24小时制HH:MM格式按发生顺序输出），并比较布放和回收阶段的燃油消耗量，输出较大的燃油消耗量（燃油消耗量单位为L，结果保留两位小数）。如果遇到数据缺失或异常情况，请根据相邻数据点进行合理估算。如果当天进行多次布放和回收，请比较所有布放和回收阶段的燃油消耗量，并输出消耗量最大的一次。"
    
    # question = "2024/05/17 一号舵桨转舵A-频率的平均值是多少（单位为Hz，四舍五入，以整数输出）？"
    # question = "5月20日征服者入水到A架摆回用了多少时间？"
    # question = "2024年5月19日，A 架的开机时间分别是几点？关机时间分别是几点？（请按 HH:MM 格式，24 小时制，以逗号分隔多个时间点）计算该日期内 A 架的总运行时间（单位：整数分钟）。在这段时间内，甲板机械系统总做功为多少（kWh，保留两位小数）"
    # question = "请统计2024年6月4日处于动力定位的时长（单位：分钟，如果没有则计为0），以及动力定位时一号、二号、三号和四号柴油发电机的运行时长（单位：分钟）。同时，请计算动力定位时一号、二号主推进器的功率最大值（单位：kW），艏侧推的功率最大值（单位：kW），以及可伸缩推的运行时长（单位：分钟）。如果某项数据缺失，请在相应字段输出\"nil\"。请按照以下格式输出结果：\n{\n    \"details\": [\n        {\n            \"date\": \"MM/DD\",\n            \"DP_time\": 分钟数,\n            \"DG1_running\": 分钟数或\"nil\",\n            \"DG2_running\": 分钟数或\"nil\",\n            \"DG3_running\": 分钟数或\"nil\",\n            \"DG4_running\": 分钟数或\"nil\",\n            \"power_AZ_1\": 功率值或\"nil\",\n            \"power_AZ_2\": 功率值或\"nil\",\n            \"BT_power\": 功率值或\"nil\",\n            \"BAZ_running\": 分钟数或\"nil\"\n        }\n        // ... 其他日期的数据\n    ]\n}"
    # question = "停泊发电机组转速为1800RPM会发生什么？"
    # question = "2024/08/23 深海作业A过程中，A架摆回到A架摆出期间，推进器的总做功是多少（单位为kWh，结果保留两位小数）？"

######################################################--------------------------------------------------------------------------------------------------------------------------------------------------------------------------#######################################################################################################################################
    # question = "1. 2024年5月23日23:00至24:00（包括这两个时间点）进行了一次特定类型的作业，请指出该次作业使用的甲板机器设备有哪些？（多个设备使用空格隔开，设备名称大小写敏感，且空格重要；每个设备仅写一次；如果没有使用任何设备，则输出N）"
    question = "2. 请指出2024年5月25日深海作业A回收阶段中，A架摆出、征服者起吊和征服者落座的确切时间（以XX:XX格式输出，时间为24小时制，时间补零，如05:03；多个时间使用单个空格隔开，按照事件顺序输出；如果某事件未发生，则在该位置输出N）"
    question = "3. 请指出2024年7月19日征服者落座以及征服者出水的时间（按发生顺序以XX:XX格式输出，时间为24小时制，时间补零，如05:03；多个时间使用单个空格隔开；如果某事件未发生，则在该位置输出N）"
    question = "4. 请统计2024年6月1日至6月30日期间，深海作业A的次数（一个完整的布放过程加一个完整的回收过程算作一次作业；如果存在只有布放或只有回收的情况，不计算在内）"
    # question = "5. 请指出2024年6月7日征服者落座和小艇入水的时间（按发生顺序以XX:XX格式输出，时间为24小时制，时间补零，如05:03；多个时间使用单个空格隔开；如果某事件未发生，则在该位置输出N）"
    # question = "6. 2024年5月23日，从征服者落座开始至当日作业结束，请计算A架右舷摆动的次数。（以整数输出，A架摆动的定义如下：A架右舷同一方向上摆动超过10°即可算作一次摆动）"
    question = "7. 请统计2024年5月1日至5月31日期间，深海作业A的次数（一个完整的布放过程加一个完整的回收过程算作一次作业；如果存在只有布放或只有回收的情况，不计算在内）"
    # question = "8. 请指出2024年6月14日小艇落座和征服者落座的时间（按发生顺序以XX:XX格式输出，时间为24小时制，时间补零，如05:03；多个时间使用单个空格隔开；如果某事件未发生，则在该位置输出N）"
    question = "9. 请指出2024年7月19日绞车A的放缆和收缆次数（放缆是指绞车A向海中释放电缆，收缆是指绞车A从海中回收电缆；放缆长度同一方向上单次变化超过5m则算一次放缆，收缆同理）"
    # question = "10. 请指出2024年6月15日小艇检查完毕的时间和小艇入水的时间（按发生顺序以XX:XX格式输出，时间为24小时制，时间补零，如05:03；多个时间使用单个空格隔开；如果某事件未发生，则在该位置输出N；注意，每个事件在一天内可能发生多次）"
    # question = "11. 假设柴油的密度为0.85kg/L，柴油热值为42.6MJ/kg。请根据提供的1~4号柴油发电机的燃油消耗量，计算2024年7月16日00:00:00至2024年7月20日00:00:00期间的理论发电量（单位转换为kWh，结果保留两位小数，四舍五入）"
    # question = "12. 根据提供的2024年6月10日00:00:00至2024年6月15日00:00:00期间甲板机械和推进器的能耗数据以及发电机的总发电数据（数据来源于作业日志和传感器记录），计算甲板机械和推进器的能耗之和占发电机总发电量的比例（结果以百分比表示，保留两位小数，四舍五入；注意，能耗和发电量数据均已转换为同一单位kWh）"
    # question = "13. 2024/06/10 00:00:00~2024/06/15 00:00:00 甲板机械和舵桨的能耗之和占发电机的总发电的比例（%，结果保留两位小数）？"
    # question = "14. 在2024年8月25日深海作业A过程中，从A架摆回至A架摆出之间相差的时长（根据设备记录的时间计算，四舍五入到整数，单位为分钟），以及在这段时间内四台发电机的总发电量（单位为kWh，保留两位小数）。如果遇到数据缺失或异常情况，请根据相邻数据点进行合理估算"
    # question = "15. 在2024年6月6日，请输出征服者起吊和征服者入水的时间（以XX:XX格式输出，时间为24小时制，时间补零，如05:03），以及在这段时间内推进器的总做功（单位为kWh，结果保留两位小数）"
    # question = "16. 在2024年6月1日深海作业A过程中，请输出两次小艇落座的时间（以XX:XX格式输出，时间为24小时制，时间补零，如05:03），以及在这两次落座期间四台发电机的总发电量（数据来源于传感器记录，单位为kWh，结果保留两位小数）。如果遇到数据缺失或异常情况，请根据相邻数据点进行合理估算。如果当天发生多次小艇落座，请仅考虑第一次和最后一次落座"
    question = "17. 在2024年8月24日深海作业A中，请输出布放和回收的开始和结束时间（以折臂吊车开机为开始，折臂吊车关机为结束，时间以24小时制HH:MM格式按发生顺序输出），并比较布放和回收阶段的燃油消耗量，输出较大的燃油消耗量（燃油消耗量单位为L，结果保留两位小数）。如果遇到数据缺失或异常情况，请根据相邻数据点进行合理估算。如果当天进行多次布放和回收，请比较所有布放和回收阶段的燃油消耗量，并输出消耗量最大的一次"
    question = "18. 在2024年6月15日深海作业A中，请输出布放和回收的开始和结束时间（以折臂吊车开机为开始，折臂吊车关机为结束，时间以24小时制HH:MM格式按发生顺序输出），并比较布放和回收阶段的燃油消耗量，输出较大的燃油消耗量（燃油消耗量单位为L，结果保留两位小数）。如果遇到数据缺失或异常情况，请根据相邻数据点进行合理估算。如果当天进行多次布放和回收，请比较所有布放和回收阶段的燃油消耗量，并输出消耗量最大的一次"
    question = "19. 2024年5月17日00:00:00至2024年5月20日00:00:00期间，推进器的总能耗为多少？（单位为kWh，结果保留两位小数，采用四舍五入规则）"
    # question = "20. 2024年5月22日00:00:00至2024年5月25日00:00:00期间，甲板机械的总能耗为多少？（单位为kWh，结果保留两位小数，采用四舍五入规则）"
    question = "21. 在2024/06/01 00:00:00至2024/06/05 00:00:00期间，请计算1~4号柴油发电机的运行时间（运行时间定义为发电机在额定转速下的运行时间，额定转速运行值为1表示发电机运行了1分钟。请输出每个发电机的运行时间，按编号顺序输出，时间按整数分钟输出，用空格隔开）"
    # question = "22. 在2024年8月24日，A架摆回和摆出的时间相隔多久（以整数分钟输出）？在这段时间内，4台柴油发电机的总发电量为多少？（kWh，保留两位小数）"
    # question = "23. 在2024年8月23日至8月25日期间，早上A架在9点前开机的有几天？在这几天内，4台柴油发电机的总燃油消耗量为多少？（L，保留两位小数）"
    # question = "24. 在2024年8月23日至8月25日期间，哪一天上午（00:00至12:00）A架的运行时长最长？（输出格式：2024/08/XX）这一天上午A架运行了多久（以整数分钟输出）？这一天上午甲板机械的总做功为多少？（kWh，保留两位小数）"
    # question = "25. 在2024年8月23日上午（00:00至12:00）和8月24日上午（00:00至12:00），A架的运行时间均值是多少（四舍五入至整数分钟输出）？主推进器的做功均值是多少？（kWh，保留两位小数）"
    question = "26. 在2024年8月23日和8月24日，征服者从起吊到入水的平均时间是多少（以分钟为单位，保留一位小数）？艏侧推的做功均值是多少？（kWh，保留两位小数）"
    question = "27. 统计2024年8月24日至8月30日期间，征服者在每日16:00前出水的比例（%，保留2位小数）。同时，输出这一周内最早的征服者出水时间（HH）以及该天00:00至征服者出水时（小时）甲板机械系统的总做功（kWh，保留两位小数）"
    question = "28. 统计 2024/08/24 - 2024/08/29 期间，“征服者”在 09:00:00 之前 入水的比例（%，以所有在该时间段内入水的征服者为基数，结果保留 2 位小数）。同时，统计该时间段内，四台发电机的日燃油消耗量最大的一天，以及对应的燃油消耗量（格式：MM/DD；单位：L，保留两位小数）。若有多天燃油消耗量相同，则输出最早的一天"
    # question = "29. 2024年5月19日，A 架的开机时间分别是几点？关机时间分别是几点？（请按 HH:MM 格式，24 小时制，以逗号分隔多个时间点）计算该日期内 A 架的总运行时间（单位：整数分钟）。在这段时间内，甲板机械系统总做功为多少（kWh，保留两位小数）"
    question = "30. 请确定2024年5月20日征服者入水的时间（格式为HH:MM，24时制），并计算入水后十分钟内艏侧推功率的最大值（单位：kW）。若该日没有入水记录或功率数据缺失，请输出“无数据”。请按照时间、最大功率的顺序输出结果"
    # question = "31. 请统计2024年6月14日处于伴航状态的总时长（单位：分钟，如果没有则计为0），以及伴航状态中一号、二号、三号和四号柴油发电机各自的运行时长（单位：分钟）。同时，请计算伴航状态中一号、二号主推进器的功率最大值（单位：kW）。如果某项数据缺失，请在相应字段输出'nil'"
    # question = "32. 请统计2024年6月12日处于停泊状态的时长（单位：分钟，如果没有则计为0），以及停泊状态时中一号、二号、三号和四号柴油发电机的运行时长（单位：分钟）。如果某项数据缺失，请在相应字段输出'nil'"
    # question = "33. 请统计2024年6月4日处于动力定位的时长（单位：分钟，如果没有则计为0），以及动力定位时一号、二号、三号和四号柴油发电机的运行时长（单位：分钟）。同时，请计算动力定位时一号、二号主推进器的功率最大值（单位：kW），艏侧推的功率最大值（单位：kW），以及可伸缩推的运行时长（单位：分钟）。如果某项数据缺失，请在相应字段输出'nil'"
    # question = "34. 请统计2024年6月4日处于航渡状态的时长（单位：分钟，如果没有则计为0），以及航渡状态时一号、二号、三号和四号柴油发电机的运行时长（单位：分钟）。同时，请计算航渡状态时一号、二号主推进器的功率最大值（单位：kW），一号舵桨转速反馈的最大值（单位：rpm），二号舵桨转速反馈的最大值（单位：rpm）。如果某项数据缺失，请在相应字段输出'nil'"
    # question = "35. 请统计2024年6月1日至2024年6月10日期间动力定位的次数。在这些动力定位的时间段内，分别统计艏侧推和可伸缩推的使用次数。此外，请按日期输出每天的动力定位次数（格式为MM/DD，如果某天没有动力定位，则次数记为0）。如果某项数据缺失，请在相应字段输出'nil'"
    # question = "36. 请统计2024年6月15日至2024年6月20日期间，每天处于航渡状态的时长（单位：分钟，如果没有则计为0），以及航渡状态时一号主推进器的功率最大值（单位：kW）、二号主推进器的功率最大值（单位：kW）、一号舵桨转速反馈的最大值（单位：rpm）、二号舵桨转速反馈的最大值（单位：rpm）。如果某项数据缺失，请在相应字段输出'nil'"
    # question = "37. 请统计2024年6月1日至2024年6月5日（包含）每天停泊的时长（单位：分钟），以及停泊状态时一号、二号、三号和四号柴油发电机的运行时长（单位：分钟）。如果某项数据缺失，请在相应字段输出'nil'"
    # question = "38. 请统计2024年8月1日至2024年8月5日（包含）停泊的天数（只要当天包含停泊状态即可算作一天），并列出这些停泊日期（格式：MM/DD）。同时，请统计每一天处于停泊状态时有多少台柴油发电机在运行，以及具体是哪几台在运行。如果某天没有柴油发电机在运行，请在相应字段输出'nil'"
    # question = "39. 请统计2024年6月15日至2024年6月20日（包含）每天处于伴航状态的时长（单位：分钟，如果没有则计为0），以及伴航状态时一号、二号、三号和四号柴油发电机的运行时长（单位：分钟）。同时，请计算伴航状态时一号、二号主推进器的功率最大值（单位：kW），以及一号舵桨转速反馈的最小值和二号舵桨转速反馈的最小值（单位：rpm）。如果某项数据缺失，请在相应字段输出'nil'"
    # question = "40. 请统计2024年6月10日至2024年6月20日（包含）动力定位的次数。在这些动力定位的时间段内，请分别统计使用艏侧推和可伸缩推的次数。按日期输出每天的动力定位次数（格式：MM/DD，如果没有则次数记为0）、每天的动力定位总时间（单位：分钟）以及艏侧推功率的最大值（单位：kW）。如果某天没有相关数据，请在相应字段输出'nil'"
    # question = "41. 请确定2024年8月19日下午A架的第一次开机时间（格式：HH:MM），并报告开机时1号柴油发电机组的功率（单位：kW，输出为整数）"
    question = "42. 请确定2024年8月20日A架的最后一次关机时间（格式：HH:MM），并判断关机时艏侧推是否处于运行状态（回答：Y/N）"
    # question = "43. 请确定2024年8月23日10:17时，甲板机械设备正在进行的具体动作，并报告进行该动作时艏侧推的功率（单位：kW，输出为整数）"
    question = "44. 请确定2024年8月20日深海作业A开始的时间（格式：HH:MM），并报告此时一号舵桨的转速反馈和方位反馈（单位：整数）"
    # question = "45. 请确定2024年8月23日18:26时，甲板机械设备正在进行的具体动作，并报告此时二号舵桨的转速反馈和方位反馈（单位：整数）"
    # question = "46. 请确定2024年8月23日19:05时，甲板机械设备正在进行的具体动作，并报告此时一号推进器和二号推进器的功率（单位：kW，输出为整数）"
    # question = "47. 请确定2024年8月23日A架的首次开机时间点（格式：HH:MM），并计算从该时间点开始后的十分钟内，甲板机械设备的总做功（单位：kWh，输出为整数）"
    # question = "48. 请确定2024年8月24日征服者入水的时间点（格式：HH:MM），并报告该时间点各个推进器的功率最大值（单位：kW，输出为整数）"
    # question = "49. 请确定2024年8月24日揽绳解除的时间点（格式：HH:MM），并报告该时间点1至4号柴油发电机组的功率（单位：kW，按编号顺序输出）"
    # question = "50. 请确定2024年8月24日A架摆回的时间点（格式：HH:MM），并计算该时间点2号和3号柴油发电机组功率差值的绝对值（单位：kW）"
    # question = "51. 请确定2024年8月24日A架摆出的时间点（格式：HH:MM），并计算该时间点一号和二号门架的总功率（单位：kW）"
    # question = "52. 请列出2024年8月24日8:45至8:55之间发生的所有关键动作，并计算该时间段内1号柴油发电机的总做功（单位：kWh，保留两位小数）"
    # question = "53. 请列出2024年8月24日16:00至16:30之间发生的所有关键动作，并计算该时间段内的总燃油消耗量（单位：升，保留两位小数）"
    # question = "54. 请确定2024年8月24日9:10时，哪些设备正在进行哪些动作，并报告此时可伸缩推的功率（单位：kW，以整数输出）"
    # question = "55. 请确定以征服者落座为标志，2024年8月23日深海作业A结束的时间（格式：HH:MM），并判断结束时舵桨转速是否低于50 RPM"
    # question = "56. 请判断2024年8月24日下午，A架开机是否发生在折臂吊车开机之前（回答Y/N）。同时，计算在这两个动作之间，1~4号柴油发电机的总做功（单位：kWh，保留两位小数）"
    # question = "57. 请确定2024年8月23日，征服者第一次出水的时间点（格式：HH:MM）。同时，计算征服者出水后10分钟内甲板机械的总做功（单位：kWh，保留两位小数）"
    # question = "58. 请确定2024年8月23日，A架最后一次关机的时间点（格式：HH:MM）。同时，计算A架关机前十分钟内折臂吊车的功率均值（单位：kW，保留两位小数）"
    # question = "59. 请确定2024年8月24日，小艇最后一次落座的时间点（格式：HH:MM）。同时，记录此时一号推进器和二号推进器的功率（单位：kW，以整数输出）"
    # question = "60. 请确定2024年8月24日，折臂吊车关机的时间点（格式：HH:MM）。同时，记录此时一号舵桨的转速反馈和方位反馈（以整数输出）"
    # question = "61. 2024/8/23上午A架的运行时长和下午A架开机时长相比，哪个时间段更长，长多少（以整数分钟输出）？"
    # question = "62. 2024/8/24 DP过程中，推进系统的总能耗是多少（单位化成kWh，保留2位小数）？"
    # question = "63. 2024/8/24 上午，折臂吊车的能耗占甲板机械设备的比例（以%输出，保留2位小数）？"
    # question = "64. 2024/8/23和2024/8/25小艇入水到小艇落座，折臂吊车的总能耗是多少（单位化成kWh，保留2位小数）？"
    # question = "65. 2024/8/23 0:00 ~ 2024/8/25 0:00推进系统能耗占总发电量的比例（以%输出，保留2位小数）？"
    # question = "66. 2024/8/23 0:00 ~ 2024/8/25 0:00甲板机械能耗占总发电量的比例（以%输出，保留2位小数）？"
    # question = "67. 假设柴油的密度为0.8448kg/L，柴油热值为42.6MJ/kg，请计算2024/8/23 0:00 ~ 2024/8/25 0:00柴油机的发电效率（%，保留2位小数）？"
    question = "68. 5月20日征服者入水到A架摆回用了多少时间？"
    # question = "69. 停泊发电机组转速为1800RPM会发生什么？"
    # question = "70. 2024/05/18~2024/05/20（含） 折臂吊车总开机时长为多少分钟（以整数输出，当天没有开机/关机动作则为0）？"
    # question = "71. 2024/05/18 折臂吊车第二次开机和第一次关机的时间差是多少（以整数输出，当天没有开机/关机动作则为0）？"
    # question = "72. 请指出2024/05/19深海作业A回收过程中小艇落座和征服者落座分别相差多少分钟（以整数分钟输出；如果没有进行深海作业A，回答N）？"
    # question = "73. 2024/08/21 深海作业A过程中，征服者起吊到缆绳解除，甲板机械做总做功为多少（单位为kWh，保留两位小数）？"
    # question = "74. 假设A架右舷同一方向上摆动超过10°即可算作一次摆动，统计2024/05/23 A架摆动的次数至少为多少次（以整数输出）？"
    # question = "75. 2024/05/18 00:00:00~2024/05/25 00:00:00 甲板机械和推进器的能耗之和占发电机的总发电的比例（%，结果保留两位小数）？"
    # question = "76. 2024/08/19 深海作业A回收过程中，从折臂吊车开机到小艇入水， 甲板机械的总做功是多少 （单位为kWh，结果保留两位小数）？"
    # question = "77. 2024/8/23 A架的摆动次数是多少次（不参考角度数据，以动作判断，任意摆出或摆回算一次）？"
    # question = "78. 2024/8/23、 2024/8/24和2024/8/25 A架的平均摆动次数是多少次？（不参考角度数据，以动作判断，任意摆出或摆回算一次）"
    # question = "79. 2024年5月20日征服者入水后A架摆回到位的时间是？（请以XX:XX输出）"
    # question = "80. 假设柴油的密度为0.85kg/L，柴油热值为42.6MJ/kg，请计算2024/05/17 00:00:00~2024/05/25 00:00:00 1~4号柴油发电机的理论发电量（单位化成kWh，保留2位小数）？"
    # question = "81. 2024/05/18 00:00:00~2024/05/25 00:00:00 舵桨的能耗占发电机的总发电的比例（%，结果保留两位小数）？"
    # question = "82. 2024/08/15~2024/08/23（含）A架第一次开机最晚的一天（以mm/dd格式输出）？"
    # question = "83. 2024/05/18 00:00:00~2024/05/25 00:00:00 甲板机械、推进器和舵桨的能耗之和占发电机的总发电的比例（%，结果保留两位小数）？"
    # question = "84. 2024/08/19 深海作业A从ON DP到OFF DP期间， 甲板机械的总做功是多少 （单位为kWh，结果保留两位小数）？"
    # question = "85. 2024/08/24 深海作业A过程中，两次小艇落座期间，四台发电机的发电量是多少（单位为kWh，结果保留两位小数）？"
    # question = "86. 2024/08/16 一号舵桨转舵B-电流不平衡度的最大值为多少（单位为%，四舍五入，以整数输出）？"
    # question = "87. 一号柴油发电机组滑油压力的范围是多少？"
    # question = "88. 假设柴油的密度为0.8448kg/L，柴油热值为42.6MJ/kg，请计算2024/8/23 0:00 ~ 2024/8/25 0:00的理论发电量（单位化成kWh，保留2位小数）？"
    question = "89. 24年8月23日征服者入水时间距离征服者出水时间是多久？（请用XX小时XX分钟表示）"
    # question = "90. 假设某一时刻二号柴油发电机组所有与温度相关的数值类型参数的值都超过了160（忽略上下限），至少会触发几个报警信号？（以整数输出）"
    # question = "91. 2024/08/24 深海作业A过程中，A架摆回到A架摆出期间，1~4号柴油发电机的燃油消耗量是多少（单位为kg，柴油的密度为0.85kg/L，保留两位小数）？"
    # question = "92. 2024/8/23和2024/8/25 平均作业时长是多久（四舍五入至整数分钟输出，下放阶段以ON DP和OFF DP为标志，回收阶段以A架开机和关机为标志）？"
    # question = "93. 2024/08/16 00:00:00~2024/08/23 00:00:00 所有推进器的总能耗为多少（单位为kWh，结果保留两位小数）？"
    # question = "94. 2024/05/17 折臂吊车开机总时长为多少分钟（以整数输出，当天没有开机/关机动作则为0）？"
    # question = "95. 2024/8/23 DP过程中，侧推的总能耗是多少（单位化成kWh，保留2位小数）？"
    # question = "96. 2024/8/23 和 2024/8/24 的DP过程中，侧推的平均能耗是多少（单位化成kWh，保留2位小数）？"
    # question = "97. 2024/05/18 00:00:00~2024/05/25 00:00:00 甲板机械的能耗占发电机的总发电的比例（%，结果保留两位小数）？"
    # question = "98. 2024/08/23 深海作业A过程中，A架摆回到A架摆出期间，推进器的总做功是多少（单位为kWh，结果保留两位小数）？"
    # question = "99. 2024/05/17 一号舵桨转舵A-频率的平均值是多少（单位为Hz，四舍五入，以整数输出）？"
    # question = "100. 2024/05/17 一号舵桨转舵A-Ua电压的平均值是多少（单位为V，四舍五入，以整数输出）？"


###########################################################################################################################
    # question = "1. 2024年5月23日23:00至24:00（包括这两个时间点）进行了一次特定类型的作业，请指出该次作业使用的甲板机器设备有哪些？（多个设备使用空格隔开，设备名称大小写敏感，且空格重要；每个设备仅写一次；如果没有使用任何设备，则输出N）"
    # question = "2. 请指出2024年5月25日深海作业A回收阶段中，A架摆出、征服者起吊和征服者落座的确切时间（以XX:XX格式输出，时间为24小时制，时间补零，如05:03；多个时间使用单个空格隔开，按照事件顺序输出；如果某事件未发生，则在该位置输出N）"
    question = "3. 请指出2024年7月19日征服者落座以及征服者出水的时间（按发生顺序以XX:XX格式输出，时间为24小时制，时间补零，如05:03；多个时间使用单个空格隔开；如果某事件未发生，则在该位置输出N）"
    question = "4. 请统计2024年6月1日至6月30日期间，深海作业A的次数（一个完整的布放过程加一个完整的回收过程算作一次作业；如果存在只有布放或只有回收的情况，不计算在内）"
    # question = "5. 请指出2024年6月7日征服者落座和小艇入水的时间（按发生顺序以XX:XX格式输出，时间为24小时制，时间补零，如05:03；多个时间使用单个空格隔开；如果某事件未发生，则在该位置输出N）"
    # question = "6. 2024年5月23日，从征服者落座开始至当日作业结束，请计算A架右舷摆动的次数。（以整数输出，A架摆动的定义如下：A架右舷同一方向上摆动超过10°即可算作一次摆动）"
    question = "7. 请统计2024年5月1日至5月31日期间，深海作业A的次数（一个完整的布放过程加一个完整的回收过程算作一次作业；如果存在只有布放或只有回收的情况，不计算在内）"
    question = "8. 请指出2024年6月14日小艇落座和征服者落座的时间（按发生顺序以XX:XX格式输出，时间为24小时制，时间补零，如05:03；多个时间使用单个空格隔开；如果某事件未发生，则在该位置输出N）"
    # question = "9. 请指出2024年7月19日绞车A的放缆和收缆次数（放缆是指绞车A向海中释放电缆，收缆是指绞车A从海中回收电缆；放缆长度同一方向上单次变化超过5m则算一次放缆，收缆同理）"
    question = "10. 请指出2024年6月15日小艇检查完毕的时间和小艇入水的时间（按发生顺序以XX:XX格式输出，时间为24小时制，时间补零，如05:03；多个时间使用单个空格隔开；如果某事件未发生，则在该位置输出N；注意，每个事件在一天内可能发生多次）"
    # question = "11. 假设柴油的密度为0.85kg/L，柴油热值为42.6MJ/kg。请根据提供的1~4号柴油发电机的燃油消耗量，计算2024年7月16日00:00:00至2024年7月20日00:00:00期间的理论发电量（单位转换为kWh，结果保留两位小数，四舍五入）"
    question = "12. 根据提供的2024年6月10日00:00:00至2024年6月15日00:00:00期间甲板机械和推进器的能耗数据以及发电机的总发电数据（数据来源于作业日志和传感器记录），计算甲板机械和推进器的能耗之和占发电机总发电量的比例（结果以百分比表示，保留两位小数，四舍五入；注意，能耗和发电量数据均已转换为同一单位kWh）"
    question = "13. 2024/06/10 00:00:00~2024/06/15 00:00:00 甲板机械和舵桨的能耗之和占发电机的总发电的比例（%，结果保留两位小数）？"
    # question = "14. 在2024年8月25日深海作业A过程中，从A架摆回至A架摆出之间相差的时长（根据设备记录的时间计算，四舍五入到整数，单位为分钟），以及在这段时间内四台发电机的总发电量（单位为kWh，保留两位小数）。如果遇到数据缺失或异常情况，请根据相邻数据点进行合理估算"
    # question = "15. 在2024年6月6日，请输出征服者起吊和征服者入水的时间（以XX:XX格式输出，时间为24小时制，时间补零，如05:03），以及在这段时间内推进器的总做功（单位为kWh，结果保留两位小数）"
    # question = "16. 在2024年6月1日深海作业A过程中，请输出两次小艇落座的时间（以XX:XX格式输出，时间为24小时制，时间补零，如05:03），以及在这两次落座期间四台发电机的总发电量（数据来源于传感器记录，单位为kWh，结果保留两位小数）。如果遇到数据缺失或异常情况，请根据相邻数据点进行合理估算。如果当天发生多次小艇落座，请仅考虑第一次和最后一次落座"
    # question = "17. 在2024年8月24日深海作业A中，请输出布放和回收的开始和结束时间（以折臂吊车开机为开始，折臂吊车关机为结束，时间以24小时制HH:MM格式按发生顺序输出），并比较布放和回收阶段的燃油消耗量，输出较大的燃油消耗量（燃油消耗量单位为L，结果保留两位小数）。如果遇到数据缺失或异常情况，请根据相邻数据点进行合理估算。如果当天进行多次布放和回收，请比较所有布放和回收阶段的燃油消耗量，并输出消耗量最大的一次"
    # question = "18. 在2024年6月15日深海作业A中，请输出布放和回收的开始和结束时间（以折臂吊车开机为开始，折臂吊车关机为结束，时间以24小时制HH:MM格式按发生顺序输出），并比较布放和回收阶段的燃油消耗量，输出较大的燃油消耗量（燃油消耗量单位为L，结果保留两位小数）。如果遇到数据缺失或异常情况，请根据相邻数据点进行合理估算。如果当天进行多次布放和回收，请比较所有布放和回收阶段的燃油消耗量，并输出消耗量最大的一次"
    # question = "19. 2024年5月17日00:00:00至2024年5月20日00:00:00期间，推进器的总能耗为多少？（单位为kWh，结果保留两位小数，采用四舍五入规则）"
    # question = "20. 2024年5月22日00:00:00至2024年5月25日00:00:00期间，甲板机械的总能耗为多少？（单位为kWh，结果保留两位小数，采用四舍五入规则）"
    # question = "21. 在2024/06/01 00:00:00至2024/06/05 00:00:00期间，请计算1~4号柴油发电机的运行时间（运行时间定义为发电机在额定转速下的运行时间，额定转速运行值为1表示发电机运行了1分钟。请输出每个发电机的运行时间，按编号顺序输出，时间按整数分钟输出，用空格隔开）"
    # question = "22. 在2024年8月24日，A架摆回和摆出的时间相隔多久（以整数分钟输出）？在这段时间内，4台柴油发电机的总发电量为多少？（kWh，保留两位小数）"
    # question = "23. 在2024年8月23日至8月25日期间，早上A架在9点前3开机的有几天？在这几天内，4台柴油发电机的总燃油消耗量为多少？（L，保留两位小数）"
    # question = "24. 在2024年8月23日至8月25日期间，哪一天上午（00:00至12:00）A架的运行时长最长？（输出格式：2024/08/XX）这一天上午A架运行了多久（以整数分钟输出）？这一天上午甲板机械的总做功为多少？（kWh，保留两位小数）"
    # question = "25. 在2024年8月23日上午（00:00至12:00）和8月24日上午（00:00至12:00），A架的运行时间均值是多少（四舍五入至整数分钟输出）？主推进器的做功均值是多少？（kWh，保留两位小数）"
    question = "26. 在2024年8月23日和8月24日，征服者从起吊到入水的平均时间是多少（以分钟为单位，保留一位小数）？艏侧推的做功均值是多少？（kWh，保留两位小数）"
    # question = "27. 统计2024年8月24日至8月30日期间，征服者在每日16:00前出水的比例（%，保留2位小数）。同时，输出这一周内最早的征服者出水时间（HH）以及该天00:00至征服者出水时（小时）甲板机械系统的总做功（kWh，保留两位小数）"
    # question = "28. 统计 2024/08/24 - 2024/08/29 期间，“征服者”在 09:00:00 之前 入水的比例（%，以所有在该时间段内入水的征服者为基数，结果保留 2 位小数）。同时，统计该时间段内，四台发电机的日燃油消耗量最大的一天，以及对应的燃油消耗量（格式：MM/DD；单位：L，保留两位小数）。若有多天燃油消耗量相同，则输出最早的一天"
    # question = "29. 2024年5月19日，A架的开机时间分别是几点？关机时间分别是几点？（请按 HH:MM 格式，24 小时制，以逗号分隔多个时间点）计算该日期内 A 架的总运行时间（单位：整数分钟）。在这段时间内，甲板机械系统总做功为多少（kWh，保留两位小数）"
    # question = "30. 请确定2024年5月20日征服者入水的时间（格式为HH:MM，24时制），并计算入水后十分钟内艏侧推功率的最大值（单位：kW）。若该日没有入水记录或功率数据缺失，请输出“无数据”。请按照时间、最大功率的顺序输出结果"
    # question = "31. 请统计2024年6月14日处于伴航状态的总时长（单位：分钟，如果没有则计为0），以及伴航状态中一号、二号、三号和四号柴油发电机各自的运行时长（单位：分钟）。同时，请计算伴航状态中一号、二号主推进器的功率最大值（单位：kW）。如果某项数据缺失，请在相应字段输出'nil'"
    # question = "32. 请统计2024年6月12日处于停泊状态的时长（单位：分钟，如果没有则计为0），以及停泊状态时中一号、二号、三号和四号柴油发电机的运行时长（单位：分钟）。如果某项数据缺失，请在相应字段输出'nil'"
    # question = "33. 请统计2024年6月4日处于动力定位的时长（单位：分钟，如果没有则计为0），以及动力定位时一号、二号、三号和四号柴油发电机的运行时长（单位：分钟）。同时，请计算动力定位时一号、二号主推进器的功率最大值（单位：kW），艏侧推的功率最大值（单位：kW），以及可伸缩推的运行时长（单位：分钟）。如果某项数据缺失，请在相应字段输出'nil'"
    # question = "34. 请统计2024年6月4日处于航渡状态的时长（单位：分钟，如果没有则计为0），以及航渡状态时一号、二号、三号和四号柴油发电机的运行时长（单位：分钟）。同时，请计算航渡状态时一号、二号主推进器的功率最大值（单位：kW），一号舵桨转速反馈的最大值（单位：rpm），二号舵桨转速反馈的最大值（单位：rpm）。如果某项数据缺失，请在相应字段输出'nil'"
    # question = "35. 请统计2024年6月1日至2024年6月10日期间动力定位的次数。在这些动力定位的时间段内，分别统计艏侧推和可伸缩推的使用次数。此外，请按日期输出每天的动力定位次数（格式为MM/DD，如果某天没有动力定位，则次数记为0）。如果某项数据缺失，请在相应字段输出'nil'"
    # question = "36. 请统计2024年6月15日至2024年6月20日期间，每天处于航渡状态的时长（单位：分钟，如果没有则计为0），以及航渡状态时一号主推进器的功率最大值（单位：kW）、二号主推进器的功率最大值（单位：kW）、一号舵桨转速反馈的最大值（单位：rpm）、二号舵桨转速反馈的最大值（单位：rpm）。如果某项数据缺失，请在相应字段输出'nil'"
    question = "37. 请统计2024年6月1日至2024年6月5日（包含）每天停泊的时长（单位：分钟），以及停泊状态时一号、二号、三号和四号柴油发电机的运行时长（单位：分钟）。如果某项数据缺失，请在相应字段输出'nil'"
    # question = "38. 请统计2024年8月1日至2024年8月5日（包含）停泊的天数（只要当天包含停泊状态即可算作一天），并列出这些停泊日期（格式：MM/DD）。同时，请统计每一天处于停泊状态时有多少台柴油发电机在运行，以及具体是哪几台在运行。如果某天没有柴油发电机在运行，请在相应字段输出'nil'"
    # question = "39. 请统计2024年6月15日至2024年6月20日（包含）每天处于伴航状态的时长（单位：分钟，如果没有则计为0），以及伴航状态时一号、二号、三号和四号柴油发电机的运行时长（单位：分钟）。同时，请计算伴航状态时一号、二号主推进器的功率最大值（单位：kW），以及一号舵桨转速反馈的最小值和二号舵桨转速反馈的最小值（单位：rpm）。如果某项数据缺失，请在相应字段输出'nil'"
    # question = "40. 请统计2024年6月10日至2024年6月20日（包含）动力定位的次数。在这些动力定位的时间段内，请分别统计使用艏侧推和可伸缩推的次数。按日期输出每天的动力定位次数（格式：MM/DD，如果没有则次数记为0）、每天的动力定位总时间（单位：分钟）以及艏侧推功率的最大值（单位：kW）。如果某天没有相关数据，请在相应字段输出'nil'"
    # question = "41. 请确定2024年8月19日下午A架的第一次开机时间（格式：HH:MM），并报告开机时1号柴油发电机组的功率（单位：kW，输出为整数）"
    # question = "42. 请确定2024年8月20日A架的最后一次关机时间（格式：HH:MM），并判断关机时艏侧推是否处于运行状态（回答：Y/N）"
    # question = "43. 请确定2024年8月23日10:17时，甲板机械设备正在进行的具体动作，并报告进行该动作时艏侧推的功率（单位：kW，输出为整数）"
    # question = "44. 请确定2024年8月20日深海作业A开始的时间（格式：HH:MM），并报告此时一号舵桨的转速反馈和方位反馈（单位：整数）"
    # question = "45. 请确定2024年8月23日18:26时，甲板机械设备正在进行的具体动作，并报告此时二号舵桨的转速反馈和方位反馈（单位：整数）"
    # question = "46. 请确定2024年8月23日19:05时，甲板机械设备正在进行的具体动作，并报告此时一号推进器和二号推进器的功率（单位：kW，输出为整数）"
    # question = "47. 请确定2024年8月23日A架的首次开机时间点（格式：HH:MM），并计算从该时间点开始后的十分钟内，甲板机械设备的总做功（单位：kWh，输出为整数）"
    # question = "48. 请确定2024年8月24日征服者入水的时间点（格式：HH:MM），并报告该时间点各个推进器的功率最大值（单位：kW，输出为整数）"
    # question = "49. 请确定2024年8月24日揽绳解除的时间点（格式：HH:MM），并报告该时间点1至4号柴油发电机组的功率（单位：kW，按编号顺序输出）"
    # question = "50. 请确定2024年8月24日A架摆回的时间点（格式：HH:MM），并计算该时间点2号和3号柴油发电机组功率差值的绝对值（单位：kW）"
    # question = "51. 请确定2024年8月24日A架摆出的时间点（格式：HH:MM），并计算该时间点一号和二号门架的总功率（单位：kW）"
    # question = "52. 请列出2024年8月24日8:45至8:55之间发生的所有关键动作，并计算该时间段内1号柴油发电机的总做功（单位：kWh，保留两位小数）"
    # question = "53. 请列出2024年8月24日16:00至16:30之间发生的所有关键动作，并计算该时间段内的总燃油消耗量（单位：升，保留两位小数）"
    # question = "54. 请确定2024年8月24日9:10时，哪些设备正在进行哪些动作，并报告此时可伸缩推的功率（单位：kW，以整数输出）"
    # question = "55. 请确定以征服者落座为标志，2024年8月23日深海作业A结束的时间（格式：HH:MM），并判断结束时舵桨转速是否低于50 RPM"
    # question = "56. 请判断2024年8月24日下午，A架开机是否发生在折臂吊车开机之前（回答Y/N）。同时，计算在这两个动作之间，1~4号柴油发电机的总做功（单位：kWh，保留两位小数）"
    # question = "57. 请确定2024年8月23日，征服者第一次出水的时间点（格式：HH:MM）。同时，计算征服者出水后10分钟内甲板机械的总做功（单位：kWh，保留两位小数）"
    # question = "58. 请确定2024年8月23日，A架最后一次关机的时间点（格式：HH:MM）。同时，计算A架关机前十分钟内折臂吊车的功率均值（单位：kW，保留两位小数）"
    # question = "59. 请确定2024年8月24日，小艇最后一次落座的时间点（格式：HH:MM）。同时，记录此时一号推进器和二号推进器的功率（单位：kW，以整数输出）"
    question = "60. 请确定2024年8月24日，折臂吊车关机的时间点（格式：HH:MM）。同时，记录此时一号舵桨的转速反馈和方位反馈（以整数输出）"
    # question = "61. 2024/8/23上午A架的运行时长和下午A架开机时长相比，哪个时间段更长，长多少（以整数分钟输出）？"
    # question = "62. 2024/8/24 DP过程中，推进系统的总能耗是多少（单位化成kWh，保留2位小数）？"
    # question = "63. 2024/8/24 上午，折臂吊车的能耗占甲板机械设备的比例（以%输出，保留2位小数）？"
    # question = "64. 2024/8/23和2024/8/25小艇入水到小艇落座，折臂吊车的总能耗是多少（单位化成kWh，保留2位小数）？"
    # question = "65. 2024/8/23 0:00 ~ 2024/8/25 0:00推进系统能耗占总发电量的比例（以%输出，保留2位小数）？"
    # question = "66. 2024/8/23 0:00 ~ 2024/8/25 0:00甲板机械能耗占总发电量的比例（以%输出，保留2位小数）？"
    # question = "67. 假设柴油的密度为0.8448kg/L，柴油热值为42.6MJ/kg，请计算2024/8/23 0:00 ~ 2024/8/25 0:00柴油机的发电效率（%，保留2位小数）？"
    # question = "68. 5月20日征服者入水到A架摆回用了多少时间？"
    # question = "69. 停泊发电机组转速为1800RPM会发生什么？"
    # question = "70. 2024/05/18~2024/05/20（含） 折臂吊车总开机时长为多少分钟（以整数输出，当天没有开机/关机动作则为0）？"
    # question = "71. 2024/05/18 折臂吊车第二次开机和第一次关机的时间差是多少（以整数输出，当天没有开机/关机动作则为0）？"
    # question = "72. 请指出2024/05/19深海作业A回收过程中小艇落座和征服者落座分别相差多少分钟（以整数分钟输出；如果没有进行深海作业A，回答N）？"
    # question = "73. 2024/08/21 深海作业A过程中，征服者起吊到缆绳解除，甲板机械做总做功为多少（单位为kWh，保留两位小数）？"
    # question = "74. 假设A架右舷同一方向上摆动超过10°即可算作一次摆动，统计2024/05/23 A架摆动的次数至少为多少次（以整数输出）？"
    # question = "75. 2024/05/18 00:00:00~2024/05/25 00:00:00 甲板机械和推进器的能耗之和占发电机的总发电的比例（%，结果保留两位小数）？"
    # question = "76. 2024/08/19 深海作业A回收过程中，从折臂吊车开机到小艇入水， 甲板机械的总做功是多少 （单位为kWh，结果保留两位小数）？"
    # question = "77. 2024/8/23 A架的摆动次数是多少次（不参考角度数据，以动作判断，任意摆出或摆回算一次）？"
    # question = "78. 2024/8/23、 2024/8/24和2024/8/25 A架的平均摆动次数是多少次？（不参考角度数据，以动作判断，任意摆出或摆回算一次）"
    # question = "79. 2024年5月20日征服者入水后A架摆回到位的时间是？（请以XX:XX输出）"
    # question = "80. 假设柴油的密度为0.85kg/L，柴油热值为42.6MJ/kg，请计算2024/05/17 00:00:00~2024/05/25 00:00:00 1~4号柴油发电机的理论发电量（单位化成kWh，保留2位小数）？"
    # question = "81. 2024/05/18 00:00:00~2024/05/25 00:00:00 舵桨的能耗占发电机的总发电的比例（%，结果保留两位小数）？"
    # question = "82. 2024/08/15~2024/08/23（含）A架第一次开机最晚的一天（以mm/dd格式输出）？"
    # question = "83. 2024/05/18 00:00:00~2024/05/25 00:00:00 甲板机械、推进器和舵桨的能耗之和占发电机的总发电的比例（%，结果保留两位小数）？"
    # question = "84. 2024/08/19 深海作业A从ON DP到OFF DP期间， 甲板机械的总做功是多少 （单位为kWh，结果保留两位小数）？"
    # question = "85. 2024/08/24 深海作业A过程中，两次小艇落座期间，四台发电机的发电量是多少（单位为kWh，结果保留两位小数）？"
    # question = "86. 2024/08/16 一号舵桨转舵B-电流不平衡度的最大值为多少（单位为%，四舍五入，以整数输出）？"
    # question = "87. 一号柴油发电机组滑油压力的范围是多少？"
    # question = "88. 假设柴油的密度为0.8448kg/L，柴油热值为42.6MJ/kg，请计算2024/8/23 0:00 ~ 2024/8/25 0:00的理论发电量（单位化成kWh，保留2位小数）？"
    # question = "89. 24年8月23日征服者入水时间距离征服者出水时间是多久？（请用XX小时XX分钟表示）"
    # question = "90. 假设某一时刻二号柴油发电机组所有与温度相关的数值类型参数的值都超过了160（忽略上下限），至少会触发几个报警信号？（以整数输出）"
    # question = "91. 2024/08/24 深海作业A过程中，A架摆回到A架摆出期间，1~4号柴油发电机的燃油消耗量是多少（单位为kg，柴油的密度为0.85kg/L，保留两位小数）？"
    # question = "92. 2024/8/23和2024/8/25 平均作业时长是多久（四舍五入至整数分钟输出，下放阶段以ON DP和OFF DP为标志，回收阶段以A架开机和关机为标志）？"
    # question = "93. 2024/08/16 00:00:00~2024/08/23 00:00:00 所有推进器的总能耗为多少（单位为kWh，结果保留两位小数）？"
    # question = "94. 2024/05/17 折臂吊车开机总时长为多少分钟（以整数输出，当天没有开机/关机动作则为0）？"
    # question = "95. 2024/8/23 DP过程中，侧推的总能耗是多少（单位化成kWh，保留2位小数）？"
    # question = "96. 2024/8/23 和 2024/8/24 的DP过程中，侧推的平均能耗是多少（单位化成kWh，保留2位小数）？"
    # question = "97. 2024/05/18 00:00:00~2024/05/25 00:00:00 甲板机械的能耗占发电机的总发电的比例（%，结果保留两位小数）？"
    # question = "98. 2024/08/23 深海作业A过程中，A架摆回到A架摆出期间，推进器的总做功是多少（单位为kWh，结果保留两位小数）？"
    # question = "99. 2024/05/17 一号舵桨转舵A-频率的平均值是多少（单位为Hz，四舍五入，以整数输出）？"
    # question = "100. 2024/05/17 一号舵桨转舵A-Ua电压的平均值是多少（单位为V，四舍五入，以整数输出）？"
    # question = "3. 请指出2024年7月19日征服者落座以及征服者出水的时间（按发生顺序以XX:XX格式输出，时间为24小时制，时间补零，如05:03；多个时间使用单个空格隔开；如果某事件未发生，则在该位置输出N）"
    # question = "1. 2024年5月23日23:00至24:00（包括这两个时间点）进行了一次特定类型的作业，请指出该次作业使用的甲板机器设备有哪些？（多个设备使用空格隔开，设备名称大小写敏感，且空格重要；每个设备仅写一次；如果没有使用任何设备，则输出N）"
    # question = "2. 请指出2024年5月25日深海作业A回收阶段中，A架摆出、征服者起吊和征服者落座的确切时间（以XX:XX格式输出，时间为24小时制，时间补零，如05:03；多个时间使用单个空格隔开，按照事件顺序输出；如果某事件未发生，则在该位置输出N）"
    # question = "27. 统计2024年8月24日至8月30日期间，征服者在每日16:00前出水的比例（%，保留2位小数）。同时，输出这一周内最早的征服者出水时间（HH/MM）以及该天00:00至征服者出水时（小时）甲板机械系统的总做功（kWh，保留两位小数）"
    # question = "23. 在2024年8月23日至8月25日期间，早上A架在9点前3开机的有几天？在这几天内，4台柴油发电机的总燃油消耗量为多少？（L，保留两位小数）"
    # question = "22. 在2024年8月24日，A架摆回和摆出的时间相隔多久（以整数分钟输出）？在这段时间内，4台柴油发电机的总发电量为多少？（kWh，保留两位小数）"
    # question = "55. 请确定以征服者落座为标志，2024年8月23日深海作业A结束的时间（格式：HH:MM），并判断结束时舵桨转速是否低于50 RPM"
    # question = "6. 2024年5月23日，从征服者落座开始至当日作业结束，请计算A架右舷摆动的次数。（以整数输出，A架摆动的定义如下：A架右舷同一方向上摆动超过10°即可算作一次摆动）"
    # question = "7. 请统计2024年5月1日至5月31日期间，深海作业A的次数（一个完整的布放过程加一个完整的回收过程算作一次作业；如果存在只有布放或只有回收的情况，不计算在内）"
    # question = "请统计2024年6月14日处于伴航状态的总时长（单位：分钟，如果没有则计为0），以及伴航状态中一号、二号、三号和四号柴油发电机各自的运行时长（单位：分钟）。同时，请计算伴航状态中一号、二号主推进器的功率最大值（单位：kW）。如果某项数据缺失，请在相应字段输出\"nil\"。\n输出格式：\n{\n    'details': [\n        {\n            'date': 'MM/DD',\n            'escort_duration': 分钟数,\n            'DG1_running': 分钟数或\"nil\",\n            'DG2_running': 分钟数或\"nil\",\n            'DG3_running': 分钟数或\"nil\",\n            'DG4_running': 分钟数或\"nil\",\n            'power_AZ_1': 功率值或\"nil\",\n            'power_AZ_2':功率值或\"nil\"\n        }\n        // ... 其他日期的数据\n    ]\n}\n（DG1_running、DG2_running、DG3_running、DG4_running为柴油发电机运行时长，powre_AZ_1一号主推进器的功率最大值，power_AZ_2二号主推进器的功率的最大值）"
    # question = "10. 请指出2024年6月15日小艇检查完毕的时间和小艇入水的时间（按发生顺序以XX:XX格式输出，时间为24小时制，时间补零，如05:03；多个时间使用单个空格隔开；如果某事件未发生，则在该位置输出N；注意，每个事件在一天内可能发生多次）"
    # question = "请统计2024年6月14日处于伴航状态的总时长（单位：分钟，如果没有则计为0），以及伴航状态中一号、二号、三号和四号柴油发电机各自的运行时长（单位：分钟）。同时，请计算伴航状态中一号、二号主推进器的功率最大值（单位：kW）。如果某项数据缺失，请在相应字段输出\"nil\"。\n输出格式：\n{\n    'details': [\n        {\n            'date': 'MM/DD',\n            'escort_duration': 分钟数,\n            'DG1_running': 分钟数或\"nil\",\n            'DG2_running': 分钟数或\"nil\",\n            'DG3_running': 分钟数或\"nil\",\n            'DG4_running': 分钟数或\"nil\",\n            'power_AZ_1': 功率值或\"nil\",\n            'power_AZ_2':功率值或\"nil\"\n        }\n        // ... 其他日期的数据\n    ]\n}\n（DG1_running、DG2_running、DG3_running、DG4_running为柴油发电机运行时长，powre_AZ_1一号主推进器的功率最大值，power_AZ_2二号主推进器的功率的最大值）"
    # question = "68. 5月20日征服者入水到A架摆回用了多少时间？"
    question = "请统计2024/10/10和2024/10/11的停泊/航渡/动力定位/伴航的时长(四舍五入到整数分钟输出)"
    # question = "请统计2024/9/25这天这A架开机时侧推和一号柴油发动机是否在运行，这天一号柴油发动机的运行时间？"
    aa = get_answer(question)
    question = enhanced(question)
    print(question+"==============总结问题")
    bb = get_end_answer(question,aa)
    print("*******************最终答案***********************")
    print(bb)


# %%
