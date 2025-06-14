import json
from zhipuai import ZhipuAI
from openai import OpenAI
import tools
import api
import os
import re

folders = ["database_in_use", "data"]

if any(not os.path.exists(folder) for folder in folders):
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    import data_process # for data process using
else:
    print("所有文件夹均已存在。不再重新预处理数据。")
    print("需要预处理数据，请删除文件夹后重新运行。")

################更改API###########################

os.environ["ZHIPUAI_API_KEY"] = "7915b06988a447c0b0bf1ef153becc67.MoW2lUeGeypqZcIR"
    
# 配置LLM接口参数
# llm_config = {
#     "api_type": "openai",
#     "model": "gpt-4-1106-preview",
#     "base_url": "https://xiaoai.plus/v1/",
#     "api_key": "sk-Hu0lH7Ac9sZwtIpjUaBZp0oxD5H2QE8Cfa7sYVBUprvKHIGA"
# }

# # 初始化客户端
# client = OpenAI(
#     base_url=llm_config["base_url"],
#     api_key=llm_config["api_key"],
#     # 如果使用自签名证书需要关闭验证
#     # http_client=httpx.Client(verify=False)
# )
def create_chat_completion(messages, model="glm-4-plus"):
# def create_chat_completion(messages, model=llm_config["model"]):
    client = ZhipuAI()
    response = client.chat.completions.create(
        model=model, stream=False, messages=messages
    )
    return response


# In[4]:


# choose_table
def choose_table(question):
    
    with open("dict.json", "r", encoding="utf-8") as file:
        context_text = str(json.load(file))
    prompt = f"""我有如下数据表：<{context_text}>
    现在基于数据表回答问题：{question}。
    分析需要哪些数据表。
    提示：可以根据数据表中的"字段含义"中是否包含问题的关键字来决定选择的数据表。
    
    仅返回需要的数据表名，无需展示分析过程。
    """
    messages = [{"role": "user", "content": prompt}]
    response = create_chat_completion(messages)
    print(str(response.choices[0].message.content))
    return str(response.choices[0].message.content)
# def choose_table(question):
#     # 读取数据表字典
#     with open("dict.json", "r", encoding="utf-8") as file:
#         tables = json.load(file)
    
#     # 提取中文关键词（连续的中文字符片段）
#     keywords = re.findall(r'[\u4e00-\u9fff]+', question)
    
#     selected_tables = []
#     for table in tables:
#         # 遍历每个字段含义
#         for meaning in table["字段含义"]:
#             # 检查是否包含任意关键词
#             if any(keyword in meaning for keyword in keywords):
#                 selected_tables.append(table["数据表名"])
#                 break  # 找到匹配即停止检查该表
    
#     # 去重并转换为字符串格式返回
#     return str(list(set(selected_tables)))

# In[5]:

def glm4_create(max_attempts, messages, tools, model="glm-4-plus"):
# def glm4_create(max_attempts, messages, tools, model=llm_config["model"]):
    client = ZhipuAI()
    for attempt in range(max_attempts):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
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

    "action_judgment": api.action_judgment

    # "calculate_energy_consumption": api.calculate_energy_consumption,
    # "calculate_total_energy_consumption": api.calculate_total_energy_consumption,
    
}


def get_answer_2(question, tools, api_look: bool = True):
    filtered_tools = tools
    try:
        messages = [
            {
                "role": "system",
                "content": "不要假设或猜测传入函数的参数值。如果用户的描述不明确，请要求用户提供必要信息。请一步一步思考，必须提供实际的数据查询结果",
            },
            {"role": "user", "content": question},
        ]
        # 第一次调用模型
        response = glm4_create(6, messages, filtered_tools)
        messages.append(response.choices[0].message.model_dump())
        function_results = []
        # 最大迭代次数
        max_iterations = 8


        #调用多个工具
        for _ in range(max_iterations):
            if not response.choices[0].message.tool_calls:
                break
            # 遍历所有工具调用
            for tool_call in response.choices[0].message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                function_name = tool_call.function.name
                # 执行工具函数
                if function_name in function_map:
                    function_result = function_map[function_name](**args)
                    function_results.append(function_result)
                    messages.append({
                        "role": "tool",
                        "content": f"{function_result}",
                        "tool_call_id": tool_call.id,
                    })
                else:
                    print(f"未找到对应的工具函数: {function_name}")
                    continue
            # 更新模型响应
            response = glm4_create(8, messages, filtered_tools)
        return response.choices[0].message.content, str(function_results)
    except Exception as e:
        print(f"Error generating answer for question: {question}, {e}")
        return None, None


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
    api_list_filter=[]
    question=question.strip()

    if "舵桨" in question and "能耗" in question:
        print("舵桨jinl")
        api_list_filter.append("calculate_tuojiang_energy")
    
    if re.search(r"A架.*能耗", question):
        print("Ajiajinl=================================")
        api_list_filter.append("calculate_ajia_energy")
        print(api_list_filter)
    #推进系统改为推进器
    if "推进器" in question and "能耗" in question:
        api_list_filter.append("calculate_total_propulsion_system_energy")
    if "甲板机械" in question and "能耗" in question:
        api_list_filter.append("calculate_total_deck_machinery_energy")
        question = question + "能耗为负值也要带入计算。"
    elif "燃油消耗量" in question:
            # api_list_filter.append("get_table_data")
        api_list_filter.append("calculate_total_fuel_volume")  
        api_list_filter.append("calculate_total_4_fuel_volume")
        if "号" not in question:
            api_list_filter.remove("calculate_total_fuel_volume")

    elif "能耗" in question and "平均能耗" not in question:
        # api_list_filter = ["calculate_total_deck_machinery_energy"]
        api_list_filter.append("calculate_total_energy")
    # elif "总能耗" in question:
    #     api_list_filter = ["calculate_total_energy"]
    #     question = question + "总能耗计算示例：一般来说，能耗等于功率*作用时间，对于一段时间，如t1-tn时刻，需要先计算t_diff，t_diff为数组(t2-t1, t3-t2,……, tn-t(n-1))，能耗=sum(功率*t_diff) 效率一般是实际做功/理论做功*100%。"
    elif "总发电量" in question or "总发电" in question:
        print("总发电量1111111111111111111111")
        api_list_filter.append("calculate_total_electricity_generation")
    # elif "理论发电量" in question:
    #     api_list_filter = ["calculate_theory_electricity_generation"]
    
    elif "动作" in question or any(value in question for value in action_list):
        api_list_filter.append("get_device_status_by_time_range")
        question = question + "##动作直接引用不要修改,如【A架摆回】##"
    elif "开机时长" in question:
        api_list_filter.append("calculate_uptime")
        if "运行时长" in question:
            question = question.replace("运行时长", "开机时长")
    elif ("运行时长" in question or "运行时间" in question) and "实际运行时长" not in question:
        api_list_filter.append("calculate_uptime")
        question = question.replace("运行时长", "开机时长")
    elif "实际运行时长" in question or "实际运行时间" in question or "实际开机时间" in question or "实际开机时长" in question:
        api_list_filter = ["compute_actual_operational_duration","calculate_uptime"]
        question = question + "##实际运行时长为有电流的时长，效率的计算示例：运行效率=实际运行时长/运行时长。##"
    elif ("时长" in question or "时间" in question or "多久" in question) and ("平均" in question) and ("折臂吊车" in question or "DP" in question or "A架" in question):
        api_list_filter = ["calculate_uptime"]
        question = "##涉及多天的平均和比例问题，均以天数为分母计算。如：【2024/8/23 和 2024/8/25 上午A架运行的平均时间多少】理解为【(2024/8/23上午A架运行时间 + 2024/8/25上午A架运行时间)/2，注意不计算24日】##" + question
    elif "数据" in question and "缺失" in question:
        table_name_string = choose_table(question)
        print(question)
        with open("dict.json", "r", encoding="utf-8") as file:
            table_data = json.load(file)

        table_name = [
            item for item in table_data if item["数据表名"] in table_name_string
        ]
        question = str(table_name)+question
        api_list_filter.append("calculate_sum_data")
    
    elif "A架摆动的次数" in question:
        api_list_filter.append("count_swings_10")
    elif "A架完整摆动的次数" in question:
        api_list_filter.append("count_swings")
    elif "数值类型参数" in question:
                # api_list_filter.remove("query_device_parameter")
        api_list_filter.append("rt_temp_or_press_parameter")
        question= question+"至少触发报警数举例：当参数值（题目的假设值）低于200，而报警值（Parameter_Information_Alarm）是低于20时，则不一定会触发报警，因为低于200不一定是低于20"
    
    else:
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
        print("okkkkkkkkkk")

        if "设备参数详情表" in [item["数据表名"] for item in table_name] or "范围" in question:
            api_list_filter = ["query_device_parameter"]
            # question = question + ""
            content_p_1 = str(table_name) + question  # 补充 content_p_1
            # print(content_p_1)
        else:
            # print("ok")
            api_list_filter=["get_table_data"]
            content_p_1 = str(table_name) + question
            # print(content_p_1)

    #移除##内的内容
    print(api_list_filter)
    ques = remove_parentheses(question)
    if "作业" in ques and ("能耗" in ques or "发电" in ques):
        api_list_filter.append("get_device_status_by_time_range")
        api_list_filter.append("calculate_total_electricity_generation")
        
        question = ques
        # question = ques + "若没有指明阶段，则默认计算一整天,例如计算从动作A到动作B期间的总能耗，则将A的时间点和B的时间点作为参数传入能耗计算。。"

    else:
        if "告警" in ques or "预警" in ques or "正常" in ques or "故障" in ques or "异常" in ques or 'RPM' in ques or "高于" in ques or "低于" in ques or "上下限" in ques or "报警" in ques:
            
            api_list_filter.append("query_device_parameter")
            # print(api_list_filter)
        if "能耗" in ques:
            api_list_filter.append("calculate_start_end")
            api_list_filter.append("calculate_total_energy")
            api_list_filter.append("get_device_status_by_time_range")
            print("能耗问题-----------")
            
        if "平均" in ques:
            question = question + "##涉及多天的平均问题，均以天数为分母计算。如：【2024/8/23 和 2024/8/25 上午A架运行的平均时间多少】理解为【(2024/8/23上午A架运行时间 + 2024/8/25上午A架运行时间)/2，注意不计算24日】。##"
            ques = remove_parentheses(question)
        if "作业" in ques and ("发电" in ques or "能耗" in ques):
            print("作业22222222222222222222222")
            api_list_filter.append("calculate_total_electricity_generation")
            # question = question + "##作业由下放阶段和回收阶段两个阶段组成.上午为下放阶段，阶段开始和结束默认为ON DP和OFF DP；下午为回收阶段，阶段开始和结束默认为A架开机和A架关机。若题目中有标志则优先使用标志。作业能耗=总发电量。##"
            # question = question + "##若题目中有标志则优先使用标志。作业能耗=总发电量。##"
            ques = remove_parentheses(question)
        if ("时间" in ques or "时长" in ques or "多久" in ques) and ("折臂吊车" in ques or "DP" in ques or "A架" in ques) and ("时间点" not in ques) and ("相隔" not in ques):
            api_list_filter.append("calculate_start_end")
            print("时间问题-----------")
        if "下放" in question or "回收" in question:
            api_list_filter.append("calculate_start_end")
            print("下放、回收问题-----------")
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
            question = question + "##理论发电量=燃油消耗量*密度*燃油热值/3.6，发电效率=四个发电机总发电量/理论发电量,##"
        if "发电机组" in ques and "号" not in ques and "应急发电机组" not in ques:
            question = question + "##发电机组即为一、二、三、四号柴油发电机组。##" 
            ques = remove_parentheses(question)
            api_list_filter.append("calculate_total_4_fuel_volume")
        if "燃油消耗量" in ques:
            # api_list_filter.append("get_table_data")
            # api_list_filter.append("calculate_total_fuel_volume")  
            api_list_filter.append("calculate_total_4_fuel_volume")
            if "号" not in ques:
                api_list_filter.remove("calculate_total_fuel_volume")
            # question = question + "##燃油消耗量=这段时间内燃油消耗率之和。##"
        # if "DP" in question:
        #     question = question + "ON DP和OFF DP的仅发生在下午，若查表有下午数据则忽略"
        if "平均值" in ques or "最大值" in ques or "最小值" in ques:
            api_list_filter.append("calculate_max_or_average")
        if "数据" in ques and "缺失" in ques:
            api_list_filter.append("calculate_sum_data")
        if "A架摆动的次数" in ques:
            api_list_filter.append("count_swings_10")
        if "A架完整摆动的次数" in ques:
            api_list_filter.append("count_swings")
    
    if "A架" in ques and "能耗" in ques:
        print("api_list_filter:", api_list_filter)
        api_list_filter.remove("calculate_total_energy")
        api_list_filter.remove("calculate_total_energy")
        print("api_list_filter:", api_list_filter)

    if "假设2024/08/20" in ques or re.search(r".*进行.*分别是", ques):
        api_list_filter= ["action_judgment"]
        print("shhdhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
        print("api_list_filter:", api_list_filter)
        question = question + "##若题目中没有指明下放还是回收阶段,则所要查询动作包括缆绳挂妥,A架摆出,征服者出水和征服者落座时,为回收阶段,否则为下放阶段.##"
    
    
    
    print("question:", question)
    print("ques:",ques)

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


def enhanced(prompt1, context=None, instructions=None, modifiers=None):
    """
    增强提示词函数
    """
    prompt1 = str(prompt1)
    print("原始 prompt1:", prompt1)
    enhanced_prompt = prompt1  # 初始化为原始字符串
    enhanced_prompt = enhanced_prompt.replace("XX小时XX分钟", "XX小时XX分钟，01小时01分钟格式,不需要秒")
    enhanced_prompt = enhanced_prompt.replace("请以XX:XX输出", "请以XX:XX输出，01:01的格式,不需要秒")
    enhanced_prompt = enhanced_prompt.replace("发生了什么", "什么设备在进行什么动作，动作直接引用不要修改,如【A架摆回】")
    enhanced_prompt = enhanced_prompt.replace("深海作业A回收过程中", "最后一次")
    enhanced_prompt = enhanced_prompt.replace("深海作业A作业", "作业")
    # enhanced_prompt = enhanced_prompt.replace("深海作业A作业结束", "OFF DP")
    # enhanced_prompt = enhanced_prompt.replace("作业开始", "作业开始（若问题中有标志则使用标志，若没有标志，则上午的作业开始标志为'ON_DP'，下午作业开始的标志为'A架开机'）")
    # enhanced_prompt = enhanced_prompt.replace("作业结束", "作业结束（若问题中有标志则使用标志，若没有标志，则上午的作业结束标志为'OFF_DP'，下午作业结束的标志为'A架关机'）")
    # enhanced_prompt = enhanced_prompt.replace("开始作业", "作业开始（若问题中有标志则使用标志，若没有标志，则上午的作业开始标志为'ON_DP'，下午作业开始的标志为'A架开机'）")
    # enhanced_prompt = enhanced_prompt.replace("结束作业", "作业结束（若问题中有标志则使用标志，若没有标志，则上午的作业结束标志为'OFF_DP'，下午作业结束的标志为'A架关机'）")
    enhanced_prompt = enhanced_prompt.replace("小艇下水", "小艇入水")
    # enhanced_prompt = enhanced_prompt.replace("假设深海作业A不会跨天进行", "##假设深海作业A不会跨天进行,一天至少两次A架开机和两次A架关机且存在其他动作才可视为作业进行##")
    enhanced_prompt = enhanced_prompt.replace("作业", "作业##作业由下放阶段以及回收阶段两个阶段组成。一般来说第一次A架开关机为布放阶段，第二次A架开关机为回收阶段，但若题目中指明布放和回收的标志，则以题目为准。若题目中没有说明阶段则默认两个阶段。##")
    
    # enhanced_prompt = enhanced_prompt.replace("下放阶段以ON DP以及OFF DP为标志，回收阶段以A架开机以及关机为标志", "")
    enhanced_prompt = enhanced_prompt.replace("是否正确", "是否正确（回答正确或不正确）")
    enhanced_prompt = enhanced_prompt.replace("期间", "期间（若为DP，仅计算第一次开关）")
    enhanced_prompt = enhanced_prompt.replace("（含）", "（含）这几天中")
    enhanced_prompt = enhanced_prompt.replace("推进器", "推进器（推进器包括：一号推进变频器+二号推进变频器+侧推+可伸缩推）")
    enhanced_prompt = enhanced_prompt.replace("甲板机械", "甲板机械（甲板机械包括：折臂吊车、一号门架、二号门架、绞车）")
    enhanced_prompt = enhanced_prompt.replace("做功", "能耗")
    enhanced_prompt = enhanced_prompt.replace("时间差是多少", "时间差是多少分钟")
    enhanced_prompt = enhanced_prompt.replace("与温度相关", "与温度相关（单位为℃）")
    enhanced_prompt = enhanced_prompt.replace("与压力相关", "与压力相关（单位为kPa）")

    action_list=["A架开机","ON DP","征服者起吊","征服者入水","缆绳解除","A架摆回","小艇落座","A架关机","OFF DP","折臂吊车开机","A架摆出","小艇检查完毕","小艇入水","缆绳挂妥","征服者出水","折臂吊车关机","征服者落座"]
    for value in action_list:
        if value in enhanced_prompt:
            enhanced_prompt=enhanced_prompt.replace(value,"【"+value+"】")
    # if re.search(r'深海.*之间', enhanced_prompt):
    #     # enhanced_prompt = remove_parentheses(enhanced_prompt)
    #     enhanced_prompt = re.sub(r"深海.*之间", "", enhanced_prompt)
    #     # print(enhanced_prompt)

    # if re.search(r'A架.*阶段', enhanced_prompt):
    #     # enhanced_prompt = remove_parentheses(enhanced_prompt)
    #     enhanced_prompt = re.sub(r"A架.*阶段", "", enhanced_prompt)
    #     # print(enhanced_prompt)
    

    # if "电流不平衡度" in enhanced_prompt:
    #     enhanced_prompt = enhanced_prompt + "最终答案为所查到数值/100,如78%为0.78%，返回整数则为1%，4466%为44.66%,返回整数则为45%"
    if "一号舵桨转舵A-Ua电压" in enhanced_prompt:
        enhanced_prompt = enhanced_prompt + "最终答案为所查到数值/10,如391=39.1V，返回整数则为40V."
    if "一号舵桨转舵A-频率" in enhanced_prompt:
        enhanced_prompt = enhanced_prompt + "最终答案为所查到数值/100,如391=3.91Hz，返回整数则为4Hz."
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
        last_answer = last_answer.replace(" ", "")
        return last_answer
    except Exception as e:
        print(f"Error occurred while executing get_answer: {e}")
        return "An error occurred while retrieving the answer."


def review_answer(answer):
    """
    验证并修正答案中的计算过程
    
    :param answer: 包含计算过程的原始答案文本
    :return: 修正后的答案文本（若无错误返回原答案）
    """
    # 构造验证提示词
    system_prompt = """您是一位严谨的数学老师，请严格按以下要求工作：
1. 仔细检查用户提供的计算过程,如加减乘除极其混合运算,特别是计算时的进位与借位，时间点的比较
2. 发现计算错误时，指出错误位置并给出正确计算
3. 若没有错误，直接返回输入
4. 若没有计算步骤,则当作正确答案处理
5. 若出现AA:BB，视为AA小时BB分钟
6. 仅需要检查计算即可，不需要检查是否合理
7. 一个数字中不要加分隔符（如用逗号分隔千位等），保持数字的完整性
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
        # print(f"GLM4验证结果：{feedback}")
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

def get_end_answer(question, answer):
    """
    根据问题总结答案内容，返回精简的回答
    
    :param question: 用户提出的问题
    :param answer: 需要总结的原始答案
    :return: 结构化整理后的最终答案
    """


    # answer = answer + "\n"+ review_answer(answer)
    answer = review_answer(answer)
    # print("start&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" + answer)
    # print("end&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    # answer = "首先计算两天的总能耗：总能耗 = 19923.6 + 10135.73 + 7795.5 + 5703.72 = 53558.55 kWh,然后计算平均能耗：平均能耗 = 总能耗 / 天数 = 53558.55 / 2 = 26779.275 kWh,保留两位小数，最终的平均作业能耗为：26779.28 kWh"
    # answer = answer + "\n"+ review_answer(answer)
    # print(f"Answer: {answer}")
    # 构造对话消息
    messages = [
        {
            "role": "system",
            "content": "你是一个专业的内容总结助手，请根据用户的问题总结提供的答案内容，直接返回能够准确回答问题和满足格式要求的精简答案,注意:1.动作直接引用不要修改,如【A架摆回】;2.需要带单位的必须带单位;3.若题目中有答案格式要求，请严格按照要求，多个答案以题目中所给格式隔开，4.禁止返回假设内容"
        },
        {
            "role": "user",
            "content": f"请根据以下问题总结答案：{question}\n原始答案内容：{answer}"
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
    # last_answer = last_answer.replace(" ", "")
    try:
        return last_answer
    except AttributeError:
        return "答案生成失败，请稍后再试"

# In[7]:


if __name__ == "__main__":
   
    # question = "假设2024/08/20 上午（0:00~12:00）征服者入水的时间是09:59，请指出小艇入水以及征服者起吊发生的时间（以XX:XX输出，如有多个请使用英文逗号隔开）。"
    question = "1. 2024年5月23日23:00至24:00（包括这两个时间点）进行了一次特定类型的作业，请指出该次作业使用的甲板机器设备有哪些？（多个设备使用空格隔开，设备名称大小写敏感，且空格重要；每个设备仅写一次；如果没有使用任何设备，则输出N）"
    question = "2. 请指出2024年5月25日深海作业A回收阶段中，A架摆出、征服者起吊和征服者落座的确切时间（以XX:XX格式输出，时间为24小时制，时间补零，如05:03；多个时间使用单个空格隔开，按照事件顺序输出；如果某事件未发生，则在该位置输出N）"


    question = "14. 在2024年8月25日深海作业A过程中，从A架摆回至A架摆出之间相差的时长（根据设备记录的时间计算，四舍五入到整数，单位为分钟），以及在这段时间内四台发电机的总发电量（单位为kWh，保留两位小数）。如果遇到数据缺失或异常情况，请根据相邻数据点进行合理估算。"
    
    question = "18. 在2024年6月15日深海作业A中，请输出布放和回收的开始和结束时间（以折臂吊车开机为开始，折臂吊车关机为结束，时间以24小时制HH:MM格式按发生顺序输出），并比较布放和回收阶段的燃油消耗量，输出较大的燃油消耗量（燃油消耗量单位为L，结果保留两位小数）。如果遇到数据缺失或异常情况，请根据相邻数据点进行合理估算。如果当天进行多次布放和回收，请比较所有布放和回收阶段的燃油消耗量，并输出消耗量最大的一次。"
    
    
    question = "2024/05/17 一号舵桨转舵A-频率的平均值是多少（单位为Hz，四舍五入，以整数输出）？"
    aa = get_answer(question)
    question = enhanced(question)
    print(question+"====================ok")
    bb = get_end_answer(question,aa)
    print("*******************最终答案***********************")
    print(bb)
    # 文件路径
    """
    question_path = "assets/question.jsonl"
    result_path = "./result.jsonl"
    intermediate_result_path = "./result_zj.jsonl"
    # 读取问题文件
    with open(question_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line.strip()) for line in f]
    # 处理每个问题并保存结果
    questions=questions[:1] # 注释掉这一行以逐个回答全部问题
    results = []
    for question in questions:
        query = question["question"]
        # 调用AI模块获取答案
        try:
            answer =get_answer(question=query)
            answer_str = str(answer)
            print(f"Question: {query}")
            print(f"Answer: {answer_str}")
            result = {
                "id": question["id"],
                "question": query,
                "answer": answer_str
            }
            results.append(result)
            # 将中间结果写入文件
            with open(intermediate_result_path, "w", encoding="utf-8") as f:
                f.write("\n".join([json.dumps(res, ensure_ascii=False) for res in results]))
        except Exception as e:
            print(f"Error processing question {question['id']}: {e}")
    # 将最终结果写入文件
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("\n".join([json.dumps(res, ensure_ascii=False) for res in results]))

"""

# %%
