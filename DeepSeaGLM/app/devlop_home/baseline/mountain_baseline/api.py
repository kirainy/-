import pandas as pd
import numpy as np
# from data_process import find_peaks, find_first_increasing_value, find_first_stable_after_peak, find_peak_after_stable
# import pandas as pd
# from data_process_new.find_peaks import find_peaks_and_time_segments
from data_process import find_peaks, find_first_increasing_value, find_first_stable_after_peak, find_peak_after_stable
  
def find_peaks(data1):
    # 数据预处理 - data1现在包含两列数据的元组
    data = data1[:-1]
    def is_normal_range(val1, val2):
        return (50 <= val1 < 60 or 50 <= val2 < 60)
        # return (50 <= val1 <= 60 and 50 <= val2 <= 68) or (50 <= val2 <= 60 and 50 <= val1 <= 68)

    peaks = []
    
    # if data[0] ==  [55.3599, 56.2103]: # 2024/5/25 19:29:49
    #     print('debug')

    # Find segments where values exceed normal range
    i = 0
    while i < len(data):
        # Look for start of abnormal segment
        # Need at least 2 normal values before
        if i < len(data) - 1 and not is_normal_range(data[i][0], data[i][1]):
            i += 1
            continue
            
        normal_before = 0
        while i < len(data) and is_normal_range(data[i][0], data[i][1]):
            normal_before += 1
            i += 1
            
        if normal_before < 1 or i >= len(data):
            i += 1
            continue
            
        # Found start of abnormal segment
        start_abnormal = i
        
        # Look for end of abnormal segment
        while i < len(data) and not is_normal_range(data[i][0], data[i][1]):
            i += 1
            
        end_abnormal = i
        
        # Need at least 2 abnormal values
        if end_abnormal - start_abnormal < 2:
            continue
            
        # Look for normal values after
        normal_after = 0
        while i < len(data) and is_normal_range(data[i][0], data[i][1]):
            normal_after += 1
            i += 1
            
        # if normal_after < 1:
        #     continue
            
        # Found a valid segment, find peak within it
        max_sum = -float('inf')
        peak_val = None
        
        for j in range(start_abnormal, end_abnormal):
            curr_sum = data[j][0] + data[j][1]
            if curr_sum > max_sum and data[j][0] > 60 and data[j][1] > 60:
                max_sum = curr_sum
                peak_val = data[j][1]
                
        if peak_val is not None:
            peaks.append(peak_val)
        i -= 1

    # Filter peaks > 80
    peaks = [peak for peak in peaks if peak > 75]
    
    # 返回峰值个数和具体峰值
    return len(peaks), peaks

def xiafang(df):
    '''
    对df时间段内的数据进行处理，找到征服者起吊、缆绳解除、征服者入水、A架摆回的时间点
    '''
    # print(df[['csvTime','check_current_presence']])
    events = df[(df['check_current_presence'].isin(['有电流', '无电流']))]
    # print(events)

    # 检查事件数量是否为偶数
    if events.shape[0] >= 2 and events.shape[0] % 2 == 0:
        # 遍历所有偶数索引的事件对
        for i in range(0, events.shape[0], 2):
            event_start = events.iloc[i]
            event_end = events.iloc[i + 1]
            # 确保第一个事件是“有电流”，第二个事件是“无电流”
            if event_start['check_current_presence'] == '有电流' and event_end['check_current_presence'] == '无电流':
                start_event_time = event_start['csvTime']
                end_event_time = event_end['csvTime']

                # 提取两个事件之间的数据
                between_events = df[(df['csvTime'] >= start_event_time) & (df['csvTime'] <= end_event_time)]
                data1 = between_events[['Ajia-3_v', 'Ajia-5_v']].values.tolist()
                len_peaks, peak_L = find_peaks(data1)
                if len_peaks == 0:
                    continue
                value_11 = find_first_increasing_value(data1)
                indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                df.loc[indices, 'status'] = '征服者起吊'
                if not indices:
                    print('征服者起吊找不到索引',value_11)
                value_11 = find_first_stable_after_peak(data1, df.loc[indices[0], 'Ajia-5_v'])
                indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                df.loc[indices, 'status'] = '缆绳解除'
                if not indices:
                    print('缆绳解除找不到索引',value_11)
                previous_indices = [idx - 1 for idx in indices if idx > 0]
                df.loc[previous_indices, 'status'] = '征服者入水'
                # A架摆回
                value_11 = find_peak_after_stable(data1, peak_L[-1])
                indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                df.loc[indices, 'status'] = 'A架摆回'
                return df

def huishou(df):
    events = df[(df['check_current_presence'].isin(['有电流', '无电流']))]
    # 检查事件数量是否为偶数
    if events.shape[0] >= 2 and events.shape[0] % 2 == 0:
        # 遍历所有偶数索引的事件对
        for i in range(0, events.shape[0], 2):
            event_start = events.iloc[i]
            event_end = events.iloc[i + 1]
            # 确保第一个事件是“有电流”，第二个事件是“无电流”
            if event_start['check_current_presence'] == '有电流' and event_end['check_current_presence'] == '无电流':
                start_event_time = event_start['csvTime']
                end_event_time = event_end['csvTime']

                # 提取两个事件之间的数据
                between_events = df[(df['csvTime'] >= start_event_time) & (df['csvTime'] <= end_event_time)]
                data1 = between_events[['Ajia-3_v', 'Ajia-5_v']].values.tolist()
                len_peaks, peak_L = find_peaks(data1)
                if len_peaks == 0:
                    continue
                indices = between_events.index[between_events['Ajia-5_v'] == peak_L[-1]].tolist()
                df.loc[indices, 'status'] = '征服者出水'
                previous_indices = [idx - 1 for idx in indices if idx > 0]
                df.loc[previous_indices, 'status'] = '缆绳挂妥'
                value_11 = find_first_stable_after_peak(data1, peak_L[-1])
                indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                df.loc[indices, 'status'] = '征服者落座'
                return df
    return 

def action_judgment(start_time, end_time, stage, time_of_action=None):
    """
    在指定时间段内查找A架和小艇的动作，如果A架找不到动作，则在该时间段内进行动作判定（小艇肯定找得到动作）。最终返回运行状态和操作建议
    :param start_time: 查询的开始时间（字符串或 datetime 类型）
    :param end_time: 查询的结束时间（字符串或 datetime 类型）
    :param stage: 深海作业A阶段名称（字符串）
    :param time_of_action: 已知的某个动作时间（字符串，可以为空），该时间点必须包含在查询的时间段内
    :return: A架相关设备动作及对应的时间（字典）
    """
    print("-------action_judgment执行***查找A架设备在指定时间段内的动作-------")
    # 读取 A架 PLC 数据
    df = pd.read_csv("database_in_use/Ajia_plc_1.csv")

    segment = None
    # 筛选指定时间段内的数据
    df_filtered = df[(df["csvTime"] >= start_time) & (df["csvTime"] <= end_time)]
    # print('过滤前的df：\n',df_filtered)
    # 如果time_of_action不为空，添加包含该时间点的数据筛选条件
    if time_of_action:
        # time_of_action = pd.to_datetime(time_of_action)
        # 确保time_of_action在查询时间段内
        if not (start_time <= time_of_action <= end_time):
            raise ValueError("time_of_action必须在查询的时间段内")
        # 找到最近的一个包含time_of_action的开关机时间段
    start_uptime = None
    end_uptime = None
    contains_action = False
    
    for index, row in df_filtered.iterrows():
        if row["status"] == "A架开机":
            start_uptime = row["csvTime"]
        elif row["status"] == "A架关机":
            end_uptime = row["csvTime"]
            if start_uptime is None:
                continue
            if time_of_action:
                if start_uptime <= time_of_action <= row["csvTime"]:
                    contains_action = True
                    df_filtered = df_filtered[(df_filtered["csvTime"] >= start_uptime) & (df_filtered["csvTime"] <= row["csvTime"])]
                    segment = (start_uptime, row["csvTime"])
                    break
            else:
                contains_action = True
                df_filtered = df_filtered[(df_filtered["csvTime"] >= start_uptime) & (df_filtered["csvTime"] <= row["csvTime"])]
                segment = (start_uptime, row["csvTime"])
                break
    if not contains_action:
        # If we only have start_uptime (only power on, no power off)
        if start_uptime is not None and end_uptime is None:
            df_filtered = df_filtered[(df_filtered["csvTime"] >= start_uptime) & (df_filtered["csvTime"] <= end_time)]
            segment = (start_uptime, end_time)
        # If we only have end_uptime (only power off, no power on)  
        elif start_uptime is None and end_uptime is not None:
            df_filtered = df_filtered[(df_filtered["csvTime"] >= start_time) & (df_filtered["csvTime"] <= end_uptime)]
            segment = (start_time, end_uptime)
        # If we have neither start_uptime nor end_uptime
        elif start_uptime is None and end_uptime is None:
            # Keep original df_filtered
            segment = (start_time, end_time)
        
    # print('过滤后的df：\n',df_filtered[['csvTime','status']])
    # 初始化变量
    action_list = []  # 动作列表
    action_time_list = []  # 动作时间列表

    # 遍历筛选后的数据，如果已经有了动作，就不再判定动作
    for index, row in df_filtered.iterrows():
        if row['status'] not in ['A架开机', 'A架关机', 'False']:
            action_list.append(row["status"])
            action_time_list.append(row["csvTime"])

    if action_list == [] or action_time_list == []:
        if stage == '布放':
            df_filtered = xiafang(df_filtered)
        else:
            df_filtered = huishou(df_filtered)

    # 添加空值检查
    if df_filtered is None or df_filtered.empty:
        return {
            "function": "action_judgment",
            "result": {
                "action_list": [],
                "action_time_list": [],
            }
        }

    for index, row in df_filtered.iterrows():
        if row['status'] not in ['A架开机', 'A架关机', 'False']:
            action_list.append(row["status"])
            action_time_list.append(row["csvTime"])

    # 查找折臂吊车的动作
    df_device = pd.read_csv("database_in_use/device_13_11_meter_1311.csv")
    df_device = df_device[(df_device["csvTime"] >= start_time) & (df_device["csvTime"] <= end_time)]
    for index, row in df_device.iterrows():
        if row['status'] not in ['折臂吊车开机', '折臂吊车关机', 'False']:
            action_list.append(row["status"])
            action_time_list.append(row["csvTime"])
    
        # 返回结果
    result = {
        "function": "action_judgment",  # 说明这个返回结果来自哪个函数
        "result": {
            "action_list": action_list,
            "action_time_list": action_time_list,
        },
    }

    print(f"Answer: {result}")
    return result




# 计算开机时长
def calculate_uptime(start_time, end_time, shebeiname="折臂吊车"):
    """
    计算指定时间段内的开机时长，并返回三种格式的开机时长
    :param start_time: 查询的开始时间（字符串或 datetime 类型）
    :param end_time: 查询的结束时间（字符串或 datetime 类型）
    :param shebeiname: 设备名称，默认为 '折臂吊车'
    :return: 返回时长（分钟）以及开关的次数
    """
    print("-------calculate_uptime执行***计算开机时间-------"+shebeiname)
    # 设备配置映射：设备名称 -> (文件路径, 开机状态, 关机状态)
    device_config = {
        "折臂吊车": (
            "database_in_use/device_13_11_meter_1311.csv",
            "折臂吊车开机",
            "折臂吊车关机",
        ),
        "A架": ("database_in_use/Ajia_plc_1.csv", "A架开机", "A架关机"),
        "DP": ("database_in_use/Port3_ksbg_9.csv", "ON DP", "OFF DP"),

        "一号柴油发电机": ("database_in_use/Port1_ksbg_3.csv", "P1_88.14"),
        "二号柴油发电机": ("database_in_use/Port1_ksbg_4.csv", "P1_90.5"),
        "三号柴油发电机": ("database_in_use/Port2_ksbg_3.csv", "P2_73.8"),
        "四号柴油发电机": ("database_in_use/Port2_ksbg_3.csv", "P2_74.15"),

        # "侧推": ("database_in_use/Port3_ksbg_9.csv", "P3_18"),
    }

    # 检查设备名称是否有效
    if shebeiname not in device_config:
        raise ValueError(f"未知的设备名称: {shebeiname}")
    elif shebeiname == "折臂吊车" or shebeiname == "A架" or shebeiname == "DP":
        # 获取设备配置
        file_path, start_status, end_status = device_config[shebeiname]

        # 读取 CSV 文件
        df = pd.read_csv(file_path)
        # print(file_path)

        # 将时间列转换为 datetime 类型
        df["csvTime"] = pd.to_datetime(df["csvTime"])

        # 将传入的开始时间和结束时间转换为 datetime 类型
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)

        # 检查并确保 start_time 早于 end_time，如果不是则对调
        if start_time > end_time:
            start_time, end_time = end_time, start_time


        
        print(f"开始时间: {start_time},结束时间: {end_time}")

        # 筛选出指定时间段内的数据
        df_filtered = df[(df["csvTime"] >= start_time) & (df["csvTime"] <= end_time)]
        # print(df_filtered)

        # 初始化变量
        total_duration = pd.Timedelta(0)
        start_uptime = None

        #计算开关的次数
        cnt = 0
        # 遍历筛选后的数据
        for index, row in df_filtered.iterrows():
            if row["status"] == start_status :
                start_uptime = row["csvTime"]
            elif row["status"] == end_status and start_uptime is not None:
                end_uptime = row["csvTime"]
                total_duration += end_uptime - start_uptime
                start_uptime = None
                cnt += 1

        # 计算三种格式的开机时长
        seconds = total_duration.total_seconds()
        minutes = int(seconds / 60)
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)

        # 将小时和分钟格式化为两位数
        hours_str = f"{hours:02d}"  # 使用格式化字符串确保两位数
        minutes_str = f"{remaining_minutes:02d}"  # 使用格式化字符串确保两位数

        # 返回三种格式的开机时长
        result = {
            "function": "calculate_uptime",  # 说明这个返回结果来自哪个函数
            "result": (
                # f'开机时长：{seconds}秒',
                f"开机时长：{minutes}分钟",
                # f'开机时长：{hours_str}小时{minutes_str}分钟'
            ),
        }
        print(f"Answer: {result}")
        # return result
        return minutes,cnt

    elif shebeiname =="一号柴油发电机" or shebeiname == "二号柴油发电机" or shebeiname == "三号柴油发电机" or shebeiname == "四号柴油发电机":
        file_path, column_name = device_config[shebeiname]
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 转换时间格式
        df["csvTime"] = pd.to_datetime(df["csvTime"])
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)

         # 检查并确保 start_time 早于 end_time，如果不是则对调
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        
        # 筛选时间范围内的数据
        mask = (df["csvTime"] >= start_time) & (df["csvTime"] <= end_time)
        df_filtered = df.loc[mask].copy()
        
        # 清洗布尔列数据（假设1表示True，0表示False）
        df_filtered[column_name] = df_filtered[column_name].astype(int)
        
        # 计算有效行数（开机时长）
        uptime_rows = df_filtered[df_filtered[column_name] == 1].shape[0]
        
        # 计算开关次数（状态从0->1的变化次数）
        df_filtered['status_change'] = (df_filtered[column_name] != df_filtered[column_name].shift(1)).astype(int)
        switch_count = df_filtered[(df_filtered[column_name] == 1) & (df_filtered['status_change'] == 1)].shape[0]
        
        # 删除临时列
        df_filtered.drop('status_change', axis=1, inplace=True)
        
        print(f"【{shebeiname}】统计结果 | 总开机时长：{uptime_rows}分钟 | 开关次数：{switch_count}次")
        return uptime_rows

# calculate_uptime("2024-06-06 00:00:00","2024-06-05 00:00:00","三号柴油发电机")
# 计算A架的实际开机时长
def compute_actual_operational_duration(start_time, end_time, device_name="A架"):
    # 设备配置映射：设备名称 -> (文件路径, 开机状态, 关机状态)
    print("-------compute_actual_operational_duration执行-------")
    device_config = {
        "A架": ("database_in_use/Ajia_plc_1.csv", "有电流", "无电流"),
    }
    # 检查设备名称是否有效
    if device_name not in device_config:
        raise ValueError(f"未知的设备名称: {device_name}")

    # 获取设备配置
    file_path, start_status, end_status = device_config[device_name]

    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 将时间列转换为 datetime 类型
    df["csvTime"] = pd.to_datetime(df["csvTime"])

    # 将传入的开始时间和结束时间转换为 datetime 类型
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    # 筛选出指定时间段内的数据
    df_filtered = df[(df["csvTime"] >= start_time) & (df["csvTime"] <= end_time)]

    # 初始化变量
    total_duration = pd.Timedelta(0)
    start_uptime = None
    # 遍历筛选后的数据

    # flag = False
    for index, row in df_filtered.iterrows():
        if row["check_current_presence"] == start_status:
            start_uptime = row["csvTime"]

        elif row["check_current_presence"] == end_status and start_uptime is not None:
            end_uptime = row["csvTime"]
            total_duration += end_uptime - start_uptime
            start_uptime = None

    # 计算三种格式的开机时长
    seconds = total_duration.total_seconds()
    minutes = int(seconds / 60)
    hours = int(seconds // 3600)
    remaining_minutes = int((seconds % 3600) // 60)

    # 将小时和分钟格式化为两位数
    hours_str = f"{hours:02d}"  # 使用格式化字符串确保两位数
    minutes_str = f"{remaining_minutes:02d}"  # 使用格式化字符串确保两位数

    result = {
        "function": "calculate_uptime",  # 说明这个返回结果来自哪个函数
        "result": (
            # f'运行时长：{seconds}秒',
            f"实际运行时长：{minutes}分钟",
            # f'运行时长：{hours_str}小时{minutes_str}分钟'
        ),
    }
    return result


def get_table_data(table_name, start_time, end_time, columns=None, status=None):
    """
    根据数据表名、开始时间、结束时间、列名和状态获取指定时间范围内的相关数据。

    参数:
    table_name (str): 数据表名
    start_time (str): 开始时间，格式为 'YYYY-MM-DD HH:MM:SS'
    end_time (str): 结束时间，格式为 'YYYY-MM-DD HH:MM:SS'
    columns (list): 需要查询的列名列表，如果为None，则返回所有列
    status (str): 需要筛选的状态（例如 "开机"、"关机"），如果为None，则不筛选状态

    返回:
    dict: 包含指定列名和对应值的字典，或错误信息
    """


    print("------------get_table_data--------------")
    # 创建一个字典来存储元数据
    metadata = {
        "table_name": table_name,
        "start_time": start_time,
        "end_time": end_time,
        "columns": columns,
        "status": status,
    }

    try:
        df = pd.read_csv(f"database_in_use/{table_name}.csv")
    except FileNotFoundError:
        return {"error": f"数据表 {table_name} 不存在", "metadata": metadata}

    # 将csvTime列从时间戳转换为datetime类型
    df["csvTime"] = pd.to_datetime(df["csvTime"], unit="ns")  # 假设时间戳是纳秒级别

    # 将开始时间和结束时间转换为datetime类型
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    # 如果开始时间和结束时间是同一分钟
    if (
            start_time.minute == end_time.minute
            and start_time.hour == end_time.hour
            and start_time.day == end_time.day
    ):
        # 将开始时间设置为这一分钟的00秒
        start_time = start_time.replace(second=0)
        # 将结束时间设置为这一分钟的59秒
        end_time = end_time.replace(second=59)
    # 筛选指定时间范围内的数据
    filtered_data = df[(df["csvTime"] >= start_time) & (df["csvTime"] <= end_time)]

    if filtered_data.empty:
        return {
            "error": f"在数据表 {table_name} 中未找到时间范围 {start_time} 到 {end_time} 的数据",
            "metadata": metadata,
        }

    # 如果传入了 status 参数，则进一步筛选状态
    if status is not None:
        filtered_data = filtered_data[filtered_data["status"] == status]
        if filtered_data.empty:
            return {
                "error": f"在数据表 {table_name} 中未找到状态为 {status} 的数据",
                "metadata": metadata,
            }

    # 如果未指定列名，则返回所有列
    if columns is None:
        columns = filtered_data.columns.tolist()

    # 检查列名是否存在
    missing_columns = [
        column for column in columns if column not in filtered_data.columns
    ]
    if missing_columns:
        return {
            "error": f"列名 {missing_columns} 在数据表 {table_name} 中不存在",
            "metadata": metadata,
        }

    # 获取指定列名和对应的值
    result = {}
    for column in columns:
        if column == "csvTime":
            # 将时间格式化为字符串
            result[column] = (
                filtered_data[column].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
            )
        else:
            result[column] = filtered_data[column].values.tolist()

    # 返回结果和元数据
    # print(result)
    return {"result": result, "metadata": metadata}


# 能耗计算
def load_and_filter_data(file_path, start_time, end_time, power_column):
    """
    加载 CSV 文件并筛选指定时间范围内的数据
    :param file_path: CSV 文件路径
    :param start_time: 开始时间
    :param end_time: 结束时间
    :param power_column: 功率列名
    :return: 筛选后的 DataFrame
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 {file_path} 未找到")

    # 确保时间列是 datetime 类型
    try:
        df["csvTime"] = pd.to_datetime(df["csvTime"])
    except Exception as e:
        raise ValueError(f"时间列转换失败: {e}")

    # 筛选特定时间范围内的数据
    filtered_data = df[
        (df["csvTime"] >= start_time) & (df["csvTime"] <= end_time)
        ].copy()

    if filtered_data.empty:
        return None

    # 计算时间差（秒）
    filtered_data.loc[:, "diff_seconds"] = (
        filtered_data["csvTime"].diff().dt.total_seconds().shift(-1)
    )

    # 计算每个时间间隔的能耗（kWh）
    filtered_data.loc[:, "energy_kWh"] = (
            filtered_data["diff_seconds"] * filtered_data[power_column] / 3600
    )

    return filtered_data

#计算总能耗
def calculate_total_energy(start_time, end_time, device_name="折臂吊车"):
    """
    计算每个指定时间段内的总能耗
    :param start_time: 查询的开始时间（字符串或 datetime 类型）
    :param end_time: 查询的结束时间（字符串或 datetime 类型）
    :param device_name: 设备名称，默认为 '折臂吊车'
    :return: 总能耗（kWh，float 类型）
    """
    # 设备配置映射：设备名称 -> (表名, 功率列名)
    # 侧推、推进器、甲板机械、推进系统、发电机
    device_config = {
        
        "一号发电机":("Port1_ksbg_3", "P1_66"), #四个柴油发电机组
        "二号发电机":("Port1_ksbg_3", "P1_75"),
        "三号发电机":("Port2_ksbg_2", "P2_51"),
        "四号发电机":("Port2_ksbg_3", "P2_60"),

        "一号推进变频器":("Port3_ksbg_8", "P3_15"), # 推进系统
        "二号推进变频器":("Port4_ksbg_7", "P4_16"),
        "侧推":("Port3_ksbg_9", "P3_18"), # 侧推的配置
        "可伸缩推":("Port4_ksbg_8", "P4_21"),

        "一号舵桨转舵A":("device_1_2_meter_102", "1-2-6_v"),
        "一号舵桨转舵B":("device_1_3_meter_103", "1-3-6_v"),
        "二号舵桨转舵A":("device_13_2_meter_1302", "13-2-6_v"),
        "二号舵桨转舵B":("device_13_3_meter_1303", "13-3-6_v"),

        "一号门架": ("device_1_5_meter_105", "1-5-6_v"),  # 一号门架的配置  （甲板机械设备）
        "二号门架": ("device_13_14_meter_1314", "13-14-6_v"),  # 二号门架的配置
        #"A架":(),
        "绞车": ("device_1_15_meter_115", "1-15-6_v"),  # 添加绞车变频器的配置
        "折臂吊车": ("device_13_11_meter_1311", "13-11-6_v"),
        
    }
    print("---------calculate_total_energy--------------"+ device_name+"---------"+start_time+"---"+end_time)

    # 检查设备名称是否有效
    if device_name not in device_config:
        raise ValueError(f"未知的设备名称: {device_name}")

    # 获取设备配置
    table_name, power_column = device_config[device_name]

    # 读取 CSV 文件并计算能耗
    file_path = f"database_in_use/{table_name}.csv"
    try:
        filtered_data = load_and_filter_data(
            file_path, start_time, end_time, power_column
        )
        if filtered_data is None:
            return None
        total_energy_kWh = filtered_data["energy_kWh"].sum()
        return round(total_energy_kWh, 2)
    except Exception as e:
        raise ValueError(f"计算能耗时出错: {e}")

# 计算dp过程对应的时间段（dp过程的开始时间和结束时间）并返回，如有多个dp过程，返回每个dp过程的开始时间和结束时间（作为列表）
def calculate_start_end(start_time, end_time, shebeiname="DP"):
    """
    找到指定时间段内设备的的开始时间和结束时间，若有多个时间段则返回多个时间段元组
    :param start_time: 查询的开始时间（字符串或 datetime 类型）
    :param end_time: 查询的结束时间（字符串或 datetime 类型）
    :param shebeiname: 设备名称，默认为 '折臂吊车'
    :return: 包含三种格式开机时长的字符串
    """
    print("-------calculate_start_end执行***计算开机时间-------"+shebeiname + start_time + end_time)
    # 设备配置映射：设备名称 -> (文件路径, 开机状态, 关机状态)
    device_config = {
        "折臂吊车": (
            "database_in_use/device_13_11_meter_1311.csv",
            "折臂吊车开机",
            "折臂吊车关机",
        ),
        "A架": ("database_in_use/Ajia_plc_1.csv", "A架开机", "A架关机"),
        "DP": ("database_in_use/Port3_ksbg_9.csv", "ON DP", "OFF DP"),
    }

    # 检查设备名称是否有效
    if shebeiname not in device_config:
        raise ValueError(f"未知的设备名称: {shebeiname}")

    # 获取设备配置
    file_path, start_status, end_status = device_config[shebeiname]

    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 将时间列转换为 datetime 类型
    df["csvTime"] = pd.to_datetime(df["csvTime"])

    # 将传入的开始时间和结束时间转换为 datetime 类型
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    
    # print(f"开始时间: {start_time},结束时间: {end_time}")

    # 筛选出指定时间段内的数据
    df_filtered = df[(df["csvTime"] >= start_time) & (df["csvTime"] <= end_time)]

    # 初始化变量
    total_duration = pd.Timedelta(0)
    start_uptime = None

    #计算开关的次数
    cnt = 0

    res = []
    # 遍历筛选后的数据
    for index, row in df_filtered.iterrows():
        if row["status"] == start_status :
            start_uptime = row["csvTime"]
        elif row["status"] == end_status and start_uptime is not None:
            end_uptime = row["csvTime"]
            total_duration += end_uptime - start_uptime
            res.append((start_uptime, end_uptime))
            start_uptime = None
            cnt += 1

    
    print(res)
    # return result
    return res
    
#计算甲板设备能耗
def calculate_total_deck_machinery_energy(start_time, end_time):
    """
    计算甲板机械（折臂吊车、一号门架、二号门架、绞车）在指定时间范围内的总能耗
    :param start_time: 查询的开始时间（字符串或 datetime 类型）
    :param end_time: 查询的结束时间（字符串或 datetime 类型）
    :return: 所有设备的总能耗（kWh，float 类型）
    """
    # 定义设备列表
    devices = ["折臂吊车", "一号门架", "二号门架", "绞车"]

    total_energy = 0  # 初始化总能耗

    print("--------------calculate_total_deck_machinery_energy-------------------------")
    # 遍历每个设备，计算能耗并累加
    for device in devices:
        try:
            energy = calculate_total_energy(start_time, end_time, device_name=device)
            if energy is not None:
                total_energy += energy
        except Exception as e:
            print(f"计算设备 {device} 能耗时出错: {e}")
    return round(total_energy, 2)  # 返回总能耗，保留两位小数

#计算A架能耗
def calculate_ajia_energy(start_time, end_time):
    """
    计算A架（一号门架、二号门架）在指定时间范围内的总能耗
    :param start_time: 查询的开始时间（字符串或 datetime 类型）
    :param end_time: 查询的结束时间（字符串或 datetime 类型）
    :return: 所有设备的总能耗（kWh，float 类型）
    """
    # 定义设备列表
    devices = ["一号门架", "二号门架"]

    total_energy = 0  # 初始化总能耗

    print("--------------calculate_ajia_energy-------------------------")
    # 遍历每个设备，计算能耗并累加
    for device in devices:
        try:
            energy = calculate_total_energy(start_time, end_time, device_name=device)
            if energy is not None:
                total_energy += energy
        except Exception as e:
            print(f"计算设备 {device} 能耗时出错: {e}")
    return round(total_energy, 2)  # 返回总能耗，保留两位小数

#计算推进系统能耗
def calculate_total_propulsion_system_energy(start_time, end_time):
    """
    计算推进器能耗(31)= 一二号推进变频器功率(反馈)+ 侧推功率(反馈)+ 可伸缩推功率(反馈)
    :param start_time: 查询的开始时间（字符串或 datetime 类型）
    :param end_time: 查询的结束时间（字符串或 datetime 类型）
    :return: 所有设备的总能耗（kWh，float 类型）
    """
    # 定义设备列表
    devices = ["一号推进变频器", "二号推进变频器", "侧推", "可伸缩推"]

    print("--------------calculate_total_propulsion_system_energy-------------------------")
    total_energy = 0  # 初始化总能耗

    # 遍历每个设备，计算能耗并累加
    for device in devices:
        try:
            energy = calculate_total_energy(start_time, end_time, device_name=device)
            if energy is not None:
                total_energy += energy
        except Exception as e:
            print(f"计算设备 {device} 能耗时出错: {e}")
    return round(total_energy, 2)  # 返回总能耗，保留两位小数

#计算舵桨的总能耗
def calculate_tuojiang_energy(start_time, end_time):
    """
    计算舵桨（一号舵桨转舵A、一号舵桨转舵B、二号舵桨转舵A、二号舵桨转舵B）在指定时间范围内的总能耗
    :param start_time: 查询的开始时间（字符串或 datetime 类型）
    :param end_time: 查询的结束时间（字符串或 datetime 类型）
    :return: 所有设备的总能耗（kWh，float 类型）
    """
    # 定义设备列表
    devices = ["一号舵桨转舵A", "一号舵桨转舵B","二号舵桨转舵A","二号舵桨转舵B"]

    total_energy = 0  # 初始化总能耗

    print("--------------calculate_tuojiang_energy-------------------------")
    # 遍历每个设备，计算能耗并累加
    for device in devices:
        try:
            energy = calculate_total_energy(start_time, end_time, device_name=device)
            if energy is not None:
                total_energy += energy
        except Exception as e:
            print(f"计算设备 {device} 能耗时出错: {e}")
    return round(total_energy, 2)  # 返回总能耗，保留两位小数


#计算单个发电机的发电量
def calculate_electricity_generation(start_time, end_time, device_name="一号发电机"):
    """
    计算每个指定时间段内的总能耗
    :param start_time: 查询的开始时间（字符串或 datetime 类型）
    :param end_time: 查询的结束时间（字符串或 datetime 类型）
    :param device_name: 设备名称，默认为 '折臂吊车'
    :return: 总能耗（kWh，float 类型）
    """
    # 设备配置映射：设备名称 -> (表名, 功率列名)
    # 侧推、推进器、甲板机械、推进系统、发电机
    device_config = {
        
        "一号发电机":("Port1_ksbg_3", "P1_66"), #四个柴油发电机组
        "二号发电机":("Port1_ksbg_3", "P1_75"),
        "三号发电机":("Port2_ksbg_2", "P2_51"),
        "四号发电机":("Port2_ksbg_3", "P2_60"),
        
    }
    print("--------------electricity_generation-------------------------"+ device_name)

    # 检查设备名称是否有效
    if device_name not in device_config:
        raise ValueError(f"未知的设备名称: {device_name}")

    # 获取设备配置
    table_name, power_column = device_config[device_name]

    # 读取 CSV 文件并计算能耗
    file_path = f"database_in_use/{table_name}.csv"
    try:
        filtered_data = load_and_filter_data(
            file_path, start_time, end_time, power_column
        )
        if filtered_data is None:
            return None
        total_energy_kWh = filtered_data["energy_kWh"].sum()
        return round(total_energy_kWh, 2)
    except Exception as e:
        raise ValueError(f"计算能耗时出错: {e}")
#计算总发电量
def calculate_total_electricity_generation(start_time, end_time):
    """
    总发电量(49)=一至四号柴油发电机组(有功)功率*时间
    :param start_time: 查询的开始时间（字符串或 datetime 类型）
    :param end_time: 查询的结束时间（字符串或 datetime 类型）
    :return: 所有设备的总能耗（kWh，float 类型）
    """
    # 定义设备列表
    devices = ["一号发电机", "二号发电机", "三号发电机", "四号发电机"]

    total_energy = 0  # 初始化总能耗


    print("----------------calculate_total_electricity_generation-------------")

    # 遍历每个设备，计算能耗并累加
    for device in devices:
        try:
            energy = calculate_total_energy(start_time, end_time, device_name=device)
            if energy is not None:
                total_energy += energy
        except Exception as e:
            print(f"计算设备 {device} 能耗时出错: {e}")
    return round(total_energy, 2)  # 返回总能耗，保留两位小数

# 计算理论发电量
def calculate_theory_electricity_generation(start_time, end_time):
    """
    理论发电量 =四个柴油发电机组燃油消耗率(P1 3)*时间*密度*热值
    :param start_time: 查询的开始时间（字符串或 datetime 类型）
    :param end_time: 查询的结束时间（字符串或 datetime 类型）
    :return: 所有设备的总能耗（kWh，float 类型）
    """
    # 定义设备列表
    devices = ["一号发电机", "二号发电机", "三号发电机", "四号发电机"]

    total_energy = 0  # 初始化总能耗

    # 遍历每个设备，计算能耗并累加
    for device in devices:
        try:
            energy = calculate_total_energy(start_time, end_time, device_name=device)
            if energy is not None:
                total_energy += energy
        except Exception as e:
            print(f"计算设备 {device} 能耗时出错: {e}")

    
    return round(total_energy, 2)  # 返回总能耗，保留两位小数

# 计算指定时间范围内的侧推的总能耗
def calculate_energy_consumption(start_time, end_time):
    """
    计算指定时间范围内的侧推的总能耗
    :param start_time: 开始时间（字符串或 datetime 类型）
    :param end_time: 结束时间（字符串或 datetime 类型）
    :return: 总能耗（kWh，float 类型），如果数据为空则返回 None
    """
    # 文件路径和功率列名直接定义在函数内部
    file_path = "database_in_use/Port3_ksbg_9.csv"
    power_column = "P3_18"  # 使用 "艏推功率反馈,单位:kW" 列

    try:
        # 加载 CSV 文件
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 {file_path} 未找到")

    # 确保时间列是 datetime 类型
    try:
        df["csvTime"] = pd.to_datetime(df["csvTime"])
    except Exception as e:
        raise ValueError(f"时间列转换失败: {e}")

    # 筛选特定时间范围内的数据
    filtered_data = df[
        (df["csvTime"] >= start_time) & (df["csvTime"] < end_time)
        ].copy()

    if filtered_data.empty:
        return None

    # 计算时间差（秒）
    filtered_data.loc[:, "diff_seconds"] = (
        filtered_data["csvTime"].diff().dt.total_seconds().shift(-1)
    )

    # 计算每个时间间隔的能耗（kWh）
    filtered_data.loc[:, "energy_kWh"] = (
            filtered_data["diff_seconds"] * filtered_data[power_column] / 3600
    )

    # 计算总能耗
    total_energy_kWh = filtered_data["energy_kWh"].sum()

    return round(total_energy_kWh, 2)


def query_device_parameter(parameter_name_cn):
    """
    通过参数中文名查询设备参数信息
    :param parameter_name_cn: 参数中文名
    :param device_parameter_file: 设备参数详情表的文件路径，默认为'设备参数详情表.csv'
    :return: 返回包含参数信息的字典
    """
    print("-------query_device_parameter执行-------")
    # 读取设备参数详情表
    # df = pd.read_excel("database_in_use/设备参数详情.xlsx")
    df = pd.read_excel("database_in_use/设备参数详情.xlsx", engine="openpyxl")
    # df = pd.read_excel(app\devlop_home\baseline\mountain_baseline\database_in_use\设备参数详情.xlsx)

    # 检查参数中文名是否包含在 Channel_Text_CN 列中
    if not df["Channel_Text_CN"].str.contains(parameter_name_cn).any():
        raise ValueError(f"未找到包含 '{parameter_name_cn}' 的参数中文名")

    # 获取包含参数中文名的所有行
    parameter_info = df[df["Channel_Text_CN"].str.contains(parameter_name_cn)].iloc[0]

    # 将参数信息转换为字典
    parameter_dict = {
        "参数名": parameter_info["Channel_Text"],
        "参数中文名": parameter_info["Channel_Text_CN"],
        "参数下限": parameter_info["Alarm_Information_Range_Low"],
        "参数上限": parameter_info["Alarm_Information_Range_High"],
        "报警值的单位": parameter_info["Alarm_Information_Unit"],
        "报警值": parameter_info["Parameter_Information_Alarm"],
        "屏蔽值": parameter_info["Parameter_Information_Inhibit"],
        "延迟值": parameter_info["Parameter_Information_Delayed"],
        "安全保护设定值": parameter_info["Safety_Protection_Set_Value"],
        "附注": parameter_info["Remarks"],
    }
    print(parameter_dict)
    return parameter_dict


def calculate_total_energy_consumption(start_time, end_time, query_type="all"):
    """
    计算指定时间范围内两个推进变频器或推进系统的总能耗
    :param start_time: 开始时间（字符串或 datetime 类型）
    :param end_time: 结束时间（字符串或 datetime 类型）
    :param query_type: 查询类型，可选值为 '1'（一号推进）、'2'（二号推进）、'all'（整个推进系统）
    :return: 总能耗（kWh，float 类型），如果数据为空则返回 None
    """
    # 文件路径和功率列名
    file_path_1 = "database_in_use/Port3_ksbg_8.csv"
    power_column_1 = "P3_15"  # 一号推进变频器功率反馈,单位:kW

    file_path_2 = "database_in_use/Port4_ksbg_7.csv"
    power_column_2 = "P4_15"  # 二号推进变频器功率反馈,单位:kW
    try:
        # 加载 CSV 文件
        df1 = pd.read_csv(file_path_1)
        df2 = pd.read_csv(file_path_2)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"文件未找到: {e}")

    # 确保时间列是 datetime 类型
    try:
        df1["csvTime"] = pd.to_datetime(df1["csvTime"])
        df2["csvTime"] = pd.to_datetime(df2["csvTime"])
    except Exception as e:
        raise ValueError(f"时间列转换失败: {e}")

    # 筛选特定时间范围内的数据
    filtered_data_1 = df1[
        (df1["csvTime"] >= start_time) & (df1["csvTime"] < end_time)
        ].copy()
    filtered_data_2 = df2[
        (df2["csvTime"] >= start_time) & (df2["csvTime"] < end_time)
        ].copy()

    if filtered_data_1.empty or filtered_data_2.empty:
        return None

    # 计算时间差（秒）
    filtered_data_1.loc[:, "diff_seconds"] = (
        filtered_data_1["csvTime"].diff().dt.total_seconds().shift(-1)
    )
    filtered_data_2.loc[:, "diff_seconds"] = (
        filtered_data_2["csvTime"].diff().dt.total_seconds().shift(-1)
    )

    # 计算每个时间间隔的能耗（kWh）
    filtered_data_1.loc[:, "energy_kWh"] = (
            filtered_data_1["diff_seconds"] * filtered_data_1[power_column_1] / 3600
    )
    filtered_data_2.loc[:, "energy_kWh"] = (
            filtered_data_2["diff_seconds"] * filtered_data_2[power_column_2] / 3600
    )

    # 计算总能耗
    total_energy_kWh_1 = filtered_data_1["energy_kWh"].sum()
    total_energy_kWh_2 = filtered_data_2["energy_kWh"].sum()

    # 根据查询类型返回相应的能耗
    if query_type == "1":
        return round(total_energy_kWh_1, 2)
    elif query_type == "2":
        return round(total_energy_kWh_2, 2)
    elif query_type == "all":
        return round(total_energy_kWh_1 + total_energy_kWh_2, 2)
    else:
        raise ValueError("query_type 参数无效，请输入 '1'、'2' 或 'all'")


#查询设备在该时间段内的状态变化，并排除 status 为 'False' 的记录。
def get_device_status_by_time_range(start_time, end_time,shebeiname):
    """ 
    根据数据表名、开始时间和结束时间，查询设备在该时间段内的状态变化，并排除 status 为 'False' 的记录。
    参数:
    start_time (str): 开始时间，格式为 'YYYY-MM-DD HH:MM:SS'
    end_time (str): 结束时间，格式为 'YYYY-MM-DD HH:MM:SS'

    返回:
    dict: 包含设备状态变化的时间点和对应状态的字典，或错误信息
    """

    def get_status_changes(table_name, device_name, status_name):
        """
        辅助函数：获取指定设备在指定时间范围内的状态变化。

        参数:
        table_name (str): 数据表名
        device_name (str): 设备名称

        返回:
        dict: 包含设备状态变化的时间点和对应状态的字典，或错误信息
        """

        
        metadata = {
            "table_name": table_name,
            "start_time": start_time,
            "end_time": end_time,
        }

        print("----------------get_status_changes---------------"+start_time+"---"+end_time+"-----"+device_name+"----"+table_name+"----"+status_name)
        try:
            df = pd.read_csv(f"database_in_use/{table_name}.csv")
        except FileNotFoundError:
            return {"error": f"数据表 {table_name} 不存在", "metadata": metadata}

        # 将csvTime列从时间戳转换为datetime类型
        # df["csvTime"] = pd.to_datetime(df["csvTime"])  # 假设时间戳是纳秒级别
        # 确保时间列是 datetime 类型
        try:
            df["csvTime"] = pd.to_datetime(df["csvTime"])
         
        except Exception as e:
            raise ValueError(f"时间列转换失败: {e}")
        # 将开始时间和结束时间转换为datetime类型
        start_time_dt = pd.to_datetime(start_time)
        end_time_dt = pd.to_datetime(end_time)

         # 检查并确保 start_time 早于 end_time，如果不是则对调
        if start_time_dt > end_time_dt:
            start_time_dt, end_time_dt = end_time_dt, start_time_dt
      
        # 筛选指定时间范围内的数据，并排除 status 为 status_name 的记录
            
        

        filtered_data = df[(df["csvTime"] >= start_time_dt) & (df["csvTime"] <= end_time_dt) & (df["status"] == status_name)].copy()
        

        if filtered_data.empty:
            return {
                "error": f"在数据表 {table_name} 中未找到时间范围 {start_time} 到 {end_time} 且 status 为 {status_name} 的数据",
                "metadata": metadata,
            }

        # 检查是否存在status列
        if "status" not in filtered_data.columns:
            return {
                "error": f"数据表 {table_name} 中不存在 'status' 列",
                "metadata": metadata,
            }

        # 获取设备状态变化的时间点和对应状态
        status_changes = filtered_data[
            ["csvTime", "status"]
        ].copy()  # 显式创建副本以避免警告

        # 使用 .loc 避免 SettingWithCopyWarning
        status_changes.loc[:, "csvTime"] = status_changes["csvTime"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # 将结果转换为字典
        return {
            "设备名称": device_name,
            "正在进行的关键动作": status_changes.to_dict(orient="records"),
        }
    

    action_list_ajia=["A架开机","征服者起吊","征服者入水","缆绳解除","A架摆回","A架关机","A架摆出","缆绳挂妥","征服者出水","征服者落座"]
    action_list_dp=["ON DP","OFF DP"]
    action_list_zhebi=["小艇落座","折臂吊车开机","小艇检查完毕","小艇入水","折臂吊车关机"]

    # 获取三个设备的状态变化
    if "DP" in shebeiname:
        if shebeiname not in action_list_dp:
            raise ValueError("设备名称‘shebeiname’必须在表中：",action_list_dp)
        res = get_status_changes("Port3_ksbg_9", "定位设备",shebeiname)
    elif "折臂吊车" in shebeiname or "小艇" in shebeiname:
        if shebeiname not in action_list_zhebi:
            raise ValueError("设备名称‘shebeiname’必须在表中：",action_list_zhebi)
        res = get_status_changes("device_13_11_meter_1311", "折臂吊车",shebeiname)
    else:
        if shebeiname not in action_list_ajia:
            raise ValueError("设备名称‘shebeiname’必须在表中：",action_list_ajia)
        res = get_status_changes("Ajia_plc_1", "A架",shebeiname)
    
    

    print(res)
    # 过滤掉包含错误的结果
    # results = [
    #     result for result in res if "error" not in result
    # ]
    # print(results)

    # 返回结果和元数据
    return {
        "result": res,
        "metadata": {"start_time": start_time, "end_time": end_time},
    }


# get_device_status_by_time_range("2024-07-19 23:59:59","2024-07-19 00:00:59","A架开机")

#燃油消耗量计算
def load_and_filter_fuel_data(file_path, start_time, end_time, power_column):
    """
    加载 CSV 文件并筛选指定时间范围内的数据
    :param file_path: CSV 文件路径
    :param start_time: 开始时间
    :param end_time: 结束时间
    :param power_column: 功率列名
    :return: 筛选后的 DataFrame
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 {file_path} 未找到")

    # 确保时间列是 datetime 类型
    try:
        df["csvTime"] = pd.to_datetime(df["csvTime"])
    except Exception as e:
        raise ValueError(f"时间列转换失败: {e}")
    # 统一时间格式处理
    try:
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)
    except Exception as e:
        raise ValueError(f"时间格式错误: {str(e)}")

    # 筛选特定时间范围内的数据
    filtered_data = df[
        (df["csvTime"] >= start_time) & (df["csvTime"] <= end_time)
        ].copy()

    if filtered_data.empty:
        return None

    # 计算时间差（秒）
    filtered_data.loc[:, "diff_seconds"] = (
        filtered_data["csvTime"].diff().dt.total_seconds().shift(-1)
    )

    # 计算每个时间间隔的（kWh）
    filtered_data.loc[:, "furl_volume_L"] = (
            filtered_data["diff_seconds"] * filtered_data[power_column] / 3600
    )

    return filtered_data

#计算时间内一至四的燃油消耗量
def calculate_total_fuel_volume(start_time, end_time, device_name="一号发电机"):
    """
    计算每个指定时间段内的燃油消耗量
    :param start_time: 查询的开始时间（字符串或 datetime 类型）
    :param end_time: 查询的结束时间（字符串或 datetime 类型）
    :param device_name: 设备名称，默认为 '折臂吊车'
    :return: 燃油消耗量（L，float 类型）
    """
    # 设备配置映射：设备名称 -> (表名, 功率列名)
    # 侧推、推进器、甲板机械、推进系统、发电机
    device_config = {
        "一号发电机":("Port1_ksbg_1", "P1_3"), #四个柴油发电机组
        "二号发电机":("Port1_ksbg_1", "P1_25"),
        "三号发电机":("Port2_ksbg_1", "P2_3"),
        "四号发电机":("Port2_ksbg_1", "P2_25"),
  
    }
    print("------------------calculate_total_fuel_volume--------------"+device_name+"--------"+start_time+"--------"+end_time)
    # 检查设备名称是否有效
    if device_name not in device_config:
        raise ValueError(f"未知的设备名称: {device_name}")

    # 获取设备配置
    table_name, power_column = device_config[device_name]

    # 读取 CSV 文件并计算能耗
    file_path = f"database_in_use/{table_name}.csv"
    try:
        filtered_data = load_and_filter_fuel_data(
            file_path, start_time, end_time, power_column
        )
        if filtered_data is None:
            return None
        total_energy_kWh = filtered_data["furl_volume_L"].sum()
        return round(total_energy_kWh, 2)
    except Exception as e:
        raise ValueError(f"计算燃油消耗量时出错: {e}")


#计算四个柴油发电机燃油消耗量
def calculate_total_4_fuel_volume(start_time, end_time):
    """
    四个柴油发电机燃油消耗量(49)=一至四号柴油发电机组在时间内的消耗率之和
    :param start_time: 查询的开始时间（字符串或 datetime 类型）
    :param end_time: 查询的结束时间（字符串或 datetime 类型）
    :return: 所有设备的总能耗（kWh，float 类型）
    """
    # 定义设备列表
    devices = ["一号发电机", "二号发电机", "三号发电机", "四号发电机"]

    total_fuel = 0  # 初始化总能耗

    # 遍历每个设备，计算能耗并累加
    for device in devices:
        try:
            energy = calculate_total_fuel_volume(start_time, end_time, device_name=device)
            if energy is not None:
                total_fuel += energy
        except Exception as e:
            print(f"计算设备 {device} 能耗时出错: {e}")
    return round(total_fuel, 2)  # 返回总能耗，保留两位小数


#计算指定列的平均值或最大值
def calculate_max_or_average(start_time, end_time, table_name, column_name, max_or_average_or_min="max"):
    """
    计算指定列的平均值或最大值或最小值
    """

    print("---------------------calculate_"+max_or_average_or_min+"-----------------")
    #读取表格数据
    file_path = f"database_in_use/{table_name}.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在")
    
    except Exception as e:
        print(f"读取文件 {file_path} 时出错：{e}")

    #获取指定时间内的数据
    filtered_data = df[
        (df["csvTime"] >= start_time) & (df["csvTime"] <= end_time)
        ].copy()

    if filtered_data.empty:
        return None
    

    print("=====column====="+column_name)

    if max_or_average_or_min == "max":
        #检查列名是否存在
        if column_name in filtered_data.columns:
            # 计算最大值
            max_value = filtered_data[column_name].max()
            return max_value
        else:
            # print(f"列 {column_name} 不存在于 {file_path}")
            raise ValueError(f"列 {column_name} 不存在于 {file_path}")
    elif max_or_average_or_min == "average":
        #检查列名是否存在
        if column_name in filtered_data.columns:
            # 计算平均值
            average_value = filtered_data[column_name].mean()
            return average_value
        else:
            # print(f"列 {column_name} 不存在于 {file_path}")
            raise ValueError(f"列 {column_name} 不存在于 {file_path}")
    elif max_or_average_or_min == 'min':
        #检查列名是否存在
        if column_name in filtered_data.columns:
            # 计算最大值
            max_value = filtered_data[column_name].min()
            return max_value
        else:
            # print(f"列 {column_name} 不存在于 {file_path}")
            raise ValueError(f"列 {column_name} 不存在于 {file_path}")
    
    else:
        raise ValueError("max_or_average 参数错误，只能是 'max' 或 'average'或 'min")

def get_value_of_time(time_point, table_name, column_name):
    """
    计算指定列在指定时间的值
    """

    print("---------------------get_value_of_time"+time_point+"--------"+table_name+"--------"+column_name)
    #读取表格数据
    file_path = f"database_in_use/{table_name}.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在")
    
    except Exception as e:
        print(f"读取文件 {file_path} 时出错：{e}")
    
    # 列名检查
    if column_name not in df.columns:
        raise KeyError(f"列 {column_name} 不存在，可用列：{', '.join(df.columns)}")
    
    try:
        # 转换时间类型
        target_time = pd.to_datetime(time_point)
        df['csvTime'] = pd.to_datetime(df['csvTime'])
        
        # 空数据检查
        if df.empty:
            raise ValueError("数据表为空")
            
        # 计算最近时间
        time_diffs = (df['csvTime'] - target_time).abs()
        nearest_idx = time_diffs.idxmin()
        
        # 获取结果并记录日志
        result = df.loc[nearest_idx, column_name]
        print(f"找到最近记录：时间：{df['csvTime'][nearest_idx]},时间差 {time_diffs[nearest_idx]}，值：{result}")
        
        return result
        
    except pd.errors.ParserError:
        raise ValueError(f"时间格式错误，请输入类似 'YYYY-MM-DD HH:MM:SS' 的格式，当前输入：{time_point}")
    except Exception as e:
        raise RuntimeError(f"查询失败: {str(e)}") from e


#计算指定表格的一定时间内的数据有多少条
def calculate_sum_data(start_time, end_time, table_name):
    print("---------------------calculate_sum_data-----------------")
    #读取表格数据
    file_path = f"database_in_use/{table_name}.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在")
    
    except Exception as e:
        print(f"读取文件 {file_path} 时出错：{e}")

    #获取指定时间内的数据
    filtered_data = df[
        (df["csvTime"] >= start_time) & (df["csvTime"] < end_time)
        ].copy()

    if filtered_data.empty:
        return None
    
    return len(filtered_data)



#计算指定时间内以同一方向10度为标准的摆动次数
def count_swings_10(start_time, end_time):
    print("---------------------count_swings_10-----------------")
    file_path = "database_in_use/Ajia_plc_1.csv"
    try:
        df = pd.read_csv(file_path, parse_dates=["csvTime"])
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在")
        return 0
    except Exception as e:
        print(f"读取文件出错: {e}")
        return 0

    mask = (df["csvTime"] >= start_time) & (df["csvTime"] < end_time)
    filtered_data = df.loc[mask].copy()
    
    if filtered_data.empty:
        return 0

    def clean_angle(value):
        try:
            if isinstance(value, str):
                cleaned = ''.join([c for c in value if c in '0123456789.-'])
                return float(cleaned) if cleaned else np.nan
            return float(value)
        except:
            return np.nan

    angles_series = filtered_data["Ajia-0_v"].apply(clean_angle)
    
    swing_count = 0
    start_value = None
    current_extreme = None
    current_dir = 0  # 0: 初始，1: 正方向，-1: 负方向

    for value in angles_series:
        # 处理无效值中断逻辑
        if pd.isna(value):
            if start_value is not None and current_extreme is not None:
                if abs(current_extreme - start_value) >= 10:
                    swing_count += 1
                    print(current_extreme,"  ",start_value)
            # 状态重置
            start_value = None
            current_extreme = None
            current_dir = 0
            continue
        
        # 初始化阶段
        if start_value is None:
            start_value = value
            current_extreme = value
            current_dir = 0
            continue
        
        # 计算方向变化
        delta = value - current_extreme
        new_dir = 0
        if delta > 1e-6:
            new_dir = 1
        elif delta < -1e-6:
            new_dir = -1
        else:
            continue
        
        # 方向判定逻辑
        if current_dir == 0:
            current_dir = new_dir
            current_extreme = value
        else:
            if new_dir != current_dir:
                # 方向变化时结算摆动
                if abs(current_extreme - start_value) >= 10:
                    swing_count += 1
                    print(current_extreme,"  ",start_value)
                # 开启新方向跟踪
                start_value = current_extreme
                current_dir = new_dir
                current_extreme = value
            else:
                # 更新极值
                if (new_dir == 1 and value > current_extreme) or (new_dir == -1 and value < current_extreme):
                    current_extreme = value
    
    # 最终结算
    if start_value is not None and current_extreme is not None:
        if abs(current_extreme - start_value) >= 10:
            swing_count += 1
            print(current_extreme,"  ",start_value)
    
    return swing_count

    
    # return cnt


#计算指定时间内完整的摆动次数
def count_swings(start_time, end_time):
    print("---------------------count_swings-----------------")
    LOW_RANGE = (-45, -41)    # 征服者出水区（-43±2°）
    HIGH_RANGE = (33, 37)     # 落座区（35±2°）
    
    # 读取数据文件
    file_path = "database_in_use/Ajia_plc_1.csv"
    try:
        df = pd.read_csv(file_path, parse_dates=["csvTime"])
    except Exception as e:
        print(f"Error loading data: {e}")
        return 0

    # 时间范围筛选
    mask = (df["csvTime"] >= start_time) & (df["csvTime"] < end_time)
    filtered_data = df.loc[mask].copy()
    
    # 数据清洗函数
    def clean_angle(value):
        """将角度值转换为浮点数，处理异常数据"""
        try:
            # 处理字符串类型数据
            if isinstance(value, str):
                # 移除非数字字符（保留负号和小数点）
                cleaned = ''.join([c for c in value if c in '0123456789.-'])
                if not cleaned:  # 空字符串处理
                    return np.nan
                return float(cleaned)
            return float(value)
        except:
            return np.nan
    
    # 生成清洗后的角度序列
    angles = filtered_data["Ajia-0_v"].apply(clean_angle)
    
    # 摆动计数逻辑
    swing_count = 0
    last_zone = None  # 跟踪上一个有效区域：'low'/'high'/None
    
    for val in angles:
        # 跳过无效值并重置状态
        if pd.isna(val):
            last_zone = None
            continue
        
        # 判断当前区域
        current_zone = None
        if LOW_RANGE[0] <= val <= LOW_RANGE[1]:
            current_zone = 'low'
        elif HIGH_RANGE[0] <= val <= HIGH_RANGE[1]:
            current_zone = 'high'
        
        # 仅处理有效区域变化
        if current_zone is not None:
            if last_zone is None:
                # 初始化区域跟踪
                last_zone = current_zone
            elif current_zone != last_zone:
                # 检测到区域切换
                swing_count += 1
                last_zone = current_zone
    
    return swing_count

#返回一至四号柴油发电机与温度或压力相关的数值类型参数的参数详情
def rt_temp_or_press_parameter(temp_or_press,device_name = "二号柴油发电机"):
    print("--------------rt_temp_or_press_parameter-------------------")
    # file_path = f"database_in_use/设备参数详情表.csv"
    # df = pd.read_excel("database_in_use/设备参数详情.xlsx", engine="openpyxl")
    try:
        # df = pd.read_csv(file_path)
        df = pd.read_excel("database_in_use/设备参数详情.xlsx", engine="openpyxl")
    except FileNotFoundError:
        print(f"文件 database_in_use/设备参数详情.xlsx 不存在")
    
    except Exception as e:
        print(f"读取文件 database_in_use/设备参数详情.xlsx 时出错：{e}")

    # 筛选出 Alarm_Information_Unit 列的值等于 temp_or_press 的行
    filtered_data_press = df[df["Alarm_Information_Unit"] == "kPa"]
    filtered_data_temp = df[df["Alarm_Information_Unit"] == "℃"]

    if temp_or_press == "press":
        # filtered_data = filtered_data_press
        filtered_data = filtered_data_press[filtered_data_press["Channel_Text_CN"].str.contains(device_name, na=False)]
    elif temp_or_press == "temp":
        # filtered_data = filtered_data_temp
        filtered_data = filtered_data_temp[filtered_data_press["Channel_Text_CN"].str.contains(device_name, na=False)]
    else:
        print("Invalid input. Please enter 'temp' or 'press'.")
    if filtered_data_press.empty:
        print(f"未找到 {temp_or_press} 的记录")
        return None
    
    print(filtered_data)
    return filtered_data
    
#查询时间段内所有设备的状态变化，并排除 status 为 'False' 的记录。
def get_all_device_status_by_time_range(start_time, end_time):
    """ 
    根据数据表名、开始时间和结束时间，查询设备在该时间段内的状态变化，并排除 status 为 'False' 的记录。
    参数:
    start_time (str): 开始时间，格式为 'YYYY-MM-DD HH:MM:SS'
    end_time (str): 结束时间，格式为 'YYYY-MM-DD HH:MM:SS'

    返回:
    dict: 包含设备状态变化的时间点和对应状态的字典，或错误信息
    """

    def get_status_changes(table_name, device_name):
        """
        辅助函数：获取指定设备在指定时间范围内的状态变化。

        参数:
        table_name (str): 数据表名
        device_name (str): 设备名称

        返回:
        dict: 包含设备状态变化的时间点和对应状态的字典，或错误信息
        """

        
        metadata = {
            "table_name": table_name,
            "start_time": start_time,
            "end_time": end_time,
        }

        print("----------------get_all_status_changes---------------"+start_time+"---"+end_time+"-----"+device_name+"----"+table_name)
        try:
            df = pd.read_csv(f"database_in_use/{table_name}.csv")
        except FileNotFoundError:
            return {"error": f"数据表 {table_name} 不存在", "metadata": metadata}

        # 将csvTime列从时间戳转换为datetime类型
        # df["csvTime"] = pd.to_datetime(df["csvTime"])  # 假设时间戳是纳秒级别
        # 确保时间列是 datetime 类型
        try:
            df["csvTime"] = pd.to_datetime(df["csvTime"])
         
        except Exception as e:
            raise ValueError(f"时间列转换失败: {e}")
        # 将开始时间和结束时间转换为datetime类型
        start_time_dt = pd.to_datetime(start_time)
        end_time_dt = pd.to_datetime(end_time)

         # 检查并确保 start_time 早于 end_time，如果不是则对调
        if start_time_dt > end_time_dt:
            start_time_dt, end_time_dt = end_time_dt, start_time_dt
      
        # 筛选指定时间范围内的数据，并排除 status 为 status_name 的记录
            
        # print(df)

        filtered_data = df[(df["csvTime"] >= start_time_dt) & (df["csvTime"] <= end_time_dt) & (df["status"] != 'False')].copy()
        

        if filtered_data.empty:
            return {
                "error": f"在数据表 {table_name} 中未找到时间范围 {start_time} 到 {end_time} 且 status 不为null 的数据",
                "metadata": metadata,
            }

        # 检查是否存在status列
        if "status" not in filtered_data.columns:
            return {
                "error": f"数据表 {table_name} 中不存在 'status' 列",
                "metadata": metadata,
            }

        # 获取设备状态变化的时间点和对应状态
        status_changes = filtered_data[
            ["csvTime", "status"]
        ].copy()  # 显式创建副本以避免警告

        # 使用 .loc 避免 SettingWithCopyWarning
        status_changes.loc[:, "csvTime"] = status_changes["csvTime"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # 将结果转换为字典
        return {
            "设备名称": device_name,
            "正在进行的关键动作": status_changes.to_dict(orient="records"),
        }
    

    action_list_ajia=["A架开机","征服者起吊","征服者入水","缆绳解除","A架摆回","A架关机","A架摆出","缆绳挂妥","征服者出水","征服者落座"]
    action_list_dp=["ON DP","OFF DP"]
    action_list_zhebi=["小艇落座","折臂吊车开机","小艇检查完毕","小艇入水","折臂吊车关机"]

    # 获取三个设备的状态变化
    # if "DP" in shebeiname:
    #     if shebeiname not in action_list_dp:
    #         raise ValueError("DP设备名称必须在表中：",action_list_dp)
    #     res = get_status_changes("Port3_ksbg_9", "定位设备",shebeiname)
    # elif "折臂吊车" in shebeiname or "小艇" in shebeiname:
    #     if shebeiname not in action_list_zhebi:
    #         raise ValueError("折臂吊车或小艇设备名称必须在表中：",action_list_zhebi)
    #     res = get_status_changes("device_13_11_meter_1311", "折臂吊车",shebeiname)
    # else:
    #     if shebeiname not in action_list_ajia:
    #         raise ValueError("A架设备名称必须在表中：",action_list_ajia)
    #     res = get_status_changes("Ajia_plc_1", "A架",shebeiname)

    result1=get_status_changes("Ajia_plc_1", "A架")
    result2=get_status_changes("Port3_ksbg_9", "定位设备")
    result3=get_status_changes("device_13_11_meter_1311", "折臂吊车")

    # print(result1)
    # print(result2)
    # print(result3)
    
    

    # print(res)
    # 过滤掉包含错误的结果
    results = [
        result for result in [result1,result2,result3] if "error" not in result
    ]
    print(results)

    # 返回结果和元数据
    return {
        "result": results,
        "metadata": {"start_time": start_time, "end_time": end_time},
    }

# get_all_device_status_by_time_range("2024-08-23 19:05:00", "2024-08-23 19:05:59")
#找到一段时间内的作业阶段
def get_zuoye_stage_by_time_range(start_time, end_time):
    """ 
    根据数据表名、开始时间和结束时间，查询该时间段内的作业阶段，并排除 status 为 'False' 的记录。
    参数:
    start_time (str): 开始时间，格式为 'YYYY-MM-DD HH:MM:SS'
    end_time (str): 结束时间，格式为 'YYYY-MM-DD HH:MM:SS'

    返回:
    dict: 包含设备状态变化的时间点和对应状态的字典，或错误信息
    """

    def get_stage_changes():
        """
        辅助函数：获取指定设备在指定时间范围内的状态变化。

        参数:
        table_name (str): 数据表名
        device_name (str): 设备名称

        返回:
        dict: 包含设备状态变化的时间点和对应状态的字典，或错误信息
        """

        
        metadata = {

            "start_time": start_time,
            "end_time": end_time,
        }

        print("----------------get_zuoye_stage_changes---------------"+start_time+"---"+end_time)
        try:
            df = pd.read_csv(f"database_in_use/Ajia_plc_1.csv")
        except FileNotFoundError:
            return {"error": f"数据表 Ajia_plc_1 不存在", "metadata": metadata}

        # 将csvTime列从时间戳转换为datetime类型
        # df["csvTime"] = pd.to_datetime(df["csvTime"])  # 假设时间戳是纳秒级别
        # 确保时间列是 datetime 类型
        try:
            df["csvTime"] = pd.to_datetime(df["csvTime"])
         
        except Exception as e:
            raise ValueError(f"时间列转换失败: {e}")
        # 将开始时间和结束时间转换为datetime类型
        start_time_dt = pd.to_datetime(start_time)
        end_time_dt = pd.to_datetime(end_time)

         # 检查并确保 start_time 早于 end_time，如果不是则对调
        if start_time_dt > end_time_dt:
            start_time_dt, end_time_dt = end_time_dt, start_time_dt
      
        # 筛选指定时间范围内的数据，并排除 status 为 status_name 的记录
            
        # print(df)

        filtered_data = df[(df["csvTime"] >= start_time_dt) & (df["csvTime"] <= end_time_dt) & (df["stage"] != 'False' ) & (df["stage"] != '伴航开始' ) & (df["stage"] != '伴航结束' )]
        

        if filtered_data.empty:
            return {
                "error": f"在数据表 Ajia_plc_1 中未找到时间范围 {start_time} 到 {end_time} 且 stage 不为null 的数据",
                "metadata": metadata,
            }

        # 检查是否存在status列
        if "status" not in filtered_data.columns or "stage" not in filtered_data.columns:
            return {
                "error": f"数据表 Ajia_plc_1 中不存在 'status' 或 'stage' 列",
                "metadata": metadata,
            }

        # 获取设备状态变化的时间点和对应状态
        status_changes = filtered_data[
            ["csvTime", "stage"]
        ].copy()  # 显式创建副本以避免警告

        # 使用 .loc 避免 SettingWithCopyWarning
        status_changes.loc[:, "csvTime"] = status_changes["csvTime"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # 将结果转换为字典
        return {
            "作业阶段": status_changes.to_dict(orient="records"),
        }


    res=get_stage_changes()
    

    # print(res)
    # 过滤掉包含错误的结果
    results = [
        result for result in [res] if "error" not in result
    ]
    print(results)

    # 返回结果和元数据
    return {
        "result": results,
        "metadata": {"start_time": start_time, "end_time": end_time},
    }


# get_zuoye_stage_by_time_range("2024-08-24 00:00:00","2024-08-24 23:59:59")

#找到一段时间内新增四个状态的变化
def calculate_new4_stage_by_time_range_test(start_time, end_time, stage_name):
    """ 
    根据数据表名、开始时间和结束时间，找到一段时间内新增四个状态的变化，并排除 status 为 'False' 的记录。
    参数:
    start_time (str): 开始时间，格式为 'YYYY-MM-DD HH:MM:SS'
    end_time (str): 结束时间，格式为 'YYYY-MM-DD HH:MM:SS'

    返回:
    dict: 包含设备状态变化的时间点和对应状态的字典，或错误信息
    """

    def get_stage_changes(table_name,column_name):
        """
        辅助函数：获取指定设备在指定时间范围内的状态变化。

        参数:
        table_name (str): 数据表名
        device_name (str): 设备名称

        返回:
        dict: 包含设备状态变化的时间点和对应状态的字典，或错误信息
        """

        
        metadata = {
            "table_name": table_name,
            "start_time": start_time,
            "end_time": end_time,
        }

        print("----------------get_new_4_stage_changes---------------"+start_time+"---"+end_time)
        try:
            df = pd.read_csv(f"database_in_use/{table_name}.csv",encoding="utf-8")
        except FileNotFoundError:
            return {"error": f"数据表 {table_name} 不存在", "metadata": metadata}

        # 将csvTime列从时间戳转换为datetime类型
        # df["csvTime"] = pd.to_datetime(df["csvTime"])  # 假设时间戳是纳秒级别
        # 确保时间列是 datetime 类型
        try:
            df["csvTime"] = pd.to_datetime(df["csvTime"])
         
        except Exception as e:
            raise ValueError(f"时间列转换失败: {e}")
        # 将开始时间和结束时间转换为datetime类型
        start_time_dt = pd.to_datetime(start_time)
        end_time_dt = pd.to_datetime(end_time)

         # 检查并确保 start_time 早于 end_time，如果不是则对调
        if start_time_dt > end_time_dt:
            start_time_dt, end_time_dt = end_time_dt, start_time_dt
      
        # 筛选指定时间范围内的数据，并排除 status 为 status_name 的记录
            
        # print(df)

        filtered_data = df[(df["csvTime"] >= start_time_dt) & (df["csvTime"] <= end_time_dt) & df[column_name].str.contains(stage_name, na=False) ]
        

        if filtered_data.empty:
            return {
                "error": f"在数据表 {table_name} 中未找到时间范围 {start_time} 到 {end_time} 且 stage 不为null 的数据",
                "metadata": metadata,
            }

        # 检查是否存在status列
        if column_name not in filtered_data.columns:
            return {
                "error": f"数据表 {table_name} 中不存在 {column_name} 列",
                "metadata": metadata,
            }

        # 获取设备状态变化的时间点和对应状态
        status_changes = filtered_data[
            ["csvTime", column_name]
        ].copy()  # 显式创建副本以避免警告

        # 使用 .loc 避免 SettingWithCopyWarning
        status_changes.loc[:, "csvTime"] = status_changes["csvTime"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # 将结果转换为字典
        return {
            "阶段": status_changes.to_dict(orient="records"),
        }
    
    # 获取三个设备的状态变化
    if "动力定位" in stage_name:
        res = get_stage_changes("Port3_ksbg_9", "stage")
    elif "航渡" in stage_name or "停泊" in stage_name or "伴航" in stage_name:
        res = get_stage_changes("status_output", "status")
    # elif "伴航" in stage_name:
    #     res = get_stage_changes("Ajia_plc_1", "stage")
    else:
        raise ValueError("Invalid stage name，仅支持四种状态查询：动力定位、航渡、伴航、停泊")

    # print(res)


    # res=get_stage_changes()
    

    # print(res)
    # 过滤掉包含错误的结果
    results = [
        result for result in [res] if "error" not in result
    ]
    # print(results)

    # 返回结果和元数据
    return {
        "result": results,
        "metadata": {"start_time": start_time, "end_time": end_time},
    }

def calculate_new4_stage_by_time_range(start_time, end_time, stage_name):
    """ 
    根据数据表名、开始时间和结束时间，找到一段时间内新增四个状态的变化，并排除 status 为 'False' 的记录。
    参数:
    start_time (str): 开始时间，格式为 'YYYY-MM-DD HH:MM:SS'
    end_time (str): 结束时间，格式为 'YYYY-MM-DD HH:MM:SS'

    返回:
    dict: 包含设备状态变化的时间点和对应状态的字典，或错误信息
    """

    def get_stage_changes(table_name,column_name):
        """
        辅助函数：获取指定设备在指定时间范围内的状态变化。

        参数:
        table_name (str): 数据表名
        device_name (str): 设备名称

        返回:
        dict: 包含设备状态变化的时间点和对应状态的字典，或错误信息
        """

        
        metadata = {
            "table_name": table_name,
            "start_time": start_time,
            "end_time": end_time,
        }

        print("----------------get_new_4_stage_changes---------------"+start_time+"---"+end_time)
        try:
            df = pd.read_csv(f"database_in_use/{table_name}.csv",encoding="utf-8")
        except FileNotFoundError:
            return {"error": f"数据表 {table_name} 不存在", "metadata": metadata}

        # 将csvTime列从时间戳转换为datetime类型
        # df["csvTime"] = pd.to_datetime(df["csvTime"])  # 假设时间戳是纳秒级别
        # 确保时间列是 datetime 类型
        try:
            df["csvTime"] = pd.to_datetime(df["csvTime"])
         
        except Exception as e:
            raise ValueError(f"时间列转换失败: {e}")
        # 将开始时间和结束时间转换为datetime类型
        start_time_dt = pd.to_datetime(start_time)
        end_time_dt = pd.to_datetime(end_time)

         # 检查并确保 start_time 早于 end_time，如果不是则对调
        if start_time_dt > end_time_dt:
            start_time_dt, end_time_dt = end_time_dt, start_time_dt
      
        # 筛选指定时间范围内的数据，并排除 status 为 status_name 的记录
            
        # print(df)
        filtered_data = df[(df["csvTime"] >= start_time_dt) & (df["csvTime"] <= end_time_dt) & df[column_name].str.contains(stage_name, na=False)]
        
        # 检查是否存在status列
        if column_name not in filtered_data.columns:
            return {
                "error": f"数据表 {table_name} 中不存在 {column_name} 列",
                "metadata": metadata,
            }
        
        # filtered_data = df[(df["csvTime"] >= start_time_dt) & (df["csvTime"] <= end_time_dt) & df[column_name].str.contains(stage_name, na=False)]
        
        
        if len(filtered_data)<2:     # 说明这段时间没有任何关于stage_name状态的记录（要么这段时间没有任何stage_name状态，要么stage_name状态涵盖了该时间段）
            # 筛选所有包含 stage_name 的行
            df_stage = df[df[column_name].str.contains(stage_name, na=False)].copy()
            # df_stage = df_stage.sort_values("csvTime")

            intervals = []
            current_start = None

            # 遍历 df_stage，配对“开始”和“结束”
            for idx, row in df_stage.iterrows():
                status_value = row[column_name]
                if status_value == stage_name + "开始":
                    current_start = row["csvTime"]
                elif status_value == stage_name + "结束" and current_start is not None:
                    current_end = row["csvTime"]
                    # 计算当前区间与 (start_time_dt, end_time_dt) 的交集
                    overlap_start = max(current_start, start_time_dt)
                    overlap_end = min(current_end, end_time_dt)
                    if current_start >= overlap_end:
                        break
                    if overlap_start < overlap_end:
                        intervals.append({"csvTime": overlap_start, column_name: stage_name + "开始"})
                        intervals.append({"csvTime": overlap_end, column_name: stage_name + "结束"})
                    current_start = None

            # 将交叠的区间构造为新的 DataFrame
            if intervals:
                filtered_data = pd.DataFrame(intervals)
            # else:
            #     filtered_data = pd.DataFrame(columns=["csvTime", column_name])

        # 如果和stage_name状态没有任何重叠，则说明该时间段内没有该状态
        if filtered_data.empty:
            return {
                "error": f"在数据表 {table_name} 中未找到时间范围 {start_time} 到 {end_time} 且 stage 不为null 的数据",
                "metadata": metadata,
            }


        # 获取设备状态变化的时间点和对应状态
        status_changes = filtered_data[
            ["csvTime", column_name]
        ].copy()  # 显式创建副本以避免警告

        # 使用 .loc 避免 SettingWithCopyWarning
        status_changes.loc[:, "csvTime"] = status_changes["csvTime"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # print({
        #     "阶段": status_changes.to_dict(orient="records"),
        # })
        # 将结果转换为字典
        return {
            "阶段": status_changes.to_dict(orient="records"),
        }
    
    # 获取三个设备的状态变化
    if "动力定位" in stage_name:
        res = get_stage_changes("Port3_ksbg_9", "stage")
    elif "航渡" in stage_name or "停泊" in stage_name or "伴航" in stage_name:
        res = get_stage_changes("status_output", "status")
    # elif "伴航" in stage_name:
    #     res = get_stage_changes("Ajia_plc_1", "stage")
    else:
        raise ValueError("Invalid stage name，仅支持四种状态查询：动力定位、航渡、伴航、停泊")

    # print(res)


    # res=get_stage_changes()
    

    # print(res)
    # 过滤掉包含错误的结果
    results = [
        result for result in [res] if "error" not in result
    ]
    print(results)

    # 返回结果和元数据
    return {
        "result": results,
        "metadata": {"start_time": start_time, "end_time": end_time},
    }

# calculate_new4_stage_by_time_range("2024-10-04 00:00:00","2024-10-04 23:59:59","停泊")

from datetime import datetime

def review_calculate(operation_type, *args):
    """
    多功能计算函数，支持四则运算和时间差值计算
    :param operation_type: 操作类型，可选 "math"（数值计算）或 "time"（时间计算）
    :param args: 
        - 当operation_type="math"时，参数格式为：(运算符, 数值1, 数值2)
        - 当operation_type="time"时，参数格式为：(时间字符串1, 时间字符串2, 时间格式)
    :return: 计算结果或错误提示
    """
    try:
        if operation_type == "math":
            operator, num1, num2 = args
            # 数值类型验证（支持整数和浮点数）
            if not all(isinstance(x, (int, float)) for x in (num1, num2)):
                raise ValueError("输入必须为数字")
                
            operations = {
                '+': lambda a, b: a + b,
                '-': lambda a, b: a - b,
                '*': lambda a, b: a * b,
                '/': lambda a, b: a / b if b != 0 else "错误：除数不能为零",
                '%': lambda a, b: a % b
            }
            return operations.get(operator, "错误：无效运算符")(num1, num2)

        elif operation_type == "time":
            time_str1, time_str2, time_format = args
            # 时间格式验证
            try:
                t1 = datetime.strptime(time_str1, time_format)
                t2 = datetime.strptime(time_str2, time_format)
            except ValueError:
                return "错误：时间格式不匹配"
                
            delta = t2 - t1
            # 返回详细时间差分解
            return {
                "小时": delta.seconds // 3600,
                "分钟": (delta.seconds % 3600) // 60,
                "秒": delta.seconds % 60
            }

        else:
            return "错误：未知操作类型"
            
    except ValueError as ve:
        return f"输入错误：{ve}"
    except ZeroDivisionError:
        return "错误：除数不能为零"
    except Exception as e:
        return f"未知错误：{str(e)}"
#api_new
def xiafang(df):
    '''
    对df时间段内的数据进行处理，找到征服者起吊、缆绳解除、征服者入水、A架摆回的时间点
    '''
    # print(df[['csvTime','check_current_presence']])
    events = df[(df['check_current_presence'].isin(['有电流', '无电流']))]
    # print(events)

    # 检查事件数量是否为偶数
    if events.shape[0] >= 2 and events.shape[0] % 2 == 0:
        # 遍历所有偶数索引的事件对
        for i in range(0, events.shape[0], 2):
            event_start = events.iloc[i]
            event_end = events.iloc[i + 1]
            # 确保第一个事件是“有电流”，第二个事件是“无电流”
            if event_start['check_current_presence'] == '有电流' and event_end['check_current_presence'] == '无电流':
                start_event_time = event_start['csvTime']
                end_event_time = event_end['csvTime']

                # 提取两个事件之间的数据
                between_events = df[(df['csvTime'] >= start_event_time) & (df['csvTime'] <= end_event_time)]
                data1 = between_events[['Ajia-3_v', 'Ajia-5_v']].values.tolist()
                len_peaks, peak_L = find_peaks(data1)
                if len_peaks == 0:
                    continue
                value_11 = find_first_increasing_value(data1)
                indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                df.loc[indices, 'status'] = '征服者起吊'
                if not indices:
                    print('征服者起吊找不到索引',value_11)
                value_11 = find_first_stable_after_peak(data1, df.loc[indices[0], 'Ajia-5_v'])
                indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                df.loc[indices, 'status'] = '缆绳解除'
                if not indices:
                    print('缆绳解除找不到索引',value_11)
                previous_indices = [idx - 1 for idx in indices if idx > 0]
                df.loc[previous_indices, 'status'] = '征服者入水'
                # A架摆回
                value_11 = find_peak_after_stable(data1, peak_L[-1])
                indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                df.loc[indices, 'status'] = 'A架摆回'
                return df


def huishou(df):
    events = df[(df['check_current_presence'].isin(['有电流', '无电流']))]
    # 检查事件数量是否为偶数
    if events.shape[0] >= 2 and events.shape[0] % 2 == 0:
        # 遍历所有偶数索引的事件对
        for i in range(0, events.shape[0], 2):
            event_start = events.iloc[i]
            event_end = events.iloc[i + 1]
            # 确保第一个事件是“有电流”，第二个事件是“无电流”
            if event_start['check_current_presence'] == '有电流' and event_end['check_current_presence'] == '无电流':
                start_event_time = event_start['csvTime']
                end_event_time = event_end['csvTime']

                # 提取两个事件之间的数据
                between_events = df[(df['csvTime'] >= start_event_time) & (df['csvTime'] <= end_event_time)]
                data1 = between_events[['Ajia-3_v', 'Ajia-5_v']].values.tolist()
                len_peaks, peak_L = find_peaks(data1)
                if len_peaks == 0:
                    continue
                indices = between_events.index[between_events['Ajia-5_v'] == peak_L[-1]].tolist()
                df.loc[indices, 'status'] = '征服者出水'
                previous_indices = [idx - 1 for idx in indices if idx > 0]
                df.loc[previous_indices, 'status'] = '缆绳挂妥'
                value_11 = find_first_stable_after_peak(data1, peak_L[-1])
                indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                df.loc[indices, 'status'] = '征服者落座'
                return df


def natural_processing():
    print("natural_processing")

#判断设备是否处于运行状态
def is_open(time_point, shebeiname):
    """
    计算指定时间段内的设备是否在运行
    :param start_time: 查询的开始时间（字符串或 datetime 类型）
    :param end_time: 查询的结束时间（字符串或 datetime 类型）
    :param shebeiname: 设备名称，默认为 '一号柴油发电机'
    :return: 返回时长（分钟）以及开关的次数
    """
    print("-------is_open***判断是否运行-------"+shebeiname)
    # 设备配置映射：设备名称 -> (文件路径, 开机状态, 关机状态)
    device_config = {
        "一号柴油发电机": ("Port1_ksbg_3", "P1_88.14"),
        "二号柴油发电机": ("Port1_ksbg_4", "P1_90.5"),
        "三号柴油发电机": ("Port2_ksbg_3", "P2_73.8"),
        "四号柴油发电机": ("Port2_ksbg_3", "P2_74.15"),

        "侧推": ("Port3_ksbg_9", "P3_18"),

    }
    shebei_list=["一号柴油发电机","二号柴油发电机","三号柴油发电机","四号柴油发电机","侧推"]

    # 检查设备名称是否有效
    if shebeiname not in device_config:
        raise ValueError(f"未知的设备名称: {shebeiname}，支持设备：‘一号柴油发电机’,‘二号柴油发电机’,‘三号柴油发电机’,‘四号柴油发电机’,‘侧推’")
    elif shebeiname =="一号柴油发电机" or shebeiname == "二号柴油发电机" or shebeiname == "三号柴油发电机" or shebeiname == "四号柴油发电机" or shebeiname =="侧推":
        table_name, column_name = device_config[shebeiname]
        is_open=get_value_of_time(time_point,table_name,column_name)
        if is_open>0:
            print(shebeiname,"在运行")
            return True
        else:
            print(shebeiname,"不在运行")
            return False

#计算收放缆次数
def count_shoufanglan(start_time, end_time, device_name):
    """
    计算每个指定时间段内的总能耗
    :param start_time: 查询的开始时间（字符串或 datetime 类型）
    :param end_time: 查询的结束时间（字符串或 datetime 类型）
    :param device_name: 设备名称，默认为 '绞车A'
    :return: 放缆和收缆次数（int 类型）
    """
    # 设备配置映射：设备名称 -> (表名, 功率列名)
    # 侧推、推进器、甲板机械、推进系统、发电机
    device_config = {
        "绞车A": ("Jiaoche_plc_1", "PLC_point0_value"),  # 添加绞车变频器的配置
        "绞车B": ("Jiaoche_plc_1", "PLC_point3_value"),
        "绞车C": ("Jiaoche_plc_1", "PLC_point6_value"),
        
    }
    print("---------count_shoufanglan--------------"+ device_name+"---------"+start_time+"---"+end_time)

    result = {
                "function": "count_shoufanglan",  # 说明这个返回结果来自哪个函数
                "result": (
                        f"收揽次数：0次",
                        f"放揽次数：0次"
                ),
            }
    print(f"result: {result}")
    return result


# calculate_uptime("2024-06-04 00:00:00","2024-06-04 23:59:59","DP")
# get_device_status_by_time_range("2024-05-23 23:00:00","2024-05-24 00:00:00")
# get_value_of_time("2024-08-19 13:34:27", "Port1_ksbg_3", "P1_66")
    
# s="sdgah ,fhskf sfj"
# s=s.replace(" ",",")
# s=s.replace(",,",",")
# print(s)