import os
import pandas as pd
from collections import defaultdict
import json
from datetime import datetime
from predict_seq import get_result
from collections import Counter



def is_normal_range(val1, val2):
    return (50 <= val1 < 60 and 50 <= val2 < 60)
    # return (50 <= val1 <= 60 and 50 <= val2 <= 68) or (50 <= val2 <= 60 and 50 <= val1 <= 68)

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
            
        if normal_after < 1:
            continue
            
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

def find_first_increasing_value(data):
    """
    找到列表中第一个从稳定值（70以下）开始增加的值。(不能只看是否大于阈值，还要看是否有连续的增加/大于阈值)

    参数:
    data (list): 输入的数值列表。data现在包含两列数据的元组(Ajia-3_v,Ajia-5_v)

    返回:
    tuple: 第一个大于70的值及其索引。如果未找到，返回 (None, None)。
    """
    # Check if data has at least 3 elements (we need current, previous, and next values)
    data = data[:-1]
    if len(data) < 3:
        return -1

    # Iterate through data from second element to second-to-last
    for i in range(1, len(data)-1):
        prev_val1, prev_val2 = data[i-1]
        curr_val1, curr_val2 = data[i]
        next_val1, next_val2 = data[i+1]
        
        # Check if previous values are in stable range (50-68)
        if prev_val1 < 60 or prev_val2 < 60:
            # Check if current values are higher than previous and outside stable range
            if ((curr_val1 > prev_val1 or curr_val2 > prev_val2) and curr_val1 > 60 and curr_val2 > 60):
                # Check if next values are higher than current and both above 80
                if next_val1 > 60 and next_val2 > 60:
                    return curr_val2

    # raise Exception(f"{data}\n No increasing value found")
    return None

def find_stable_value(data, peak1, peak2):
    """
    找到两个峰值之间的数据中，回落到稳定值的第一个值。
    假设稳定值在 50 到 60 之间。

    参数:
    data (list): 数据列表
    peak1 (float): 第一个峰值
    peak2 (float): 第二个峰值

    返回:
    float or None: 稳定值，如果未找到则返回 None
    """
    data = data[:-1]
    # 找到峰值之间的数据
    try:
        # Find indices where second element of tuple matches peaks

        start_index = next(i for i, x in enumerate(data) if x[1] == peak1)
        end_index = next(i for i, x in enumerate(data) if x[1] == peak2)
    except ValueError:
        # 如果峰值不在列表中，返回 None
        return None

    between_peaks = data[start_index:end_index + 1]
    # print(type(data[0][0]), type(data[0][1]))
    # print('data')
    # 找到回落到稳定值的第一个值（假设稳定值在 50 到 60 之间）（该值的后一个应该也处于稳定值）
    for current_idx, value in enumerate(between_peaks):  # 使用 enumerate 直接获取索引和值
        if current_idx < len(between_peaks) - 1:
            next_value = between_peaks[current_idx + 1]
            if (is_normal_range(value[0], value[1])): #and is_normal_range(next_value[0], next_value[1])
                return value[1]

    # 如果未找到稳定值，返回 None
    # raise Exception(f"{data}\n No stable value found between peak1:{peak1} and peak2:{peak2}")
    return None

def find_first_stable_after_peak(data, peak, stable_min=50, stable_max=59):
    """
    从峰值到列表末尾的数据中，找到第一个回落到稳定值的值。

    参数:
    data (list): 数据列表
    peak (float): 峰值
    stable_min (float): 稳定值的最小值
    stable_max (float): 稳定值的最大值

    返回:
    float or None: 稳定值，如果未找到则返回 None
    """
    data = data[:-1]
    try:
        # 找到峰值的索引
        start_index = next(i for i, x in enumerate(data) if x[1] == peak)
    except ValueError:
        # 如果峰值不在列表中，返回 None
        return None

    # 切片获取从峰值到列表末尾的数据
    after_peak = data[start_index:]

    # 找到回落到稳定值的第一个值（该值的后一个应该也处于稳定值）
    for current_idx, value in enumerate(after_peak):  # 使用 enumerate 直接获取索引和值
            if current_idx < len(after_peak) - 1:
                next_value = after_peak[current_idx + 1]
                if (is_normal_range(value[0], value[1])):
                    return value[1]

    # 如果未找到稳定值，返回 None
    # If no stable value found after peak, raise an exception
    # raise Exception(f"{data}\n No stable value found after peak {peak}")
    return None

def find_peak_after_stable(data, stable, stable_min=50, stable_max=59):
    """
    从峰值到列表末尾的数据中，找到第一个回落到稳定值的值。
    参数:
    data (list): 数据列表
    peak (float): 峰值
    stable_min (float): 稳定值的最小值
    stable_max (float): 稳定值的最大值
    返回:
    float or None: 稳定值，如果未找到则返回 None
    """
    def is_normal_range(val1, val2):
        return (50 <= val1 < 60 or 50 <= val2 < 60)
    data = data[:-1]
    try:
        # 找到峰值的索引
        start_index = next(i for i, x in enumerate(data) if x[1] == stable)
    except ValueError:
        # 如果峰值不在列表中，返回 None
        return None
    
    # 切片获取从峰值到列表末尾的数据
    after_stable = data[start_index:]
    max_sum = -float('inf')
    peak_val = None
    for value in after_stable:
        if is_normal_range(value[0], value[1]):
            continue
        curr_sum = value[0] + value[1]
        if curr_sum > max_sum:
            max_sum = curr_sum
            peak_val = value[1]
    return peak_val
        
    # 如果未找到稳定值，返回 None
    # If no stable value found after peak, raise an exception
    # raise Exception(f"{data}\n No stable value found after stable {stable}")
    return None

if __name__ == "__main__":
    
    data_path = '../../assets/初赛数据/'
    # 合并相同前缀的CSV文件，以及将Excel文件转换为CSV格式。
    def merge_csv_files(folder_path, out_path):
        # Store files by their prefix
        file_groups = defaultdict(list)
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv') and '字段释义' not in file_name:
                prefix = file_name.rsplit('_', 1)[0]
                file_groups[prefix].append(os.path.join(folder_path, file_name))

        # Merge files with the same prefix
        for prefix, file_list in file_groups.items():
            merged_df = pd.concat((pd.read_csv(file) for file in file_list), ignore_index=True)
            output_file = os.path.join(out_path, f'{prefix}.csv')

            print('-----------')
            print(output_file)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            merged_df.to_csv(output_file, index=False)
            print('---完成---')
            print(f'Merged files with prefix "{prefix}" into {output_file}')

        # Convert Excel to CSV
        os.makedirs('database_in_use', exist_ok=True)
        df_device = pd.read_excel(f'{data_path}设备参数详情.xlsx')
        df_device.to_csv('data/设备参数详情表.csv', index=False)
        df_device.to_csv('database_in_use/设备参数详情表.csv', index=False)


    # merge_csv_files(data_path, 'data')
    # merge_csv_files(data_path, 'database_in_use')


    # In[3]:
    # %%
    # Pre2: Status and Events Detection
    # 定义一个函数，将值转换为数值类型，无法转换的返回 -1
    def convert_to_numeric(value):
        try:
            return float(value)
        except ValueError:
            return -1


    # 读取A架CSV文件
    df = pd.read_csv('data/Ajia_plc_1.csv')
    # 将 Ajia-3_v 和 Ajia-5_v 列转换为数值类型，无法转换的设为 -1
    df['Ajia-3_v'] = df['Ajia-3_v'].apply(convert_to_numeric)
    df['Ajia-5_v'] = df['Ajia-5_v'].apply(convert_to_numeric)


    # 初始化 status 列，默认值为 'False'
    df['status'] = 'False'
    df['check_current_presence'] = 'False'

    # 读取设备CSV文件
    df_device = pd.read_csv('data/device_13_11_meter_1311.csv')

    # 将13-11-6_v列转换为数值类型，无法转换的保留原值
    df_device['13-11-6_v'] = pd.to_numeric(df_device['13-11-6_v'], errors='coerce')

    # 初始化status和action列
    df_device['status'] = 'False'
    df_device['action'] = 'False'

    df_device['13-11-6_v_new'] = df_device['13-11-6_v'].tolist()
    # 检测折臂吊车的开机和关机事件
    segments = []
    start_time = None

    for i in range(1, df_device.shape[0]):
        # 开机
        if df_device.iloc[i - 1]['13-11-6_v'] == 0 and df_device.iloc[i]['13-11-6_v'] > 0:
            df_device.at[df_device.index[i], 'status'] = '折臂吊车开机'
        # 关机
        if df_device.iloc[i - 1]['13-11-6_v'] > 0 and df_device.iloc[i]['13-11-6_v'] == 0:
            df_device.at[df_device.index[i], 'status'] = '折臂吊车关机'

        # 检测由待机进入工作和由工作进入待机的事件
        if df_device.iloc[i - 1]['13-11-6_v_new'] <= 9 and df_device.iloc[i]['13-11-6_v_new'] > 9 and df_device.iloc[i]['status'] != '折臂吊车开机': 
            df_device.at[df_device.index[i], 'action'] = '由待机进入工作'
        if df_device.iloc[i - 1]['13-11-6_v_new'] > 9 and df_device.iloc[i]['13-11-6_v_new'] <= 9 and df_device.iloc[i-1]['status'] != '折臂吊车开机':
            df_device.at[df_device.index[i], 'action'] = '由工作进入待机'
        # 遍历DataFrame
        # if df_device.iloc[i]['status'] == None:
        #     df_device.iloc[i]['status'] = 'False'

    for index, row in df_device.iterrows():
        if row['status'] == '折臂吊车开机':
            start_time = row['csvTime']
        elif row['status'] == '折臂吊车关机' and start_time is not None:
            end_time = row['csvTime']
            segments.append((start_time, end_time))
            start_time = None


    # 提取每个区段内的“由待机进入工作”和“由工作进入待机”事件
    for segment in segments:
        start, end = segment
        # if start == '2024-05-26 16:04:42':
        #     print('debug')
        events = df_device[
            (df_device['csvTime'] >= start) & (df_device['csvTime'] <= end) & (df_device['action'].isin(['由待机进入工作', '由工作进入待机']))]
        events_2 = df_device[(df_device['csvTime'] >= start) & (df_device['csvTime'] <= end)]
        # 检查事件数量是否为偶数且等于6
        if 0 < events.shape[0] <= 6:
            print(f"开机时间: {start}, 关机时间: {end}")
            print(f"事件数量: {events.shape[0]}")
            # 处理每一对事件，从后往前遍历
            for i in range(events.shape[0]-2, -1, -2):  # 从最后一对事件开始，每次减2
                event_start = events.iloc[i]
                event_end = events.iloc[i + 1]
                
                if event_start['action'] == '由待机进入工作' and event_end['action'] == '由工作进入待机':
                    start_event_time = event_start['csvTime']
                    end_event_time = event_end['csvTime']
                    between_events = df_device[(df_device['csvTime'] >= start_event_time) & (df_device['csvTime'] <= end_event_time)]
                    data1 = list(between_events['13-11-6_v'])

                    # 找到最后一个大于9的值
                    last_value_above_9 = next((x for x in reversed(data1) if x > 9), None)

                    if last_value_above_9 is not None:
                        all_indices = between_events.index[between_events['13-11-6_v_new'] == last_value_above_9].tolist()
                        last_index = all_indices[-1] if all_indices else None

                        # 根据事件对的顺序更新status，从后往前判定
                        if last_index is not None and df_device.loc[last_index, 'status'] == "False":
                            if i == events.shape[0]-2:  # 最后一对事件
                                df_device.loc[last_index, 'status'] = '小艇落座'
                            elif i == events.shape[0]-4:  # 倒数第二对事件
                                df_device.loc[last_index, 'status'] = '小艇入水'
                            elif i == events.shape[0]-6:  # 倒数第三对事件
                                df_device.loc[last_index, 'status'] = '小艇检查完毕'
                    else:
                        print("列表中没有大于 9 的值")

    df_device = df_device.drop(columns=['action'])
    df_device = df_device.drop(columns=['13-11-6_v_new'])

    '''
    动作判定：
    上午判定为下放阶段和下午判定为回收阶段，上午下午的区分点为12:00
    不管上午还是下午，判定逻辑如下：
    1、根据error-》非error 、非error-》error的状态判定开关机状态
    2、根据上午下午，判定是下放阶段还是回收阶段

    3、若为下放阶段
    3.1、先确定征服者起吊，然后往前在13-11-6_v找回落前的最后一个值（小艇入水）、再往前找回落前的最后一个值（小艇检查完毕）、再往前找13-11-6_v数值从0增加（折臂吊车开机）
    3.2、再确定缆绳解除，然后往前找回落前的最后一个值（折臂吊车关机）
    3.3、缆绳解除后，找峰值（A架摆回）
    3.4、确定dp：在小艇检查完毕动作前找ON DP，确定了ON DP后，再找OFF DP（数值归零）
    若征服者起吊不好确定，可以反着来，从后往前找，A架摆回->缆绳解除->征服者入水->征服者起吊->(换表)小艇入水->小艇检查完毕->折臂吊车开机   

    4、若为回收阶段
    4.1、关键判断两个峰值：第一个峰值（A架摆出，在小艇入水前）和第二个峰值（征服者出水）
    4.2、征服者出水->缆绳挂妥->(换表)小艇入水->小艇检查完毕->折臂吊车开机（回落前的最后一个值：前一个>7，后一个<=7）

    5、折臂吊车关机->小艇落座


    考虑数据缺失情况（A架开机、关机的时间是否是连续的）

    假设Ajia-3_v,Ajia-5_v两个表是同步变化的（0->非0、非0->0）
    如果不是同步变化的，则不处理（以后改进优化）
    '''


    def sliding_window_5(arr):
        """滑动窗口大小为5的逻辑"""
        window_size = 5
        modified_arr = arr.copy()
        for i in range(len(arr) - window_size + 1):
            window = arr[i:i + window_size]
            if window[1] < 0 and window[2] < 0 and window[3] < 0 and window[0] >= 0 and window[4] >= 0:
                modified_arr[i + 1:i + 4] = [0] * 3
            elif window[1] >= 0 and window[2] >= 0 and window[3] >= 0 and window[0] < 0 and window[4] < 0:
                modified_arr[i + 1:i + 4] = [-1] * 3
        return modified_arr

    def sliding_window_4(arr):
        """滑动窗口大小为4的逻辑"""
        window_size = 4
        modified_arr = arr.copy()
        for i in range(len(arr) - window_size + 1):
            window = arr[i:i + window_size]
            if window[1] < 0 and window[2] < 0 and window[0] >= 0 and window[3] >= 0:
                modified_arr[i + 1:i + 3] = [0] * 2
            elif window[1] >= 0 and window[2] >= 0 and window[0] < 0 and window[3] < 0:
                modified_arr[i + 1:i + 3] = [-1] * 2
        return modified_arr

    def sliding_window_3(arr):
        """滑动窗口大小为3的逻辑"""
        window_size = 3
        modified_arr = arr.copy()
        for i in range(len(arr) - window_size + 1):
            window = arr[i:i + window_size]
            if window[1] < 0 and window[0] >= 0 and window[2] >= 0:
                modified_arr[i + 1] = 0
            if window[1] >= 0 and window[0] < 0 and window[2] < 0:
                modified_arr[i + 1] = -1
            if window[1] > 90 and 60 > window[0] >50 and 50 < window[2] < 60:
                modified_arr[i + 1] = (window[0]+window[2])/2
        return modified_arr

    # 应用滑动窗口修改Ajia中的电流噪声
    df['Ajia-3_v'] = sliding_window_5(df['Ajia-3_v'].tolist())
    df['Ajia-3_v'] = sliding_window_4(df['Ajia-3_v'].tolist())
    df['Ajia-3_v'] = sliding_window_3(df['Ajia-3_v'].tolist())

    df['Ajia-5_v'] = sliding_window_5(df['Ajia-5_v'].tolist())
    df['Ajia-5_v'] = sliding_window_4(df['Ajia-5_v'].tolist())
    df['Ajia-5_v'] = sliding_window_3(df['Ajia-5_v'].tolist())


    cnt = 0
    index = None
    segments = []  # 存储开机关机时间段

    # 遍历每一行，判断A架开关机状态及是否有电流，并存储开机关机时间段
    for i in range(1, df.shape[0]):
        # 由于数据缺失，开关机并不一定是成对的，有可能只有开机没有关机，或者只有关机没有开机
        # 所以需要判断是否有未匹配的开机或关机时间

        # 开机条件：电流值从error变为0（取0）(上一行和下一行的时间点是否连续)
        if df.loc[i - 1, 'Ajia-3_v'] == -1 and df.loc[i, 'Ajia-3_v'] >= 0 and df.loc[i-1, 'Ajia-5_v'] == -1 and df.loc[i, 'Ajia-5_v'] >= 0:
            cnt = 1
            current_time = pd.to_datetime(df.loc[i,'csvTime'])
            prev_time = pd.to_datetime(df.loc[i-1,'csvTime']) 
            # 检查时间差是否为1分钟
            if (current_time - prev_time).total_seconds() == 60 or 1 == 1: # 1 == 1 用于取消判定
                df.loc[i, 'status'] = 'A架开机'
                index = i
                

        # 关机条件：电流值变为error（取error）(上一行和下一行的时间点是否连续)
        if df.loc[i, 'Ajia-3_v'] == -1 and df.loc[i - 1, 'Ajia-3_v'] >= 0 and df.loc[i, 'Ajia-5_v'] == -1 and df.loc[i - 1, 'Ajia-5_v'] >= 0:
            current_time = pd.to_datetime(df.loc[i,'csvTime'])
            prev_time = pd.to_datetime(df.loc[i-1,'csvTime']) 
            # 检查时间差是否为1分钟
            if (current_time - prev_time).total_seconds() == 60 or 1 == 1:
                if current_time.date() != prev_time.date(): # 如果不是同一天，则将前一行的状态设为'A架关机'
                    df.loc[i-1, 'status'] = 'A架关机'
                    # 如果有error->0的变化，且有过开机状态
                    if cnt == 1 and index is not None:
                        # 如果开关机时间段内有电流，则存储该时间段 
                        if any(df.loc[index:i-1, 'check_current_presence'] == '有电流'):
                            segments.append((df.iloc[index]['csvTime'], df.iloc[i-1]['csvTime']))
                            index = None
                else:    
                    df.loc[i, 'status'] = 'A架关机'
                    # 如果有error->0的变化，且有过开机状态
                    if cnt == 1 and index is not None:
                        # 如果开关机时间段内有电流，则存储该时间段 
                        if any(df.loc[index:i, 'check_current_presence'] == '有电流'):
                            segments.append((df.iloc[index]['csvTime'], df.iloc[i]['csvTime']))
                            index = None
            cnt = 0


        # 判断有无电流(两边同时有才算有电流，若只有一边有则不算)，用于后续提取实际运行时间，判断逻辑如下：
        # 开始时间是3v!=0 and 5v!=的第一个点 结束时间为 3v==0 and 5v==0的第一个点
        if df.loc[i, 'Ajia-3_v'] > 0 and df.loc[i, 'Ajia-5_v'] > 0 and not (df.loc[i - 1, 'Ajia-3_v'] > 0 and df.loc[i - 1, 'Ajia-5_v'] > 0):
            df.loc[i, 'check_current_presence'] = '有电流'
        if df.loc[i - 1, 'Ajia-3_v'] > 0 and df.loc[i - 1, 'Ajia-5_v'] > 0 and not (df.loc[i, 'Ajia-3_v'] > 0 and df.loc[i, 'Ajia-5_v'] > 0):
            df.loc[i, 'check_current_presence'] = '无电流'


    # 已知开关机时间段的df，判断其中的电流是否有增减
    # 若无明显波动（都是正常值，没有从稳定值（50多）开始增加，也没有从高值回落至稳定值（50多），更没有峰值），则该时间段内不进行动作判定

    # 从CSV文件中提取一天内有两次开机的第一次和第二次开机时间。
    def extract_daily_power_on_times(df):
        """
        参数:
        file_path (str): CSV文件的路径，包含 'csvTime' 和 'status' 列。

        返回:
        first_start_times (list): 一天内有两次开机的第一次开机时间列表。
        second_start_times (list): 一天内有两次开机的第二次开机时间列表。
        """
        # 读取CSV文件
        df = df

        # 将 csvTime 转换为 datetime 类型
        df['csvTime'] = pd.to_datetime(df['csvTime'])

        # 按天分组
        df['date'] = df['csvTime'].dt.date

        # 初始化一个字典来存储每天的开机关机时间段
        daily_segments = {}

        # 遍历每一天的数据
        for date, group in df.groupby('date'):
            segments = []
            start_time = None

            # 遍历每一天的记录
            for index, row in group.iterrows():
                if row['status'] == '开机':
                    start_time = row['csvTime']
                elif row['status'] == '关机' and start_time is not None:
                    end_time = row['csvTime']
                    segments.append((start_time, end_time))
                    start_time = None

            # 将每天的开机关机时间段存入字典
            daily_segments[date] = segments

        # 统计每天的开机关机次数
        daily_counts = {date: len(segments) for date, segments in daily_segments.items()}

        # 筛选出一天内有两次开机关机的情况
        two_times_days = [date for date, count in daily_counts.items() if count == 2]

        # 初始化两个列表来存储第一次和第二次的开机时间
        first_start_times = []
        second_start_times = []

        # 遍历这些日期，提取第一次和第二次的开机时间
        for date in two_times_days:
            segments = daily_segments[date]
            first_start_times.append(segments[0][0])  # 第一次开机时间
            second_start_times.append(segments[1][0])  # 第二次开机时间

        return first_start_times, second_start_times


    def extract_events(df, segment):
        start, end = segment
        # start = pd.to_datetime(start)
        # end = pd.to_datetime(end)
        # print(f"开机时间: {start}, 关机时间: {end}")
        events = df[
            (df['csvTime'] >= start) & (df['csvTime'] <= end) & (df['check_current_presence'].isin(['有电流', '无电流']))]
        # print(f"事件数量: {events.shape[0]}")
        L3 = []
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
                    # print(between_events[['Ajia-3_v', 'Ajia-5_v']].values)
                    # print(data1)
                    # print(f"事件对 ({i}, {i + 1}) 之间的数据: {data1}")

                    # 调用 find_peaks 函数（假设已定义）
                    len_peaks, peak_L = find_peaks(data1)
                    # print(f'峰值为{peak_L}')
                    L3.append(len_peaks)
        return L3

    # Store segments by date
    segments_by_date = {}

    # Create a copy of segments and only keep valid ones
    segments = [segment for segment in segments if sum(extract_events(df, segment=segment)) >= 2]

    # Process each segment
    # for segment in segments:
    #     start_time = pd.to_datetime(segment[0])
    #     date = start_time.date()
        
    #     # Add segment to date's list
    #     if date not in segments_by_date:
    #         segments_by_date[date] = []
    #     segments_by_date[date].append(segment)

    # Print dates with 2 or more segments
    # for date, date_segments in segments_by_date.items():
    #     if len(date_segments) < 2:
    #         print(f"\nDate: {date}")
    #         for segment in date_segments:
    #             print(f"Segment: {segment[0]} to {segment[1]}")

    # 根据A架设备状态时间戳，确定小艇和折臂吊车的动作
    def device(csvTime):
        '''
        参数：
        csvTime: 设备状态的时间戳
        返回：
        小艇入水的时间，用于确定回收阶段的A架摆出
        '''
        # print('csvTime')
        # print(csvTime)
        # Find rows in df_device with matching csvTime to the minute
        target_time = pd.to_datetime(csvTime)
        
        # Convert df_device csvTime to datetime if not already
        df_device['csvTime'] = pd.to_datetime(df_device['csvTime'])
        
        # Get matching rows where minute components match
        matching_rows = df_device[df_device['csvTime'].dt.floor('min') == target_time.floor('min')]
        # print(matching_rows)
        # print(matching_rows.index)
        row_idx = matching_rows.index[0]
        # print(row_idx)
        # Find previous crane shutdown time
        start_row = row_idx
        for i in range(row_idx-1, -1, -1):
            if df_device.loc[i, 'status'] == '折臂吊车开机':
                start_row = i
                break
                
        # Find next crane shutdown time  
        end_row = row_idx
        for i in range(row_idx+1, len(df_device)):
            if df_device.loc[i, 'status'] == '折臂吊车关机':
                end_row = i
                break
        
        cnt = 0
        boat_water_time = ''
        # Search backwards from the matching row
        for i in range(row_idx-1, start_row, -1):
            # Check for '小艇入水'
            if cnt == 0 and df_device.loc[i, '13-11-6_v'] <= 7 and df_device.loc[i-1, '13-11-6_v'] > 7:
                df_device.loc[i-1, 'status'] = '小艇入水'
                boat_water_time = str(df_device.loc[i-1, 'csvTime'])
                cnt += 1
                continue
                
            # Check for '小艇检查完毕'
            if cnt == 1 and i > start_row + 1 and df_device.loc[i, '13-11-6_v'] <= 7 and df_device.loc[i-1, '13-11-6_v'] > 7:
                df_device.loc[i-1, 'status'] = '小艇检查完毕'
                cnt += 1
                
        # Then search backwards from crane shutdown to find boat landing
        if end_row is not None:
            for i in range(end_row, row_idx, -1):
                if df_device.loc[i, '13-11-6_v'] <= 7 and df_device.loc[i-1, '13-11-6_v'] > 7:
                    df_device.loc[i-1, 'status'] = '小艇落座'
                    break
        
        if cnt < 2 :
            print('找不到小艇入水或小艇检查完毕')
            return None
        return boat_water_time  # Return the time when boat entered water

    def xiafang(segment):
        start, end = segment
        L4 = extract_events(df, segment=segment)
        events = df[(df['csvTime'] >= start) & (df['csvTime'] <= end) & (df['check_current_presence'].isin(['有电流', '无电流']))]
        # Get first non-zero index from L4 starting from the end
        first_nonzero_idx = next((i for i in range(len(L4)-1, -1, -1) if L4[i] != 0), None)

        if first_nonzero_idx is not None:
            # Calculate start and end indices for the events
            event_start_idx = first_nonzero_idx * 2  # Each L4 element corresponds to 2 events
            event_end_idx = event_start_idx + 1

            # Get the corresponding events
            start_event_time = events.iloc[event_start_idx]['csvTime']
            end_event_time = events.iloc[event_end_idx]['csvTime']

            if L4[first_nonzero_idx] >= 2:
                between_events = df[(df['csvTime'] >= start_event_time) & (df['csvTime'] <= end_event_time)]
                data1 = between_events[['Ajia-3_v', 'Ajia-5_v']].values.tolist()
                # print(data1)
                len_peaks, peak_L = find_peaks(data1)

                value_11 = find_first_increasing_value(data1)
                indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                df.loc[indices, 'status'] = '征服者起吊'

                value_11 = find_first_stable_after_peak(data1, df.loc[indices[0], 'Ajia-5_v'])
                # value_11 = find_stable_value(data1, peak_L[-2], peak_L[-1])
                indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                df.loc[indices, 'status'] = '缆绳解除'
                if not indices:
                    print('找不到索引',value_11)
                previous_indices = [idx - 1 for idx in indices if idx > 0]
                df.loc[previous_indices, 'status'] = '征服者入水'
                device(df.loc[previous_indices, 'csvTime'].iloc[0])
                # 找到 between_events 中 Ajia-5_v 等于 target_value 的索引
                indices = between_events.index[between_events['Ajia-5_v'] == peak_L[-1]].tolist()
                df.loc[indices, 'status'] = 'A架摆回'
            else:
                between_events = df[(df['csvTime'] >= start_event_time) & (df['csvTime'] <= end_event_time)]
                data1 = between_events[['Ajia-3_v', 'Ajia-5_v']].values.tolist()
                # print(data1)
                len_peaks, peak_L = find_peaks(data1)
                indices = between_events.index[between_events['Ajia-5_v'] == peak_L[-1]].tolist()
                df.loc[indices, 'status'] = 'A架摆回'

                # Get second non-zero index by searching backwards from first_nonzero_idx
                second_nonzero_idx = next((i for i in range(first_nonzero_idx-1, -1, -1) if L4[i] != 0), None) if first_nonzero_idx else None

                # Calculate start and end indices for the events
                event_start_idx = second_nonzero_idx * 2  # Each L4 element corresponds to 2 events 
                event_end_idx = event_start_idx + 1

                # Get the corresponding events
                start_event_time = events.iloc[event_start_idx]['csvTime']
                end_event_time = events.iloc[event_end_idx]['csvTime']

                between_events = df[(df['csvTime'] >= start_event_time) & (df['csvTime'] <= end_event_time)]
                data1 = between_events[['Ajia-3_v', 'Ajia-5_v']].values.tolist()
                # print(data1)
                len_peaks, peak_L = find_peaks(data1)
                value_11 = find_first_increasing_value(data1)
                indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                df.loc[indices, 'status'] = '征服者起吊'
                if not indices:
                    print('征服者起吊找不到索引',value_11)
                    print(segment)
                    print(data1)
                    print(peak_L)
                value_11 = find_first_stable_after_peak(data1, df.loc[indices[0], 'Ajia-5_v'])
                indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                if not indices:
                    print('缆绳解除找不到索引',value_11)
                    print(segment)
                    print(data1)
                    print(peak_L)
                df.loc[indices, 'status'] = '缆绳解除'
                previous_indices = [idx - 1 for idx in indices if idx > 0]
                df.loc[previous_indices, 'status'] = '征服者入水'
                device(df.loc[previous_indices, 'csvTime'].iloc[0])


    def huishou(segment):
        start, end = segment
        L4 = extract_events(df, segment=segment)
        events = df[(df['csvTime'] >= start) & (df['csvTime'] <= end) & (df['check_current_presence'].isin(['有电流', '无电流']))]
        
        first_nonzero_idx = next((i for i in range(len(L4)-1, -1, -1) if L4[i] != 0), None)
        if first_nonzero_idx is not None:
            # Calculate start and end indices for the events
            event_start_idx = first_nonzero_idx * 2  # Each L4 element corresponds to 2 events
            event_end_idx = event_start_idx + 1

            # Get the corresponding events
            start_event_time = events.iloc[event_start_idx]['csvTime']
            end_event_time = events.iloc[event_end_idx]['csvTime']

            if L4[first_nonzero_idx] >= 2:
                
                between_events = df[(df['csvTime'] >= start_event_time) & (df['csvTime'] <= end_event_time)]
                data1 = between_events[['Ajia-3_v', 'Ajia-5_v']].values.tolist()
                # print(data1)
                len_peaks, peak_L = find_peaks(data1)

                # 找到 between_events 中 Ajia-5_v 等于 target_value 的索引
                indices = between_events.index[between_events['Ajia-5_v'] == peak_L[-1]].tolist()
                df.loc[indices, 'status'] = '征服者出水'
                previous_indices = [idx - 1 for idx in indices if idx > 0]
                df.loc[previous_indices, 'status'] = '缆绳挂妥'
                csvTime = device(df.loc[previous_indices, 'csvTime'].iloc[0])
                if csvTime:
                    value_11 = find_first_stable_after_peak(data1, peak_L[-1])
                    indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                    df.loc[indices, 'status'] = '征服者落座'


                    for i in range(len(peak_L)-2, -1, -1):
                        indices = between_events.index[between_events['Ajia-5_v'] == peak_L[i]].tolist()
                        peak_time = df.loc[indices[0], 'csvTime']
                        
                        if pd.to_datetime(peak_time) < pd.to_datetime(csvTime):
                            # If peak is before csvTime, use it for 'A架摆出'
                            df.loc[indices, 'status'] = 'A架摆出'
                            return

                    second_nonzero_idx = next((i for i in range(first_nonzero_idx-1, -1, -1) if L4[i] != 0), None) if first_nonzero_idx else None
                    
                    while pd.to_datetime(peak_time) >= pd.to_datetime(csvTime) and second_nonzero_idx is not None:
                        if second_nonzero_idx is not None:
                            # Calculate new event window
                            next_start_idx = second_nonzero_idx * 2
                            next_end_idx = next_start_idx + 1
                            
                            next_start_time = events.iloc[next_start_idx]['csvTime'] 
                            next_end_time = events.iloc[next_end_idx]['csvTime']
                            
                            # Get data and peaks for new window
                            next_events = df[(df['csvTime'] >= next_start_time) & (df['csvTime'] <= next_end_time)]
                            next_data = next_events[['Ajia-3_v', 'Ajia-5_v']].values.tolist()
                            next_len_peaks, next_peak_L = find_peaks(next_data)
                            
                            if next_len_peaks > 0:
                                # Find first peak before csvTime
                                for peak in next_peak_L:
                                    indices = next_events.index[next_events['Ajia-5_v'] == peak].tolist()
                                    peak_time = df.loc[indices[0], 'csvTime']
                                    
                                    if pd.to_datetime(peak_time) < pd.to_datetime(csvTime):
                                        df.loc[indices, 'status'] = 'A架摆出'
                                        return
                        # Need to look in next non-zero L4 segment
                        second_nonzero_idx = next((i for i in range(first_nonzero_idx-1, -1, -1) if L4[i] != 0), None) if first_nonzero_idx else None
                    # Get indices where csvTime equals start and end
                    # start_idx = df.index[df['csvTime'] == start].tolist()[0]
                    # end_idx = df.index[df['csvTime'] == end].tolist()[0]

                    # Reset status to 'False' for all rows between start+1 and end-1
                    df.loc[(df['csvTime'] > start) & (df['csvTime'] < end), 'status'] = 'False'
                    xiafang(segment)
                else:
                    df.loc[indices, 'status'] = 'False'
                    df.loc[previous_indices, 'status'] = 'False'
                    xiafang(segment)
                    return
            else:
                between_events = df[(df['csvTime'] >= start_event_time) & (df['csvTime'] <= end_event_time)]
                data1 = between_events[['Ajia-3_v', 'Ajia-5_v']].values.tolist()
                # print(data1)
                len_peaks, peak_L = find_peaks(data1)

                indices = between_events.index[between_events['Ajia-5_v'] == peak_L[-1]].tolist()
                df.loc[indices, 'status'] = '征服者出水'
                previous_indices = [idx - 1 for idx in indices if idx > 0]
                df.loc[previous_indices, 'status'] = '缆绳挂妥'
                csvTime = device(df.loc[previous_indices, 'csvTime'].iloc[0])
                if csvTime:
                    value_11 = find_first_stable_after_peak(data1, peak_L[-1])
                    indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                    df.loc[indices, 'status'] = '征服者落座'
                    
                    # Get second non-zero index by searching backwards from first_nonzero_idx
                    second_nonzero_idx = next((i for i in range(first_nonzero_idx-1, -1, -1) if L4[i] != 0), None) if first_nonzero_idx else None
                    while second_nonzero_idx is not None:
                        if second_nonzero_idx is not None:
                            # Calculate new event window
                            next_start_idx = second_nonzero_idx * 2
                            next_end_idx = next_start_idx + 1
                            
                            next_start_time = events.iloc[next_start_idx]['csvTime'] 
                            next_end_time = events.iloc[next_end_idx]['csvTime']
                            
                            # Get data and peaks for new window
                            next_events = df[(df['csvTime'] >= next_start_time) & (df['csvTime'] <= next_end_time)]
                            next_data = next_events[['Ajia-3_v', 'Ajia-5_v']].values.tolist()
                            next_len_peaks, next_peak_L = find_peaks(next_data)
                            
                            if next_len_peaks > 0:
                                # Find first peak before csvTime
                                for peak in reversed(next_peak_L):
                                    indices = next_events.index[next_events['Ajia-5_v'] == peak].tolist()
                                    peak_time = df.loc[indices[0], 'csvTime']
                                    
                                    if pd.to_datetime(peak_time) < pd.to_datetime(csvTime):
                                        df.loc[indices, 'status'] = 'A架摆出'
                                        return
                        # Need to look in next non-zero L4 segment
                        second_nonzero_idx = next((i for i in range(second_nonzero_idx-1, -1, -1) if L4[i] != 0), None) if second_nonzero_idx else None
                    df.loc[(df['csvTime'] > start) & (df['csvTime'] < end), 'status'] = 'False'
                    xiafang(segment)
                else:
                    df.loc[indices, 'status'] = 'False'
                    df.loc[previous_indices, 'status'] = 'False'
                    xiafang(segment)
                    return


    # 对每个有效开关机时间段进行分类处理（下放、回收）
    for segment in segments:
        print('当天segment：',segment)
        # if segment[0] == '2024-08-24 16:03:08':
        # if segment[0] == '2024-08-19 13:34:27':
        # if segment[0] == '2024-05-23 23:05:49':
        # if segment[0] == '2024-05-25 09:55:49':
            # print('debug')

        # Skip if segment spans multiple days
        # start_time = pd.to_datetime(segment[0])
        # end_time = pd.to_datetime(segment[1])
        # if start_time.date() != end_time.date():
        #     print('跨天', segment)
        #     continue
        if segment == segments[0]:
            # For first segment, check if start time is before noon
            start_time = pd.to_datetime(segment[0])
            if start_time.hour < 12:
                xiafang(segment)
            else:
                huishou(segment)
        else:
            # Get previous segment's start time
            prev_segment = segments[segments.index(segment)-1]
            prev_start = pd.to_datetime(prev_segment[0])
            curr_start = pd.to_datetime(segment[0])
            
            # Check if current and previous segments are on same day
            if prev_start.date() == curr_start.date():
                huishou(segment)
            else:
                if curr_start.hour < 12:
                    xiafang(segment)
                else:
                    huishou(segment)

        
        '''
        L4 = extract_events(df, segment=segment)
        print(L4)
        
        start, end = segment
        if not (start >= '2024-08-25' and end <= '2024-08-25 23:59:08'):
            continue
        print('debug')

        events_2 = df[(df['csvTime'] >= start) & (df['csvTime'] <= end)]
        # print('-----------------事件--------------------')
        # print(list(events_2['Ajia-5_v']))
        # print('-----------------事件--------------------')

        if not L4 or sum(L4) < 2:  # if L4 is empty or sum of elements is less than 2
            continue
        
        if L4 == [0, 2]:     #下放阶段
            events = df[(df['csvTime'] >= start) & (df['csvTime'] <= end) & (
                df['check_current_presence'].isin(['有电流', '无电流']))]
            if events.shape[0] % 2 == 0:
                if events.iloc[0]['check_current_presence'] == '有电流' and events.iloc[1][
                    'check_current_presence'] == '无电流':
                    start_event_time = events.iloc[0]['csvTime']
                    end_event_time = events.iloc[1]['csvTime']
                    between_events = df[(df['csvTime'] >= start_event_time) & (df['csvTime'] <= end_event_time)]
                    data1 = list(between_events[['Ajia-3_v', 'Ajia-5_v']].values)

                    len_peaks, peak_L = find_peaks(data1)

                    if len_peaks == 0:
                        if events.iloc[2]['check_current_presence'] == '有电流' and events.iloc[3][
                            'check_current_presence'] == '无电流':
                            start_event_time = events.iloc[2]['csvTime']
                            end_event_time = events.iloc[3]['csvTime']
                            between_events = df[(df['csvTime'] >= start_event_time) & (df['csvTime'] <= end_event_time)]
                            data1 = list(between_events[['Ajia-3_v', 'Ajia-5_v']].values)
                            print(data1)
                            len_peaks, peak_L = find_peaks(data1)
                            if len_peaks == 2:
                                value_11 = find_first_increasing_value(data1)
                                indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                                df.loc[indices, 'status'] = '征服者起吊'

                                value_11 = find_stable_value(data1, peak_L[0], peak_L[1])
                                indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                                df.loc[indices, 'status'] = '缆绳解除'
                                previous_indices = [idx - 1 for idx in indices if idx > 0]
                                df.loc[previous_indices, 'status'] = '征服者入水'

                                # 找到 between_events 中 Ajia-5_v 等于 target_value 的索引
                                indices = between_events.index[between_events['Ajia-5_v'] == peak_L[1]].tolist()
                                df.loc[indices, 'status'] = 'A架摆回'
        elif L4 == [2]:    #下放阶段
            events = df[(df['csvTime'] >= start) & (df['csvTime'] <= end) & (
                df['check_current_presence'].isin(['有电流', '无电流']))]
            if events.shape[0] % 2 == 0:
                if events.iloc[0]['check_current_presence'] == '有电流' and events.iloc[1][
                    'check_current_presence'] == '无电流':
                    start_event_time = events.iloc[0]['csvTime']
                    end_event_time = events.iloc[1]['csvTime']
                    between_events = df[(df['csvTime'] >= start_event_time) & (df['csvTime'] <= end_event_time)]
                    data1 = list(between_events[['Ajia-3_v', 'Ajia-5_v']].values)

                    len_peaks, peak_L = find_peaks(data1)

                    if len_peaks == 2:
                        if events.iloc[0]['check_current_presence'] == '有电流' and events.iloc[1][
                            'check_current_presence'] == '无电流':
                            start_event_time = events.iloc[0]['csvTime']
                            end_event_time = events.iloc[1]['csvTime']
                            between_events = df[(df['csvTime'] >= start_event_time) & (df['csvTime'] <= end_event_time)]
                            data1 = list(between_events[['Ajia-3_v', 'Ajia-5_v']].values)
                            len_peaks, peak_L = find_peaks(data1)
                            if len_peaks == 2:
                                value_11 = find_first_increasing_value(data1)
                                indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                                df.loc[indices, 'status'] = '征服者起吊'

                                value_11 = find_stable_value(data1, peak_L[0], peak_L[1])
                                indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                                df.loc[indices, 'status'] = '缆绳解除'
                                previous_indices = [idx - 1 for idx in indices if idx > 0]
                                df.loc[previous_indices, 'status'] = '征服者入水'

                                # 找到 between_events 中 Ajia-5_v 等于 target_value 的索引
                                indices = between_events.index[between_events['Ajia-5_v'] == peak_L[1]].tolist()
                                df.loc[indices, 'status'] = 'A架摆回'

        elif L4 == [0, 3]:  #下放阶段
            events = df[(df['csvTime'] >= start) & (df['csvTime'] <= end) & (
                df['check_current_presence'].isin(['有电流', '无电流']))]
            if events.shape[0] % 2 == 0:
                if events.iloc[0]['check_current_presence'] == '有电流' and events.iloc[1][
                    'check_current_presence'] == '无电流':
                    start_event_time = events.iloc[0]['csvTime']
                    end_event_time = events.iloc[1]['csvTime']
                    between_events = df[(df['csvTime'] >= start_event_time) & (df['csvTime'] <= end_event_time)]
                    data1 = list(between_events[['Ajia-3_v', 'Ajia-5_v']].values)

                    len_peaks, peak_L = find_peaks(data1)

                    if len_peaks == 0:
                        if events.iloc[2]['check_current_presence'] == '有电流' and events.iloc[3][
                            'check_current_presence'] == '无电流':
                            start_event_time = events.iloc[2]['csvTime']
                            end_event_time = events.iloc[3]['csvTime']
                            between_events = df[(df['csvTime'] >= start_event_time) & (df['csvTime'] <= end_event_time)]
                            data1 = list(between_events[['Ajia-3_v', 'Ajia-5_v']].values)
                            print(data1)
                            len_peaks, peak_L = find_peaks(data1)
                            if len_peaks == 3:
                                value_11 = find_first_increasing_value(data1)
                                indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                                df.loc[indices, 'status'] = '征服者起吊'

                                value_11 = find_stable_value(data1, peak_L[1], peak_L[2])
                                indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                                df.loc[indices, 'status'] = '缆绳解除'
                                previous_indices = [idx - 1 for idx in indices if idx > 0]
                                df.loc[previous_indices, 'status'] = '征服者入水'

                                # 找到 between_events 中 Ajia-5_v 等于 target_value 的索引
                                indices = between_events.index[between_events['Ajia-5_v'] == peak_L[2]].tolist()
                                df.loc[indices, 'status'] = 'A架摆回'
        elif L4 == [0, 1, 3]:   #下放阶段
            events = df[(df['csvTime'] >= start) & (df['csvTime'] <= end) & (
                df['check_current_presence'].isin(['有电流', '无电流']))]
            if events.shape[0] % 2 == 0:
                if events.iloc[0]['check_current_presence'] == '有电流' and events.iloc[1][
                    'check_current_presence'] == '无电流':
                    start_event_time = events.iloc[0]['csvTime']
                    end_event_time = events.iloc[1]['csvTime']
                    between_events = df[(df['csvTime'] >= start_event_time) & (df['csvTime'] <= end_event_time)]
                    data1 = list(between_events[['Ajia-3_v', 'Ajia-5_v']].values)

                    len_peaks, peak_L = find_peaks(data1)

                    if len_peaks == 0:
                        if events.iloc[4]['check_current_presence'] == '有电流' and events.iloc[5][
                            'check_current_presence'] == '无电流':
                            start_event_time = events.iloc[4]['csvTime']
                            end_event_time = events.iloc[5]['csvTime']
                            between_events = df[(df['csvTime'] >= start_event_time) & (df['csvTime'] <= end_event_time)]
                            data1 = list(between_events[['Ajia-3_v', 'Ajia-5_v']].values)

                            len_peaks, peak_L = find_peaks(data1)
                            if len_peaks == 3:
                                value_11 = find_first_increasing_value(data1)
                                indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                                df.loc[indices, 'status'] = '征服者起吊'
                                value_11 = find_stable_value(data1, peak_L[1], peak_L[2])
                                indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                                df.loc[indices, 'status'] = '缆绳解除'
                                previous_indices = [idx - 1 for idx in indices if idx > 0]
                                df.loc[previous_indices, 'status'] = '征服者入水'

                                # 找到 between_events 中 Ajia-5_v 等于 target_value 的索引
                                indices = between_events.index[between_events['Ajia-5_v'] == peak_L[2]].tolist()
                                df.loc[indices, 'status'] = 'A架摆回'
        elif L4 == [1, 2] or L4 == [1, 1]:  #回收阶段
            events = df[(df['csvTime'] >= start) & (df['csvTime'] <= end) & (
                df['check_current_presence'].isin(['有电流', '无电流']))]
            if events.shape[0] % 2 == 0:
                if events.iloc[0]['check_current_presence'] == '有电流' and events.iloc[1]['check_current_presence'] == '无电流':
                    start_event_time = events.iloc[0]['csvTime']
                    end_event_time = events.iloc[1]['csvTime']
                    between_events = df[(df['csvTime'] >= start_event_time) & (df['csvTime'] <= end_event_time)]
                    data1 = list(between_events[['Ajia-3_v', 'Ajia-5_v']].values)

                    len_peaks, peak_L = find_peaks(data1)

                    if len_peaks == 1:
                        value_11 = find_first_increasing_value(data1)
                        indices = between_events.index[between_events['Ajia-5_v'] == peak_L[0]].tolist()
                        df.loc[indices, 'status'] = 'A架摆出'
                        if events.iloc[2]['check_current_presence'] == '有电流' and events.iloc[3][
                            'check_current_presence'] == '无电流':
                            start_event_time = events.iloc[2]['csvTime']
                            end_event_time = events.iloc[3]['csvTime']
                            between_events = df[(df['csvTime'] >= start_event_time) & (df['csvTime'] <= end_event_time)]
                            data1 = list(between_events[['Ajia-3_v', 'Ajia-5_v']].values)

                            len_peaks, peak_L = find_peaks(data1)
                            max_value = max([x[1] for x in data1 if x[0] <= 200 and x[1] <= 200])

                            if len_peaks == 2:
                                # 找到 between_events 中 Ajia-5_v 等于 target_value 的索引
                                indices = between_events.index[between_events['Ajia-5_v'] == max_value].tolist()
                                df.loc[indices, 'status'] = '征服者出水'
                                previous_indices = [idx - 1 for idx in indices if idx > 0]
                                df.loc[previous_indices, 'status'] = '缆绳挂妥'

                                value_11 = find_first_stable_after_peak(data1, max_value)
                                indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                                df.loc[indices, 'status'] = '征服者落座'
                            elif len_peaks == 1:
                                # 找到 between_events 中 Ajia-5_v 等于 target_value 的索引
                                indices = between_events.index[between_events['Ajia-5_v'] == max_value].tolist()
                                df.loc[indices, 'status'] = '征服者出水'
                                previous_indices = [idx - 1 for idx in indices if idx > 0]
                                df.loc[previous_indices, 'status'] = '缆绳挂妥'

                                value_11 = find_first_stable_after_peak(data1, max_value)
                                indices = between_events.index[between_events['Ajia-5_v'] == value_11].tolist()
                                df.loc[indices, 'status'] = '征服者落座'
        else:#################其他情况

            events_2 = events_2.copy()
            events_2.loc[:, 'csvTime'] = pd.to_datetime(events_2['csvTime'])
            # 获取第一个值
            first_value = events_2['csvTime'].iloc[0]

            # 定义目标日期
            target_date = datetime(2024, 8, 19)  # 2024年8月19日   大模型预测测试
            is_target_date = (first_value.date() == target_date.date())

            target_date_1 = datetime(2024, 8, 19)  # 2024年8月19日   大模型预测测试
            is_target_date_1 = (first_value.date() == target_date.date())

            # 判断小时是否大于12点
            is_hour_greater_than_12 = first_value.hour > 17
            first_start_times, second_start_times = extract_daily_power_on_times(df=df)

            if is_target_date and is_hour_greater_than_12:  # 全部预测可以去掉is_target_date条件 或者根据问题传入
                events_2['new_column'] = events_2.apply(
                    lambda row: row['Ajia-3_v'] if row['Ajia-5_v'] == 0 and row['Ajia-3_v'] > 0 else row['Ajia-5_v'],
                    axis=1
                )
                print('----------------LLM预测的列表---------------------')
                print(str(list(events_2['new_column'])))

                try:
                    a, b, c = get_result(str(list(events_2['new_column'])), 1)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    a, b, c = -100, -100, -100
                print('----------------预测的值---------------------')
                print(a, b, c)

                indices = events_2.index[events_2['new_column'] == a].tolist()
                df.loc[indices, 'status'] = 'A架摆出'

                indices = events_2.index[events_2['new_column'] == b].tolist()
                df.loc[indices, 'status'] = '征服者出水'
                previous_indices = [idx - 1 for idx in indices if idx > 0]
                df.loc[previous_indices, 'status'] = '缆绳挂妥'

                indices = events_2.index[events_2['new_column'] == c].tolist()
                df.loc[indices, 'status'] = '征服者落座'

            elif first_value in first_start_times and 1 == 0:  # 去掉1==0由LLM判断状态  默认不开启 给大家分享参考思路。
                events_2['new_column'] = events_2.apply(
                    lambda row: row['Ajia-3_v'] if row['Ajia-5_v'] == 0 and row['Ajia-3_v'] > 0 else row['Ajia-5_v'],
                    axis=1
                )
                print('----------------LLM预测的列表---------------------')
                print(str(list(events_2['new_column'])))

                try:
                    a, b, c = get_result(str(list(events_2['new_column'])), 0)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    a, b, c = -100, -100, -100
                print('----------------预测的值---------------------')
                print(a, b, c)

                indices = events_2.index[events_2['new_column'] == a].tolist()
                df.loc[indices, 'status'] = '征服者起吊'

                indices = events_2.index[events_2['new_column'] == b].tolist()
                df.loc[indices, 'status'] = '缆绳解除'
                previous_indices = [idx - 1 for idx in indices if idx > 0]
                df.loc[previous_indices, 'status'] = '征服者入水'

                indices = events_2.index[events_2['new_column'] == c].tolist()
                df.loc[indices, 'status'] = 'A架摆回'

            elif first_value in second_start_times and 1 == 0:  # 去掉1==0由LLM判断状态
                events_2['new_column'] = events_2.apply(
                    lambda row: row['Ajia-3_v'] if row['Ajia-5_v'] == 0 and row['Ajia-3_v'] > 0 else row['Ajia-5_v'],
                    axis=1
                )
                print('----------------LLM预测的列表---------------------')
                print(str(list(events_2['new_column'])))

                try:
                    a, b, c = get_result(str(list(events_2['new_column'])), 1)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    a, b, c = -100, -100, -100
                print('----------------预测的值---------------------')
                print(a, b, c)

                indices = events_2.index[events_2['new_column'] == a].tolist()
                df.loc[indices, 'status'] = 'A架摆出'

                indices = events_2.index[events_2['new_column'] == b].tolist()
                df.loc[indices, 'status'] = '征服者出水'
                previous_indices = [idx - 1 for idx in indices if idx > 0]
                df.loc[previous_indices, 'status'] = '缆绳挂妥'

                indices = events_2.index[events_2['new_column'] == c].tolist()
                df.loc[indices, 'status'] = '征服者落座'

            print('------------------')
            print(L4)
        '''

    # df = df.drop(columns=['date'])  # 删除 'date' 列
    # df = df.drop(columns=['check_current_presence'])  # 删除 'date' 列
    df.to_csv('database_in_use/Ajia_plc_1.csv', index=False)
    df_device.to_csv('database_in_use/device_13_11_meter_1311.csv', index=False)
    # In[4]:


    import pandas as pd
    #############################################################dp 判断有问题
    # 读取CSV文件
    df = pd.read_csv('data/Port3_ksbg_9.csv')
    # 将P3_33列转换为数值类型，无法转换的保留原值
    df['P3_33'] = pd.to_numeric(df['P3_33'], errors='coerce')
    # 初始化status列
    df['status'] = 'False'
    # DP开关机事件检测
    # DP开启判断逻辑 P3_18 > 0 or P3_33 >0  DP关闭逻辑   P3_18 == 0 or P3_33  == 0
    for i in range(1, df.shape[0]):
        # DP开启
        if df.loc[i - 1, 'P3_33'] == 0 and df.loc[i - 1, 'P3_18'] == 0 and (df.loc[i, 'P3_33'] > 0 or df.loc[i - 1, 'P3_18'] > 0):
            df.loc[i, 'status'] = 'ON DP'
        # DP关闭
        if df.loc[i - 1, 'P3_33'] > 0 and df.loc[i - 1, 'P3_18'] > 0 and (df.loc[i, 'P3_33'] == 0 or df.loc[i - 1, 'P3_18'] == 0):
            df.loc[i, 'status'] = 'OFF DP'
    # 保存结果
    df.to_csv('database_in_use/Port3_ksbg_9.csv', index=False)

    # In[5]:

    '''
    # 读取CSV文件
    df = pd.read_csv('data/device_13_11_meter_1311.csv')

    # 将13-11-6_v列转换为数值类型，无法转换的保留原值
    df['13-11-6_v'] = pd.to_numeric(df['13-11-6_v'], errors='coerce')

    # 初始化status和action列
    df['status'] = 'False'
    df['action'] = 'False'


    def sliding_window_5(arr):
        """滑动窗口大小为5的逻辑"""
        window_size = 5
        modified_arr = arr.copy()
        for i in range(len(arr) - window_size + 1):
            window = arr[i:i + window_size]
            if window[1] < 10 and window[2] < 10 and window[3] < 10 and window[0] > 10 and window[4] > 10:
                # 将 window[0] 包装成列表进行赋值
                modified_arr[i + 1:i + 4] = [window[0]] * 3
        return modified_arr

    def sliding_window_4(arr):
        """滑动窗口大小为4的逻辑"""
        window_size = 4
        modified_arr = arr.copy()
        for i in range(len(arr) - window_size + 1):
            window = arr[i:i + window_size]
            if window[1] < 10 and window[2] < 10 and window[0] > 10 and window[3] > 10:
                # 将 window[0] 包装成列表进行赋值
                modified_arr[i + 1:i + 3] = [window[0]] * 2
        return modified_arr

    def sliding_window_3(arr):
        """滑动窗口大小为3的逻辑"""
        window_size = 3
        modified_arr = arr.copy()
        for i in range(len(arr) - window_size + 1):
            window = arr[i:i + window_size]
            if window[1] < 10 and window[0] > 10 and window[2] > 10:
                # 直接赋值，因为只修改一个值
                modified_arr[i + 1] = window[0]
        return modified_arr

    # 应用滑动窗口逻辑到 DataFrame 的某一列
    # df['13-11-6_v_new'] = sliding_window_5(df['13-11-6_v'].tolist())
    # df['13-11-6_v_new'] = sliding_window_4(df['13-11-6_v_new'].tolist())
    df['13-11-6_v_new'] = sliding_window_3(df['13-11-6_v'].tolist())

    # 检测折臂吊车的开机和关机事件
    segments = []
    start_time = None

    for i in range(1, df.shape[0]):
        # 开机
        if df.iloc[i - 1]['13-11-6_v'] == 0 and df.iloc[i]['13-11-6_v'] > 0:
            df.at[df.index[i], 'status'] = '折臂吊车开机'
        # 关机
        if df.iloc[i - 1]['13-11-6_v'] > 0 and df.iloc[i]['13-11-6_v'] == 0:
            df.at[df.index[i], 'status'] = '折臂吊车关机'

        # 检测由待机进入工作和由工作进入待机的事件
        if df.iloc[i - 1]['13-11-6_v_new'] < 10 and df.iloc[i]['13-11-6_v_new'] > 10:
            df.at[df.index[i], 'action'] = '由待机进入工作'
        if df.iloc[i - 1]['13-11-6_v_new'] > 10 and df.iloc[i]['13-11-6_v_new'] < 10:
            df.at[df.index[i], 'action'] = '由工作进入待机'
        # 遍历DataFrame
        if df.iloc[i]['status'] == None:
            df.iloc[i]['status'] = 'False'

    for index, row in df.iterrows():
        if row['status'] == '折臂吊车开机':
            start_time = row['csvTime']
        elif row['status'] == '折臂吊车关机' and start_time is not None:
            end_time = row['csvTime']
            segments.append((start_time, end_time))
            start_time = None


    # 提取每个区段内的“由待机进入工作”和“由工作进入待机”事件
    for segment in segments:
        start, end = segment
        events = df[
            (df['csvTime'] >= start) & (df['csvTime'] <= end) & (df['action'].isin(['由待机进入工作', '由工作进入待机']))]
        events_2 = df[(df['csvTime'] >= start) & (df['csvTime'] <= end)]
        # 检查事件数量是否为偶数且等于6
        if events.shape[0] == 6:
            print(f"开机时间: {start}, 关机时间: {end}")
            print(f"事件数量: {events.shape[0]}")
            # 处理每一对事件
            for i in range(0, 6, 2):

                event_start = events.iloc[i]
                event_end = events.iloc[i + 1]

                if event_start['action'] == '由待机进入工作' and event_end['action'] == '由工作进入待机':
                    start_event_time = event_start['csvTime']
                    end_event_time = event_end['csvTime']
                    between_events = df[(df['csvTime'] >= start_event_time) & (df['csvTime'] <= end_event_time)]
                    data1 = list(between_events['13-11-6_v'])

                    # 找到最后一个大于9的值
                    last_value_above_9 = next((x for x in reversed(data1) if x > 9), None)

                    if last_value_above_9 is not None:
                        all_indices = between_events.index[between_events['13-11-6_v_new'] == last_value_above_9].tolist()
                        last_index = all_indices[-1] if all_indices else None

                        # 根据事件对的顺序更新status
                        if last_index is not None:
                            if i == 0:
                                df.loc[last_index, 'status'] = '小艇检查完毕'
                            elif i == 2:
                                df.loc[last_index, 'status'] = '小艇入水'
                            elif i == 4:
                                df.loc[last_index, 'status'] = '小艇落座'
                    else:
                        print("列表中没有大于 9 的值")
        if events.shape[0] == 4:
            print(f"开机时间: {start}, 关机时间: {end}")
            print(f"事件数量: {events.shape[0]}")
            # 处理每一对事件
            for i in range(0, 4, 2):
                event_start = events.iloc[i]
                event_end = events.iloc[i + 1]
                if event_start['action'] == '由待机进入工作' and event_end['action'] == '由工作进入待机':
                    start_event_time = event_start['csvTime']
                    end_event_time = event_end['csvTime']
                    between_events = df[(df['csvTime'] >= start_event_time) & (df['csvTime'] <= end_event_time)]
                    data1 = list(between_events['13-11-6_v'])

                    # 找到最后一个大于9的值
                    last_value_above_9 = next((x for x in reversed(data1) if x > 9), None)

                    if last_value_above_9 is not None:
                        all_indices = between_events.index[between_events['13-11-6_v_new'] == last_value_above_9].tolist()
                        last_index = all_indices[-1] if all_indices else None

                        # 根据事件对的顺序更新status

                        if last_index is not None and df.loc[last_index, 'status'] == "FALSE":
                            if i == 0:
                                df.loc[last_index, 'status'] = '小艇入水'
                            elif i == 2:
                                df.loc[last_index, 'status'] = '小艇落座'
                    else:
                        print("列表中没有大于 9 的值")
                    # 保存结果
    df = df.drop(columns=['action'])
    df = df.drop(columns=['13-11-6_v_new'])

    df.to_csv('database_in_use/device_13_11_meter_1311.csv', index=False)
    '''

    # In[6]:


    # Pre3: Data Annotations
    '''

    def create_annotations():
        df_desc = pd.read_csv(f'{data_path}字段释义.csv', encoding='gbk')
        df_desc['字段含义_new'] = df_desc['字段含义'] + df_desc['单位'].apply(
            lambda x: f",单位:{x}" if pd.notnull(x) else "")
        field_dict = df_desc.set_index('字段名')['字段含义_new'].to_dict()

        folder_path = 'database_in_use'
        files = [f.split('.')[0] for f in os.listdir(folder_path) if f.endswith('.csv')]

        descriptions = []
        for file_name in files:
            df = pd.read_csv(f'{folder_path}/{file_name}.csv')
            columns = df.columns.tolist()
            annotations = [field_dict.get(col, '无注释') for col in columns]
            descriptions.append({'数据表名': file_name, '字段名': columns, "字段含义": annotations})

        # Customize specific tables
        for item in descriptions:
            if item['数据表名'] == 'Ajia_plc_1':
                item['字段含义'][-2] = 'A架动作,包括关机、开机、A架摆出、缆绳挂妥、征服者出水、征服者落座、征服者起吊、征服者入水、缆绳解除、A架摆回'
            if item['数据表名'] == 'device_13_11_meter_1311':
                item['字段含义'][-1] = '折臂吊车及小艇动作,包括折臂吊车关机,折臂吊车开机,小艇检查完毕,小艇入水,小艇落座'
            if item['数据表名'] == 'Port3_ksbg_9':
                item['字段含义'][-1] = 'DP动作,包括OFF_DP,ON_DP'

        with open('dict.json', 'w', encoding='utf-8') as f:
            json.dump(descriptions, f, ensure_ascii=False, indent=4)


    create_annotations()
    # In[7]:
    import pandas as pd

    df = pd.read_csv(f'{data_path}字段释义.csv', encoding='gbk')
    # 检查某一列是否有重复值
    column_name = '字段名'
    value_counts = df[column_name].value_counts()
    if any(value_counts > 1):
        print(f"列 '{column_name}' 中存在重复值。")
    else:
        print(f"列 '{column_name}' 中没有重复值。")
    df['字段含义_new'] = df['字段含义'] + df['单位'].apply(lambda x: ",单位:" + x if pd.notnull(x) else "")
    # 将两列转换为字典
    field_dict = df.set_index('字段名')['字段含义_new'].to_dict()


    # 读取CSV文件

    def aa(filename):
        df = pd.read_csv(f'database_in_use/{filename}.csv')
        if 'Unnamed: 0' in df.columns:
            del df['Unnamed: 0']
        # 获取列名
        column_names = df.columns.tolist()
        # 定义列名与中文注释的映射字典
        column_name_to_chinese = field_dict
        # 获取中文注释
        chinese_annotations = [column_name_to_chinese.get(col, '无注释') for col in column_names]

        last_dict = {'数据表名': filename, '字段名': column_names, "字段含义": chinese_annotations}
        return last_dict


    def process_folder(folder_path):
        # 获取文件夹中的所有CSV文件
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        # 初始化结果列表
        result_list = []

        # 遍历每个CSV文件
        for csv_file in csv_files:
            # 去掉文件扩展名，获取文件名
            filename = os.path.splitext(csv_file)[0]
            # 调用aa函数处理文件
            result_dict = aa(filename)
            # 将结果字典添加到列表中
            result_list.append(result_dict)

        return result_list


    # 使用示例
    folder_path = 'database_in_use'
    os.makedirs(folder_path, exist_ok=True)
    result = process_folder(folder_path)
    result = [item for item in result if item['数据表名'] != '设备参数详情表']
    for item in result:
        if item['数据表名'] == 'Ajia_plc_1':
            item['字段含义'][-1] = 'A架动作,包括关机、开机、A架摆出、缆绳挂妥、征服者出水、征服者落座、征服者起吊、征服者入水、缆绳解除、A架摆回'
        if item['数据表名'] == 'device_13_11_meter_1311':
            item['字段含义'][-1] = '折臂吊车及小艇动作,包括折臂吊车关机,折臂吊车开机,小艇检查完毕,小艇入水,小艇落座'
        if item['数据表名'] == 'Port3_ksbg_9':
            item['字段含义'][-1] = 'DP动作,包括OFF_DP,ON_DP'
    # %%
    df1 = pd.read_excel(f'{data_path}设备参数详情.xlsx', sheet_name='字段释义')
    df1['含义1'] = df1['含义'].fillna('') + ',' + df1['备注'].fillna('')
    dict_shebei = {'数据表名': '设备参数详情表', '字段名': list(df1['字段']), "字段含义": list(df1['含义1'])}
    # 修改字段含义列表的第二个值
    dict_shebei['字段含义'][1] = "参数中文名,值包含一号柴油发电机组滑油压力、停泊/应急发电机组、一号柴油发电机组滑油压力等"
    result.append(dict_shebei)

    # 假设这是你的列表数据
    data_list = result
    # 将列表存入 JSON 文件
    with open('dict.json', 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

    '''