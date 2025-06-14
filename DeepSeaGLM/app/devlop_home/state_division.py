import pandas as pd
import os
from collections import defaultdict
# pd.set_option('future.no_silent_downcasting', True)

def convert_to_numeric(value):
        try:
            return float(value)
        except ValueError:
            return -1

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

df_Ajia = pd.read_csv('database_in_use/Ajia_plc_1.csv')
# 将 Ajia-3_v 和 Ajia-5_v 列转换为数值类型，无法转换的设为 -1
df_Ajia['Ajia-3_v_new'] = df_Ajia['Ajia-3_v'].apply(convert_to_numeric)
df_Ajia['Ajia-5_v_new'] = df_Ajia['Ajia-5_v'].apply(convert_to_numeric)

# 应用滑动窗口修改Ajia中的电流噪声
# df_Ajia['Ajia-3_v_new'] = sliding_window_4(df_Ajia['Ajia-3_v_new'].tolist())
# df_Ajia['Ajia-3_v_new'] = sliding_window_3(df_Ajia['Ajia-3_v_new'].tolist())

# df_Ajia['Ajia-5_v_new'] = sliding_window_4(df_Ajia['Ajia-5_v_new'].tolist())
# df_Ajia['Ajia-5_v_new'] = sliding_window_3(df_Ajia['Ajia-5_v_new'].tolist())

df_Ajia_filtered = df_Ajia[df_Ajia['status'].isin(['A架开机', 'A架关机'])]
df_Ajia['stage'] = 'False'

ajia_segments = []
start_time = None
for _, row in df_Ajia_filtered.iterrows():
    if row['status'] == 'A架开机':
        start_time = row['csvTime']
    elif row['status'] == 'A架关机' and start_time is not None:
        ajia_segments.append((start_time, row['csvTime']))
        start_time = None
# ajia_segments_dt = [(pd.to_datetime(start_time), pd.to_datetime(end_time)) for start_time, end_time in ajia_segments]

df_DP = pd.read_csv('database_in_use/Port3_ksbg_9.csv')
# 筛选DP开关机时间段
df_DP_filtered = df_DP[df_DP['status'] != 'False']
df_DP['stage'] = 'False'
dp_segments = []
df_device = pd.read_csv('database_in_use/device_13_11_meter_1311.csv')

df_device_segments = []
start_time = None
# 存储折臂吊车开机关机时间段
for index, row in df_device.iterrows():
    if row['status'] == '折臂吊车开机':
        start_time = row['csvTime']
    elif row['status'] == '折臂吊车关机' and start_time is not None:
        end_time = row['csvTime']
        df_device_segments.append((start_time, end_time))
        start_time = None

df_device_segments = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in df_device_segments]

i = 0
while i < len(df_DP_filtered)-1:
    if df_DP_filtered.iloc[i]['status'] == 'ON DP' and df_DP_filtered.iloc[i+1]['status'] == 'OFF DP':
        dp_segments.append((df_DP_filtered.iloc[i]['csvTime'], df_DP_filtered.iloc[i+1]['csvTime']))
        i += 2
    else:
        i += 1
dp_segments_dt = [(pd.to_datetime(on_dp), pd.to_datetime(off_dp)) for on_dp, off_dp in dp_segments]

def merge_csv_files():
    # Read the CSV files
    df1 = pd.read_csv('database_in_use/Port3_ksbg_8.csv')
    df2 = pd.read_csv('database_in_use/Port4_ksbg_7.csv')

    # Convert csvTime to datetime
    df1['csvTime'] = pd.to_datetime(df1['csvTime'])
    df2['csvTime'] = pd.to_datetime(df2['csvTime'])

    # Process seconds only when consecutive rows are in the same minute
    # df1['csvTime'] = df1['csvTime'].copy()
    # for i in range(1, len(df1)):
    #     if df1['csvTime'].iloc[i].strftime('%Y-%m-%d %H:%M') == df1['csvTime'].iloc[i-1].strftime('%Y-%m-%d %H:%M'):
    #         # If consecutive rows are in same minute, round based on seconds
    #         if df1['csvTime'].iloc[i].second >= 30:
    #             df1.loc[df1.index[i], 'csvTime'] = df1['csvTime'].iloc[i].ceil('min')
    #         else:
    #             df1.loc[df1.index[i], 'csvTime'] = df1['csvTime'].iloc[i].floor('min')
    #         if df1['csvTime'].iloc[i-1].second >= 30:
    #             df1.loc[df1.index[i-1], 'csvTime'] = df1['csvTime'].iloc[i-1].ceil('min')
    #         else:
    #             df1.loc[df1.index[i-1], 'csvTime'] = df1['csvTime'].iloc[i-1].floor('min')
    df1['merge_time'] = df1['csvTime'].dt.strftime('%Y-%m-%d %H:%M') #:%S
    df2['merge_time'] = df2['csvTime'].dt.strftime('%Y-%m-%d %H:%M') #:%S

    # Merge dataframes
    merged_df = pd.merge(
        df1[['merge_time', 'P3_15', 'csvTime']],
        df2[['merge_time', 'P4_16']],
        on='merge_time',
        how='outer'
    )

    # Find unmatched rows
    unmatched = merged_df[merged_df['P3_15'].isna() | merged_df['P4_16'].isna()]
    if not unmatched.empty:
        print("Unmatched rows:")
        print(unmatched)

    # Fill missing values in 'P3_15' and 'P4_16' using the average of the previous and next valid values
    merged_df['P3_15'] = merged_df['P3_15'].interpolate(method='linear', limit_direction='both')
    merged_df['P4_16'] = merged_df['P4_16'].interpolate(method='linear', limit_direction='both')


    # Remove adjacent duplicate rows based on csvTime:
    # If two consecutive rows have the same csvTime value, drop the earlier (first) one.
    merged_df = merged_df.reset_index(drop=True)
    indices_to_drop = []
    for i in range(len(merged_df) - 1):
        if merged_df.loc[i, 'merge_time'] == merged_df.loc[i + 1, 'merge_time']:
            indices_to_drop.append(i)
    merged_df.drop(indices_to_drop, inplace=True)
    merged_df = merged_df.reset_index(drop=True)

    merged_df.drop('csvTime', axis=1, inplace=True)

    # 缺失merge_time填充
    merged_df['merge_time'] = pd.to_datetime(merged_df['merge_time'], format='%Y-%m-%d %H:%M')
    merged_df = merged_df.sort_values('merge_time').reset_index(drop=True)
    new_rows = []
    for i in range(len(merged_df) - 1):
        curr = merged_df.iloc[i]
        nxt = merged_df.iloc[i + 1]
        new_rows.append(curr.to_dict())  # 使用to_dict()保证列结构一致
        diff_minutes = int((nxt['merge_time'] - curr['merge_time']).total_seconds() // 60)
        # If gap exists (i.e. not consecutive minutes)
        # 在生成缺失时间的new_row时，确保所有列都存在
        if diff_minutes > 1:
            for offset in range(1, diff_minutes):
                missing_time = curr['merge_time'] + pd.Timedelta(minutes=offset)
                new_row = {}
                for col in merged_df.columns:
                    if col == 'merge_time':
                        new_row[col] = missing_time
                    else:
                        try:
                            # 强制转换为float进行数值计算
                            new_row[col] = (float(curr[col]) + float(nxt[col])) / 2
                        except (ValueError, TypeError):
                            # 非数值列直接取前值
                            new_row[col] = curr[col]
                new_rows.append(new_row)
    new_rows.append(merged_df.iloc[-1].to_dict())  # 最后一行也用to_dict()
    merged_df = pd.DataFrame(new_rows)  # 此时new_rows所有元素都是结构一致的字典
    # Convert merge_time back to the original string format
    merged_df['merge_time'] = merged_df['merge_time'].dt.strftime('%Y-%m-%d %H:%M')

    # Drop helper column and save result
    # merged_df = merged_df.dropna()
    
    merged_df.to_csv('database_in_use/merged_output.csv', index=False)

# Check if DP segment overlaps with Ajia segment
# def check(dp_segment,zuoye):
#     dp_start, dp_end = dp_segment
#     dp_start = str(dp_start)
#     dp_end = str(dp_end)
#     j = 0  # Index for ajia_segments
#     while j < len(ajia_segments) and ajia_segments[j][1] <= dp_start:
#         j += 1
        
#     # Check if current DP segment overlaps with current Ajia segment
#     if j >= len(ajia_segments) or ajia_segments[j][0] >= dp_end:
#         return False
#     df_Ajia.loc[df_Ajia['csvTime'] == ajia_segments[j][0], 'stage'] = zuoye + '开始'
#     df_Ajia.loc[df_Ajia['csvTime'] == ajia_segments[j][1], 'stage'] = zuoye + '结束'
#     # df_Ajia[df_Ajia['csvTime']==dp_start]['stage'] = 'ON dp'
#     # df_Ajia[df_Ajia['csvTime']==dp_end]['stage'] = 'OFF dp'
    
#     return True

def find_peaks(data1,zuoye,sum):
    # 数据预处理 - data1现在包含两列数据的元组
    # data = data1[:-1]
    data = data1

    
    def is_normal_range(val1, val2):
        return (0 < val1 < 60 and 0 < val2 < 60)
    
    def is_abnormal_range(val1, val2):
        return (val1 >= 60 and val2 >= 60)

    peaks = []

    # Find segments where values exceed normal range
    i = 0
    while i < len(data):
        # Look for start of abnormal segment
        # Need at least 2 normal values before
        if i < len(data) - 1 and not is_normal_range(data[i][0], data[i][1]):
            i += 1
            continue
            
        normal_before = 0
        # while i < len(data) and is_normal_range(data[i][0], data[i][1]):
        while i < len(data) and not is_abnormal_range(data[i][0], data[i][1]):
            normal_before += 1
            i += 1
            
        if normal_before < 1 or i >= len(data):
            i += 1
            continue
            
        # Found start of abnormal segment
        start_abnormal = i
        cnt = 0
        # Look for end of abnormal segment
        while i < len(data) and not is_normal_range(data[i][0], data[i][1]):
            if is_abnormal_range(data[i][0], data[i][1]):
                cnt += 1
            i += 1
            
        end_abnormal = i
        
        # Need at least 2 abnormal values
        if (end_abnormal - start_abnormal < 2 and zuoye == '布放') or ((end_abnormal - start_abnormal < 1)): # and (data[end_abnormal-1][0] < 70 and data[end_abnormal-1][1] < 70)
            continue
            
        # Look for normal values after
        normal_after = 0
        while i < len(data) and is_normal_range(data[i][0], data[i][1]):
            normal_after += 1
            i += 1
            
        if normal_after < 1 and ((zuoye == '布放' and (sum == 0 and len(peaks) == 0)) or (zuoye == '回收' and (sum > 0 or len(peaks) > 0))):  #and i < len(data)
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
    # peaks = [peak for peak in peaks if peak > 70]
    
    # 返回峰值个数和具体峰值
    return len(peaks), peaks

def find_peaks(data1,zuoye,sum):
    # 数据预处理 - data1现在包含两列数据的元组
    data = data1[:-1]

    def is_normal_range(val1, val2):
        return (0 < val1 < 60 and 0 < val2 < 60)

    peaks = []

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
        if (end_abnormal - start_abnormal < 1) and (data[end_abnormal-1][0] < 70 and data[end_abnormal-1][1] < 70):
            continue
            
        # Look for normal values after
        normal_after = 0
        while i < len(data) and is_normal_range(data[i][0], data[i][1]):
            normal_after += 1
            i += 1
            
        if normal_after < 1 and ((zuoye == '布放' and (sum == 0 and len(peaks) == 0)) or (zuoye == '回收' and (sum > 0 or len(peaks) > 0))):  #and i < len(data)
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
    # peaks = [peak for peak in peaks if peak > 70]
    
    # 返回峰值个数和具体峰值
    return len(peaks), peaks

action_ajia_segments = []
# 根据A架开关机时间段内的判别是否有动作（有电流，峰值个数>=2）
def check_Ajia(start_time,end_time,zuoye):
    if start_time == '2024-10-13 16:01:50': #2024-05-23 23:05:49   2024-05-31 21:12
        print('debug')
    segment = df_Ajia[(df_Ajia['csvTime'] >= start_time) & (df_Ajia['csvTime'] <= end_time) & (df_Ajia['check_current_presence'].isin(['有电流', '无电流']))]
    if segment.empty:
        print(f'check_Ajia：A架开关机时间{start_time},{end_time}里没有电流')
        return False
    

    # 判断A架开关机时间段与DP开启和折臂吊车开关机是否有重叠
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)
    
    dp_overlap = any(dp_end >= start_dt and dp_start <= end_dt for dp_start, dp_end in dp_segments_dt)
    device_overlap_segments = [
        (dev_start, dev_end)
        for dev_start, dev_end in df_device_segments
        if dev_end >= start_dt and dev_start <= end_dt
    ]
    device_overlap = bool(device_overlap_segments)  # 判断列表是否为空

    if not (dp_overlap and device_overlap):
        print(f'check_Ajia：A架开关机时间{start_time},{end_time}里没有与折臂吊车或DP开关机时间存在重叠')
        return False
    
    valid_segment_found = False
    # 遍历每个折臂吊车开关机时间段（df_device_segments中存储的是(crane_start, crane_end)元组）
    for crane_start, crane_end in device_overlap_segments:
        work_events = df_device[
            (df_device['csvTime'] >= crane_start.strftime("%Y-%m-%d %H:%M:%S")) &
            (df_device['csvTime'] <= crane_end.strftime("%Y-%m-%d %H:%M:%S")) &
            (df_device['action'].isin(['由待机进入工作', '由工作进入待机']))
        ]
        if work_events.shape[0] >= 6:
            valid_segment_found = True
            break

    if not valid_segment_found:
        print('在所有折臂吊车开关机时间段内，都未找到单个时间段内有6个以上事件')
        return False
    

    
    # print(segment)
    # print(f"事件数量: {segment.shape[0]}")
    L3 = []
    # 检查事件数量是否为偶数
    if segment.shape[0] >= 2 and segment.shape[0] % 2 == 0:
        # 遍历所有偶数索引的事件对
        for i in range(0, segment.shape[0], 2):
            event_start = segment.iloc[i]
            event_end = segment.iloc[i + 1]
            # 确保第一个事件是“有电流”，第二个事件是“无电流”
            if event_start['check_current_presence'] == '有电流' and event_end['check_current_presence'] == '无电流':
                start_event_time = event_start['csvTime']
                end_event_time = event_end['csvTime']

                # 提取两个事件之间的数据
                between_events = df_Ajia[(df_Ajia['csvTime'] >= start_event_time) & (df_Ajia['csvTime'] <= end_event_time)]
                data1 = between_events[['Ajia-3_v_new', 'Ajia-5_v_new']].values.tolist()

                # 调用 find_peaks 函数（假设已定义）
                len_peaks, peak_L = find_peaks(data1,zuoye,sum(L3))
                # print(f'峰值为{peak_L}')
                L3.append(len_peaks)
    if sum(L3) >= 2:
        return True
    else:
        return False

zuoye = None
# 根据已知的DP时间段，对A架开关机时间段判别是布放还是回收
def stage_division(dp_after,dp_before):
    if dp_after == dp_before:
        print('dp_after和dp_before相同')
    global action_ajia_segments
    global zuoye  # 声明使用全局变量
    dp_after = tuple(map(str, dp_after)) if dp_after is not None else None
    dp_before = tuple(map(str, dp_before)) if dp_before is not None else None
    # print(f"dp_after: {dp_after}, dp_before: {dp_before}")
    if dp_after == None:
        zuoye = '回收' if zuoye == None else zuoye
        dp_start, dp_end = dp_before
        j = 0
        # Skip Ajia segments that end before current DP segment
        while j < len(ajia_segments) and ajia_segments[j][1] <= dp_start:
            j += 1
            
        # Check if current DP segment overlaps with current Ajia segment
        if not(j >= len(ajia_segments) or ajia_segments[j][0] >= dp_end):
            segment = df_Ajia[(df_Ajia['csvTime'] >= ajia_segments[j][0]) & (df_Ajia['csvTime'] <= ajia_segments[j][1])]
            if segment['check_current_presence'].eq('False').all():
                print(f'传入的dp_before{dp_before}对应的A架开关机时间里没有电流',ajia_segments[j][0],ajia_segments[j][1])
                print('查找dp_before在dp_segments里的前一个dp时间段')
                # 查找dp_before在dp_segments里的前一个dp时间段
                previous_dp_segment = None
                for idx in range(1, len(dp_segments_dt)):
                    # 注意：这里比较时需要将dp_before转换为datetime格式
                    if dp_segments_dt[idx] == (pd.to_datetime(dp_before[0]), pd.to_datetime(dp_before[1])):
                        previous_dp_segment = dp_segments_dt[idx - 1]
                        break
                if previous_dp_segment:
                    print(f"dp_before的前一个DP时间段为：{previous_dp_segment}")
                    stage_division(None,previous_dp_segment)
                else:
                    print("未找到dp_before的前一个DP时间段,stage_division停止")

            else:
                while(j>=0):
                    if check_Ajia(ajia_segments[j][0],ajia_segments[j][1],'布放') or check_Ajia(ajia_segments[j][0],ajia_segments[j][1],'回收'):
                        # df_Ajia.loc[df_Ajia['csvTime'] == ajia_segments[j][0], 'stage'] = zuoye + '开始'
                        # df_Ajia.loc[df_Ajia['csvTime'] == ajia_segments[j][1], 'stage'] = zuoye + '结束'
                        # if zuoye == '回收': zuoye = '布放'
                        # else: zuoye = '回收'
                        action_ajia_segments.insert(0, ajia_segments[j])
                    j -= 1
                zuoye = '回收'
        else:
            print(f'传入的dp_before{dp_before}找不到对应的A架开关机时间段')
            # 查找dp_before在dp_segments里的前一个dp时间段
            previous_dp_segment = None
            for idx in range(1, len(dp_segments_dt)):
                # 注意：这里比较时需要将dp_before转换为datetime格式
                if dp_segments_dt[idx] == (pd.to_datetime(dp_before[0]), pd.to_datetime(dp_before[1])):
                    previous_dp_segment = dp_segments_dt[idx - 1]
                    break
            if previous_dp_segment:
                print(f"dp_before的前一个DP时间段为：{previous_dp_segment}")
                stage_division(None,previous_dp_segment)
            else:
                print("未找到dp_before的前一个DP时间段,stage_division停止")

    elif dp_before == None:
        
        dp_start, dp_end = dp_after
        j = 0
        # Skip Ajia segments that end before current DP segment
        while j < len(ajia_segments) and ajia_segments[j][1] <= dp_start:
            j += 1
            
        # Check if current DP segment overlaps with current Ajia segment
        if not(j >= len(ajia_segments) or ajia_segments[j][0] >= dp_end):
            segment = df_Ajia[(df_Ajia['csvTime'] >= ajia_segments[j][0]) & (df_Ajia['csvTime'] <= ajia_segments[j][1])]
            if segment['check_current_presence'].eq('False').all():
                print(f'传入的dp_after{dp_after}对应的A架开关机时间里没有电流',ajia_segments[j][0],ajia_segments[j][1])
                print('查找dp_after在dp_segments里的下一个dp时间段')
                # 查找dp_after在dp_segments里的下一个dp时间段
                next_dp_segment = None
                for idx in range(0, len(dp_segments_dt)-1):
                    # 注意：这里比较时需要将dp_after转换为datetime格式
                    if dp_segments_dt[idx] == (pd.to_datetime(dp_after[0]), pd.to_datetime(dp_after[1])):
                        next_dp_segment = dp_segments_dt[idx + 1]
                        break
                if next_dp_segment:
                    print(f"dp_after的下一个DP时间段为：{next_dp_segment}")
                    stage_division(next_dp_segment,None)
                else:
                    print("未找到dp_after的下一个DP时间段,stage_division停止")
            else:
                # zuoye = '布放' if zuoye == '回收' else '回收'
                while(j < len(ajia_segments)):
                    if check_Ajia(ajia_segments[j][0],ajia_segments[j][1],'布放') or check_Ajia(ajia_segments[j][0],ajia_segments[j][1],'回收'):
                        # if zuoye == '回收': zuoye = '布放'
                        # else: zuoye = '回收'
                        # df_Ajia.loc[df_Ajia['csvTime'] == ajia_segments[j][0], 'stage'] = zuoye + '开始'
                        # df_Ajia.loc[df_Ajia['csvTime'] == ajia_segments[j][1], 'stage'] = zuoye + '结束'
                        action_ajia_segments.append(ajia_segments[j])
                    j += 1
        else:
            print(f'传入的dp_after{dp_after}找不到对应的A架开关机时间段')
            print('查找dp_after在dp_segments里的下一个dp时间段')
            # 查找dp_after在dp_segments里的下一个dp时间段
            next_dp_segment = None
            for idx in range(0, len(dp_segments_dt)-1):
                # 注意：这里比较时需要将dp_after转换为datetime格式
                if dp_segments_dt[idx] == (pd.to_datetime(dp_after[0]), pd.to_datetime(dp_after[1])):
                    next_dp_segment = dp_segments_dt[idx + 1]
                    break
            if next_dp_segment:
                print(f"dp_after的下一个DP时间段为：{next_dp_segment}")
                stage_division(next_dp_segment,None)
            else:
                print("未找到dp_after的下一个DP时间段,stage_division停止")

    else:
        if dp_after != dp_before and not(dp_after[1] < dp_before[0]):
            print('dp_after的结束时间早于dp_before的开始时间,stage_division停止')
            return
        dp_after_start, dp_after_end = dp_after
        j = 0
        # Skip Ajia segments that end before current DP segment
        while j < len(ajia_segments) and ajia_segments[j][1] <= dp_after_start: #.strftime('%Y-%m-%d %H:%M')
            j += 1
        if j >= len(ajia_segments) or ajia_segments[j][0] >= dp_after_end:
            print(f'传入的dp_after{dp_after}找不到对应的A架开关机时间段')
            # 查找dp_after在dp_segments里的下一个dp时间段
            next_dp_segment = None
            for idx in range(0, len(dp_segments_dt)-1):
                # 注意：这里比较时需要将dp_after转换为datetime格式
                if dp_segments_dt[idx] == (pd.to_datetime(dp_after[0]), pd.to_datetime(dp_after[1])):
                    next_dp_segment = dp_segments_dt[idx + 1]
                    break
            if next_dp_segment:
                print(f"dp_after的下一个DP时间段为：{next_dp_segment}")
                stage_division(next_dp_segment,dp_before)
            else:
                print("未找到dp_after的下一个DP时间段,stage_division停止")
        else:
            segment = df_Ajia[(df_Ajia['csvTime'] >= ajia_segments[j][0]) & (df_Ajia['csvTime'] <= ajia_segments[j][1])]
            if segment['check_current_presence'].eq('False').all():
                print(f'传入的dp_after{dp_after}对应的A架开关机时间里没有电流',ajia_segments[j][0],ajia_segments[j][1])
                print('查找dp_after在dp_segments里的后一个dp时间段')
                # 查找dp_after在dp_segments里的后一个dp时间段
                next_dp_segment = None
                for idx in range(0, len(dp_segments_dt)-1):
                    # 注意：这里比较时需要将dp_after转换为datetime格式
                    if dp_segments_dt[idx] == (pd.to_datetime(dp_after[0]), pd.to_datetime(dp_after[1])):
                        next_dp_segment = dp_segments_dt[idx + 1]
                        break
                if next_dp_segment:
                    print(f"dp_after的下一个DP时间段为：{next_dp_segment}")
                    stage_division(next_dp_segment,dp_before)
                else:
                    print("未找到dp_after的下一个DP时间段,stage_division停止")
            else:
                dp_before_start, dp_before_end = dp_before
                k = j
                if dp_after == dp_before:
                    k = j+1
                # Skip Ajia segments that end before current DP segment
                while k < len(ajia_segments) and ajia_segments[k][1] <= dp_before_start:
                    k += 1
                if k >= len(ajia_segments) or ajia_segments[k][0] >= dp_before_end:
                    print(f'传入的dp_before{dp_before}找不到对应的A架开关机时间段')
                    # 查找dp_before在dp_segments里的前一个dp时间段
                    previous_dp_segment = None
                    for idx in range(1, len(dp_segments_dt)):
                        # 注意：这里比较时需要将dp_before转换为datetime格式
                        if dp_segments_dt[idx] == (pd.to_datetime(dp_before[0]), pd.to_datetime(dp_before[1])):
                            previous_dp_segment = dp_segments_dt[idx - 1]
                            break
                    if previous_dp_segment:
                        print(f"dp_before的前一个DP时间段为：{previous_dp_segment}")
                        stage_division(dp_after,previous_dp_segment)
                    else:
                        print("未找到dp_before的前一个DP时间段,stage_division停止")
                else:
                    segment = df_Ajia[(df_Ajia['csvTime'] >= ajia_segments[k][0]) & (df_Ajia['csvTime'] <= ajia_segments[k][1])]
                    if segment['check_current_presence'].eq('False').all():
                        print(f'传入的dp_before{dp_before}对应的A架开关机时间里没有电流',ajia_segments[k][0],ajia_segments[k][1])
                        print('查找dp_before在dp_segments里的前一个dp时间段')
                        # 查找dp_before在dp_segments里的前一个dp时间段
                        previous_dp_segment = None
                        for idx in range(1, len(dp_segments_dt)):
                            # 注意：这里比较时需要将dp_before转换为datetime格式
                            if dp_segments_dt[idx] == (pd.to_datetime(dp_before[0]), pd.to_datetime(dp_before[1])):
                                previous_dp_segment = dp_segments_dt[idx - 1]
                                break
                        if previous_dp_segment:
                            print(f"dp_before的前一个DP时间段为：{previous_dp_segment}")
                            stage_division(dp_after,previous_dp_segment)
                        else:
                            print("未找到dp_before的前一个DP时间段,stage_division停止")
                    else:
                        
                        for i in range(j,k+1):
                            if check_Ajia(ajia_segments[i][0],ajia_segments[i][1],'布放') or check_Ajia(ajia_segments[i][0],ajia_segments[i][1],'回收'):
                            #     if(df_Ajia.loc[df_Ajia['csvTime'] == ajia_segments[i][0], 'stage'].values[0] == 'False'):
                            #         if zuoye == '回收': zuoye = '布放'
                            #         else: zuoye = '回收'
                            #         df_Ajia.loc[df_Ajia['csvTime'] == ajia_segments[i][0], 'stage'] = zuoye + '开始'
                            #         df_Ajia.loc[df_Ajia['csvTime'] == ajia_segments[i][1], 'stage'] = zuoye + '结束'
                                action_ajia_segments.append(ajia_segments[i])
                        # if len(action_ajia_segments) % 2 != 0:
                        #     print(f'传入的dp_after{dp_after}和dp_before{dp_before}对应的A架开关机时间段内的动作不成对')

# 对处于停泊状态的DP段进行修正
def dp_revise(action_ajia_segments):
    # for start_time, end_time in action_ajia_segments:
    #     df_DP.loc[(df_DP['csvTime'] > str(start_time)) & (df_DP['csvTime'] < str(end_time)), 'stage'] = 'False'
    # df_DP['stage'] = df_DP['stage'].replace({'ON DP': '动力定位开启', 'OFF DP': '动力定位结束'})
    overlapping_dp = []
    j = 0  # Index for ajia_segments
    for dp_start, dp_end in dp_segments:
        # Skip Ajia segments that end before current DP segment
        while j < len(action_ajia_segments) and action_ajia_segments[j][1] <= dp_start:
            j += 1
            
        # Check if current DP segment overlaps with current Ajia segment
        if not(j >= len(action_ajia_segments) or action_ajia_segments[j][0] >= dp_end):
            df_DP.loc[(df_DP['csvTime'] == dp_start), 'stage'] = '动力定位开始'
            df_DP.loc[(df_DP['csvTime'] == dp_end), 'stage'] = '动力定位结束'
            # mask_start = merged_df['merge_time'].str[:16] == dp_start[:16]
            # merged_df.loc[mask_start, 'status'] = merged_df.loc[mask_start, 'status'].apply(
            #     lambda s: '动力定位开始' if s == 'False' else s + '动力定位开始'
            # )
            # mask_end = merged_df['merge_time'].str[:16] == dp_end[:16]
            # merged_df.loc[mask_end, 'status'] = merged_df.loc[mask_end, 'status'].apply(
            #     lambda s: '动力定位结束' if s == 'False' else s + '动力定位结束'
            # )


    df_DP.to_csv('database_in_use/Port3_ksbg_9.csv', index=False, encoding='utf-8-sig')
    return 

# 当前传入的A架开关机segment里，有多个通电流时间段，可能存在布放和回收。需要更细粒度的判定
def process(seg,label):
    df_Ajia.loc[df_Ajia['csvTime'] == seg[0], 'stage'] = '特殊处理开始'
    df_Ajia.loc[df_Ajia['csvTime'] == seg[1], 'stage'] = '特殊处理结束'


def action_revise(action_ajia_segments):
    return

if __name__ == '__main__':
    # global action_ajia_segments
    if not os.path.exists('database_in_use/merged_output.csv'):
        merge_csv_files()
    merged_df = pd.read_csv('database_in_use/merged_output.csv')

    merged_df['status'] = 'False'
    merged_df['stage'] = 'False'
    high_speed_segments = []


    start_time = None
    threshold = 750

    # 遍历每一行，根据一二号主推的功率来判断航渡状态，并存储对应关机时间段
    for i in range(1, merged_df.shape[0]):
        # if (merged_df['P3_15'].iloc[i-1] <= 0 and merged_df['P4_16'].iloc[i-1] <= 0) and \
        #     (merged_df['P3_15'].iloc[i] > 0 or merged_df['P4_16'].iloc[i] > 0):
        #     merged_df.loc[i, 'status'] = '主推进器运行'
        # elif (merged_df['P3_15'].iloc[i-1] > 0 or merged_df['P4_16'].iloc[i-1] > 0) and \
        #     (merged_df['P3_15'].iloc[i] <= 0 and merged_df['P4_16'].iloc[i] <= 0):
        #     merged_df.loc[i, 'status'] = '主推进器关闭'
        if ((merged_df['P3_15'].iloc[i-1] < threshold and merged_df['P4_16'].iloc[i-1] < threshold) and \
            (merged_df['P3_15'].iloc[i] >= threshold and merged_df['P4_16'].iloc[i] >= threshold)) or \
            (start_time is None) and (merged_df['P3_15'].iloc[i] >= threshold and merged_df['P4_16'].iloc[i] >= threshold):
            # merged_df.loc[i, 'status'] = '高速运行开始'
            start_time = merged_df['merge_time'].iloc[i]
        elif ((merged_df['P3_15'].iloc[i-1] >= threshold and merged_df['P4_16'].iloc[i-1] >= threshold) and \
            (merged_df['P3_15'].iloc[i] < 400 and merged_df['P4_16'].iloc[i] < 400)) or \
            (start_time is not None) and (merged_df['P3_15'].iloc[i] < threshold and merged_df['P4_16'].iloc[i] < threshold):
            # merged_df.loc[i, 'status'] = '高速运行结束'
            if start_time:
                high_speed_segments.append((start_time,merged_df['merge_time'].iloc[i]))
                start_time = None

    # 筛除时长小于1小时的时间段
    high_speed_segments = [
        (start, end) for start, end in high_speed_segments
        if (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() >= 3600
    ]

    # 合并前后间隔小于0.5小时的时间段
    if high_speed_segments:
        i = 0
        while i < len(high_speed_segments) - 1:
            current_end = pd.to_datetime(high_speed_segments[i][1])
            next_start = pd.to_datetime(high_speed_segments[i + 1][0])
            
            # If time difference is less than 6 hours
            if (next_start - current_end).total_seconds() / 3600 <= 0.5:
                # Merge segments
                high_speed_segments[i] = (high_speed_segments[i][0], high_speed_segments[i + 1][1])
                # Remove the next segment
                high_speed_segments.pop(i + 1)
            else:
                i += 1

    
    anchor = False
    anchor_segment = []
    # for start_time, end_time in high_speed_segment:
    for index, (start_time, end_time) in enumerate(high_speed_segments):
        if start_time == '2024-10-22 17:28': #2024-05-23 23:05:49   2024-05-31 21:12
            print('debug')
        # if end_time == '2024-10-24 07:49':
        #     print('debug')    
        merged_df.loc[merged_df['merge_time'] == start_time, 'stage'] = '高速运行开始'
        merged_df.loc[merged_df['merge_time'] == end_time, 'stage'] = '高速运行结束'
        # merged_df.loc[merged_df['merge_time'] == end_time, 'status'] = '航渡结束'

        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)

        '''
        遍历每个high_speed_segment，然后只看前面
        1、找前一个OFF DP 标记为航渡开始，计算该DP和A架开关机时间有重叠（如果没有重叠则OFF DP为航渡开始，如果有重叠，以OFF DP还是A架关机为航渡开始）
        2、如果DP和A架开关机时间没有重叠，则遍历上一个和这个之间的空隙（上一个的end_time，这个的start_time），如果有上一个high_speed_segment
        2.1、计算其中P3_15和P4_16都小于等于0的比例，如果大于0.7，则寿命该段为停泊状态（停泊状态期间的ON DP和OFF DP不能算作动力定位）。并进行停泊状态开始结束的判别
        3、如果有重叠（不是停泊状态），则进行作业状态的判别（计算其中A架有效开关机次数是否为偶数，因为下放和回收应该成对）
        4、对于最后一个high_speed_segment，还要看后面（是停泊状态还是作业状态）
        '''


        # 查找start_time之前最近的DP时间段
        # 条件：该段的OFF DP时间早于start_dt，并且下一段的ON DP时间晚于start_dt
        dp_before = None
        for i in range(len(dp_segments_dt) - 1):
            if dp_segments_dt[i][0] <= start_dt and dp_segments_dt[i+1][0] > start_dt:
                dp_before = dp_segments_dt[i]
                break
        
        if dp_before:
            # print(type(merged_df['merge_time'][0]))
            if index == 0:
                merged_df.loc[merged_df['merge_time'] == dp_before[1].strftime('%Y-%m-%d %H:%M'), 'status'] = '航渡开始'
                merged_df.loc[merged_df['merge_time'] == end_time, 'status'] = '航渡结束'

            # if check(dp_before,'回收'):
            #     stage_division('回收')
            # else:
                # print(dp_before[0].strftime('%Y-%m-%d %H:%M'))
            if index:
                pre_end_time = high_speed_segments[index-1][1]
                pre_end_time_dt = pd.to_datetime(pre_end_time)
                if dp_before[0] <= pre_end_time_dt:
                    print('dp_before和上一个high_speed_segment有重叠') 
                    #说明当前high_speed_segment和上一个high_speed_segment之间没有DP开启，即这两者之间没有停泊和作业
                    # 应该合并这两段high_speed_segment为一段航渡状态
                    merged_df.loc[merged_df['merge_time'] == end_time, 'status'] = '航渡结束'
                    merged_df.loc[merged_df['merge_time'] == pre_end_time, 'status'] = 'False'
                    continue
                merged_df.loc[merged_df['merge_time'] == dp_before[1].strftime('%Y-%m-%d %H:%M'), 'status'] = '航渡开始'
                merged_df.loc[merged_df['merge_time'] == end_time, 'status'] = '航渡结束'
                # 在pre_end_time和start_time之间筛选数据
                interval_mask = (
                    (pd.to_datetime(merged_df['merge_time']) >= pd.to_datetime(pre_end_time)) &
                    (pd.to_datetime(merged_df['merge_time']) <= start_dt)
                )
                interval_data = merged_df.loc[interval_mask]
                total_rows = len(interval_data)
                if total_rows > 0:
                    # 计算P3_15和P4_16同时小于等于0的行数
                    negative_rows = ((interval_data['P3_15'] <= 0) & (interval_data['P4_16'] <= 0)).sum()
                    ratio = negative_rows / total_rows
                else:
                    ratio = None  # 没有数据时返回None
                print(f'Between {pre_end_time} and {start_time}, the ratio is: {ratio}')
                dp_after = None
                for i in range(1, len(dp_segments_dt)):
                    if dp_segments_dt[i][1] >= pre_end_time_dt and dp_segments_dt[i-1][1] < pre_end_time_dt:
                        dp_after = dp_segments_dt[i]
                        break
                if dp_after:
                    # if dp_after[0] >= start_dt:
                    #     print('dp_after和high_speed_segment有重叠')
                    #     continue
                    if dp_before[0] <= dp_after[1]:
                        print('dp_before和dp_after有重叠') 
                        # 当前high_speed_segment和上一个high_speed_segment之间仅有一个DP开启，即这两者之间没有停泊和作业
                        # 应该合并这两段high_speed_segment为一段航渡状态
                        merged_df.loc[merged_df['merge_time'] == end_time, 'status'] = '航渡结束'
                        merged_df.loc[merged_df['merge_time'] == pre_end_time, 'status'] = 'False'
                        merged_df.loc[merged_df['merge_time'] == dp_before[1].strftime('%Y-%m-%d %H:%M'), 'status'] = 'False'
                        continue
                else: print('dp_after不存在')
                if ratio >= 0.7:
                    merged_df.loc[merged_df['merge_time'] == dp_before[0].strftime('%Y-%m-%d %H:%M'), 'status'] = '停泊结束'
                    if dp_after:
                        merged_df.loc[merged_df['merge_time'] == dp_after[1].strftime('%Y-%m-%d %H:%M'), 'status'] = '停泊开始'
                        anchor_segment.append((dp_after[1],dp_before[0]))
                else: stage_division(dp_after,dp_before)
            else: 
                interval_mask = (
                    (pd.to_datetime(merged_df['merge_time']) <= start_dt)
                )
                interval_data = merged_df.loc[interval_mask]
                total_rows = len(interval_data)
                if total_rows > 0:
                    # 计算P3_15和P4_16同时小于等于0的行数
                    negative_rows = ((interval_data['P3_15'] <= 0) & (interval_data['P4_16'] <= 0)).sum()
                    ratio = negative_rows / total_rows
                else:
                    ratio = None  # 没有数据时返回None
                print(f'Before {start_time}, the ratio is: {ratio}')
                if ratio >= 0.7:
                    merged_df.loc[merged_df['merge_time'] == dp_before[0].strftime('%Y-%m-%d %H:%M'), 'status'] = '停泊结束'
                else: stage_division(None,dp_before)
        else:
            print('No DP segment found before', start_time, end_time)
        
        if index == len(high_speed_segments) - 1:
            # 如果是最后一个segment，则看后面
            dp_after = None
            for i in range(1, len(dp_segments_dt)):
                if dp_segments_dt[i][0] > end_dt and dp_segments_dt[i-1][0] < end_dt:
                    dp_after = dp_segments_dt[i]
                    break
            if dp_after:
                # if check(dp_after,'布放'):
                #     stage_division('布放')
                # else:
                    # 在pre_end_time和start_time之间筛选数据
                interval_mask = (
                    (pd.to_datetime(merged_df['merge_time']) >= end_dt)
                )
                interval_data = merged_df.loc[interval_mask]
                total_rows = len(interval_data)
                if total_rows > 0:
                    # 计算P3_15和P4_16同时小于等于0的行数
                    negative_rows = ((interval_data['P3_15'] <= 0) & (interval_data['P4_16'] <= 0)).sum()
                    ratio = negative_rows / total_rows
                else:
                    ratio = None  # 没有数据时返回None
                print(f'After {end_time}, the ratio is: {ratio}')
                if ratio >= 0.7:
                    merged_df.loc[merged_df['merge_time'] == dp_after[1].strftime('%Y-%m-%d %H:%M'), 'status'] = '停泊开始'
                else: stage_division(dp_after, None)

    # 按照开机时间当天归类 action_ajia_segments，然后交替打标签
    segments_by_day = defaultdict(list)
    # print('action_ajia_segments:')
    # for seg in action_ajia_segments:
    #     print(seg)
    unique_segments = []
    for seg in action_ajia_segments:
        if seg not in unique_segments:
            unique_segments.append(seg)
    action_ajia_segments = unique_segments
    for seg in action_ajia_segments:
        # 假设 seg[0] 是开机时间的字符串，将其转换为日期对象
        day = pd.to_datetime(seg[0]).date()
        segments_by_day[day].append(seg)

    for day, segments in segments_by_day.items():
        # 按开机时间升序排序
        segments.sort(key=lambda x: pd.to_datetime(x[0]))
        # 如果当天只有一个有效时间段（确定有动作），是否需要根据开机时间来判定是布放还是回收？而不是布放回收交替？
        # print(f"Day: {day}, Segments长度: {len(segments)}")
        # if len(segments) > 1 and segments[0][0].split(' ')[1] >= '12:00:00':
        #     print(f"有多个Segments时，第一个segment的开机时间在下午: {segments[0][0]} - {segments[0][1]}")
        if len(segments) == 1:
            label = '布放' if segments[0][0].split(' ')[1] < '12:00:00' else '回收'
        else: label = '布放'
        for i, seg in enumerate(segments):
            duration = pd.to_datetime(seg[1]) - pd.to_datetime(seg[0])
            if duration.total_seconds() > 6 * 3600: 
                print('segment时长大于6小时',segments)
            #     process(seg, label)
            #     # Count device on/off events between seg[0] and seg[1]
            #     device_events = df_device[
            #         (df_device['csvTime'] >= seg[0]) &
            #         (df_device['csvTime'] <= seg[1]) &
            #         (df_device['status'].isin(['折臂吊车开机', '折臂吊车关机']))
            #     ]
            #     num_events = device_events.shape[0]
            #     print(f"Segment {seg[0]} to {seg[1]} duration {duration}: {num_events} device on/off events.")
            #     # if len(segments) > 1 :
            #         # print('该天有多个segments。其中有一个segment时长大于6小时，',segments)
            #     # print("Segment:", seg, "duration:", duration)
            # else:
            df_Ajia.loc[df_Ajia['csvTime'] == seg[0], 'stage'] = label + '开始'
            df_Ajia.loc[df_Ajia['csvTime'] == seg[1], 'stage'] = label + '结束'
            label = '布放' if label == '回收' else '回收'
    
    # 标记伴航状态
    for i in range(0,len(action_ajia_segments)-1):
        seg = action_ajia_segments[i]
        seg_next = action_ajia_segments[i+1]
        if df_Ajia.loc[df_Ajia['csvTime'] == seg[0], 'stage'].values[0] == '布放开始' and df_Ajia.loc[df_Ajia['csvTime'] == seg_next[0], 'stage'].values[0] == '回收开始':
            # 查找和seg时间段重叠的dp_segment，并取最后一个dp_segment，
            # 将该dp_segment的OFF DP对应的时间点标记为'伴航开始'
            # overlapping_dp_seg = [
            #     dp for dp in dp_segments 
            #     if pd.to_datetime(dp[0]) <= pd.to_datetime(seg[1]) and pd.to_datetime(dp[1]) >= pd.to_datetime(seg[0])
            # ]
            # if overlapping_dp_seg:
            #     last_dp_seg = overlapping_dp_seg[-1]
            #     # df_Ajia.loc[df_Ajia['csvTime'].str[:16] == last_dp_seg[1][:16], 'stage'] = '伴航开始'
            #     merged_df.loc[merged_df['merge_time'].str[:16] == last_dp_seg[1][:16], 'status'] = '伴航开始'

            merged_df.loc[merged_df['merge_time'].str[:16] == seg[1][:16], 'status'] = '伴航开始'
            merged_df.loc[merged_df['merge_time'].str[:16] == seg_next[0][:16], 'status'] = '伴航结束'

            # 查找和seg_next时间段重叠的dp_segment，并取最后一个dp_segment，
            # 将该dp_segment的ON DP对应的时间点标记为'伴航结束'
            # overlapping_dp_seg_next = [
            #     dp for dp in dp_segments 
            #     if pd.to_datetime(dp[0]) <= pd.to_datetime(seg_next[1]) and pd.to_datetime(dp[1]) >= pd.to_datetime(seg_next[0])
            # ]
            # if overlapping_dp_seg_next:
            #     last_dp_seg_next = overlapping_dp_seg_next[0]
            #     # df_Ajia.loc[df_Ajia['csvTime'].str[:16] == last_dp_seg_next[0][:16], 'stage'] = '伴航结束'
            #     merged_df.loc[merged_df['merge_time'].str[:16] == last_dp_seg_next[0][:16], 'status'] = '伴航结束'
    
    start_indices = merged_df.index[merged_df['status'] == '伴航开始'].tolist()
    end_indices = merged_df.index[merged_df['status'] == '伴航结束'].tolist()
    
    # 将伴航开始和伴航结束之间的status全部设为False   （伴航状态之内不能有其他状态）
    for start in start_indices:
        # 找到第一个行号大于start的“伴航结束”
        end = next((idx for idx in end_indices if idx > start), None)
        if end is not None:
            mask = (merged_df.index > start) & (merged_df.index < end)
            if not (merged_df.loc[mask, 'status'] == 'False').all():
                merged_df.loc[mask, 'status'] = 'False'

    dp_revise(action_ajia_segments)
    
    action_revise(action_ajia_segments)

    # Save the results
    # Rename merge_time column to csvTime in merged_df
    merged_df.rename(columns={'merge_time': 'csvTime'}, inplace=True)
    merged_df.to_csv('database_in_use/status_output.csv', index=False, encoding='utf-8-sig')
    df_Ajia.drop(['Ajia-3_v_new', 'Ajia-5_v_new'], axis=1, inplace=True)
    df_Ajia.to_csv('database_in_use/Ajia_plc_1.csv', index=False, encoding='utf-8-sig')