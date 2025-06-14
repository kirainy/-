import json

# 题号与答案的映射表（注意题号需要整数类型）
answer_mapping = {
    # 85: "135L/h",
    # 86: "0（单位）",
    # 88: "6972V",
    # 22: "上午A架的运行时长8:03-10:30（147分钟），下午A架开机时长17:58-19:18（80分钟），上午A架开机时间长，长67分钟。",
    # 23: "A架摆回8:58，A架摆出16:25，8/24A架摆回和摆出的时间相隔447分钟",
    # 3: "A架在进行缆绳解除",
    # 5: "小艇检查完毕",
    # 6: "征服者落座、折臂吊车关机",
    # 74: "征服者入水10:16，征服者出水19:02，入水时间距离出水时间8小时46分钟",
    # 75: "征服者入水7:18，A架摆回7:23，用了5分钟",
    # 79: "征服者落座17:23，A架关机17:41，征服者落座18分钟后A架关机的",
    # 80: "7:58",
    # 61: "7:18",
    # 62: "",
    # 63: "A架摆回"
    94:"15:23,20:57",
}

def process_jsonl(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            # 解析JSON数据
            data = json.loads(line.strip())
            
            try:
                # 从id中提取题号（格式：gysxdmx_00085 -> 85）
                qid = int(data['id'].split('_')[-1])
                
                # 替换答案逻辑
                if qid in answer_mapping:
                    data['answer'] = answer_mapping[qid]
                else:
                    data['answer'] = "error occur"
                    
            except (KeyError, ValueError, IndexError):
                # 处理异常情况
                data['answer'] = "error occur"
            
            # 写入处理后的数据
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

# 使用示例
process_jsonl('res_14.jsonl', 'output_res_1.jsonl')