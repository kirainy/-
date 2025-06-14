import json
import concurrent.futures as cf
# import ai_brain
import ai_agents
import time
import sys


def process_one(question_json):
    line = question_json
    query = line["question"]
    try:
        # query = enhanced(query)
        print(f"Processing question ID {line['id']}: {query}")
        aa = ai_agents.get_answer(question=query)
        question = ai_agents.enhanced(query)
        answer = ai_agents.get_end_answer(question,aa)
        ans = str(answer)
        print(f"Answer for question ID {line['id']}: {ans}")
        return {"id": line["id"], "question": query, "answer": ans}
    except Exception as e:
        print(f"Error processing question ID {line['id']}: {e}")
        return {"id": line["id"], "question": query, "answer": "Error: " + str(e)}


def main():
    # 从命令行参数获取输入
    if len(sys.argv) < 3:
        print("需要两个参数: input_param 和 result_path")
        return
    
    input_param_str = sys.argv[1]
    result_path = sys.argv[2]

    # 调试：打印输入参数
    print(f"输入参数内容: {input_param_str}")
    print(f"输入参数长度: {len(input_param_str)}")
    
    # 解析输入参数
    # input_param = json.loads(input_param_str)
    # 从文件中读取 JSON 内容
    try:
        with open(input_param_str, "r", encoding="utf-8-sig") as f:
            input_param = json.load(f)
    except Exception as e:
        print(f"读取输入参数文件失败: {e}")
        return
    question_file_path = input_param["fileData"]["questionFilePath"]
    
    # 读取问题文件
    with open(question_file_path, "r", encoding="utf-8-sig") as f:
        q_json_list = [json.loads(line.strip()) for line in f]
    
    # 多线程处理（保持原有逻辑）
    result_json_list = []
    with cf.ThreadPoolExecutor(max_workers=1) as executor:
        future_list = [executor.submit(process_one, q_json) for q_json in q_json_list]
        for future in cf.as_completed(future_list):
            result_json_list.append(future.result())
    
    # 按 ID 排序并写入结果
    result_json_list.sort(key=lambda x: x["id"])
    with open(result_path, "w", encoding="utf-8") as f:
        for result in result_json_list:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"程序运行时间: {(end_time - start_time)/60:.2f} 分钟")


#--------------------------------------------------------------------------------------------------------------------
# import json
# import concurrent.futures as cf
# import ai_agents as ai_brain
# import time
# import sys


# def process_one(question_json):
#     line = question_json
#     query = line["question"]
#     try:
#         print(f"Processing question ID {line['id']}: {query}")
        
#         aa = ai_brain.get_answer(question=query)
#         query = ai_brain.enhanced(query)
#         answer = ai_brain.get_end_answer(query, aa)
#         ans = str(answer)
#         print(f"Answer for question ID {line['id']}: {ans}")
#         return {"id": line["id"], "question": query, "answer": ans}
#     except Exception as e:
#         print(f"Error processing question ID {line['id']}: {e}")
#         return {"id": line["id"], "question": query, "answer": "Error: " + str(e)}


# def main():
#     if len(sys.argv) < 3:
#         print("需要两个参数: input_param 和 result_path")
#         return
    
#     input_param_str = sys.argv[1]
#     result_path = sys.argv[2]

#     print(f"输入参数内容: {input_param_str}")
    
#     try:
#         with open(input_param_str, "r", encoding="utf-8") as f:
#             input_param = json.load(f)
#     except Exception as e:
#         print(f"读取输入参数文件失败: {e}")
#         return
    
#     # 获取题号列表（新增数字转换逻辑）
#     # selected_ids = input_param.get("selected_ids", [])
#     selected_ids = [3]
#     selected_ids_set = set(selected_ids) if selected_ids else None
    
#     question_file_path = input_param["fileData"]["questionFilePath"]
    
#     with open(question_file_path, "r", encoding="utf-8") as f:
#         q_json_list = [json.loads(line.strip()) for line in f]
    
#     process_list = []
#     default_list = []
    
#     for q_json in q_json_list:
#         # 提取题号中的数字部分（核心修改）
#         q_id_str = q_json["id"]
#         numeric_part = ''.join(filter(str.isdigit, q_id_str))  # 提取所有数字字符
#         q_id = int(numeric_part) if numeric_part else -1  # 转换为整数
        
#         # 判断逻辑
#         if (selected_ids_set is None) or (q_id in selected_ids_set):
#             process_list.append(q_json)
#         else:
#             default_list.append({
#                 "id": q_id_str,  # 保持原始ID格式
#                 "question": q_json["question"],
#                 "answer": "answer wrong"
#             })
    
#     # 多线程处理
#     processed_results = []
#     if process_list:
#         with cf.ThreadPoolExecutor(max_workers=1) as executor:
#             future_list = [executor.submit(process_one, q_json) for q_json in process_list]
#             for future in cf.as_completed(future_list):
#                 processed_results.append(future.result())
    
#     # 合并结果
#     result_json_list = processed_results + default_list
#     result_json_list.sort(key=lambda x: x["id"])  # 按原始ID字符串排序
    
#     with open(result_path, "w", encoding="utf-8") as f:
#         for result in result_json_list:
#             f.write(json.dumps(result, ensure_ascii=False) + "\n")

# if __name__ == "__main__":
#     start_time = time.time()
#     main()
#     end_time = time.time()
#     print(f"程序运行时间: {(end_time - start_time)/60:.2f} 分钟")