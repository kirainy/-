# GLM深远海船舶作业大模型应用赛

## 1. 团队信息 & 项目介绍
**团队成绩**：
初赛模型复赛b榜第8(准确率79%)；复赛b榜第3(59%)  

**成员组成**：  
• 模型开发:(https://github.com//kirainy)
• 数据开发:(https://github.com//code-farmer233)

**项目简介**：  
面向深远海作业船舶的智能化需求，本赛题解决方案针对船舶作业场景的三大挑战：复杂问题适应性、决策准确率优化，基于GLM大模型构建端到端决策方案。通过创新性的[Prompt工程优化]+[轻量化部署]+[数据科学智能体构建]技术组合，在保持原始数据合规性的前提下，实现最终复赛b榜准确率约60%的显著效果。

## 2. 项目文件引导栏
```
DEEPSEGML_UPLOAD/
├── app/
│   ├── develop_data/            # 开发输入数据
│   │   ├── input_param.json     # 参数配置文件
│   │   └── question.json        # 问题定义文件
│   ├── devlop_home/             # 开发环境中枢
│   │   ├── assets/              # 资源管理中心
│   │   │   ├── 复赛数据/         # 复赛数据集
│   │   │   └── history_res/     # 历史实验记录（按日期归档）
│   │   ├── baseline/            # 基线模型
│   │   │   └── mountain_baseline/  # 旗舰基线实现
│   │   │       ├── _pycache_/   # Python编译缓存
│   │   │       ├── data/        # 数据处理中间数据
│   │   │       ├── database_in_use/   # 数据处理最终数据，agent要用的数据
│   │   │       ├── history_ai_agent/  # 智能体交互历史
│   │   │       ├── ai_agents.py   # 智能体调度中心
│   │   │       ├── ai_brain.py    # 核心算法（初版）
│   │   │       ├── api.py         # 为智能体提供的API
│   │   │       ├── data_process_fusai.py     # 复赛数据处理主程序1
│   │   │       ├── data_process_fusai_new.py # 复赛数据处理主程序2
│   │   │       ├── state_division.py         # 复赛阶段划分主程序
│   │   │       ├── dict.json      # 数据词典映射
│   │   │       ├── main.py        # 系统入口（CLI交互）
│   │   │       ├── README.md      # 模块技术手册
│   │   │       ├── run.py         # 训练任务启动器
│   │   │       ├── test.py        # 单元测试套件
│   │   │       └── tools.py       # 开发者工具包
│   │   ├── pyproject.toml      # 项目元数据配置
│   │   ├── README.md           # 全局说明文档
│   │   └── requirements.txt    # 全量依赖清单
│   ├── devop_result/           # 运维输出沙盒
│   └── py_devop.sh             # 环境初始化脚本
└── Dockerfile                 # 容器化部署蓝图

```
## 3. 快速开始
### 环境要求
• Python 3.8+
• docker

### 创建并运行镜像
```bash
# 1、登录比赛账号(或者其他账号，环境相同也可以)
docker login hubdocker.aminer.cn -u "*********" -p "******"

# 2、创建镜像（版本号0.1镜像名test可改）
docker build -t hubdocker.aminer.cn/****/test:0.1 .

# 3、运行镜像
docker run -it hubdocker.aminer.cn/****/test:0.1 /bin/bash
```
### 数据处理
1. 数据预处理，默认路径为`app\devlop_home\assets\复赛数据`
```bash
python data_process_fusai.py
```
2. 处理数据，进行动作判定：
```bash
python data_process_fusai_new.py
```
3. 进行复赛四个新的阶段划分
```bash
python state_division.py
```
**<span style="color:red">注意：</span>**
请勿直接删除`database_in_use`目录，若需重新数据预处理，则需要保留源文件`设备参数详情.xlsx`，因为此文件不涉及预处理，故重新运行时不会添加文件到此目录，但后期会使用到。若需要重新添加此文件，请勿更改名称`设备参数详情.xlsx`，否则会找不到此文件导致回答错误。

### 运行程序
```bash
python main.py "/app/devlop_data/input_param.json" "/app/devlop_result/answer.jsonl"
```



## 4. 常见问题
**Q：未安装docker**  
A：进入docker官网安装docker后重试，安装docker参考(https://blog.csdn.net/weixin_71699295/article/details/137387383)的第一部分。

**Q：出现warning：Please set the environment variable.**  
A：确保环境与复赛一直，并设置环境变量，即`base_url_env`


## 5. 注意事项
**更改API_KEY**  
进入`app\devlop_home\baseline\mountain_baseline\ai_agents.py`中第`122`行进行更改


---


🐛 问题反馈：1449027231@qq.com，或直接联系wx：18919755652

---

> 本README遵循[CommonMark规范](https://commonmark.org/)，最后更新于2025-04-06
