FROM hubdocker.aminer.cn/library/python-base:1.0.1

#缺哪个库安装哪个库即可
RUN pip install jsonlines pandas loguru openai sqlparse --index-url=https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

RUN pip install requests zhipuai --index-url=https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

# 复制代码到镜像仓库
COPY app /app

# 指定工作目录
WORKDIR /app/devlop_home/baseline/mountain_baseline


CMD ["/app/py_devlop.sh", "/app/devlop_data/input_param.json", "/app/devlop_result/answer.json" ]
