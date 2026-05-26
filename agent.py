"""
轻量级双 AI 辩论框架 - Agent 类
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件中的环境变量
load_dotenv()


class Agent:
    """辩论 Agent 类，负责与大模型交互并维护对话历史"""

    def __init__(self, name: str, sys_prompt: str):
        """
        初始化 Agent

        参数:
            name: Agent 的名字（如"正方"、"反方"）
            sys_prompt: 系统提示词，用于设定 Agent 的立场和行为
        """
        self.name = name                      # Agent 名称
        self.sys_prompt = sys_prompt          # 系统提示词（决定立场）
        self.history = []                     # 对话历史列表，存储所有交互记录

        # 从环境变量读取 API 配置并初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),  # 从环境变量读取 API Key
            base_url=os.getenv("OPENAI_BASE_URL"),  # 可选：自定义 API 地址（兼容各种 OpenAI 格式服务）
        )

    def chat(self, user_input: str) -> str:
        """
        与 Agent 进行对话

        流程:
            1. 将用户输入追加到 history 中
            2. 调用大模型 API，传入系统提示词和完整历史
            3. 获取模型回复后，将回复也存入 history
            4. 返回模型回复

        参数:
            user_input: 用户输入的文本

        返回:
            str: 大模型生成的回复内容
        """
        # 第一步：将用户输入添加到对话历史
        self.history.append({"role": "user", "content": user_input})

        # 第二步：构建完整的消息列表（系统提示词 + 历史对话）
        messages = [
            {"role": "system", "content": self.sys_prompt},  # 系统提示词，设定 Agent 立场
            *self.history                                    # 展开历史对话记录
        ]

        # 第三步：调用大模型 API
        response = self.client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),  # 模型名称，可从环境变量配置
            messages=messages,
            temperature=0.5,           # 降低创造性，减少重复
            max_tokens=500,            # 限制输出长度
            presence_penalty=0.5,      # 惩罚重复话题
            frequency_penalty=0.5,     # 惩罚重复词语
        )

        # 第四步：获取模型回复文本
        reply = response.choices[0].message.content

        # 第五步：将模型回复也存入历史，保持对话连续性
        self.history.append({"role": "assistant", "content": reply})

        return reply


class Judge(Agent):
    """裁判 Agent 类，继承自 Agent，用于辩论总结与评判"""

    def __init__(self, name: str = "裁判"):
        sys_prompt = (
            "你是一个客观中立的辩论裁判。"
            "你的任务是：\n"
            "1. 总结双方的核心论点\n"
            "2. 分析双方的逻辑严密性和论据充分性\n"
            "3. 指出双方的优点和不足\n"
            "4. 给出最终的评判结果和理由\n\n"
            "请保持公正客观，用专业的裁判语言进行总结。"
        )
        super().__init__(name, sys_prompt)

    def evaluate(self, debate_history: list) -> str:
        """
        根据完整辩论历史进行评判

        参数:
            debate_history: 包含所有对话历史的列表

        返回:
            str: 裁判的综合评判报告
        """
        # 将完整的历史传给裁判进行分析
        self.history = [{"role": "user", "content": str(debate_history)}]

        messages = [
            {"role": "system", "content": self.sys_prompt},
            *self.history
        ]

        response = self.client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=messages,
            temperature=0.3,           # 低创造性，保持客观
            max_tokens=1000,           # 允许较长评判
            presence_penalty=0.2,
            frequency_penalty=0.2,
        )

        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply
