"""
双 AI 辩论框架 - 主程序
"""

from agent import Agent, Judge


def debate(topic: str, rounds: int = 3):
    """
    执行 AI 辩论

    参数:
        topic: 辩论主题
        rounds: 辩论轮数
    """
    # 创建正反两个 Agent
    pro = Agent(
        name="正方",
        sys_prompt="你是辩论赛的正方辩手。你需要支持给定观点，给出有力的论据和逻辑推理。语言要简洁有力，每次发言不超过200字。"
    )

    con = Agent(
        name="反方",
        sys_prompt="你是辩论赛的反方辩手。你需要反对给定观点，给出有力的反驳和逻辑推理。语言要简洁有力，每次发言不超过200字。"
    )

    # 收集完整辩论历史
    all_history = []

    print("=" * 50)
    print(f"辩论主题：{topic}")
    print("=" * 50)
    print()

    # 正方先发言
    reply = pro.chat(f"请就以下观点发表正方陈词：{topic}")
    print(f"【{pro.name}】：{reply}\n")
    all_history.append({"name": pro.name, "content": reply})

    # 进行多轮辩论
    for i in range(rounds):
        print(f"--- 第 {i + 1} 轮 ---\n")

        # 反方发言
        reply = con.chat(reply)
        print(f"【{con.name}】：{reply}\n")
        all_history.append({"name": con.name, "content": reply})

        # 正方发言
        reply = pro.chat(reply)
        print(f"【{pro.name}】：{reply}\n")
        all_history.append({"name": pro.name, "content": reply})

    # 裁判评判环节
    print("=" * 50)
    print("裁判评判中...")
    print("=" * 50)
    print()

    judge = Judge()
    evaluation = judge.evaluate(all_history)
    print(f"【{judge.name}评判报告】：\n{evaluation}\n")

    print("=" * 50)
    print("辩论结束")
    print("=" * 50)


if __name__ == "__main__":
    # 设置辩论主题
    topic = "人工智能的发展利大于弊"

    # 开始辩论（3轮）
    debate(topic, rounds=3)
