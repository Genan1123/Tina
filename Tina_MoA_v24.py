#!/usr/bin/env python3
# Tina Real Execution – v17.0 〈改进版〉
# ------------------------------------------------------------
# 核心改进
# 1. 评分显示优化：显示每个模型的名称和具体分数
# 2. 错误处理增强：确保所有5个模型都参与评分
# 3. 执行效率提升：简化任务直接执行，减少循环
# 4. 日志记录改进：记录每个模型的详细响应
# 5. 智能任务识别：简单任务跳过复杂流程
# (Revisions from Coding Partner to fix premature completion)
# ------------------------------------------------------------
import os, sys, asyncio, json, re, subprocess, glob, shutil
from datetime import datetime
from typing import List, Literal, Dict, Any, Tuple
from collections import Counter, defaultdict
import traceback

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.columns import Columns
import together

# ---------- 常量 & 配置 ----------
REFERENCE_MODELS = [
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "deepseek-ai/DeepSeek-V3",
]

AGGREGATOR_MODEL = "deepseek-ai/DeepSeek-V3"
CRITIC_MODELS = [
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "deepseek-ai/DeepSeek-V3",
]

EXPERT_MODEL = "deepseek-ai/DeepSeek-V3"

State = Literal["DISCOVER_FILES", "INSTALL_TOOLS", "RUN_QC", "REPORT", "DONE"]
PIPELINE_ORDER: List[State] = ["DISCOVER_FILES","INSTALL_TOOLS","RUN_QC","REPORT","DONE"]

# 简化的提示词
PLANNER_SYSTEM_PROMPT = """You are a bioinformatics consultant. Output ONE concise sub-task to advance the pipeline.
States: DISCOVER_FILES → INSTALL_TOOLS → RUN_QC → REPORT → DONE
If state=="DONE" and validated==true, output TERMINATE."""

AGGREGATOR_SYSTEM_PROMPT = "Merge suggestions into ONE concise actionable sub-task."
SELF_CRITIQUE_SYSTEM_PROMPT = "Point out one concrete flaw or risk."
CRITIC_SYSTEM_PROMPT = """Score the plan 0-10. Return ONLY: {"score":N,"reason":"one sentence"}"""
EXPERT_SYSTEM_PROMPT = """React agent. Return ONLY JSON:
{"thought":"...","action":{"tool_name":"execute_bash|task_complete","parameters":{...}}}
Only use 'task_complete' when the specific sub-task is fully accomplished.
"""

# 简单任务检测
SIMPLE_TASK_PATTERNS = [
    r"generate.*fastq",
    r"create.*file",
    r"生成.*文件",
    r"make.*reads"
]

API_KEY = os.getenv("TOGETHER_API_KEY") or os.getenv("TINA_API_KEY") 
if not API_KEY:
    API_KEY = "92d5979f5ffc69d344a37cc7b2cbef622b6d51b9a24f984d97a68ece985384fb"

TIMEOUT_CMD = 300  # 减少超时时间
MAX_PLANNER_ROUNDS = 10  # 减少最大轮数
MAX_REACT_STEPS = 15  # 减少React步骤
CRITIC_PASS = 7  # 降低通过门槛
MAX_TOKENS_PER_TURN = 3000  # 减少token数

console = Console()
log_dir = "tina_logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"tina_v17_{datetime.now():%Y%m%d_%H%M%S}.log")

def log(msg: str):
    """增强的日志功能"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_entry = f"[{timestamp}] {msg}"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")
    # 也打印到控制台用于调试
    if "ERROR" in msg or "WARN" in msg:
        console.print(f"[red]{log_entry}[/red]", highlight=False)

def panel(content: str, title: str, style: str = "white"):
    """显示面板并记录日志"""
    console.print(Panel(Text(content, overflow="fold"), title=title, border_style=style, expand=True))
    log(f"{title}: {content[:2000]}")

def run_bash(cmd: str) -> str:
    """执行bash命令"""
    panel(f"$ {cmd}", "💻 执行命令", "cyan")
    try:
        proc = subprocess.Popen(
            cmd, shell=True, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            executable="/bin/bash", text=True, 
            encoding="utf-8", errors="replace"
        )
        out, err = proc.communicate(timeout=TIMEOUT_CMD)
        result = f"[STDOUT]\n{out}\n[STDERR]\n{err}".strip()
        style = "red" if err.strip() and proc.returncode != 0 else "green"
        panel(result[:1000], "🖥️ 命令输出", style)
        return result
    except subprocess.TimeoutExpired:
        proc.kill()
        return "[ERROR] Command timeout"
    except Exception as e:
        return f"[ERROR] {str(e)}"

# 异步客户端
async_client = together.AsyncClient(api_key=API_KEY, timeout=60)

async def llm(model: str, system: str, user: str, temperature: float = 0.3, max_tokens: int = 1000) -> str:
    """调用LLM"""
    try:
        resp = await async_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        msg = resp.choices[0].message.content.strip()
        log(f"[{model}] Response: {msg[:200]}...")
        return msg
    except Exception as e:
        error_msg = f"ERROR calling {model}: {str(e)}"
        log(error_msg)
        return error_msg

def is_simple_task(goal: str) -> bool:
    """检测是否为简单任务"""
    goal_lower = goal.lower()
    return any(re.search(pattern, goal_lower) for pattern in SIMPLE_TASK_PATTERNS)

def parse_json_score(txt: str) -> Tuple[int, str]:
    """解析评分JSON，返回(分数, 原因)"""
    try:
        # 首先尝试提取JSON
        json_match = re.search(r'\{[^}]*"score"[^}]*\}', txt, re.DOTALL)
        if json_match:
            obj = json.loads(json_match.group(0))
            return int(obj.get("score", -1)), obj.get("reason", "No reason")
    except:
        pass
    
    # 备用：查找数字
    score_match = re.search(r'\b([0-9]|10)\b', txt)
    score = int(score_match.group(1)) if score_match else -1
    return score, "Failed to parse reason"

async def critic_vote_enhanced(goal: str, plan: str) -> Tuple[bool, List[Dict]]:
    """增强的评审团投票，返回详细信息"""
    prompt = f"Goal: {goal}\nPlan: {plan}"
    
    # 创建评分表格
    score_table = Table(title="🗳️ 评审团详细评分", show_header=True, expand=True)
    score_table.add_column("模型", style="cyan", width=40)
    score_table.add_column("分数", style="yellow", width=10)
    score_table.add_column("理由", style="white", width=50)
    
    with Live(Spinner("dots", text="评审团投票中..."), console=console, transient=True):
        # 并行调用所有评审模型
        tasks = []
        for i, model in enumerate(CRITIC_MODELS):
            task = llm(model, CRITIC_SYSTEM_PROMPT, prompt, temperature=0, max_tokens=200)
            tasks.append((i, model, task))
        
        # 收集结果
        results = []
        for i, model, task in tasks:
            try:
                response = await task
                score, reason = parse_json_score(response)
                results.append({
                    "model": model,
                    "score": score,
                    "reason": reason,
                    "response": response
                })
                
                # 添加到表格
                model_short = model.split('/')[-1][:35] + "..." if len(model.split('/')[-1]) > 35 else model.split('/')[-1]
                score_table.add_row(
                    model_short,
                    str(score) if score >= 0 else "ERROR",
                    reason[:47] + "..." if len(reason) > 47 else reason
                )
                
            except Exception as e:
                log(f"ERROR in critic {i} ({model}): {str(e)}")
                results.append({
                    "model": model,
                    "score": -1,
                    "reason": f"Error: {str(e)}",
                    "response": str(e)
                })
                score_table.add_row(
                    model.split('/')[-1],
                    "ERROR",
                    str(e)[:50]
                )
    
    # 显示详细评分表
    console.print(score_table)
    
    # 计算有效分数
    valid_scores = [r["score"] for r in results if r["score"] >= 0]
    
    if valid_scores:
        mean = np.mean(valid_scores)
        std = np.std(valid_scores) if len(valid_scores) > 1 else 0
        median = np.median(valid_scores)
        
        # 显示统计信息
        stats_panel = Panel(
            f"有效评分: {len(valid_scores)}/{len(CRITIC_MODELS)}\n"
            f"分数列表: {valid_scores}\n"
            f"平均分: {mean:.2f} ± {std:.2f}\n"
            f"中位数: {median:.1f}\n"
            f"{'✅ 通过' if mean >= CRITIC_PASS else '❌ 未通过'} (及格线: {CRITIC_PASS})",
            title="📊 评分统计",
            border_style="green" if mean >= CRITIC_PASS else "red"
        )
        console.print(stats_panel)
        
        passed = mean >= CRITIC_PASS
    else:
        panel("所有评审模型都失败了！", "⚠️ 警告", "yellow")
        passed = False
    
    return passed, results

async def simple_task_execution(goal: str) -> bool:
    """直接执行简单任务"""
    panel(f"检测到简单任务，直接执行：{goal}", "🚀 快速模式", "green")
    
    # 对于生成fastq文件的任务
    if "fastq" in goal.lower() or "generate" in goal.lower():
        # 生成一个简单的fastq文件
        cmd = """
# 生成随机fastq文件
cat > generate_random_fastq.py << 'EOF'
import random
import sys

def generate_random_seq(length):
    return ''.join(random.choice('ACGT') for _ in range(length))

def generate_random_qual(length):
    return ''.join(chr(random.randint(33, 73)) for _ in range(length))

# 生成1000个reads
num_reads = 1000
read_length = 150

with open('random_reads_R1.fastq', 'w') as f1, open('random_reads_R2.fastq', 'w') as f2:
    for i in range(num_reads):
        seq1 = generate_random_seq(read_length)
        seq2 = generate_random_seq(read_length)
        qual1 = generate_random_qual(read_length)
        qual2 = generate_random_qual(read_length)
        
        # R1
        f1.write(f'@read_{i}/1\\n{seq1}\\n+\\n{qual1}\\n')
        # R2
        f2.write(f'@read_{i}/2\\n{seq2}\\n+\\n{qual2}\\n')

print(f"Generated {num_reads} paired-end reads")
print("Files: random_reads_R1.fastq, random_reads_R2.fastq")
EOF

python generate_random_fastq.py
ls -lh random_reads_*.fastq
head -n 8 random_reads_R1.fastq
"""
        result = run_bash(cmd)
        
        if "Generated" in result and "random_reads" in result:
            panel("✅ 成功生成随机FASTQ文件！", "🎉 任务完成", "green")
            return True
    
    return False

async def react_execute_improved(sub_task: str, project_state: Dict[str, Any]) -> bool:
    """改进的ReAct执行，返回是否成功"""
    history = ""
    
    # Let's give the agent a few more steps to work with.
    for step in range(1, MAX_REACT_STEPS + 1): 
        if step == 1:
            panel(f"子任务: {sub_task}", "🎯 开始执行", "bold green")
        
        # Add more context to the prompt for better decisions
        prompt = (f"Current State: {project_state['state']}\n"
                  f"Overall Goal: {project_state['user_goal']}\n"
                  f"Sub-Task: {sub_task}\n\n"
                  f"History of previous steps (use this to inform your next action):\n{history[-1500:]}")
        
        with Live(Spinner("bouncingBar", text=f"思考步骤 {step}/{MAX_REACT_STEPS}..."), console=console, transient=True):
            resp = await llm(EXPERT_MODEL, EXPERT_SYSTEM_PROMPT, prompt, temperature=0)
        
        # --- Action Parsing Logic (remains the same) ---
        try:
            data = None
            json_match = re.search(r'\{.*"thought".*"action".*\}', resp, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                # Handle cases where the model doesn't return valid JSON
                panel(f"Could not find valid JSON in response. Trying to continue.\nResponse: {resp[:200]}", f"⚠️ 警告 ({step})", "yellow")
                history += f"\n[Step {step}] Invalid response from model."
                continue
        except Exception as e:
            panel(f"JSON解析失败: {str(e)}\n响应: {resp[:200]}", "❌ 错误", "red")
            history += f"\n[Step {step}] Failed to parse JSON."
            continue
        
        thought = data.get("thought", "No thought provided.")
        action = data.get("action", {})
        tool = action.get("tool_name", "")
        params = action.get("parameters", {})
        
        panel(thought, f"🤔 思考 ({step})", "green")
        
        # --- Core Logic Change is Here ---
        if tool == "execute_bash":
            cmd = params.get("cmd") or params.get("command") or params.get("script", "")
            if cmd:
                result = run_bash(cmd)
                # Append the outcome to the history for the next step
                history += f"\n[Step {step} Observation]: Executed `{cmd}`\n{result[:500]}"
            else:
                history += f"\n[Step {step} Observation]: Model chose execute_bash but provided no command."

        elif tool == "task_complete":
            reason = params.get("reason", "No reason given.")
            panel(f"Agent decided the sub-task is complete. Reason: {reason}", "✅ 子任务确认完成", "bold green")
            return True # This is the ONLY place we should return True

        else:
            panel(f"Unknown tool: {tool}", f"❌ 错误 ({step})", "red")
            history += f"\n[Step {step} Observation]: Agent chose an invalid tool '{tool}'."

    # If the loop finishes without `task_complete` being called, the sub-task failed.
    panel("Agent reached max steps without completing the sub-task.", "❌ 子任务失败", "red")
    return False

async def main():
    """主函数"""
    console.print(Panel(
        "[bold cyan]Tina v17.0 – 改进版 MoA Agent[/bold cyan]\n"
        "• 详细评分显示\n"
        "• 简单任务快速执行\n"
        "• 更高效的执行流程",
        border_style="blue"
    ))
    
    while True:
        try:
            goal = console.input("\n[bold yellow]🧑 请输入任务目标：[/bold yellow]").strip()
            if goal.lower() in ("exit", "quit", "q"):
                break
            
            log(f"User goal: {goal}")
            
            # 检测简单任务
            if is_simple_task(goal):
                success = await simple_task_execution(goal)
                if success:
                    continue
            
            # 复杂任务流程
            project_state = {
                "user_goal": goal,
                "state": "DISCOVER_FILES",
                "validated": False
            }
            
            for round_id in range(1, MAX_PLANNER_ROUNDS + 1):
                panel(f"第 {round_id}/{MAX_PLANNER_ROUNDS} 轮", "⭐ 规划轮次", "bold magenta")
                
                # 规划步骤
                payload = json.dumps(project_state, indent=2)
                
                # 获取专家意见（简化版）
                with Live(Spinner("dots", text="专家思考中..."), console=console, transient=True):
                    opinions = await asyncio.gather(*[
                        llm(m, PLANNER_SYSTEM_PROMPT, 
                            f"State: {project_state}\nGoal: {goal}\nGive ONE next action.")
                        for m in REFERENCE_MODELS[:2]  # 只用2个模型加快速度
                    ])
                
                # 聚合意见
                final_plan = await llm(
                    AGGREGATOR_MODEL, 
                    AGGREGATOR_SYSTEM_PROMPT,
                    "\n".join(opinions)
                )
                
                panel(final_plan, "📄 最终方案", "blue")
                
                # 检查是否完成
                if "TERMINATE" in final_plan.upper():
                    panel("🏆 任务完成！", "🎉 成功", "bold green")
                    break
                
                # 评审投票
                passed, vote_details = await critic_vote_enhanced(goal, final_plan)
                
                if not passed:
                    panel("方案未通过，继续优化...", "❌ 需要改进", "red")
                    continue
                
                # 执行计划
                success = await react_execute_improved(final_plan, project_state)
                
                # 更新状态
                if success:
                    panel(f"Sub-task for state '{project_state['state']}' was successful.", "✅ 状态推进", "bold green")
                    current_idx = PIPELINE_ORDER.index(project_state['state'])
                    
                    if current_idx < len(PIPELINE_ORDER) - 1:
                        next_state = PIPELINE_ORDER[current_idx + 1]
                        project_state['state'] = next_state
                        panel(f"Advancing to next state: {next_state}", "🔄 流程更新", "magenta")
                        
                        # Check if we have just moved to the final state
                        if next_state == "DONE":
                            project_state["validated"] = True
                            panel("✅ All pipeline states completed. Task is successful!", "🎊 任务成功", "bold green")
                            break
                    else:
                        # This means we were already at the DONE state
                        project_state["validated"] = True
                        panel("✅ Final state 'DONE' was re-validated. Task is successful!", "🎊 任务成功", "bold green")
                        break
                else:
                    # This branch is new. It handles the case where a sub-task fails.
                    panel(f"Sub-task for state '{project_state['state']}' failed to complete.", "❌ 状态停滞", "red")
                    panel("The agent will try to re-plan in the next round.", "🤔 重新规划", "yellow")

            else: # This else belongs to the for loop
                panel("达到最大轮次限制", "⚠️ 超时", "yellow")
                
        except KeyboardInterrupt:
            panel("用户中断", "⚠️ 退出", "yellow")
            break
        except Exception as e:
            panel(f"错误: {str(e)}\n{traceback.format_exc()}", "❌ 异常", "red")
            
    # 显示日志位置
    panel(f"详细日志保存在: {log_file}", "📝 日志文件", "dim")

if __name__ == "__main__":
    asyncio.run(main())
