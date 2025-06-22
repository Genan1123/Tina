#!/usr/bin/env python3
# Tina Real Execution – v16.0 〈铁人版〉
# ------------------------------------------------------------
# 核心升级
# 1. State Machine ：明确 DISCOVER → INSTALL → QC → REPORT → DONE
# 2. 双重确认    ：终稿带 "DONE" 仍需评审团 + 结果验收双通过
# 3. 结果验收    ：自动检查 FastQC/MultiQC 报告是否生成
# 4. 评分 JSON   ：Critic 返回 {"score":8,"reason":"..."}，零解析错误
# 5. 全异步调用  ：planner / critic / react 全用 async Together
# ------------------------------------------------------------
import os, sys, asyncio, json, re, subprocess, glob, shutil
from datetime import datetime
from typing import List, Literal, Dict, Any
from collections import Counter

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
import together                             # pip install together

# ---------- 常量 & 配置 ----------
REFERENCE_MODELS = [
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "deepseek-ai/DeepSeek-V3",
    "Qwen/Qwen3-235B-A22B-fp8-tput",
]
AGGREGATOR_MODEL = "deepseek-ai/DeepSeek-V3"
CRITIC_MODELS = [
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "deepseek-ai/DeepSeek-V3",
    "Qwen/Qwen3-235B-A22B-fp8-tput",
    ]

EXPERT_MODEL = "deepseek-ai/DeepSeek-V3"

State = Literal["DISCOVER_FILES", "INSTALL_TOOLS", "RUN_QC", "REPORT", "DONE"]
PIPELINE_ORDER: List[State] = ["DISCOVER_FILES","INSTALL_TOOLS","RUN_QC","REPORT","DONE"]

PLANNER_SYSTEM_PROMPT = """You are a committee of senior bioinformatics consultants.
You receive:
* a JSON project_state
* the user's ultimate goal

Output exactly ONE high-level sub-task in natural language that advances the pipeline
from its current state to the **next** state (see allowed states below).

Allowed states flow sequentially:
DISCOVER_FILES → INSTALL_TOOLS → RUN_QC → REPORT → DONE

If project_state["state"] == "DONE" **and** the required validation flag is true,
output the single word TERMINATE (in upper-case).  
Never output bash code.
"""
AGGREGATOR_SYSTEM_PROMPT = "You are the Chief Scientist. Merge and refine the committee's suggestions into ONE concise sub-task."
SELF_CRITIQUE_SYSTEM_PROMPT = "You are a ruthless reviewer. Point out concrete flaws or risks in the plan."
CRITIC_SYSTEM_PROMPT = """You are an academic grading rubric.
Return a JSON object: {"score":<0-10 integer>,"reason":"one short sentence"}."""
EXPERT_SYSTEM_PROMPT = """You are an AI engineer (ReAct). Respond **only** with JSON:
{"thought":"...","action":{"tool_name":"execute_bash|task_complete","parameters":{...}}}
"""

API_KEY = os.getenv("TOGETHER_API_KEY") or os.getenv("TINA_API_KEY")
TIMEOUT_CMD = 3600
MAX_PLANNER_ROUNDS = 10
MAX_REACT_STEPS = 15
CRITIC_PASS = 7
MAX_TOKENS_PER_TURN = 4096
console = Console()
log_dir = "tina_logs"; os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"tina_v16_{datetime.now():%Y%m%d_%H%M%S}.log")

def log(msg:str):
    with open(log_file,"a",encoding="utf-8") as f:
        f.write(f"[{datetime.now():%H:%M:%S}] {msg}\n")

def panel(content:str,title:str,style:str="white"):
    console.print(Panel(Text(content,overflow="fold"),title=title,border_style=style,expand=True))
    log(f"{title}: {content[:5000]}")

def run_bash(cmd:str)->str:
    panel(f"$ {cmd}","💻 执行命令","cyan")
    proc = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                            executable="/bin/bash",text=True,encoding="utf-8",errors="replace")
    try:
        out,err = proc.communicate(timeout=TIMEOUT_CMD)
    except subprocess.TimeoutExpired:
        proc.kill(); out,err="","TIMEOUT"
    result = (f"[STDOUT]\n{out}\n[STDERR]\n{err}").strip()
    style = "red" if err.strip() else "green"
    panel(result,"🖥️ 命令输出",style)
    return result

def validate_completion()->bool:
    fastqc_html = glob.glob("**/fastqc_reports/*html",recursive=True)
    multiqc = glob.glob("**/multiqc_report.html",recursive=True)
    success = bool(fastqc_html or multiqc)
    panel(f"验收结果: {'✅ 通过' if success else '❌ 未通过'}","🔍 结果验收",
          "bold green" if success else "bold red")
    return success

if not API_KEY:
    console.print("[bold red]❌ 未检测到 TOGETHER_API_KEY / TINA_API_KEY[/bold red]")
    sys.exit(1)
async_client = together.AsyncClient(api_key=API_KEY, timeout=300)

async def llm(model:str,system:str,user:str,temperature:float=0.4,max_tokens:int=1500)->str:
    try:
        resp = await async_client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        msg = resp.choices[0].message.content.strip()
        log(f"[{model}] {msg[:500]}")
        return msg
    except Exception as e:
        log(f"[{model}] ERROR {e}")
        return f"ERROR: {e}"

def next_state(cur:State)->State:
    idx = PIPELINE_ORDER.index(cur)
    return PIPELINE_ORDER[min(idx+1,len(PIPELINE_ORDER)-1)]

async def plan_step(project_state:Dict[str,Any])->str:
    user_goal = project_state["user_goal"]
    payload = json.dumps(project_state,indent=2)
    with Live(Spinner("dots",text="委员会独立思考中..."),console=console,transient=True):
        opinions = await asyncio.gather(*[
            llm(m,PLANNER_SYSTEM_PROMPT,
                f"project_state:\n{payload}\nultimate_goal:\n{user_goal}") 
            for m in REFERENCE_MODELS
        ])
    panel("\n\n---\n".join(opinions),"✍️ 各专家独立意见","dim blue")
    draft    = await llm(AGGREGATOR_MODEL,AGGREGATOR_SYSTEM_PROMPT,"\n\n".join(opinions))
    critique = await llm(AGGREGATOR_MODEL,SELF_CRITIQUE_SYSTEM_PROMPT,f"Plan:\n{draft}")
    final    = await llm(AGGREGATOR_MODEL,AGGREGATOR_SYSTEM_PROMPT,
                        f"Original Plan:\n{draft}\n\nCritique:\n{critique}\n\nRewrite improved version.")
    panel(final,"📄 终稿方案","blue")
    return final

def parse_json_score(txt:str)->int:
    try:
        obj = json.loads(txt)
        return int(obj.get("score",-1))
    except Exception:
        m = re.search(r"\b([0-9]|10)\b",txt)
        return int(m.group(1)) if m else -1

async def critic_vote(goal:str,plan:str)->bool:
    prompt = json.dumps({"goal":goal,"plan":plan})
    with Live(Spinner("dots",text="评审团投票中..."),console=console,transient=True):
        votes = await asyncio.gather(*[
            llm(m,CRITIC_SYSTEM_PROMPT,prompt,temperature=0) for m in CRITIC_MODELS
        ])
    scores = [parse_json_score(v) for v in votes if not v.startswith("ERROR")]
    panel("\n".join(votes),"🗳️ 评审团原始评分","magenta")
    mean = np.mean(scores) if scores else -1
    panel(f"有效分数: {scores} | 平均: {mean:.2f}","🗳️ 评分汇总",
          "green" if mean>=CRITIC_PASS else "red")
    return mean>=CRITIC_PASS

# ---------- ReAct 执行（JSON 解析容错 & script/command/cmd 兼容） ----------
async def react_execute(sub_task:str,project_state:Dict[str,Any])->None:
    history = ""
    for step in range(1,MAX_REACT_STEPS+1):
        if step==1:
            panel(f"{sub_task}\n\n{history}","🎯 专家收到子任务","bold green")
        prompt = json.dumps({"step":step,"history":history,"sub_task":sub_task})
        with Live(Spinner("bouncingBar",text=f"ReAct step {step}..."),console=console,transient=True):
            resp = await llm(EXPERT_MODEL,EXPERT_SYSTEM_PROMPT,prompt,temperature=0)
        # 容错解析 JSON
        try:
            data = json.loads(resp)
        except json.JSONDecodeError:
            m = re.search(r'(\{.*\})', resp, re.DOTALL)
            if m:
                data = json.loads(m.group(1))
            else:
                panel(f"解析失败，无法提取JSON:\n{resp}","❌ JSON 格式错误","red")
                continue
        thought = data.get("thought","(no thought)")
        action  = data.get("action",{})
        tool    = action.get("tool_name","")
        param   = action.get("parameters",{})
        panel(thought,f"🤔 专家思考 ({step})","green")
        if tool=="execute_bash":
            # 兼容 cmd / command / script 三种字段
            cmd_str = param.get("cmd") or param.get("command") or param.get("script") or ""
            obs = run_bash(cmd_str)
            history += f"\n[Thought]{thought}\n[Cmd]{cmd_str}\n[Obs]{obs}"
        elif tool=="task_complete":
            panel(f"理由: {param.get('reason','(无)')}","✅ 子任务完成","bold green")
            break

# ---------- 主流程 ----------
async def main():
    console.print(Panel("[bold cyan]Tina v16.0 – State-Machine MoA Agent[/bold cyan]",border_style="blue"))
    while True:
        try:
            goal = console.input("\n[bold yellow]🧑 请输入最终目标：[/bold yellow]").strip()
            if goal.lower() in ("exit","quit","q"): break
            project_state = {"user_goal":goal,"state":"DISCOVER_FILES","validated":False}
            for round_id in range(1,MAX_PLANNER_ROUNDS+1):
                panel(f"第 {round_id}/{MAX_PLANNER_ROUNDS} 轮","⭐️ 总指挥官决策轮","bold magenta")
                plan = await plan_step(project_state)
                if plan.strip().upper()=="TERMINATE":
                    if not project_state["validated"]:
                        panel("模型声称完成，但还未通过自动验收！","⚠️ 伪终止拦截","yellow")
                        plan = "Generate final QC/summary report and set validation flag when done."
                    else:
                        panel("🏆 项目真正完成！","🏆 完成","bold green")
                        break
                ok = await critic_vote(goal,plan)
                if not ok:
                    panel("方案被否决，继续下一轮...","❌ 方案被否决","red")
                    continue
                await react_execute(plan,project_state)
                if project_state["state"]!="DONE":
                    project_state["state"] = next_state(project_state["state"])
                    if project_state["state"]=="DONE":
                        project_state["validated"] = validate_completion()
            else:
                panel("达到最大轮次，流程终止","⚠️ 终止","yellow")
        except KeyboardInterrupt:
            break

if __name__=="__main__":
    asyncio.run(main())
