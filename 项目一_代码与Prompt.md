# 项目一：代码实现 + Vibe Coding Prompt

> 用法：每个模块提供两种方式
> - **方式一 · Vibe Coding Prompt**：复制到 AI 工具（Claude/ChatGPT/Cursor），AI 吐代码，你审+跑
> - **方式二 · 直接代码**：手动抄写理解，或直接复制运行
> 环境：`da-venv`，依赖 `langchain chromadb openai streamlit pandas numpy matplotlib seaborn`

---

## 模块 A：数据理解（CSV 上传 → 自动数据概览）

### 方式一 · Vibe Coding Prompt

```
写一个 Python 函数 analyze_csv(file_path)，功能如下：

1. 用 pandas 读取 CSV 文件
2. 提取以下数据特征：
   - 行数列数
   - 列名列表
   - 每列数据类型
   - 每列缺失值数量和缺失率
   - 数值列的 describe() 统计摘要
   - 前5行样本数据
3. 把以上特征格式化成一段可读的文本，通过 OpenAI GPT-4o-mini API 生成自然语言数据概览报告
4. 报告包含：数据整体情况、字段含义推测、数据质量问题、建议下一步分析方向

要求：
- 用 openai 库（不是 langchain），方便直接在 Streamlit 里调用
- API key 从环境变量 OPENAI_API_KEY 读取
- model 用 gpt-4o-mini，temperature=0.3
- 返回 LLM 生成的文本字符串
- 加 try-except 错误处理，API 报错时返回友好的错误提示
```

### 方式二 · 直接代码

```python
import os
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def analyze_csv(file_path: str) -> str:
    """上传CSV → AI生成数据概览报告"""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return f"文件读取失败: {e}"

    # 提取数据特征
    data_info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
        "missing": df.isnull().sum().to_dict(),
        "missing_rate": {col: f"{v/len(df)*100:.1f}%" for col, v in df.isnull().sum().items()},
        "numeric_stats": df.describe().to_dict(),
        "sample_rows": df.head(3).to_dict("records"),
    }

    prompt = f"""
你是一个资深数据分析师。以下是上传的CSV数据信息：

- 数据量: {data_info['shape']}（行, 列）
- 列名: {data_info['columns']}
- 数据类型: {data_info['dtypes']}
- 缺失值情况: {data_info['missing_rate']}
- 数值列统计摘要: {data_info['numeric_stats']}
- 前3行样本: {data_info['sample_rows']}

请生成一份用户友好的数据概览报告，分四个部分：
1. 📊 数据整体情况（多少行多少列，时间跨度如有）
2. 📋 字段含义推测（从列名和数据类型推断每列是什么）
3. ⚠️ 数据质量问题（缺失值、可能的异常值）
4. 💡 建议下一步分析方向（3-5个具体建议）
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI 调用失败: {e}"
```

---

## 模块 B：自动分析（自然语言 → 代码生成 → exec 执行 → 错误重试）

### 方式一 · Vibe Coding Prompt

```
写一个 Python 类 DataAnalyst，包含以下功能：

1. __init__(self, df): 接收一个 pandas DataFrame，提取列名、类型、前5行样本
2. generate_code(self, user_query): 
   - 用 OpenAI API (GPT-4o-mini, temperature=0.1) 生成分析代码
   - prompt 里包含列名类型信息 + 前5行 + 用户问题
   - 要求 LLM 只返回 Python 代码，不要额外解释
   - 代码自动 import pandas/numpy/matplotlib/seaborn，用户问题里的关键词作为变量或注释保留
3. execute(self, code):
   - 在本地用 exec() 执行代码
   - 用 io.StringIO 捕获 stdout/stderr 输出
   - 把 df, pd, np, plt, sns 注入执行环境
   - 捕获异常，返回 {success: bool, stdout: str, stderr: str, error: str}
4. analyze(self, user_query, max_retries=3):
   - 完整流程：生成代码 → 执行 → 如果失败，把错误信息发回 LLM 修正 → 重试最多 max_retries 次
   - 返回最终结果

安全要求：
- exec 前用 ast.parse 检查语法
- 禁止代码里出现 __import__, open, os, subprocess, sys 等危险关键词
- 在代码注释里说明「这是简化版沙箱，生产环境用 Docker」
```

### 方式二 · 直接代码

```python
import ast
import io
import traceback
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端，Streamlit 环境必需
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import redirect_stdout, redirect_stderr
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 中文显示
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FORBIDDEN_KEYWORDS = {"__import__", "open(", "os.", "subprocess", "sys.", "eval(", "exec(", "compile("}


class DataAnalyst:
    """自然语言 → Python 代码 → 执行 → 错误重试"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.column_info = self._build_column_info()
        self.sample_rows = df.head(5).to_dict("records")

    def _build_column_info(self) -> str:
        lines = []
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            sample = self.df[col].dropna().head(3).tolist()
            missing = self.df[col].isnull().sum()
            lines.append(f"  {col} ({dtype}), 缺失{missing}个, 样本: {sample}")
        return "\n".join(lines)

    def _check_safety(self, code: str) -> tuple[bool, str]:
        """安全检查：拒绝危险代码"""
        for kw in FORBIDDEN_KEYWORDS:
            if kw in code:
                return False, f"代码包含禁止关键词: {kw}"
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"语法错误: {e}"

    def _generate_code(self, user_query: str) -> str:
        prompt = f"""
你是数据分析专家。数据集信息：
列名和类型:
{self.column_info}

前5行数据:
{self.sample_rows}

用户想分析: "{user_query}"

请生成完整可执行的 Python 代码（pandas/matplotlib/seaborn）。
要求：
1. 代码必须完整、可直接 exec() 执行
2. 数据分析结果用 print() 输出中文说明
3. 图表保存为 plt.savefig('chart.png')，不要 plt.show()
4. 先用简单方法，不要过度复杂
5. df 变量已存在于执行环境中，不要重新定义
6. 🚫 只返回 Python 代码，不要 markdown 代码块标记，不要任何解释文字
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        code = response.choices[0].message.content.strip()
        # 清理可能的 markdown 代码块标记
        for prefix in ["```python", "```"]:
            if code.startswith(prefix):
                code = code[len(prefix):].strip()
        if code.endswith("```"):
            code = code[:-3].strip()
        return code

    def execute(self, code: str) -> dict:
        """安全执行 Python 代码，返回 {success, stdout, stderr, error}"""
        safe, reason = self._check_safety(code)
        if not safe:
            return {"success": False, "stdout": "", "stderr": reason, "error": reason}

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        local_vars = {"df": self.df, "pd": pd, "np": np, "plt": plt, "sns": sns}

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                compiled = compile(code, "<analysis>", "exec")
                exec(compiled, local_vars)
            return {
                "success": True,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }

    def analyze(self, user_query: str, max_retries: int = 3) -> dict:
        """完整分析流程：生成 → 执行 → 失败修正 → 重试"""
        code = self._generate_code(user_query)
        logs = []

        for attempt in range(max_retries):
            result = self.execute(code)
            result["attempt"] = attempt + 1
            result["code"] = code
            logs.append(result)

            if result["success"]:
                result["logs"] = logs
                return result

            if attempt < max_retries - 1:
                fix_prompt = f"""
上一轮代码执行报错：
{result.get('traceback', result['error'])}

原始需求: "{user_query}"

请修正代码确保能正常运行。只返回修正后的 Python 代码，不要解释。
"""
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.1,
                    messages=[{"role": "user", "content": fix_prompt}],
                )
                code = response.choices[0].message.content.strip()
                for prefix in ["```python", "```"]:
                    if code.startswith(prefix):
                        code = code[len(prefix):].strip()
                if code.endswith("```"):
                    code = code[:-3].strip()

        logs[-1]["logs"] = logs
        logs[-1]["max_retries_exceeded"] = True
        return logs[-1]
```

---

## 模块 C：向量知识库（ChromaDB + RAG 问答）

### 方式一 · Vibe Coding Prompt

```
写一个 Python 类 KnowledgeBase，实现 RAG 问答功能：

1. __init__(self, persist_dir="./chroma_db"):
   - 初始化 ChromaDB PersistentClient
   - 创建或获取 collection "analysis_history"
   - embedding 用 OpenAI text-embedding-3-small

2. add_analysis(self, text, metadata: dict):
   - 把分析结论文本存入向量库
   - metadata 包含: 日期、数据源文件名、分析类型（概览/相关性/异常检测等）
   - 自动去重：如果已有相似度 > 0.95 的文档就跳过

3. search(self, query, top_k=3):
   - 语义检索最相关的 top_k 条历史分析
   - 返回文档内容和 metadata

4. ask(self, question):
   - 先 search 检索相关历史
   - 把检索结果作为上下文，通过 GPT-4o-mini 生成回答
   - prompt 里要求「如果没有相关信息就说不知道」
   - 返回 {answer, sources}，sources 列出引用了哪些历史分析

技术要求：
- 用 chromadb 库（不是 langchain 的 chroma wrapper）
- API key 从环境变量读取
- 持久化目录如果不存在自动创建
```

### 方式二 · 直接代码

```python
import os
import datetime
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class KnowledgeBase:
    """ChromaDB 向量库 + RAG 问答"""

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small",
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="analysis_history",
            embedding_function=self.embedding_fn,
        )

    def add_analysis(self, text: str, metadata: dict = None) -> bool:
        """存入分析结论，自动去重"""
        if metadata is None:
            metadata = {}
        metadata.setdefault("date", datetime.datetime.now().isoformat())
        metadata.setdefault("type", "general")

        # 去重检查
        existing = self.collection.query(query_texts=[text], n_results=1)
        if existing["distances"] and existing["distances"][0] and existing["distances"][0][0] < 0.05:
            return False  # 已存在高度相似文档，跳过

        doc_id = f"analysis_{datetime.datetime.now().timestamp()}"
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id],
        )
        return True

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """语义检索历史分析"""
        results = self.collection.query(query_texts=[query], n_results=top_k)
        docs = []
        for i, doc in enumerate(results["documents"][0]):
            docs.append({
                "content": doc,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
            })
        return docs

    def ask(self, question: str) -> dict:
        """RAG 问答：检索 → 增强 → 生成"""
        sources = self.search(question, top_k=3)

        if not sources:
            return {
                "answer": "知识库中暂无相关分析记录，请先上传数据并完成分析。",
                "sources": [],
            }

        context = "\n\n---\n\n".join(
            f"[{s['metadata'].get('date', '未知日期')}] {s['content'][:500]}"
            for s in sources
        )

        prompt = f"""
基于以下历史分析结论，回答用户问题。

历史分析结论：
{context}

用户问题：{question}

要求：
- 如果历史分析中有相关信息，基于这些信息回答
- 如果没有相关信息，直接说「没有找到相关信息」，不要编造
- 回答末尾列出引用了哪些分析记录
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": [{"date": s["metadata"].get("date"), "type": s["metadata"].get("type")} for s in sources],
        }
```

---

## 模块 D：评估闭环

> 这个模块没有「流水线代码」——它是一组评估脚本和分析方法。先用模块 A/B/C 积累至少 10 条真实分析记录，再跑评估。

### 方式一 · Vibe Coding Prompt

```
我有以下数据：
- 用户问题列表（例如: "哪些因素和流失最相关", "销售额趋势如何"）
- 系统对每个问题返回的答案
- 每个答案检索到的文档 ID 列表

请写一个 Python 脚本 evaluator.py，功能如下：

1. 构建离线评估集：
   - 从历史日志中提取至少 20 组 (问题, 系统答案, 人工标注正确答案)
   - 覆盖至少 3 个分析场景：数据概览、相关性分析、趋势分析

2. 检索质量指标：
   - Hit Rate@3: top-3 检索结果中至少一条相关的比例
   - MRR (Mean Reciprocal Rank): 第一个相关文档排名的倒数均值

3. 生成质量指标：
   - 用 GPT-4o-mini 比较系统答案和标准答案，给出 1-5 分
   - 幻觉率: 答案中含未检索到信息的句子比例

4. Bad Case 分析：
   - 把失败案例的输出用 text-embedding-3-small 向量化
   - 用 KMeans 聚类（k=3），输出每个聚类的典型失败模式
   - 归因分类: chunk不合理/知识缺失/prompt边界

输出: 评估报告 + Bad Case 聚类结果
```

### 方式二 · 直接代码

```python
"""模块 D：评估闭环（离线评估 + Bad Case 聚类）
运行前提: 模块 A/B/C 已产出了至少 20 组分析记录
"""

import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ========== D1 离线评估集 ==========

def build_eval_set(analysis_logs: list[dict]) -> list[dict]:
    """
    从分析日志构建评估集
    analysis_logs: [{"question": str, "answer": str, "retrieved_docs": [...], "ground_truth": str}]
    """
    eval_set = []
    for log in analysis_logs:
        if log.get("ground_truth"):
            eval_set.append({
                "question": log["question"],
                "system_answer": log["answer"],
                "retrieved_docs": log.get("retrieved_docs", []),
                "ground_truth": log["ground_truth"],
                "scenario": log.get("scenario", "通用"),
            })
    return eval_set


def compute_retrieval_metrics(eval_set: list[dict]) -> dict:
    """计算检索层指标：Hit Rate@3, MRR"""
    hit_count = 0
    reciprocal_ranks = []

    for item in eval_set:
        retrieved = item["retrieved_docs"]
        # 简化：检索到的文档 ID 和 ground_truth 关联的文档 ID 有交集 = 命中
        gt_doc_id = item.get("gt_doc_id")
        if gt_doc_id and gt_doc_id in retrieved:
            hit_count += 1
            rank = retrieved.index(gt_doc_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return {
        "Hit_Rate@3": hit_count / len(eval_set) if eval_set else 0,
        "MRR": np.mean(reciprocal_ranks) if reciprocal_ranks else 0,
        "total_queries": len(eval_set),
    }


def compute_generation_quality(system_answer: str, ground_truth: str) -> dict:
    """用 LLM 评估生成质量"""
    prompt = f"""
请评估以下系统回答的质量。

【标准答案】: {ground_truth}
【系统回答】: {system_answer}

评分标准 (1-5):
1 = 完全错误或答非所问
2 = 部分相关但主要信息错误
3 = 基本正确但有遗漏或偏差
4 = 正确且较完整
5 = 完全正确、完整、表达清晰

请只返回 JSON 格式: {{"score": 数字, "is_hallucination": true/false, "reason": "一句话原因"}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"score": 3, "is_hallucination": False, "reason": "评分解析失败"}


# ========== D3 Bad Case 聚类归因 ==========

def cluster_bad_cases(bad_answers: list[str], n_clusters: int = 3) -> dict:
    """
    对失败案例做语义聚类，找高频失败模式
    归因类型: chunk不合理 / 知识缺失 / prompt边界
    """
    if len(bad_answers) < n_clusters:
        return {"error": f"失败案例不足{n_clusters}个"}

    # 向量化
    embeddings = []
    for text in bad_answers:
        resp = client.embeddings.create(model="text-embedding-3-small", input=text)
        embeddings.append(resp.data[0].embedding)

    embeddings = np.array(embeddings)

    # KMeans 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # 每类找一个代表性案例
    clusters = {}
    for i in range(n_clusters):
        indices = np.where(labels == i)[0]
        representative = bad_answers[indices[0]][:200]
        clusters[f"cluster_{i}"] = {
            "count": len(indices),
            "sample": representative,
        }

    return {"n_clusters": n_clusters, "clusters": clusters}
```

---

## 模块 E：多智能体架构（Orchestrator-Workers）

> 本模块依赖模块 B 和 C 已完成。在 Streamlit 中作为「高级模式」开关引入。

### 方式一 · Vibe Coding Prompt

```
我需要为一个数据分析助手系统实现多智能体协作架构。已有以下组件：
- DataAnalyst 类：自然语言 → 生成代码 → 执行 → 返回分析结果
- KnowledgeBase 类：向量库存储 + RAG 检索问答

请实现一个 Orchestrator 编排层，实现以下功能：

1. 任务拆解：用户输入复杂请求（如"分析销售趋势，然后和上个月对比，最后给出补货建议"），
   Orchestrator 拆解成子任务列表 [分析销售趋势, 对比上月, 生成补货建议]

2. Worker 分派：
   - AnalysisWorker: 调用 DataAnalyst.analyze() 做数据分析
   - RetrievalWorker: 调用 KnowledgeBase.search() 检索历史
   - ReviewWorker: 检查前两个 Worker 结果的完整性，决定是否需要重试或转人工

3. 结果仲裁：
   - 如果有多个 Worker 冲突，按优先级：检索结果 > 生成结果
   - ReviewWorker 检查结果置信度，低于阈值标记「需人工确认」

4. 防死循环：
   - 单任务最多 3 轮工具调用
   - 超时 30 秒自动终止
   - 连续 2 轮输出无变化则终止

5. 用 LangGraph 实现状态图：
   入口 → 拆解 → 分派 {AnalysisWorker, RetrievalWorker} → ReviewWorker → 
   如果通过 → 汇总输出；如果不过 → 重试（最多3次）→ 输出/兜底

要求：
- 用 langgraph 库（pip install langgraph）
- 状态用 TypedDict 定义，包含 messages, task_list, worker_results, retry_count
- 最终返回 {final_answer, worker_logs}
```

### 方式二 · 直接代码（LangGraph 状态图）

```python
"""模块 E：多智能体编排层（Orchestrator-Workers）
依赖: pip install langgraph
前提: 模块 B DataAnalyst 和模块 C KnowledgeBase 已就绪
"""

from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END
from openai import OpenAI
import os
import json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ========== 状态定义 ==========

class OrchestratorState(TypedDict):
    messages: Annotated[list, operator.add]          # 对话历史
    user_query: str                                    # 原始用户输入
    task_list: list                                    # 拆解后的子任务
    worker_results: dict                               # {worker_name: result}
    retry_count: int                                   # 当前重试次数
    review_passed: bool                                # 审校是否通过
    final_answer: str                                  # 最终输出
    error: str                                         # 兜底错误信息


# ========== 节点函数 ==========

def decompose_task(state: OrchestratorState) -> OrchestratorState:
    """E1: 任务拆解"""
    prompt = f"""
你是任务调度专家。用户的原始请求是：
"{state['user_query']}"

系统中可用的 Worker:
- AnalysisWorker: 做数据分析和代码执行
- RetrievalWorker: 从历史知识库检索相关结论

请把用户请求拆解成子任务列表，每个子任务指明：
- 应由哪个 Worker 执行
- 具体要做什么

只返回 JSON 数组:
[{{"worker": "AnalysisWorker", "task": "..."}}, {{"worker": "RetrievalWorker", "task": "..."}}]
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0, messages=[{"role": "user", "content": prompt}],
    )
    try:
        task_list = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        task_list = [{"worker": "AnalysisWorker", "task": state["user_query"]}]

    return {"task_list": task_list}


def dispatch_workers(state: OrchestratorState, analyst: "DataAnalyst", knowledge_base: "KnowledgeBase") -> OrchestratorState:
    """E2: 分派 Worker 执行"""
    results = {}
    for task in state.get("task_list", []):
        worker = task["worker"]
        task_desc = task["task"]

        if worker == "AnalysisWorker":
            result = analyst.analyze(task_desc)
            results["AnalysisWorker"] = {
                "success": result.get("success", False),
                "output": result.get("stdout", ""),
                "error": result.get("error"),
            }
        elif worker == "RetrievalWorker":
            docs = knowledge_base.search(task_desc, top_k=3)
            results["RetrievalWorker"] = {
                "success": len(docs) > 0,
                "output": [d["content"][:300] for d in docs],
            }

    return {"worker_results": results, "retry_count": state.get("retry_count", 0)}


def review_results(state: OrchestratorState) -> OrchestratorState:
    """E2: 审校 Worker 检查结果完整性"""
    results = state.get("worker_results", {})

    # 检查所有 Worker 是否成功
    all_success = all(r.get("success", False) for r in results.values())
    has_output = any(r.get("output") for r in results.values())

    if all_success and has_output:
        return {"review_passed": True}
    elif state.get("retry_count", 0) >= 2:
        # 重试上限已到，强制通过
        return {"review_passed": True, "error": "部分任务失败，已用兜底回答"}
    else:
        return {"review_passed": False, "retry_count": state["retry_count"] + 1}


def summarize_results(state: OrchestratorState) -> OrchestratorState:
    """汇总生成最终回答"""
    results = state.get("worker_results", {})
    parts = []

    if "AnalysisWorker" in results:
        parts.append(f"【数据分析】\n{results['AnalysisWorker'].get('output', '无')[:500]}")
    if "RetrievalWorker" in results:
        docs = results["RetrievalWorker"].get("output", [])
        if docs:
            parts.append(f"【历史参考】\n" + "\n".join(f"· {d}" for d in docs[:2]))

    if not parts:
        return {"final_answer": f"抱歉，分析未能完成。{state.get('error', '')}"}

    prompt = f"""
将以下分析结果汇总成一段通顺的中文回答（300字以内）：

{chr(10).join(parts)}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0.3, messages=[{"role": "user", "content": prompt}],
    )
    return {"final_answer": response.choices[0].message.content}


# ========== 状态图构建 ==========

def build_orchestrator_graph(analyst: "DataAnalyst", knowledge_base: "KnowledgeBase") -> StateGraph:
    """E3: 构建 LangGraph 状态图"""
    workflow = StateGraph(OrchestratorState)

    # 定义节点
    workflow.add_node("decompose", decompose_task)
    workflow.add_node("dispatch", lambda state: dispatch_workers(state, analyst, knowledge_base))
    workflow.add_node("review", review_results)
    workflow.add_node("summarize", summarize_results)

    # 定义边
    workflow.set_entry_point("decompose")
    workflow.add_edge("decompose", "dispatch")
    workflow.add_edge("dispatch", "review")

    # 条件分支：review 通过 → 汇总输出；不通过 → 重试 dispatch
    workflow.add_conditional_edges(
        "review",
        lambda state: "summarize" if state["review_passed"] else "dispatch",
        {"summarize": "summarize", "dispatch": "dispatch"},
    )
    workflow.add_edge("summarize", END)

    return workflow.compile()


def run_orchestrator(user_query: str, analyst: "DataAnalyst", knowledge_base: "KnowledgeBase") -> dict:
    """E1-E3: 一站式多智能体调用"""
    graph = build_orchestrator_graph(analyst, knowledge_base)
    initial_state = {
        "user_query": user_query,
        "task_list": [],
        "worker_results": {},
        "retry_count": 0,
        "review_passed": False,
        "final_answer": "",
        "error": "",
    }
    result = graph.invoke(initial_state)
    return {
        "answer": result.get("final_answer", "系统未能生成回答"),
        "task_list": result.get("task_list", []),
        "worker_results": result.get("worker_results", {}),
    }
```

---

## Streamlit 界面（整合所有模块）

### 方式一 · Vibe Coding Prompt

```
写一个 Streamlit 应用 app.py，整合以下功能：

模块 A: 侧边栏上传 CSV → 自动显示数据概览
模块 B: 主界面输入框 → 输入分析问题 → 自动生成代码执行并显示结果
模块 C: 分析完成后自动存入向量库 → 支持追问历史分析
额外功能: 
- 侧边栏有「高级模式」开关，开启后使用多智能体编排（模块 E）
- 分析结果显示在可折叠区域
- 图表自动展示

要求：
- 用 st.session_state 管理 DataAnalyst 和 KnowledgeBase 实例
- API key 从 st.secrets 读取（适配 Streamlit Cloud部署）
- 界面简洁，中文
```

### 方式二 · 直接代码

```python
"""app.py — Streamlit 主界面
部署: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os

# 初始化会话状态
if "analyst" not in st.session_state:
    st.session_state.analyst = None
if "kb" not in st.session_state:
    from module_c import KnowledgeBase  # 假设已保存为 module_c.py
    st.session_state.kb = KnowledgeBase()

st.set_page_config(page_title="智能数据分析助手", layout="wide")
st.title("🤖 智能数据分析助手")

# 侧边栏
with st.sidebar:
    st.header("📁 数据上传")
    uploaded_file = st.file_uploader("上传 CSV 文件", type="csv")

    use_multi_agent = st.checkbox("🧠 高级模式（多智能体）", value=False)

    st.divider()
    st.caption("API Key 从 Streamlit Secrets 读取")

# 主界面
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # 模块 A: 数据概览
    with st.expander("📊 数据概览", expanded=True):
        from module_a import analyze_csv
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df.to_csv(tmp.name, index=False)
            overview = analyze_csv(tmp.name)
        st.markdown(overview)

    # 初始化 DataAnalyst
    from module_b import DataAnalyst
    st.session_state.analyst = DataAnalyst(df)

    # 模块 B/C: 分析 + 知识库
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("💬 你想分析什么？", placeholder="例如：销售趋势如何？哪些品类增长最快？")
    with col2:
        search_query = st.text_input("🔍 追问历史", placeholder="上次XX分析的结果？")

    if query:
        with st.spinner("分析中..."):
            if use_multi_agent:
                from module_e import run_orchestrator
                result = run_orchestrator(query, st.session_state.analyst, st.session_state.kb)
                st.markdown(result["answer"])
            else:
                result = st.session_state.analyst.analyze(query)
                if result["success"]:
                    st.markdown(result["stdout"])
                    # 图表展示
                    if os.path.exists("chart.png"):
                        st.image("chart.png")
                    # 存入知识库
                    st.session_state.kb.add_analysis(
                        f"问: {query}\n答: {result['stdout'][:500]}",
                        {"type": "analysis", "source": uploaded_file.name},
                    )
                else:
                    st.error(f"分析失败 (尝试{result.get('attempt', '?')}次): {result.get('error')}")

    if search_query:
        with st.spinner("检索历史分析..."):
            answer = st.session_state.kb.ask(search_query)
            st.info(answer["answer"])
```

---

## 快速启动清单

```bash
# 1. 激活环境
source ~/da-venv/bin/activate

# 2. 安装依赖（缺什么补什么）
pip install streamlit langchain chromadb openai pandas numpy matplotlib seaborn scikit-learn langgraph

# 3. 设置 API Key（二选一）
export OPENAI_API_KEY="sk-xxx"           # 本地运行
# 或在 Streamlit Cloud → Settings → Secrets 里添加 OPENAI_API_KEY

# 4. 本地测试
streamlit run app.py

# 5. 部署（推 GitHub 后在 Streamlit Cloud 一键部署）
```

---

## Vibe Coding 避坑清单

| 坑 | 现象 | 解法 |
|------|------|------|
| AI 生成代码用 `plt.show()` | Streamlit 白屏 | 改 `plt.savefig('chart.png')`，在代码 prompt 里写死这条 |
| `exec()` 找不到 df | NameError | prompt 里明确说「df 已存在于执行环境中」 |
| ChromaDB 每次重启丢数据 | 检索不到 | 用 `PersistentClient(path="./chroma_db")`，不用 `Client()` |
| AI 返回 markdown 包裹的代码 | exec 报错 | 代码里 strip 掉 ` ```python ``` ` 标记 |
| Streamlit 每次交互重跑整个脚本 | 状态丢失 | 用 `st.session_state` 缓存 analyst/kb 实例 |
| `matplotlib` 在 macOS 弹窗 | 卡死 | 导入前加 `matplotlib.use("Agg")` |
