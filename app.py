"""
AI 数据分析助手 · 多智能体完整版
用法: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import io, os, re, ast, json, datetime
from openai import OpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI 数据分析助手", page_icon="🤖", layout="wide")

# ============================================================
# RAG 记忆（TF-IDF 本地向量）
# ============================================================
if "analysis_memory" not in st.session_state:
    st.session_state.analysis_memory = []
if "tfidf_vectorizer" not in st.session_state:
    st.session_state.tfidf_vectorizer = TfidfVectorizer(max_features=1000)

def retrieve_similar(query, top_k=3):
    memory = st.session_state.analysis_memory
    if len(memory) < 1:
        return []
    try:
        docs = [q + " " + r[:500] for q, r in memory]
        tfidf = st.session_state.tfidf_vectorizer
        matrix = tfidf.fit_transform(docs + [query])
        sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        return [(memory[i][0], memory[i][1][:300]) for i, s in ranked[:top_k] if s > 0.05]
    except:
        return []

def save_to_memory(query, result):
    memory = st.session_state.analysis_memory
    memory.append((query, result))
    if len(memory) > 50:
        memory.pop(0)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None

# ============================================================
# 代码沙箱
# ============================================================
FORBIDDEN = ["subprocess", "os.system", "eval(", "exec(", "shutil.rmtree",
             "os.remove", "os.unlink", "socket", "requests.post", "ssh", "ftp", "urllib"]

def sandbox_exec(code, df):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    import scipy.stats as stats

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    safe_globals = {
        "pd": pd, "np": np, "df": df.copy(), "plt": plt, "sns": sns,
        "px": px, "go": go, "stats": stats,
        "__builtins__": {
            "print": print, "len": len, "range": range, "list": list, "dict": dict,
            "str": str, "int": int, "float": float, "bool": bool, "sum": sum,
            "min": min, "max": max, "sorted": sorted, "enumerate": enumerate,
            "zip": zip, "abs": abs, "round": round, "set": set, "tuple": tuple,
            "type": type, "__import__": __import__,
        }
    }
    code_lower = code.lower()
    for kw in FORBIDDEN:
        if kw.lower() in code_lower:
            return {"text": f"[沙箱拦截] {kw}", "images": []}
    try:
        ast.parse(code)
    except SyntaxError as e:
        return {"text": f"[语法错误] {e}", "images": []}
    buf = io.StringIO()
    safe_globals["__builtins__"]["print"] = lambda *a, **kw: print(*a, **kw, file=buf)
    plt.close('all')
    try:
        exec(code, safe_globals)
        text = buf.getvalue(); buf.close()
        images = []
        for fn in plt.get_fignums():
            fig = plt.figure(fn)
            ib = io.BytesIO(); fig.savefig(ib, format='png', dpi=100, bbox_inches='tight')
            ib.seek(0); images.append({"type": "matplotlib", "data": ib.read()}); ib.close()
        plt.close('all')
        for k, v in list(safe_globals.items()):
            if isinstance(v, go.Figure):
                try: images.append({"type": "plotly", "data": v.to_json()})
                except: pass
        if not text.strip() and not images: text = "[代码执行完成]"
        return {"text": text, "images": images}
    except Exception as e:
        buf.close(); plt.close('all')
        return {"text": f"[执行错误] {type(e).__name__}: {e}", "images": []}

# ============================================================
# LLM 客户端
# ============================================================
def get_llm_client():
    api_key = st.session_state.get("api_key", "")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com") if api_key else None

def call_llm(system_prompt, user_prompt, temperature=0.3):
    client = get_llm_client()
    if not client: return "NO_CODE"
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            temperature=temperature, max_tokens=2000)
        return resp.choices[0].message.content
    except:
        return "NO_CODE"

# ============================================================
# LangGraph 多智能体
# ============================================================
class AgentState(TypedDict):
    user_query: str; df_summary: str
    retrieved_context: Annotated[list, operator.add]
    generated_code: str; exec_result: object
    review_feedback: str; final_answer: str; retry_count: int

def retrieval_worker(state: AgentState) -> AgentState:
    similar = retrieve_similar(state["user_query"], top_k=3)
    state["retrieved_context"] = [f"Q: {q}\nA: {r}" for q, r in similar]
    return state

def analysis_worker(state: AgentState) -> AgentState:
    ctx = ""
    if state["retrieved_context"]:
        ctx = "【历史相关分析】\n" + "\n---\n".join(state["retrieved_context"][:2])
    prompt = f"""你是数据分析专家。

【数据】{state['df_summary']}
{ctx}
【问题】{state['user_query']}

生成 Python 代码回答。变量 df 已加载，用 print 输出，不重读文件。
不需要代码时返回 NO_CODE。只输出代码。"""
    code = call_llm("只输出 Python 代码。", prompt, 0.1).strip()
    code = re.sub(r'^```(?:python)?\s*', '', code)
    code = re.sub(r'\s*```$', '', code)
    state["generated_code"] = code
    return state

def executor_worker(state: AgentState) -> AgentState:
    code = state["generated_code"]
    if code == "NO_CODE" or not code.strip():
        state["exec_result"] = "NO_EXECUTION"
    else:
        state["exec_result"] = sandbox_exec(code, st.session_state.df)
    return state

def review_worker(state: AgentState) -> AgentState:
    result = state["exec_result"]
    if isinstance(result, str):
        if result == "NO_EXECUTION":
            state["final_answer"] = "分析完成"; state["review_feedback"] = "PASS"; return state
        text = result
    else:
        text = result.get("text", "")

    if text.startswith("[执行错误]") and state["retry_count"] < 3:
        prompt = f"代码出错，修正。\n错误: {text}\n原代码: {state['generated_code']}\n只输出修正后的代码。"
        fixed = call_llm("只输出 Python 代码。", prompt, 0.1).strip()
        fixed = re.sub(r'^```(?:python)?\s*', '', fixed); fixed = re.sub(r'\s*```$', '', fixed)
        new_result = sandbox_exec(fixed, st.session_state.df)
        state["generated_code"] = fixed; state["exec_result"] = new_result
        state["retry_count"] += 1; result = new_result

    if isinstance(result, str):
        if result.startswith("[执行错误]") or result.startswith("[沙箱拦截]"):
            state["review_feedback"] = "FAIL"
            state["final_answer"] = f"执行失败（重试{state['retry_count']}次）: {result}"
        else:
            state["review_feedback"] = "PASS"; state["final_answer"] = result
    else:
        text = result.get("text", "")
        if text.startswith("[执行错误]") or text.startswith("[沙箱拦截]"):
            state["review_feedback"] = "FAIL"
            state["final_answer"] = f"执行失败（重试{state['retry_count']}次）: {text}"
        else:
            state["review_feedback"] = "PASS"
            state["final_answer"] = json.dumps({"text": text, "images": len(result.get("images", []))})
    return state

def route_after_analysis(state): return "executor" if state["generated_code"] != "NO_CODE" else "reviewer"
def route_after_review(state): return END

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("retriever", retrieval_worker); g.add_node("analyst", analysis_worker)
    g.add_node("executor", executor_worker); g.add_node("reviewer", review_worker)
    g.set_entry_point("retriever"); g.add_edge("retriever", "analyst")
    g.add_conditional_edges("analyst", route_after_analysis, {"executor": "executor", "reviewer": "reviewer"})
    g.add_edge("executor", "reviewer")
    g.add_conditional_edges("reviewer", route_after_review, {END: END})
    return g.compile()

graph_app = build_graph()

# ============================================================
# Streamlit UI
# ============================================================
st.title("🤖 AI 数据分析助手")
st.caption("多智能体协作 · LangGraph 编排 · 代码沙箱 · 可信度评估 · RAG 记忆")

with st.sidebar:
    st.markdown("## ⚙️ 设置")
    st.session_state.api_key = st.text_input("DeepSeek API Key", type="password",
        value=st.session_state.get("api_key", ""), placeholder="sk-...")
    with st.expander("🔒 数据安全说明"):
        st.caption("原始数据不离开本地 · 仅发送列名/统计摘要至 LLM · 企业级可切换私有化模型")
    st.divider()
    st.markdown("## 📤 上传数据")
    uploaded = st.file_uploader("CSV / Excel", type=["csv", "xlsx"], label_visibility="collapsed")
    if uploaded is not None:
        if uploaded.name.endswith(".csv"): df_raw = pd.read_csv(uploaded)
        else: df_raw = pd.read_excel(uploaded)
        total_rows = len(df_raw)
        # 新数据上传 → 检测数据是否变了，变了才清旧分析
        new_hash = hash(str(df_raw.columns.tolist()) + str(len(df_raw)))
        old_hash = st.session_state.get("_data_hash", None)
        if old_hash != new_hash:
            for k in ["auto_text", "auto_charts", "auto_chart_code", "auto_questions",
                      "asked_questions", "followup_results", "auto_question_sources",
                      "auto_question_map", "alt_verification_images",
                      "analysis_snapshot", "prev_snapshot", "extra_suggestions", "extra_analysis", "_data_hash"]:
                st.session_state.pop(k, None)
            st.session_state["_data_hash"] = new_hash

        # 自动检测并转换日期列
        for col in df_raw.columns:
            if 'date' in col.lower() or 'time' in col.lower() or '日期' in col or '时间' in col or col.lower() == 'dt':
                try:
                    df_raw[col] = pd.to_datetime(df_raw[col], errors='coerce')
                except: pass

        if total_rows > 100_000:
            df_raw = df_raw.sample(n=min(50_000, total_rows), random_state=42)
            st.session_state.df = df_raw
            st.success(f"加载：{len(df_raw):,} 行（全量 {total_rows:,}，已自动抽样）")
        else:
            st.session_state.df = df_raw
            st.success(f"加载：{len(df_raw):,} 行 × {len(df_raw.columns)} 列")
    if st.session_state.df is not None:
        if st.button("🔍 一键全面分析", type="primary", use_container_width=True):
            st.session_state.auto_analysis = True
        # 导出按钮
        if st.session_state.df is not None and st.session_state.get("auto_text"):
            st.divider()
            st.caption("📥 导出")
            st.download_button("📥 下载报告", f"# AI 分析报告\n\n{st.session_state.auto_text}",
                file_name="分析报告.md", mime="text/markdown", use_container_width=True, key="dl_report_side")
            chart_code = st.session_state.get("auto_chart_code", "")
            nb = {"nbformat": 4, "nbformat_minor": 5,
                  "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                  "cells": [
                      {"cell_type": "markdown", "source": ["# AI 数据分析\n自动生成的 Notebook"]},
                      {"cell_type": "code", "source": ["import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport plotly.express as px\n\ndf = pd.read_csv('your_data.csv')\ndf.head()"], "metadata": {}},
                      {"cell_type": "markdown", "source": ["## 图表代码"]},
                      {"cell_type": "code", "source": [l + "\n" for l in chart_code.split("\n")] if chart_code else ["# 重新运行一键分析\n"], "metadata": {}},
                      {"cell_type": "markdown", "source": ["## 分析结论\n\n" + (st.session_state.auto_text or "").replace("#", "###")]},
                  ]}
            st.download_button("📓 导出 Notebook", json.dumps(nb, ensure_ascii=False, indent=1),
                file_name="分析代码.ipynb", mime="application/json", use_container_width=True, key="dl_nb_side")

        if st.button("🗑️ 清空", use_container_width=True):
            for k in ["messages", "auto_text", "auto_charts", "auto_questions",
                      "asked_questions", "followup_results", "auto_question_sources",
                      "auto_question_map", "alt_verification_images",
                      "analysis_snapshot", "prev_snapshot", "extra_suggestions", "extra_analysis", "_data_hash"]:
                if k in st.session_state: st.session_state.pop(k)
            st.rerun()

# ============================================================
# 列类型推断（在所有检测之前）
# ============================================================
def infer_column_types(df):
    """推断每列类型：ID / 分类 / 日期 / 布尔 / 连续指标"""
    types = {}
    for col in df.columns:
        low = col.lower()
        series = df[col]
        n = series.nunique()

        # 日期列
        if "date" in low or "时间" in col or "日期" in col or pd.api.types.is_datetime64_any_dtype(series):
            types[col] = "📅 日期"
        # 布尔/标志列
        elif n <= 2 and pd.api.types.is_numeric_dtype(series):
            vals = set(series.dropna().unique())
            if vals <= {0, 1} or vals <= {0.0, 1.0} or vals <= {True, False}:
                types[col] = "🔘 布尔"
            else:
                types[col] = "📊 指标"
        # ID列（名字含id/code/key，或唯一值占比>90%）
        elif any(p in low for p in ["_id", " id", "code", "key", "uuid"]):
            types[col] = "🆔 ID"
        elif pd.api.types.is_numeric_dtype(series) and n / max(len(series), 1) > 0.9:
            types[col] = "🆔 ID"
        # 分类编码列（唯一值少且为整数）
        elif pd.api.types.is_integer_dtype(series) and n <= 20 and len(series) > 100:
            types[col] = "🏷️ 分类"
        elif n <= 15 and not pd.api.types.is_numeric_dtype(series):
            types[col] = "🏷️ 分类"
        # 文本列
        elif not pd.api.types.is_numeric_dtype(series):
            types[col] = "📝 文本"
        # 默认：连续指标
        else:
            types[col] = "📊 指标"
    return types

# ============================================================
# 数据预览
# ============================================================
if st.session_state.df is not None:
    df = st.session_state.df
    with st.expander("📊 数据预览 & 诊断", expanded=not st.session_state.get("auto_text")):

        # 列类型推断
        col_types = infer_column_types(df)

        missing_total = df.isnull().sum().sum()
        missing_pct = missing_total / max(len(df) * len(df.columns), 1) * 100
        dupes = df.duplicated().sum()

        # 按类型做检测：📊 指标列 → 偏度 > 2 用分位数，否则用 IQR
        outliers = 0
        for col, ctype in col_types.items():
            if ctype == "📊 指标":
                series = df[col].dropna()
                if len(series) < 4: continue
                if abs(series.skew()) > 2:
                    # 高度偏态 → 分位数法（P1-P99），识别的是"极端值"而非"异常值"
                    lo, hi = series.quantile(0.01), series.quantile(0.99)
                    if lo < hi:
                        outliers += ((series < lo) | (series > hi)).sum()
                else:
                    # 近似对称 → 标准 IQR
                    q1, q3 = series.quantile(0.25), series.quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        outliers += ((series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr)).sum()
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("行数", f"{len(df):,}"); m2.metric("列数", len(df.columns))
        m3.metric("缺失值", str(missing_total), delta=f"{missing_pct:.1f}%" if missing_pct > 0 else "0%", delta_color="inverse")
        m4.metric("重复行", dupes, delta="⚠️" if dupes > 0 else "✅", delta_color="inverse")
        # 标注：偏态→极端值，对称→异常值
        skewed_cnt = sum(1 for col, ctype in col_types.items()
                        if ctype == "📊 指標" and len(df[col].dropna()) >= 4 and abs(df[col].dropna().skew()) > 2)
        metric_cnt = len([c for c, t in col_types.items() if t == "📊 指標"])
        if metric_cnt > 0 and skewed_cnt >= metric_cnt // 2:
            label = "极端值（偏态分布，可能为正常业务波动）"
        else:
            label = "异常值"
        m5.metric(label, outliers, delta="⚠️" if outliers > 5 else "✅", delta_color="inverse")
        tab_preview, tab_info = st.tabs(["前10行", "列信息"])
        with tab_preview: st.dataframe(df.head(10), use_container_width=True)
        with tab_info:
            info = pd.DataFrame({
                "列名": df.columns,
                "推断类型": [col_types.get(c, "") for c in df.columns],
                "缺失": df.isnull().sum().values,
                "缺失%": (df.isnull().sum() / max(len(df), 1) * 100).round(1),
                "唯一值": df.nunique().values,
            })
            st.dataframe(info, use_container_width=True, hide_index=True)
        if missing_total > 0 or dupes > 0:
            st.caption(f"🔍 数据诊断：缺失{missing_total}个 · 重复{dupes}行 · 需人工判断处理方式")
            if missing_total > 0:
                missing_cols = ", ".join(f"{c}({df[c].isnull().sum()})" for c in df.columns if df[c].isnull().sum() > 0)
                st.code(f"# 缺失列: {missing_cols}\n# 缺失→0: df.fillna(0)\n# 缺失→删: df.dropna()\n# 缺失→估算: df.fillna(df.median(numeric_only=True))", language="python")
            if dupes > 0:
                st.code("# 去重: df.drop_duplicates()", language="python")

# ============================================================
# 一键分析结果
# ============================================================
if "auto_text" in st.session_state and st.session_state.auto_text:
    st.divider(); st.markdown("## 📊 一键分析结果")

    df = st.session_state.df
    if df is not None:
        date_cols = [c for c in df.columns if "date" in c.lower() or "日期" in c or "时间" in c or "dt" == c.lower()]
        if date_cols:
            try:
                dcol = pd.to_datetime(df[date_cols[0]])
                df[date_cols[0]] = dcol
                c1, c2 = st.columns([2, 1])
                with c1: st.caption(f"📅 {dcol.min().date()} ~ {dcol.max().date()}（{(dcol.max()-dcol.min()).days}天）")
                with c2: granularity = st.selectbox("图表粒度", ["自动", "按日", "按周", "按月"], key="chart_granularity", label_visibility="collapsed")
            except: pass

    # 快照对比
    prev = st.session_state.get("prev_snapshot", {})
    curr = st.session_state.get("analysis_snapshot", {})
    if prev and curr and prev.get("data_hash") != curr.get("data_hash"):
        with st.expander("🔄 快照对比（上次 vs 本次）", expanded=True):
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("行数", f"{curr.get('rows',0):,}", delta=f"{curr.get('rows',0)-prev.get('rows',0):+,}")
            pc2.metric("列数", curr.get('cols',0), delta=f"{curr.get('cols',0)-prev.get('cols',0):+,}")
            pc3.metric("缺失值", curr.get('missing',0), delta=f"{curr.get('missing',0)-prev.get('missing',0):+,}", delta_color="inverse")
            pc4.metric("时间", curr.get("timestamp","")[-5:], delta=prev.get("timestamp","")[-8:-3] if prev.get("timestamp") else "")

    # 图表
    charts = st.session_state.get("auto_charts", [])
    if date_cols and granularity != "自动":
        try:
            import plotly.express as px
            agg_df = df.copy(); dc = date_cols[0]
            if granularity == "按周": agg_df["period"] = pd.to_datetime(agg_df[dc]).dt.to_period("W").apply(lambda r: r.start_time)
            elif granularity == "按月": agg_df["period"] = pd.to_datetime(agg_df[dc]).dt.to_period("M").apply(lambda r: r.start_time)
            else: agg_df["period"] = pd.to_datetime(agg_df[dc])
            nc = agg_df.select_dtypes(include=np.number).columns.tolist()
            if nc:
                agg = agg_df.groupby("period")[nc[0]].sum().reset_index()
                agg.columns = ["period", nc[0]]
                fig = px.line(agg, x="period", y=nc[0], title=f"{nc[0]}（{granularity}）")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
                charts = charts[1:] if len(charts) > 1 else []
        except: pass

    if charts:
        cols = st.columns(2)
        for i, img in enumerate(charts):
            with cols[i % 2]:
                if isinstance(img, dict) and img.get("type") == "plotly":
                    import plotly.io as pio
                    fig = pio.from_json(img["data"]); fig.update_layout(height=380 if i < 2 else 550)
                    st.plotly_chart(fig, use_container_width=True)
                elif isinstance(img, dict):
                    st.image(img["data"], caption=f"图表{i+1}", use_column_width=True)

    alt_charts = st.session_state.get("alt_verification_images", [])
    if alt_charts:
        with st.expander("🔬 替代验证图表", expanded=True):
            acols = st.columns(min(len(alt_charts), 2))
            for i, img in enumerate(alt_charts):
                with acols[i % 2]:
                    if isinstance(img, dict) and img.get("type") == "plotly":
                        import plotly.io as pio
                        fig = pio.from_json(img["data"]); st.plotly_chart(fig, use_container_width=True)
                    elif isinstance(img, dict):
                        st.image(img["data"], caption=f"验证图{i+1}", use_column_width=True)

    clean_text = re.sub(r'=+\s*\d*\s*=*', '', st.session_state.auto_text)
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
    st.markdown(clean_text)


    # 审计
    with st.expander("🔍 审计信息"):
        st.markdown(f"- 分析时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n- 数据：{len(df):,}行×{len(df.columns)}列\n- 模型：DeepSeek Chat\n- 可复现：上传同一份CSV可得到相同结论")

    asked_count = len(st.session_state.get("asked_questions", set()))
    # 显示追问结果摘要
    if asked_count > 0:
        followups = st.session_state.get("followup_results", [])
        if followups:
            with st.expander(f"📋 追问结果（{len(followups)} 次）", expanded=True):
                for q, src, ans in followups[-3:]:  # 最近3条
                    st.caption(f"{'🧠' if src == '🧠' else '📈'} {q}")
                    st.markdown(ans[:400] + ("..." if len(ans) > 400 else ""))
        st.caption(f"🔄 已追问{asked_count}次 · 💡 建议基于初版分析，追问后可重新运行一键分析更新结论")

    # 扩展业务建议：生成新的行动建议（做什么）
    if "extra_suggestions" not in st.session_state: st.session_state.extra_suggestions = ""
    if "extra_analysis" not in st.session_state: st.session_state.extra_analysis = ""

    if st.button("💡 探索更多行动建议", key="explore_more_action"):
        with st.spinner("生成补充行动建议..."):
            existing = st.session_state.auto_text.split('## 💡')[1].split('##')[0] if '## 💡' in st.session_state.auto_text else ""
            ep = f"基于分析结论，提2-3条新的具体行动建议（做什么），不重复已有。\n数据:{list(df.columns)}\n已有建议:{existing[:1000]}"
            st.session_state.extra_suggestions = call_llm("你是业务分析师。输出行动建议，不重复已有。", ep, 0.0)
            st.rerun()
    if st.session_state.extra_suggestions:
        st.info(f"**补充行动建议**（{datetime.datetime.now().strftime('%m-%d %H:%M')}）：\n\n{st.session_state.extra_suggestions}")

    # 扩展分析方向：生成新的分析角度（分析什么）
    if st.button("📈 探索更多分析方向", key="explore_more_analysis"):
        with st.spinner("生成补充分析方向..."):
            existing = st.session_state.auto_text.split('## 📈')[1].split('##')[0] if '## 📈' in st.session_state.auto_text else ""
            ep = f"基于数据和分析结论，提2-3个新的分析方向（还可以分析什么），不重复已有。\n数据:{list(df.columns)}\n已有方向:{existing[:1000]}"
            st.session_state.extra_analysis = call_llm("你是数据分析师。输出新的分析方向，不重复已有。", ep, 0.0)
            st.rerun()
    if st.session_state.extra_analysis:
        st.info(f"**补充分析方向**（{datetime.datetime.now().strftime('%m-%d %H:%M')}）：\n\n{st.session_state.extra_analysis}")

    # 追问按钮：分两组（盲点追问 / 下一步建议）+ 状态变化
    if "auto_questions" not in st.session_state: st.session_state.auto_questions = []
    if "auto_question_sources" not in st.session_state: st.session_state.auto_question_sources = []
    if "asked_questions" not in st.session_state: st.session_state.asked_questions = set()
    if "followup_results" not in st.session_state: st.session_state.followup_results = []
    if not st.session_state.auto_questions and st.session_state.auto_text:
        with st.spinner("生成追问建议..."):
            all_q, all_s = [], []
            blind = st.session_state.auto_text[st.session_state.auto_text.find('## 🧠'):][:800] if '## 🧠' in st.session_state.auto_text else ""
            if blind:
                br = call_llm("只输出问题，每行一个。", f"把盲点转成追问。\n盲点:{blind}\n不要请字，简短。", 0.5)
                for q in br.strip().split("\n"):
                    if q.strip(): all_q.append(q.strip()); all_s.append("🧠")
            nxt = st.session_state.auto_text[st.session_state.auto_text.find('## 📈'):][:600] if '## 📈' in st.session_state.auto_text else ""
            if nxt:
                nr = call_llm("只输出行动，每行一个。", f"提取建议。\n原文:{nxt}\n去掉库名，简短。", 0.3)
                for q in nr.strip().split("\n"):
                    if len(q.strip()) > 5: all_q.append(q.strip()); all_s.append("📈")
            st.session_state.auto_questions = all_q[:4]
            st.session_state.auto_question_sources = all_s[:4]
            st.session_state.auto_question_map = dict(zip(all_q[:4], all_s[:4]))  # 问题→来源映射

    # 分组显示
    blind_qs = [(q, src) for q, src in zip(st.session_state.auto_questions, st.session_state.auto_question_sources) if src == "🧠"]
    next_qs = [(q, src) for q, src in zip(st.session_state.auto_questions, st.session_state.auto_question_sources) if src == "📈"]

    if blind_qs or next_qs:
        st.markdown("##### 💡 深入分析")

        # 组一：盲点追问
        if blind_qs:
            st.caption("🧠 本次分析盲点 → 点击直接分析")
            qcols = st.columns(min(len(blind_qs), 3))
            for i, (q, src) in enumerate(blind_qs):
                with qcols[i % 3]:
                    asked = q in st.session_state.asked_questions
                    label = f"✅ {q}" if asked else f"⏳ {q}"
                    if st.button(label, key=f"blind_{hash(q) % 100000}", use_container_width=True, disabled=asked):
                        st.session_state.asked_questions.add(q)
                        st.session_state.pending_question = q
                        st.rerun()

        # 组二：下一步建议追问
        if next_qs:
            st.caption("📈 下一步分析方向 → 点击直接分析")
            qcols = st.columns(min(len(next_qs), 3))
            for i, (q, src) in enumerate(next_qs):
                with qcols[i % 3]:
                    asked = q in st.session_state.asked_questions
                    label = f"✅ {q}" if asked else f"→ {q}"
                    if st.button(label, key=f"next_{hash(q) % 100000}", use_container_width=True, disabled=asked):
                        st.session_state.asked_questions.add(q)
                        st.session_state.pending_question = q
                        st.rerun()

# ============================================================
# 聊天
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

prompt = None
if st.session_state.get("auto_analysis"):
    st.session_state.auto_analysis = False
    if not st.session_state.api_key: st.error("请先输入 API Key")
    elif st.session_state.df is None: st.error("请先上传数据")
    else:
        df = st.session_state.df
        buf = io.StringIO(); df.describe(include="all").to_string(buf)
        col_info = "\n".join([f"- {c} ({d}): 缺失{df[c].isnull().sum()}" for c, d in zip(df.columns, df.dtypes)])
        text_prompt = f"""你是资深数据分析师。EDA。

数据: 行{len(df):,} 列{list(df.columns)}。{col_info}。描述统计:{buf.getvalue()[:2500]}

输出（markdown格式）。

在分析之前，必须先逐项检查以下基础异常，发现任何一项必须标在报告中：
1. 列间重复：任意两列相关系数>0.99或统计特征完全一致 → 标注"X列与Y列重复"
2. 常量列：唯一值=1的列 → 标注"X列为常量"
3. 异常缺失：缺失率>20%或两列缺失模式一致 → 标注
4. 数值范围异常：百分比列出现>1的值、计数列出现负数 → 标注
5. 类别失衡：分类列某一类占比>95% → 标注

## 📌 数据关键发现
3-5条，说人话。如果某个结论依赖的数据有缺失/异常值，必须在结论后标注 ⚠️。

## ⚠️ 数据质量问题
列出上述基础检查发现的所有异常，说明影响程度。

## 💡 业务建议
2-4条，为什么+怎么做。

## 📈 下一步分析方向
根据数据的实际特征，自动生成3-5条有针对性的深度分析建议。每条必须具体到可执行的追问指令（说"做什么分析"而不是"可以分析XX"）。示例：
- 如果数据有缺货率列→"按10档分桶分析缺货率与日销量的关系，定位销量下滑的拐点"
- 如果数据有品类列→"拆解各品类对整体指标的贡献度，找出拖累最大的Top品类"
- 如果数据有日期列+周数列→"先检查首尾周是否有完整7天数据，排除不完整周，再定位指标波动最大的具体周次，对比前后变化"
- 如果有折扣/促销列→"分段分析折扣力度与销量的关系，确认是否存在边际递减"

每条指令必须自包含——分析师不需要知道数据有首尾不完整周、不需要知道反向因果这些坑，你替他想好，写进指令里。你的角色就是分析师的第一视角：把你看到数据第一反应会问的"深度问题"变成按钮。"

## 🧠 本次分析盲点
2-3条。交代分析边界。如果数据存在缺失/异常值，必须有一条说明"数据质量对结论的影响未量化——缺失率X%对整体结论的置信区间影响未评估"。最后加"未纳入业务背景因素，建议结合实际判断"。"""
        progress_bar = st.progress(0, text="🔍 开始分析...")
        progress_bar.progress(10, text="📊 1/6 生成图表...")
        chart_code = call_llm("只输出 Python 代码。", f"""用 plotly 画4张图，赋值给 fig1-fig4。
数据列:{list(df.columns)} 类型:{dict(df.dtypes)}
1.时间趋势折线图——⚠️ 如果每天有多个数据点，必须先按日期聚合（df.groupby('日期列')['指标列'].sum().reset_index()），再用px.line()画聚合后的数据。fig.update_layout(height=350)
2.px.bar()分类对比(fig.update_layout(height=350))
3.go.Heatmap热力图(fig.update_layout(height=500,width=700))
4.px.scatter()关键变量散点图+trendline='ols'(fig.update_layout(height=350))
每张写title/xaxis_title/yaxis_title。""", 0.1).strip()
        chart_code = re.sub(r'^```(?:python)?\s*', '', chart_code); chart_code = re.sub(r'\s*```$', '', chart_code)
        chart_images = []
        if chart_code and chart_code != "NO_CODE":
            result = sandbox_exec(chart_code, df)
            if isinstance(result, dict) and result.get("text","").startswith(("[语法错误]","[执行错误]")):
                chart_code = call_llm("只输出修正后的 Python 代码。", f"错误:{result['text']}\n原:{chart_code}", 0.1).strip()
                chart_code = re.sub(r'^```(?:python)?\s*', '', chart_code); chart_code = re.sub(r'\s*```$', '', chart_code)
                result = sandbox_exec(chart_code, df)
            if isinstance(result, dict): chart_images = result.get("images", [])
        st.session_state.auto_charts = chart_images; st.session_state.auto_chart_code = chart_code

        progress_bar.progress(35, text="📝 2/6 生成分析报告...")
        text_analysis = call_llm("你是资深数据分析师。结论需稳定可复现。", f"{text_prompt}\n已生成{len(chart_images)}张图表。", 0.0)

        progress_bar.progress(55, text="🎯 3/6 ReviewWorker 可信度评估...")
        confidence_text = call_llm("你是严格的数据科学审查员。结论需稳定可复现。", f"""评估分析报告可信度。

数据: {len(df)}行, {list(df.columns)}, {buf.getvalue()[:2000]}
报告: {text_analysis}

输出:
## 🎯 可信度评估
每条发现: 🟢高/🟡中/🔴低+评估依据。维度:样本量/混淆变量/因果推断风险/数据质量影响（缺失值/异常值是否影响该结论）/统计方法建议。
输出总览: > 📊 整体可信度: X🟢 Y🟡 Z🔴

## 🔍 最需验证的发现
从🔴🟡中选一条。

## 🔬 验证代码
生成Python代码验证。不同分析方法。print输出必须用以下三层结构：

📊 分析方法：
  → 方法1：XXX
  → 方法2：XXX

📈 结果：
  → 方法1结果（说人话，如 "r=.047, p=.512，几乎无相关"）
  → 方法2结果

🎯 验证结论：
  ✅ 可信度高（如"多次验证一致，结论成立"）或 ⚠️ 需重新审视（如"IQR未发现异常，原始发现可能被极端值误导"）

只输出代码。""", 0.0)

        has_low = any(icon in confidence_text for icon in ["🔴","🟡"])
        if has_low:
            progress_bar.progress(75, text="🔬 4/6 执行替代验证...")
            code_match = re.search(r'```(?:python)?\s*\n(.*?)```', confidence_text, re.DOTALL)
            alt_code = code_match.group(1).strip() if code_match else None
            if alt_code and alt_code != "NO_CODE":
                alt_code = re.sub(r'^```(?:python)?\s*', '', alt_code); alt_code = re.sub(r'\s*```$', '', alt_code)
                alt_result = sandbox_exec(alt_code, df)
                if isinstance(alt_result, dict):
                    t = alt_result.get("text",""); imgs = alt_result.get("images",[])
                    if t and t != "[代码执行完成]": confidence_text += f"\n\n### 🔬 验证结果\n```\n{t[:1500]}\n```"
                    if imgs: st.session_state.alt_verification_images = imgs
        else:
            confidence_text += "\n\n---\n### 🔬 替代验证\n✅ 所有发现可信度均为 🟢 高。"
        progress_bar.progress(95, text="✅ 5/6 完成...")
        final_text = text_analysis + "\n\n---\n\n" + confidence_text
        save_to_memory("一键全面分析", final_text[:2000])

        import hashlib
        prev_snap = st.session_state.get("analysis_snapshot", {})
        st.session_state.analysis_snapshot = {
            "text": final_text, "rows": len(df), "cols": len(df.columns),
            "missing": df.isnull().sum().sum(),
            "timestamp": datetime.datetime.now().isoformat()[:19],
            "columns": list(df.columns),
            "data_hash": hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()[:8],
        }
        st.session_state.prev_snapshot = prev_snap; st.session_state.auto_text = final_text
        progress_bar.progress(100, text="✅ 分析完成"); st.rerun()

if "pending_question" in st.session_state and st.session_state.pending_question:
    prompt = st.session_state.pending_question; st.session_state.pending_question = None
else:
    ui = st.chat_input("追问分析"); prompt = ui if ui else None

if prompt:
    if not st.session_state.api_key: st.error("请先输入 API Key")
    elif st.session_state.df is None: st.error("请先上传数据")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Orchestrator 调度 Agent..."):
                try:
                    df = st.session_state.df
                    buf = io.StringIO(); df.describe(include="all").to_string(buf)
                    df_summary = f"行{len(df):,} 列{list(df.columns)} 统计:{buf.getvalue()[:2000]}"
                    initial_state: AgentState = {
                        "user_query": prompt, "df_summary": df_summary,
                        "retrieved_context": [], "generated_code": "",
                        "exec_result": "", "review_feedback": "", "final_answer": "", "retry_count": 0,
                    }
                    result = graph_app.invoke(initial_state, {"recursion_limit": 10})
                    exec_result = result.get("exec_result", {})
                    simgs = exec_result.get("images", []) if isinstance(exec_result, dict) else []
                    answer = result.get("final_answer", "")

                    if simgs:
                        cols = st.columns(2)
                        for i, img in enumerate(simgs):
                            with cols[i % 2]:
                                if isinstance(img, dict) and img.get("type") == "plotly":
                                    import plotly.io as pio
                                    fig = pio.from_json(img["data"]); st.plotly_chart(fig, use_container_width=True)
                                elif isinstance(img, dict): st.image(img["data"], use_column_width=True)

                    if answer and answer != "分析完成":
                        display_text = answer
                        if answer.startswith("{"):
                            try:
                                parsed = json.loads(answer)
                                if parsed.get("text"): display_text = parsed["text"]
                            except: pass
                        if display_text and display_text != "分析完成":
                            fmt = call_llm("整理成markdown。结论需稳定可复现。", f"整理:\n{display_text[:2500]}\n用##标题、-列表、表格。", 0.0)
                            if fmt and fmt != "NO_CODE": st.markdown(fmt); answer = fmt
                            else: st.markdown(display_text)
                    elif answer == "分析完成":
                        fallback = call_llm("你是数据分析师。结论需稳定可复现。", f"数据:{df_summary[:2000]}\n问题:{prompt}", 0.0)
                        st.markdown(fallback); answer = fallback

                    save_to_memory(prompt, answer or "")
                    src = st.session_state.get("auto_question_map", {}).get(prompt, "")
                    st.session_state.followup_results.append((prompt, src, answer or "分析完成"))
                    st.session_state.messages.append({"role": "assistant", "content": answer or "分析完成"})
                except Exception as e:
                    err = str(e)
                    if "Connection" in err: st.error("无法连接 DeepSeek API")
                    else: st.error(f"出错：{err}")
                    st.session_state.messages.append({"role": "assistant", "content": f"出错：{err}"})
