"""
AI 数据分析助手 · 最小可演示版
用法: streamlit run demo_app.py
"""

import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="AI 数据分析助手", page_icon="🤖", layout="wide")
st.title("🤖 AI 数据分析助手")
st.caption("上传 CSV → AI 自动理解数据、生成分析结论")

# Sidebar: API Key
with st.sidebar:
    st.markdown("## ⚙️ 设置")
    api_key = st.text_input("DeepSeek / OpenAI API Key", type="password", placeholder="sk-...")
    api_provider = st.selectbox("API", ["DeepSeek（推荐）", "OpenAI"], index=0)
    st.caption("DeepSeek：deepseek-chat，一次分析约 ¥0.001 | OpenAI：gpt-4o-mini")
    st.divider()
    st.markdown("## 🔧 功能")
    st.markdown("""
    - 📤 上传 CSV / Excel
    - 📊 自动数据概览
    - 🔍 AI 深度分析
    - 💡 业务建议输出
    """)

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "上传数据文件",
        type=["csv", "xlsx"],
        help="支持 CSV 和 Excel 格式"
    )

if uploaded_file is not None:
    # Load data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Data overview
    st.divider()
    st.markdown("## 📊 数据概览")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("行数", f"{len(df):,}")
    col_b.metric("列数", len(df.columns))
    col_c.metric("缺失值", df.isnull().sum().sum())
    col_d.metric("重复行", df.duplicated().sum())

    with st.expander("🔍 点击查看列信息"):
        info_df = pd.DataFrame({
            "列名": df.columns,
            "类型": df.dtypes.values,
            "缺失": df.isnull().sum().values,
            "唯一值": df.nunique().values,
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)

    with st.expander("📋 前 5 行预览"):
        st.dataframe(df.head(), use_container_width=True)

    # AI Analysis
    st.divider()
    st.markdown("## 🤖 AI 分析")

    analysis_type = st.selectbox(
        "分析类型",
        ["自动分析（推荐）", "相关性分析", "异常检测", "趋势分析", "数据分布分析"]
    )

    if st.button("🚀 开始分析", type="primary", use_container_width=True):
        if not api_key:
            st.error("请先输入 OpenAI API Key（左侧设置栏）")
        else:
            with st.spinner("AI 正在分析你的数据..."):
                try:
                    from openai import OpenAI

                    if "DeepSeek" in api_provider:
                        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                        model = "deepseek-chat"
                    else:
                        client = OpenAI(api_key=api_key)
                        model = "gpt-4o-mini"

                    # Build data summary for prompt
                    buffer = io.StringIO()
                    df.describe(include="all").to_string(buffer)
                    describe_str = buffer.getvalue()

                    col_info = "\n".join([
                        f"- {col} ({dtype}): {df[col].nunique()} 唯一值, "
                        f"{df[col].isnull().sum()} 缺失, "
                        f"示例: {df[col].dropna().head(3).tolist()}"
                        for col, dtype in zip(df.columns, df.dtypes)
                    ])

                    prompt = f"""你是一位资深数据分析师。用户上传了一个数据集，请根据以下信息进行分析。

【数据基本信息】
- 行数: {len(df):,}
- 列数: {len(df.columns)}
- 列名: {list(df.columns)}

【各列详情】
{col_info}

【描述性统计】
{describe_str[:3000]}

【分析类型】{analysis_type}

请输出（用中文，面向业务人员，不要术语堆砌）：

## 📌 数据关键发现
3-5 条，每条一句话，说人话。突出业务含义。

## ⚠️ 数据质量问题
如果有缺失值、异常值、重复数据等问题，指出来。

## 💡 业务建议
基于数据发现，给出 2-4 条可落地的建议。每条建议说明"为什么"和"怎么做"。

## 📈 下一步分析方向
建议接下来可以做什么深度分析，做什么类型的图表最合适。
"""

                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=2000,
                    )

                    result = resp.choices[0].message.content

                    st.markdown("---")
                    st.markdown(result)
                    st.success("✅ 分析完成")

                except Exception as e:
                    st.error(f"AI 调用失败")
                    st.code(str(e))
else:
    # No file uploaded yet - show sample/public data option
    st.divider()
    st.markdown("### 💡 没有数据？试试示例数据集")
    if st.button("📥 加载电商示例数据"):
        # Generate sample e-commerce data
        import numpy as np
        np.random.seed(42)
        n = 200
        sample_df = pd.DataFrame({
            "日期": pd.date_range("2026-01-01", periods=n, freq="D"),
            "订单数": np.random.poisson(150, n) + np.sin(np.arange(n) * 0.1) * 30,
            "销售额": np.random.normal(25000, 5000, n).astype(int),
            "客单价": np.random.normal(180, 30, n).round(0),
            "新客占比": np.random.beta(3, 7, n).round(2),
            "退货率": np.random.beta(1, 9, n).round(3),
            "折扣率": np.random.uniform(0.05, 0.25, n).round(2),
        })
        st.session_state.sample_data = sample_df
        st.dataframe(sample_df.head(10), use_container_width=True)
        st.info("👆 下载这个数据（点击表格右上角）→ 上传到上方 → 点「开始分析」")
