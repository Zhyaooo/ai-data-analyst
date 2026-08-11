# AI Data Analysis Assistant / AI 数据分析助手

Multi-Agent Orchestration · Confidence Scoring · Auto-Cross-Validation · Automated EDA / 多智能体协作 · 可信度评估 · 自动化 EDA

Built with LangGraph + Streamlit + Plotly

---

## Features / 功能

Upload a CSV → One-click 7-section analysis report + interactive charts + confidence scoring + follow-up deep dive / 上传 CSV → 一键生成 7 板块分析报告 + 交互式图表 + 可信度评估 + 追问深挖

- 📌 Key Findings / 数据关键发现
- ⚠️ Data Quality Issues / 数据质量问题（含基础异常检查清单）
- 💡 Business Recommendations / 业务建议
- 📈 Next Steps / 下一步分析方向（自动生成深度追问按钮）
- 🧠 Analysis Blind Spots / 分析盲点
- 🎯 Confidence Scoring / 可信度评估（🟢🟡🔴 逐条评分）
- 🔬 Cross-Validation / 替代验证（中低可信度自动触发交叉检验）

## Run Locally / 本地运行

```bash
pip install streamlit pandas numpy matplotlib seaborn plotly scipy scikit-learn langgraph openai
streamlit run app.py
```

## Deployment / 部署

Deployed on Streamlit Cloud / 已部署至 Streamlit Cloud：[link to be added / 链接待添加]

## Tech Stack / 技术栈

Python · LangGraph · Streamlit · Plotly · Pandas · DeepSeek API · scikit-learn (TF-IDF)
