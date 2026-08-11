# AI 数据分析助手

多智能体协作 · 可信度评估 · 自动化 EDA · LangGraph + Streamlit

---

## 功能

上传 CSV → 一键生成 7 板块分析报告 + 交互式图表 + 可信度评估 + 追问深挖

- 📌 数据关键发现
- ⚠️ 数据质量问题（含基础异常检查清单）
- 💡 业务建议
- 📈 下一步分析方向（自动生成深度追问按钮）
- 🧠 分析盲点
- 🎯 可信度评估（🟢🟡🔴 逐条评分）
- 🔬 替代验证（中低可信度自动触发交叉检验）

## 本地运行

```bash
pip install streamlit pandas numpy matplotlib seaborn plotly scipy scikit-learn langgraph openai
streamlit run app.py
```

## 部署

已部署至 Streamlit Cloud：[链接待添加]

## 技术栈

Python · LangGraph · Streamlit · Plotly · Pandas · DeepSeek API · scikit-learn (TF-IDF)
