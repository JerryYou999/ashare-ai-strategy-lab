# A股 AI 策略实验室

一个面向 A 股交易学习者的 AI 策略实验平台，支持：
- 策略模板市场
- 自然语言生成策略
- 沪深300（日频）回测
- 市场情报摘要与行业映射
- 论文/研报上传后的策略提炼（Research Copilot）

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## API Key 配置方式

项目不再内置任何 API Key。

应用会按以下顺序读取配置：
1. 环境变量
2. `st.secrets`
3. 默认值（仅 `LLM_BASE_URL` 和 `LLM_MODEL` 有默认值）

### 方式 A：本地 `secrets.toml`
在项目根目录创建 `.streamlit/secrets.toml`，内容示例见 `.streamlit/secrets.toml.example`。

### 方式 B：Streamlit Community Cloud Secrets
部署到 Streamlit 后，在应用的 **Settings -> Secrets** 中粘贴下面内容：

```toml
LLM_API_KEY = "你的智谱 key"
LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
LLM_MODEL = "glm-4.7-flash"

MARKETAUX_API_KEY = "你的 Marketaux key"
THENEWSAPI_API_KEY = "你的 The News API key"
```

## 推荐演示路径
1. 在“模板市场”选择一个模板
2. 在“策略实验室”设置参数并保存到当前会话
3. 在“回测结果”运行回测并查看收益曲线与指标
4. 在“市场情报”查看自动抓取的市场消息与 AI 解读
5. 在“策略研究员”上传一篇 PDF，让 AI 提炼成候选策略

## 安全说明
- 不要把 `.streamlit/secrets.toml` 提交到 GitHub。
- 部署版请优先使用 Streamlit Community Cloud 的 Secrets 面板。
