# IPO RAG Assistant（最小可运行版）

Python + **OpenAI**（chat + embedding）+ **Chroma** + **LangChain**（检索链 + 多轮会话式 RAG）。设计见 [DESIGN.md](./DESIGN.md)。

## 环境

- Python 3.9+
- `OPENAI_API_KEY`

```bash
cd "/path/to/IPO"
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY
```

## 用法

将招股书 PDF 放入 `data/raw/`，例如 `data/raw/prospectus.pdf`。

```bash
# 1) 入库（默认 collection 名 default）
ipo-rag ingest data/raw/prospectus.pdf -c default

# 2) 生成中文 Markdown 报告（基于检索）
ipo-rag report -c default

# 3) 多轮问答（history-aware retrieval；exit 退出）
ipo-rag chat -c default

# 4) 检索评测：编辑 benchmarks/gold_qa.jsonl 后
ipo-rag eval -c default
```

也可：`python -m ipo_rag --help`

## 配置（环境变量）

| 变量 | 含义 | 默认 |
|------|------|------|
| `OPENAI_API_KEY` | 必填 | — |
| `OPENAI_CHAT_MODEL` | 对话模型 | `gpt-4o-mini` |
| `OPENAI_EMBEDDING_MODEL` | 嵌入模型 | `text-embedding-3-small` |
| `IPO_CHUNK_SIZE` | 分块大小 | `900` |
| `IPO_CHUNK_OVERLAP` | 重叠 | `120` |
| `IPO_TOP_K` | 检索条数 | `6` |
| `IPO_CHROMA_DIR` | Chroma 持久化目录 | `./data/chroma` |
| `IPO_DATA_DIR` | 数据根目录（影响默认 chroma 路径） | `./data` |

## 免责声明

本工具仅基于招股书检索做信息整理演示，**不构成投资建议**。
