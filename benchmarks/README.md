# Benchmarks

`gold_qa.jsonl`：每行一条 JSON。

- `question`：用于检索的短问题（与真实用户问法接近）。
- `gold_pages`：答案所在**页码**列表（与你 PDF 解析后的 `metadata.page` 一致）；用于计算 Recall@k。
- `prospectus_id`：可选，便于区分多份招股书。

先 `ipo-rag ingest your.pdf`，根据招股书把 `gold_pages` 改成真实页码，再运行：

```bash
ipo-rag eval
# 或
ipo-rag eval -g benchmarks/gold_qa.jsonl -c default
```

`eval` 会检查：对每个问题，**top-k 检索结果里是否出现过任一 gold 页**（k 由环境变量 `IPO_TOP_K` 控制，默认 6）。
