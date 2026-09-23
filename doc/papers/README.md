# 论文 PDF

本目录下的 PDF **没有纳入 git**。

原因：这两篇是 arXiv 上的第三方论文，放进公开仓等于转载，版权上不干净。
编号已经记在 [`REFS.md`](../../REFS.md) §二，随时能重新下载，所以不入库也不丢东西。

重新下载（几秒钟）：

```bash
bash scripts/fetch-papers.sh
```

| 文件 | arXiv |
|---|---|
| `FT-HSDP-2026-Training-LLMs-with-Fault-Tolerant-HSDP-on-100k-GPUs.pdf` | [2602.00277](https://arxiv.org/abs/2602.00277) |
| `NCCLX-2025-Collective-Communication-for-100k+GPUs.pdf` | [2510.20171](https://arxiv.org/abs/2510.20171) |

> 想改成入库也可以：把 `.gitignore` 里的 `doc/papers/*.pdf` 那一行删掉，然后
> `git add -f doc/papers/*.pdf`。但公开仓不建议这么做。
