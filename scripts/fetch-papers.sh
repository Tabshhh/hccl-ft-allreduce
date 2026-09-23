#!/bin/bash
# 下载 doc/papers/ 下的 arXiv 论文 PDF。
#
# 这些 PDF 没有纳入 git：它们是 arXiv 上的第三方论文，放进公开仓等于转载。
# 在任意一台机器上跑这个脚本即可重新拿到（几秒钟）。
#
# 用法： bash scripts/fetch-papers.sh
#       默认从 arxiv.org 直连下载；国内网络慢的话可加环境变量走代理：
#       ARXIV_PROXY=http://127.0.0.1:7897 bash scripts/fetch-papers.sh

set -u

cd "$(dirname "$0")/.." || exit 1
OUT="doc/papers"
mkdir -p "$OUT"

PROXY="${ARXIV_PROXY:-}"
CURL_OPTS=(-fL --retry 3 --retry-delay 2 --connect-timeout 30 -sS)
[ -n "$PROXY" ] && CURL_OPTS+=(-x "$PROXY")

# 文件名|arXiv 编号
PAPERS=(
  "FT-HSDP-2026-Training-LLMs-with-Fault-Tolerant-HSDP-on-100k-GPUs.pdf|2602.00277"
  "NCCLX-2025-Collective-Communication-for-100k+GPUs.pdf|2510.20171"
)

fail=0
for entry in "${PAPERS[@]}"; do
  name="${entry%%|*}"
  id="${entry##*|}"
  dest="$OUT/$name"

  if [ -s "$dest" ]; then
    echo "SKIP  $name  (已存在, $(du -h "$dest" | cut -f1))"
    continue
  fi

  echo "GET   $name  <- arXiv:$id"
  # arxiv.org/pdf/<id> 会 302 到具体版本；-L 跟随
  if curl "${CURL_OPTS[@]}" -o "$dest.part" "https://arxiv.org/pdf/$id"; then
    # 简单校验：不是 HTML 错误页，且体积合理
    if head -c 4 "$dest.part" | grep -q '%PDF'; then
      mv "$dest.part" "$dest"
      echo "OK    $name  ($(du -h "$dest" | cut -f1))"
    else
      rm -f "$dest.part"
      echo "FAIL  $name  —— 拿到的不是 PDF（可能是 HTML 错误页）"
      fail=1
    fi
  else
    rm -f "$dest.part"
    echo "FAIL  $name  —— 下载失败；试试 ARXIV_PROXY=http://127.0.0.1:7897 $0"
    fail=1
  fi
done

exit $fail
