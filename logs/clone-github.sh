#!/bin/bash
cd "C:/Users/Tata/Desktop/claude-hccl/code"
CANON=""
clone_one() {  # name  canonical-url  depth
  name="$1"; canon="$2"; depth="$3"
  if [ -d "$name/.git" ]; then echo "SKIP $name"; return; fi
  for m in "https://ghfast.top/" "https://ghproxy.net/" "https://gh-proxy.com/"; do
    echo "=== $name via $m (depth=$depth) ==="
    git clone --depth "$depth" --progress "${m}${canon}" "$name" 2>&1 | tail -3
    if [ -d "$name/.git" ]; then
      git -C "$name" remote set-url origin "$canon"
      echo "OK $name  origin=$(git -C "$name" remote get-url origin)  size=$(du -sh "$name" | cut -f1)"
      return
    fi
    rm -rf "$name"
  done
  echo "FAIL $name"
}
clone_one torch-npu        "https://github.com/Ascend/pytorch.git" 1
clone_one triton-ascend-ops "https://github.com/Ascend/triton-ascend-ops.git" 1
echo "ALL-GITHUB-DONE"
