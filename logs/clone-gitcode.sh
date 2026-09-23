#!/bin/bash
cd "C:/Users/Tata/Desktop/claude-hccl/code"
set -x
for pair in "driver|https://gitcode.com/cann/driver.git" "hixl|https://gitcode.com/cann/hixl.git" "mind-cluster|https://gitcode.com/Ascend/mind-cluster.git" "runtime|https://gitcode.com/cann/runtime.git"; do
  name="${pair%%|*}"; url="${pair##*|}"
  if [ -d "$name/.git" ]; then echo "SKIP $name (exists)"; continue; fi
  echo "=== CLONE $name <- $url ==="
  git clone --progress "$url" "$name" 2>&1 | tail -5
  echo "EXIT=$? $name"
done
echo "ALL-GITCODE-DONE"
