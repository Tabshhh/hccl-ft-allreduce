# 资源索引 REFS

> 记录所有外部资源的出处、版本、下载情况。**新加资源请补到这里**，并注明下载日期。
> 最后更新：2026-09-23

## 一、代码仓库（`code/`）

### 1.1 导师 2026-09-23 给的清单（10 个）—— 已全部拉到本地，链接逐条核对完毕

| 仓库 | remote（与导师清单核对） | 本地目录 | 体积 | 文件数 | 状态 |
|---|---|---|---|---|---|
| driver | `https://gitcode.com/cann/driver` ✅ | `code/driver/` | 90 MB | 4909 | 🆕 09-23 克隆 |
| hccl | `https://gitcode.com/cann/hccl` ✅ | `code/hccl/` | 47 MB | 1842 | ♻️ 09-21 克隆 → **09-23 已更新至最新** `ba04fcb`（+12）|
| hcomm | `https://gitcode.com/cann/hcomm` ✅ | `code/hcomm/` | 167 MB | 8349 | ♻️ 09-21 克隆 → **09-23 已更新至最新** `3f014f8`（+21）|
| hixl | `https://gitcode.com/cann/hixl` ✅ | `code/hixl/` | 20 MB | 809 | 🆕 09-23 克隆 |
| mind-cluster | `https://gitcode.com/Ascend/mind-cluster` ✅ | `code/mind-cluster/` | 163 MB | 5550 | 🆕 09-23 克隆 |
| ops-transformer | `https://gitcode.com/cann/ops-transformer` ✅ | `code/ops-transformer/` | 291 MB | 13954 | ♻️ 09-21 克隆 → **09-23 已更新至最新** `7700f4b8`（+72）|
| runtime | `https://gitcode.com/cann/runtime` ✅ | `code/runtime/` | 237 MB | 6695 | 🆕 09-23 克隆 |
| torch_npu | `https://github.com/Ascend/pytorch` ✅ | `code/torch-npu/` | 102 MB | 3193 | 🆕 09-23 克隆（浅克隆 `--depth 1`）|
| triton-ascend-ops | `https://github.com/Ascend/triton-ascend-ops` ✅ | `code/triton-ascend-ops/` | 379 KB | 31 | 🆕 09-23 克隆（浅克隆 `--depth 1`）|
| triton-ascend | `https://github.com/triton-lang/triton-ascend` ✅ | `code/triton-ascend/` | 62 MB | 3264 | ♻️ 09-21 克隆 → **09-23 已更新至最新** `a706695`（+5）|

**链接核对结论（2026-09-23）**：
- ✅ **10/10 的 `git remote origin` 与导师清单逐字一致**——4 个已有仓库的 remote 本来就是官方地址；6 个新克隆的仓库，克隆时走了 GitHub 镜像，**克隆完成后已把 origin 改回官方地址**。→ **不存在"链接不一致、需要比对内容"的情况**。
- ✅ **版本已对齐**：4 个已有仓库原本落后上游几天（hccl 12 / hcomm 21 / ops-transformer 69 / triton-ascend 5 个提交），**2026-09-23 已全部 `pull` 到远端 tip** 并逐一比对确认（见上表括号里的 `+N` = 本次并入的提交数）。新并入的提交以修复/性能为主，**未改变本文档的技术结论**。
- ℹ️ 更新后已**重新核对文档里引用的关键行号**，只有 hcomm 有 3 处 1~2 行的漂移（已改）：`coll_comm.cc` 的 Suspend/Clean/Resume 由 1054/1072/1097 → **1053/1071/1096**；`cluster_monitor.cc` 的 `lostThreshold_` 由 729 → **731**。**hccl 与 ops-transformer 的全部锚点（含 `hccl.h:188/211/230`、`all_to_all_v.cc:23/84/152`、elastic 常量 `:57-63`、README `:327/399/420`）逐一核对未变**。

### 1.2 我们自己补充的对照库（非导师清单，均为 09-21~09-23 克隆）

| 目录 | 仓库地址 | 内容 | 体积 | 文件数 |
|---|---|---|---|---|
| `code/nccl/` | https://github.com/NVIDIA/nccl | NCCL 源码（v2.32.3，对照学习对象；`--filter=blob:none`） | 29 MB | 1453 |
| `code/deep-ep/` | https://github.com/deepseek-ai/DeepEP | DeepSeek 的 MoE EP 通信库（`deep_ep` 是模块名） | 4.0 MB | 108 |
| `code/torchft/` | https://github.com/meta-pytorch/torchft | Meta 容错训练库，FT-HSDP 论文的配套实现 | 11 MB | 141 |
| `code/nixl/` | https://github.com/ai-dynamo/nixl | NVIDIA 传输库；**`examples/device/ep/` 内有 `nixl_ep`（弹性 all2all）** | 26 MB | 865 |
| `code/mooncake/` | https://github.com/kvcache-ai/Mooncake | Moonshot 传输引擎；**`mooncake-ep/` 内有 elastic 实现** | 117 MB | 2173 |
| `code/vllm/` | https://github.com/vllm-project/vllm | vLLM 主仓（**Elastic EP 路线的源头**，见 `doc/09`） | 176 MB | 7315 |
| `code/vllm-ascend/` | https://github.com/vllm-project/vllm-ascend | vLLM 的昇腾后端（我们落点所在的框架层） | 82 MB | 4194 |

### 易踩的坑（重要）

- ⚠️ **官方仓库在 GitCode（`gitcode.com/cann/*`），不在 GitHub**。GitHub 上的同名 `cann-hccl` / `ops-transformer` **全是个人镜像**，不要用。
- ⚠️ `gitee.com/ascend/cann-hccl` 是**早期镜像，已停止维护**（自述）。不要用。
- ⚠️ `triton-ascend` 的旧仓 `github.com/Ascend/triton-ascend` **2026-05 后停更**，当前官方仓是 `github.com/triton-lang/triton-ascend`。
- ⚠️ 官方 HCCL 相关的门户/文档站在 **hiascend.com**：`https://www.hiascend.com/cann/hccl`
- ⚠️ **组织名不统一**：`driver`/`hixl`/`runtime`/`hccl`/`hcomm`/`ops-transformer` 在 **`gitcode.com/cann/`**，而 **`mind-cluster` 在 `gitcode.com/Ascend/`**（不是 cann）。找仓库时别只盯一个组织。
- ⚠️ **`torch_npu` 的仓库名是 `pytorch`**：地址是 `github.com/Ascend/pytorch`（昇腾版 PyTorch 插件），没有叫 `Ascend/torch_npu` 的仓。主开发仓在 `gitcode.com/Ascend/pytorch`，GitHub 只是镜像。
- ⚠️ **`triton-ascend-ops` ≠ `triton-ascend`**：前者是**教学样例仓**（31 个文件，只有 `tutorial/`，**无任何通信代码**），后者才是 Triton 的昇腾后端本体。
- ℹ️ **`hixl` 是单边通信库（one-sided），不是集合通信库**——官方定位是"与集合通信库（HCCL）并列"。**它没有 rank / 通信域 / AllToAll 概念**，不要把它当容错落点（详见 `doc/10`）。
- ℹ️ `deep-ep` / `nixl-ep` / `mooncake-ep` **都不是独立仓库名**：`deep-ep` 是 DeepEP 的口头简称；`nixl-ep` 是 NIXL 仓库里的子目录；`mooncake-ep` 是 Mooncake 仓库里的子目录。

### 更新方法（版本落后时）

```bash
# GitCode（直连即可）
cd code/<仓库> && git pull --ff-only

# GitHub 仓库：本地代理 127.0.0.1:7897 开着时用 -c http.proxy；代理没开就用镜像：
git -C code/<仓库> pull --ff-only https://ghfast.top/https://github.com/<org>/<repo>.git <branch>
```
> 2026-09-23 实测：GitCode 直连 ✅；GitHub 直连 ❌、本地代理未开启 ❌、镜像站 `ghfast.top` / `ghproxy.net` / `gh-proxy.com` ✅。

## 二、论文（`doc/papers/`）

| 文件 | 编号 | 标题 | 说明 |
|---|---|---|---|
| `FT-HSDP-2026-Training-LLMs-with-Fault-Tolerant-HSDP-on-100k-GPUs.pdf` | arXiv:2602.00277 | *Training LLMs with Fault Tolerant HSDP on 100,000 GPUs* | ⭐ 导师说的"Meta FT-FSDP 论文"**大概率就是这篇**（简称是 FT-HSDP，不是 FT-FSDP）。arXiv 编号已独立核实 ✅ |
| `NCCLX-2025-Collective-Communication-for-100k+GPUs.pdf` | arXiv:2510.20171 | *Collective Communication for 100k+ GPUs* | 同族论文，Meta 的 NCCLX，含 FTAR 与 Fault Analyzer，arXiv 编号已核实 ✅ |

> 两篇 PDF 均于 2026-09-21 从 arxiv.org 下载。

## 三、官方文档（在线）

| 资源 | 地址 | 用途 |
|---|---|---|
| CANN / HCCL 官方门户 | https://www.hiascend.com/cann/hccl | 官方文档、版本说明 |
| Ascend C 算子开发指南 | https://www.hiascend.com/document | 写算子的主流路线（C++） |
| 昇腾 Triton 算子路线 | https://www.hiascend.com/developer/operator?tag=triton | 辅助提效路线 |
| triton-ascend 文档 | https://triton-ascend.readthedocs.io | Triton 昇腾后端文档 |
| NCCL 用户指南（API） | https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html | 通信子 API |
| NCCL RAS 文档 | https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html | 故障检测子系统 |
| NCCL 容错应用开发 | https://developer.nvidia.com/blog/building-scalable-and-fault-tolerant-nccl-applications/ | 官方容错推荐做法 |
| torchcomms 中的 FT collectives | https://pytorch.org/blog/torchcomms/ | 生态里的"容错集合通信"实现 |

## 四、本机环境（2026-09-21 实测）

| 项 | 状态 |
|---|---|
| 身份 | 非管理员（`DESKTOP-86KDRFS\Tata`），安装需走用户级 |
| OS / Shell | Windows 11 Pro（26200），PowerShell 5.1 |
| git | ✅ 2.55.0.windows.5，装于 `C:\Program Files\Git`（本次安装） |
| python | ✅ 3.11.9，装于 `%LOCALAPPDATA%\Programs\Python\Python311`（本次安装） |
| winget | 🔄 安装中（`Repair-WinGetPackageManager` 失败后改用官方 msixbundle 手动安装） |
| 本地代理 | ✅ `127.0.0.1:7897` 可用（Clash 类工具在跑；**Windows 系统代理是关闭的**） |
| 本机 GPU/NPU | ❌ 无（`nvidia-smi` / `npu-smi` 都不存在）→ **本机只能看代码写文档，跑不了实验** |
| 昇腾实验环境 | ✅ 导师说"有的，都会有，但卡不多"（细节待确认，见 `doc/99-问题清单.md`） |

### 网络经验（血泪）

- GitCode（国内站）**稳定快速**；GitHub **时通时不通**。
- 第一次批量克隆 GitHub 时 6 个仓库**全部失败**（`Failed to connect to github.com:443 after 21s`），但同一时段下载 62MB 的 git 安装包却成功 → 典型的间歇性连通问题。
- **解法**：机器上有本地代理 `127.0.0.1:7897`。给 git 加 `-c http.proxy=http://127.0.0.1:7897` 即可。
- 如需长期生效（可选，自行决定）：
  ```
  git config --global http.proxy http://127.0.0.1:7897
  git config --global https.proxy http://127.0.0.1:7897
  ```
  ⚠️ 代理不常开的话别设全局，否则 git 会连不上。
- 备用镜像站（实测可达，作为代理失效时的兜底）：`ghfast.top`、`ghproxy.net`、`gh-proxy.com`、`gitcode.com/gh_mirrors/*`。
