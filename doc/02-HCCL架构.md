# 02 · HCCL 架构与容错现状

> 来源：直接勘察 `code/hccl`（GitCode `cann/hccl`，2026-09-21 克隆，**2026-09-23 更新至 `ba04fcb`**，47 MB，约 1800 文件）+ `docs/zh/` 官方文档。
> **本文件里的路径都已核对过代码，可以直接照着打开。**
> 当前工作目标已转为 FT-AllReduce；本文仍含此前 AllToAll 方向的定位描述，仅作代码架构参考，具体 AllReduce 路径见 `doc/11-AllReduce路径洞察.md`。
> 最后更新：2026-09-21

---

## 1. 仓库结构（实测）

```
code/hccl/
├── include/
│   ├── hccl.h          ← 对外 C API（全库只有两个头文件）
│   └── hccl_mc2.h      ← MC2（通算融合）接口
├── src/
│   ├── ops/            ← ★ 通信算子实现（本任务主战场）
│   ├── common/         ← 环境变量解析、日志、dlsym 动态加载
│   └── algo_plugin/    ← 算法插件
├── test/ut, test/st/algorithm
├── docs/zh/{api_ref, architecture, build, rfcs, user_guide}
├── examples/01..07
└── experimental/ops     ← 需 ENABLE_EXPERIMENTAL 才编译
```

### ⚠️ 最重要的一条结构性认知
**本仓库不含通信域的生命周期管理。** 没有 `HcclCommInitRootInfo` 的实现，通信域相关调用是通过 `src/common/hcomm_dlsym/*` 的**弱符号**动态加载到外部 **hcomm** 库的。

→ **含义**：如果方案涉及"通信子缩容/重建/挂起恢复"，要改的代码在 `code/hcomm`，不在 `code/hccl`。这一点直接决定工作量归属，值得和导师确认。

---

## 2. AllToAll 的实现位置 ★

**所有 AllToAll 变体都在同一个目录**：`src/ops/all_to_all_v/`

| 文件 | 内容 |
|---|---|
| `src/ops/all_to_all_v/all_to_all_v.cc` | **6 个入口函数**：`HcclAlltoAll`(:23)、`HcclAlltoAllV`(:84)、`HcclAlltoAllVC`(:152)，以及三个 GraphMode 版本(:212 / :271 / :330) |

**执行层（executor）**：
- `src/ops/all_to_all_v/algorithm/executor/ins_v2_all_to_all_v_{sole,concurrent,sequence}_executor.*`
- `ins_v2_all_to_all_concurrent_executor.*`
- `ins_v2_aiv_all_to_all_v_sole_executor.*`

**算法模板**：`src/ops/all_to_all_v/algorithm/template/{aicpu, aiv, ccu}/…all_to_all*`
变体包括：Mesh1D、NHR、2Die、multi_jetty、hier（分层）、ubx、dpu
CCU 侧另有：`…/ccu/kernel/ccu_kernel_all_to_all*`

**算法选择器**：`src/ops/all_to_all_v/selector/{alltoall, alltoallv, alltoallvc}_auto_selector.*`
**图模式原型**：`src/ops/all_to_all_v/op_graph/all_to_all_v_proto.cc`

> 对照：其他算子是 `src/ops/{all_reduce, all_gather, all_gather_v, reduce_scatter, reduce_scatter_v, reduce, broadcast, scatter, send, recv, barrier, batch_send_recv}/`

**公共模板框架**（所有算子共用）：`src/ops/op_common/algorithm/template/{aicpu, aiv, ccu, dpu, registry, wrapper}/` + `alg_template_base.*`、`alg_v2_template_base.*`

---

## 3. AllToAll 的语义边界（读 `include/hccl.h` 实测）

| 接口 | 位置 | 语义 | 是否支持变长 |
|---|---|---|---|
| `HcclAlltoAll` | `include/hccl.h:230` | 单个 `sendCount/recvCount` | ❌ **只支持等分片** |
| `HcclAlltoAllV` | `include/hccl.h:211` | `sendCounts/sdispls/recvCounts/rdispls` 四数组 | ✅ 支持不等分片 |
| `HcclAlltoAllVC` | `include/hccl.h:188` | 二维 `sendCountMatrix`，且收发 dtype 可不同 | ✅ 最灵活 |

实现细节：`HcclAlltoAll` 内部通过 `ConvertAlltoAllParam` 构造等分矩阵后**复用 AllToAllV 流程**（`all_to_all_v.cc:59-74`）；不等分片时 `all_to_all_v.cc:120` 会算 `maxSendRecvCount` 来决定缓冲大小。

**兼容分支**：当 `GetHcommVersion() < CANN_VERSION(9,0,0)` 或 `IsOutPlaceDevice()==false` 时，回退到老的 `HcclAlltoAll*Inner` 流程（走外部库）。**做实验时要确认自己落在哪个分支上。**

示例代码：`examples/02_collectives/{06_alltoall, 07_alltoallv, 08_alltoallvc}`

---

## 4. 算法层（怎么选算法）

| 组件 | 位置 |
|---|---|
| 选择引擎 | `src/ops/op_common/selector/{selector_engine, execute_selector, auto_selector_base, selector_registry, alg_attrs_registry}.cc` |
| **成本模型** | `src/ops/op_common/selector/{cost_model, cost_table}.cc` |
| 拓扑匹配 | `src/ops/op_common/algorithm/topo_match/topo_match_*.cc`（base/base_v2/1d/two_level/three_level/nlevel/multilevel/concurrent(_v2)/pcie_mix/squeeze_2d/ubx(_1d)） |
| 拓扑构造 | `src/ops/op_common/topo_info/{topo, topo_host, physical_level_build, physical_level_normalize}.cc` |
| **总入口 `Selector()`** | `src/ops/op_common/op_common.cc:159` |

**流程**：查通信域状态 → `HcclCalcTopoInfo` → 若 level0 是 MESH_1D 走 `SelectorEngine`，否则走 `ExecuteSelector` → `SetCommEngine` 决定加载 AICPU 还是 AIV kernel。调优器在 `src/common/tuner/`。

> 💡 **做容错要动这里**：缩容后成员数和拓扑都变了，**必须重新走一遍算法选择**（重新匹配拓扑、重新选算法、重新算 tiling）。这是"容错不能只改 kernel"的技术原因。

---

## 5. 容错现状 ★★（历史 AllToAll 方向调研）

### 结论先说
HCCL 里**已经有容错的骨架**，但**执行体不在这个仓库**；此前调研聚焦于没有针对 AllToAll 的容错。当前 FT-AllReduce 需要重新沿 AllReduce 路径判断这些设施的适用性，不能直接把本节的 AllToAll 结论当作当前任务定义。

### 5.1 算子重执行 OpRetry（**唯一现成的"重试"机制**）

**行为**：AI CPU 检测到 SDMA/RDMA 的 CQE 错误 → 通过 host socket 全 rank 协商 → **重新下发 SQE/WQE，把整个算子重跑一遍**。

- 粒度：**以通信域为单位**
- 分级：L1 / L2 两级开关；**L2 支持"借轨通信"**（用备用网卡绕开故障链路）
- ⚠️ **约束（极其重要，直接限制你的方案空间）**：
  - 必须 **AI_CPU 展开**（不能是纯 SDMA 路径）
  - 必须**全 rank 停在同一个算子**上
  - **输入内存不可被污染** → **零拷贝 / In-Place / 图模式一律不支持**
- **代码痕迹（只有配置解析，没有执行体）**：
  - `src/common/alg_env_config.cc:890` `SplitHcclRetryEnable`
  - `src/common/alg_env_config.cc:916` `CollectRetryEnableFromConfig`
  - `src/common/alg_env_config.h:54` `hcclRetryConfig[]`
  - → **重执行的实现体推断在 hcomm 里，待查**
- **文档**（先读这三份）：
  - `docs/zh/user_guide/hccl_env/HCCL_OP_RETRY_ENABLE.md`
  - `docs/zh/user_guide/hccl_env/HCCL_OP_RETRY_PARAMS.md`
  - `docs/zh/user_guide/hccl_env/comm_retry_perf_impact.md`（性能影响）

### 5.2 Suspend / Resume（快恢）

- 状态定义：`src/common/hcomm_dlsym/dlsym_common.h:40` → `HcclCommStatus{READY, SUSPENDING}`
- 阶段：`dlsym_common.h:108` → `RESUME_PRE / RESUME_POST`
- 弱符号接口：`src/common/hcomm_dlsym/hccl_host_comm_dl.h:42/63` → `HcclCommGetStatus` / `HcclCommResume`
- **算子侧行为**：通信域非 READY 时**直接返回 `HCCL_E_SUSPENDING`**（`src/ops/op_common/op_common.cc:163-170`）
- AICPU 侧：轮询到 SUSPENDING 返回 301（`src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc:426 / 807 / 1076`）
- 子通信域**级联恢复**：`src/ops/op_common/ccu_fallback.cc:96-134`（失败时降级为"销毁 + 懒重建"）

> ⚠️ 这套机制**和 NCCL 的 `ncclCommSuspend/Resume` 不是一回事**（后者是显存 offload）。HCCL 这个是**真·快恢**。详见 `99-问题清单.md` E3。

### 5.3 超时管理

- `src/ops/op_common/exec_timeout_manager.{h,cc}`（单例）
- `src/ops/op_common/aicpu_timeout.h:49` `DeriveAicpuTimeout`（+20/+50/+30 的推导）
- 解析入口：`src/common/alg_env_config.cc:75` `ParseExecTimeout`
- 网络侧的连接/重传超时**不在本仓库**：
  `docs/zh/user_guide/hccl_env/HCCL_CONNECT_TIMEOUT.md`、`HCCL_RDMA_RETRY_CNT.md`、`HCOMM_TA_*_TIMEOUT.md`
  （**RDMA 自身的重传机制也不在本仓库**）

### 5.4 错误上报（**只上报，不恢复**）

- `src/common/adapter_error_manager_pub.{h,cc}`（`RptInputErr` / `RptEnvErr`，弱符号）
- `inconsistent_check.{h,cc}`、`param_check.{h,cc}`
- 故障诊断文档目录：`docs/zh/user_guide/fault_diagnosis/`
  - `link_timeout_EI0006.md`（链路超时）
  - `notify_wait_timeout_EI0002.md`（notify/wait 超时）
  - `error_cqe_report_EI0013.md`（CQE 错误上报）
  - `cluster_heartbeat_mechanism_troubleshooting.md`（**集群心跳机制**——说明心跳机制存在，但实现不在本仓库）

### 5.5 本仓库**没有**的东西（做方案时的"缺口清单"）

- ❌ 心跳 / rank 失效检测（**本仓库内**没有；存在集群心跳机制，实现在别处）
- ❌ shrink / elastic / 动态成员变更
- ❌ 链路故障后降级到备用链路（除了 OpRetry 的"借轨通信"）
- ❌ **用户态数据重传**（丢包重传靠 RDMA 自身，不在本仓库）
- ❌ 算子级状态保存 / 回滚
- ⚠️ `src/common/static_restore.cc` 是 rank table 文件的**文件锁**安全（flock 重试），**与容错无关**——别被文件名骗了

---

## 6. 建议的代码阅读顺序

1. `include/hccl.h:188/211/230` —— AllToAll 三个接口长什么样（先建立对外语义）
2. `src/ops/all_to_all_v/all_to_all_v.cc` —— 6 个入口，看 host 侧怎么编排
3. `src/ops/op_common/op_common.cc:159` —— `Selector()` 总入口，理解"算法怎么被选出来"
4. `src/ops/all_to_all_v/selector/*_auto_selector.*` —— AllToAll 的算法选择
5. `src/ops/all_to_all_v/algorithm/executor/*` —— 执行层怎么下发
6. `docs/zh/user_guide/hccl_env/HCCL_OP_RETRY_ENABLE.md` —— **容错的起点**
7. `docs/zh/user_guide/hccl_env/HCCL_OP_RETRY_PARAMS.md` + `comm_retry_perf_impact.md`
8. `docs/zh/api_ref/` 与 `docs/zh/architecture/` ——官方视角的架构说明

---

## 7. 待补（TODO）

**已解决（2026-09-23，详见 `doc/10` §3.2）**
- ✅ `code/hcomm` 的结构勘察 —— 三层架构（L1 算子层 / L2 `coll_communicator_mgr` / L3 `base_comm`）；`AGENTS.md:13-15`
- ✅ 通信域生命周期 —— `Suspend/Resume/Clean` 在 hcomm；A2/A3 与 A5 **两代不同实现**（**A5 的 `HcclCommSuspendV2` 返回 `HCCL_E_NOT_SUPPORT`**，A5 走快照恢复）
- ✅ **OpRetry 执行体位置** —— `hcomm/src/legacy/ascend910/framework/cluster_maintenance/recovery/operator_retry/`（Server-Agent 状态机，`OP_RETRY_MAX_CNT=3`；**A5 没有 OpRetry，grep 零命中**）
- ✅ 心跳机制 —— 两代并存：一代 `legacy/ascend910/.../heartbeat/`（50ms 周期 / 30s LOST / STUCK 5min），二代 `coll_communicator_mgr/dfx/cluster_monitor/`（README 有时序图）
- ✅ "借轨通信" —— `SwitchNic` / `backupDeviceIp` / `PrepareLinkForSwitchNic`；**发生借轨后不支持回切**（`opretry_base.cc:702`）
- 🆕 额外发现：hcomm **无 elastic/shrink**（grep 零命中）；有"等量替换" `DiffRankUpdater`（整框 / 64+1）+ 子通信域 `team/`

**待办**
- [ ] CCU / AICPU / AIV 三条执行路径的区别与适用场景
- [ ] `ccu_fallback.cc` 的"销毁 + 懒重建"路径对 AllToAll 语义的影响
- [ ] CANN 版本兼容分支（`GetHcommVersion() < 9.0.0`）在目标环境上落在哪一边
- [ ] hcomm 的 `legacy/` 是否会对我们的改造构成"禁新增特性"的硬约束（`AGENTS.md:48`）—— 见 `doc/10` §四.3
