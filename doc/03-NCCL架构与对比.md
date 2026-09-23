# 03 · NCCL 架构与对比

> 来源：`code/nccl`（`github.com/NVIDIA/nccl`，2026-09-21 克隆，**版本 2.32.3**，`--filter=blob:none` 模式，25 MB）。
> `【源码核实】`= 在克隆代码里直接看到的（附路径/行号）；`【调研】`= 网络调研所得，未逐条核对；`【通识】`= 领域通用知识。
> 最后更新：2026-09-21

---

## 1. NCCL 是什么

NVIDIA 的集合通信库，CUDA 生态的事实标准，被 PyTorch 的 `torch.distributed` 作为默认后端。**我们克隆到的是 2.32.3**（`makefiles/version.mk:9-13` 核实）。

**和 HCCL 的关系**：同层竞品——都提供 AllReduce/AllGather/AllToAll 这类集合通信，但分别面向 NVIDIA GPU 与昇腾 NPU。
**为什么值得精读**：它的容错设计（RAS / Shrink / Grow / 异步错误）是"新一代容错"的**源头**，也是本任务要参考的对象。

---

## 2. NCCL 架构要点【通识】

| 概念 | 说明 |
|---|---|
| **communicator** | 通信上下文（`ncclComm_t`），持有 rank 映射、通道、缓冲等 |
| **channel** | 通信子内部把连接切成多个通道以并行利用多网卡/多链路 |
| **ring / tree** | 内置的两族集合通信算法，按消息大小与规模自动选择 |
| **transport** | P2P（NVLink/PCIe）、IB verbs（InfiniBand/RoCE）、Socket |
| **GPU-driven** | 集合通信由 **device kernel** 发起和推进，CPU 只做编排（对比 HCCL 里 AICPU/AIV 的角色）|
| **collnet / SHARP** | 借助交换机做在网归约的加速路径 |

> 💡 **理解这条对做容错很重要**：NCCL 是 **GPU 驱动**的，所以"通信出错"表现为 **device 侧 kernel 报错/挂住**，CPU 只能通过**异步错误查询 + 超时**感知。这也解释了为什么 NCCL 提供的是"挂起/中止 + 重建"式的粗粒度 API，而不是细粒度重传。

---

## 3. ⭐ NCCL 官方 EP 库（`contrib/nccl_ep/`）——本轮最大发现

**NVIDIA 已经把专家并行（EP）的 dispatch/combine 做成了官方库**，在 NCCL 仓库的 contrib 下单独发布（`libnccl_ep.so`，版本 **0.1.0**）。

| 项 | 内容 |
|---|---|
| 头文件 | `contrib/nccl_ep/include/nccl_ep.h`、`include/nccl_ep/ep_enums.h` |
| 实现 | `contrib/nccl_ep/nccl_ep.cc`：**`ncclEpDispatch`:2820**、**`ncclEpCombine`:3653**、`ncclEpGroupDestroy`:1785 |
| 测试/基准 | `ep_test.cu`、`ep_bench.cu`（约 4600 行） |
| 文档 | `contrib/nccl_ep/README.md`（481+ 行，含概览/快速开始/性能数据/使用场景） |
| 关键结构 | `ncclEpTensor_t`（N 维张量描述符）、`ncclEpGroup_t`（EP 组）、`ncclEpDispatchInputs_t`、`ncclEpGroupConfig_t` |
| 两种模式【README 核实】 | **LL 模式（low latency，低延迟）** 与 **HT 模式（high throughput，高吞吐）** |
| 调优 | `NCCL_EP_TOKENS_PER_CHUNK`（HT 模式，须为 32 的倍数）、`NCCL GIN configuration`（多机 RDMA 推荐） |
| 构建依赖 | 需要 NCCL 以 **Device API support** 编译 |
| Python 绑定 | `nccl_lib.ncclEpDispatch` / `ncclEpCombine` |
| ABI 设计 | 所有跨 API 结构体首字段是 `size` + `magic`，做 ABI/layout 校验（值得借鉴的工程实践）|

> 🎯 **对本任务的价值**：这是**最接近你目标的第一方实现**——同一套 dispatch/combine 语义，在另一个硬件生态里是**怎么定义接口、怎么分 LL/HT 两种模式、怎么组织 group 生命周期**的。
> **建议：把 `contrib/nccl_ep/README.md` 列为优先阅读材料**，和 `ops-transformer` 的 dispatch v2 设计文档对照着读。

---

## 4. NCCL 的容错设施【源码核实，2.32.3】

| 设施 | 证据 | 说明 |
|---|---|---|
| **RAS 子系统** | ✅ `src/ras/ras.cc`、`src/ras/diagnostics_paths.cc`、`src/include/ras.h`；文档 `docs/userguide/source/troubleshooting/ras.rst`、`docs/userguide/source/env.rst`；**官方例子** `docs/examples/08_ras/01_fault_detection/` | 检测挂死/掉队的 rank（细节待读文档） |
| **`ncclCommShrink`** | `docs/userguide/source/usage/communicators.rst:143-184`；flag `NCCL_SHRINK_DEFAULT` / `NCCL_SHRINK_ABORT` | 排除若干 rank 创建新通信子。**只有会留在新通信子里的 rank 才调用它** |
| **`ncclCommGrow`** | 同上 `:189-283` | 加入新 rank。老 rank 传 NULL uniqueId，新 rank 用 `growId`；**父通信子上不能有未完成操作，否则死锁** |
| **`ncclCommGetAsyncError`** | `docs/userguide/source/usage/communicators.rst:102`、`contrib/nccl_checkpoint/shim_core.h:169` | 异步错误查询（配非阻塞通信子） |
| **官方容错配方** | 官方文档 | 非阻塞通信子 + 轮询异步错误 + `ncclCommAbort` + **重建通信子** |
| **`contrib/nccl_checkpoint/`** | `shim.cc`、`shim_checkpoint.cc`、`gen_shim.py` | **通信子 checkpoint/restore 垫片**：记录 `ncclCommShrink` 的父通信子/被排除 rank/flags 以便恢复 |

### 4.1 ⚠️ 名字陷阱：`ncclCommSuspend/Resume` **不是**容错

- 实现：**`src/mem_manager.cc:992-1070`**（是**内存管理**模块）
- 定义：`src/nccl.h.in:416` → `#define NCCL_SUSPEND_MEM 0x01  // Suspend memory (release dynamic allocations)`
- 日志："suspending memory" / "resuming all resources"

→ **它是显存 offload 手段，与错误恢复无关。** 而 **HCCL 的 Suspend/Resume 才是真·快恢**。名字相同、语义相反，务必分清。

### 4.2 【调研，待核】各能力引入的版本
RAS ≈ 2.24（默认开启，`NCCL_RAS_ENABLE`）；Shrink ≈ 2.27；Grow ≈ 2.29.2。**这三项在我们手上的 2.32.3 中均已确认存在。**

---

## 5. ⭐ NCCL ↔ HCCL 概念对照表

| 概念 | NCCL（NVIDIA） | HCCL（昇腾） | 备注 |
|---|---|---|---|
| 通信域 | communicator（`ncclComm_t`） | 通信域（`HcclComm`） | 生命周期实现位置：NCCL 在库内；**HCCL 在 hcomm** |
| 拓扑获取 | 自动探测 NVLink/PCIe/IB | rank table + `topo_info`（`physical_level_build` 等） | |
| 算法选择 | 内置自动（ring/tree） | `selector_engine` + `cost_model` + `topo_match` | HCCL 显式建模成本，更可干预 |
| 数据搬运 | P2P / NVLink / IB verbs | **SDMA / HCCS / RoCE / UB** | |
| 谁驱动通信 | **device kernel（GPU-driven）** | AICPU / AIV / CCU 三种执行引擎 | 决定了错误感知方式 |
| 同步原语 | flag / atomic 计数 | **notify / wait** | |
| 出错检测 | **RAS**（`src/ras/`）+ 异步错误 + 超时 | 超时管理（`exec_timeout_manager`）+ 集群心跳（实现在别处）+ 错误上报（只报不恢复） | |
| 通信域缩容/扩容 | ✅ `ncclCommShrink`(.27) / `ncclCommGrow`(.29.2) | ❌ **本仓库没有**（需查 hcomm） | ← **关键差距** |
| 在途失败处理 | 中止 + 重建；`NCCL_SHRINK_ABORT` 可先中止在途操作 | **OpRetry**：整算子重跑（执行体在 hcomm） | HCCL 的粒度更粗 |
| 挂起/恢复 | `ncclCommSuspend/Resume` = **显存 offload（不是 FT）** | Suspend/Resume = **真·快恢**（`HcclCommStatus{READY,SUSPENDING}`） | ⚠️ 名字陷阱 |
| 通信子检查点 | ✅ `contrib/nccl_checkpoint/` | ❌ 未找到 | |
| 官方 EP 库 | ✅ `contrib/nccl_ep/`（0.1.0，LL/HT 双模式） | ✅ `ops-transformer/mc2/*`（dispatch v1/v2/v3 + elastic） | 两边都在做，**对照着读** |
| 官方 EP 弹性/容错 | 待查（`nccl_ep` 是否含 elastic 未确认） | ✅ `moe_distribute_elastic.h`（EP 域缩容） | |

---

## 6. 对本任务的启示

1. **NCCL 的容错是"粗粒度 + 框架驱动"**：检测靠 RAS/超时，恢复靠 shrink/grow/abort+重建，**库自己不重建数据**。这与 A4 定的边界一致。
2. **"缩容"是 NCCL 提供的核心原语**，而 **HCCL 缺这一环**（本仓库内）。**这可能正是你任务里最有价值的部分**：为 AllToAll 提供等价的成员变更能力。
3. **官方 EP 库值得抄接口设计**：LL/HT 双模式、group 生命周期、ABI 校验（`size`+`magic`）。
4. **注意异常路径的语义**：官方文档明确写了 shrink 的"只有留下的 rank 才调用"、grow 的"父通信子不能有未完成操作" —— 这类**边界约束**是容错实现最容易出错的地方，你的设计文档必须写明对应约束。

---

## 7. 待补（TODO）

- [ ] 读 `docs/userguide/source/usage/communicators.rst` 全文，把 shrink/grow 的**完整约束与错误码**整理出来
- [ ] 读 `docs/examples/08_ras/01_fault_detection/` —— **官方故障检测示例**，看 RAS 到底怎么用
- [ ] 读 `contrib/nccl_ep/README.md` 全文（尤其 Common scenarios 的 LL/HT 两节）
- [ ] 查 `contrib/nccl_ep` 是否含 elastic/容错能力（对比 `moe_distribute_elastic.h`）
- [ ] 读 `contrib/nccl_checkpoint/`，理解"通信子可恢复"是怎么做的
- [ ] 核对 §4.2 的版本号（NCCL release notes）
- [ ] NCCL 的 channel/ring 构造细节（做对照时可能需要）
