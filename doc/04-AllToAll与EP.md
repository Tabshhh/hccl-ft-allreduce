# 04 · AllToAll 与 MoE 专家并行（EP）

> 来源：`code/ops-transformer`（GitCode `cann/ops-transformer`，2026-09-21 克隆，**2026-09-23 更新至 `7700f4b8`**，291 MB，约 14000 文件）+ `code/hccl`。
> **本文件里的算子路径都已核对过实际目录结构。**
> 最后更新：2026-09-21

---

## 1. AllToAll 的语义变体

（接口定义见 `code/hccl/include/hccl.h`，详见 `02-HCCL架构.md` §3）

| 变体 | 语义 | MoE 场景是否常用 |
|---|---|---|
| `HcclAlltoAll` | **等分片**：每人给其他每人的数据量相同 | 理想情况 |
| `HcclAlltoAllV` | **不等分片**：四个数组指定各人的 count 与 displace | ✅ **真实场景就用这个**（token 分布不均） |
| `HcclAlltoAllVC` | 二维矩阵指定，收发类型可不同 | ✅ 更灵活的场景 |

> **一句话**：MoE 里每个专家收到的 token 数天然不均，所以**你的目标算子几乎必然是 AllToAllV 型**，不是等分的 AlltoAll。

---

## 2. 为什么 MoE 需要 AllToAll

MoE（Mixture of Experts）里，每张卡上放的不是完整模型，而是**若干个专家**：

```
一张卡上的计算：token → 路由器打分 → 选中 topK 个专家
                ↓
   但被选中的专家可能不在本卡上！ → 需要把 token 发过去
                ↓
   dispatch（分发）：按路由结果，把 token 发给拥有对应专家的卡   ← AllToAll
                ↓
   专家在远端算完
                ↓
   combine（回收）：把结果收回本地                                ← AllToAll
```

**所以"MoE 通信"= 两次 AllToAll（dispatch + combine）**，这也是 EP（专家并行）域的通信主体。

**数据布局的复杂性**（为什么这类算子代码很难读）：
- token 要按 `expert_id` 分组后再发（**permute / 重排**）
- 每个专家收到的 token 数不同 → 变长 → 需要 `counts` / `offsets` 数组
- 返回时要知道"每个 token 从哪些专家收回、按什么顺序拼回去"
- 常做 FP8 量化以省带宽

### 2.1 ★ dispatch / combine 的实际数据流（源码级，2026-09-23 补）

> 来源：`mc2/moe_distribute_dispatch_v2/docs/MoeDistributeDispatch-Combine算子设计介绍.md`（457 行，based on Atlas A2）+ `README.md`。
> 这份设计文档是目前读到的**最有价值的一份材料**，下面是它的干货提炼。

**总体架构：通算融合 + AIV/AICPU 分工**

| 角色 | 干什么 |
|---|---|
| **AIV**（Vector 核）| 索引计算、Token 重排、轮询接收 Flag、后处理、加权求和 |
| **AICPU**（Device 侧）| **直接驱动 RDMA**，摒弃"Host 构造子图 + 调度"的传统流程 → 消除 Host Bound |
| **HCCL** | 提供 `HCCLBUFFER` = `WindowsIn`（收）/ `WindowsOut`（发），算子用 **BatchWrite** 接口通信 |

⚠️ **关键认知**：这两个算子是**"算子 + HCCL window 缓冲"的紧耦合设计**。容错方案必须同时考虑算子侧和 HCCL 侧。

**为什么不用裸 AllToAllV**（设计文档 §1.1.2）：
1. 动态路由下每个 token 的目标专家离散 → 数据分发不均 → 只能用 AllToAllV
2. 传统方案需要**前置 AllGather 收路由表 + Host 侧同步** → 额外通信 + Stream 同步延迟
3. 推理场景 token 小 → Host 驱动下发时延**随 EP 规模线性增长**
4. RDMA 前后同步引入额外 RTT

**Dispatch 的三阶段**

| 阶段 | 内容 |
|---|---|
| **① 索引计算与 Token 重排** | 输入 `expertIds(BS×K)`；计算 `expandIdx(BS×K)` = "token i 是发给专家 expertIds(i,j) 的第几个 token"（全局视角）；目标 rank = `ceil(expertId / localExpertNum)`，局部专家 = `expertId % localExpertNum`；用 `sendStatus` 矩阵（`worldSize × 32`，前 `FLAG_OFFSET=24` 记录发往各卡各专家的 token 数，第 24 位是同步标志，**约束 `localExpertNum ≤ 24`**）计数；**Token 重排（= MoePermute）**：把发往同一 rank 的数据在 GM 中连续存放，使每个目标 rank 只需 1 次下发 |
| **② 数据发送与接收同步** | 用 **BatchWrite**（输入是 GM 指针指向任务结构体数组）。**BatchWrite 无内置同步，每次下发固定开销 1~2 μs → 必须最小化下发次数**。把 `WindowsIn/Out` 各均分为 `worldSize` 个窗口，每窗口 = token 数量数组 + `FLAG1` + tokens + `FLAG2`。**接收同步用"分核双循环轮询"**：把 rank 分配给各核 → 循环 1 轮询 `FLAG1`（确认数量数组到齐）→ 求和定位 `FLAG2` 位置 → 循环 2 轮询 `FLAG2`（确认数据到齐） |
| **③ 发送后处理** | 数据重排（同一专家的 token 在 GM 中连续）；生成 `epRecvCount`（`w×e` 矩阵**转置 + 行主序前缀和**）与 `expertTokenNum`（= `epRecvCount` 最后一列）；用 `Add + GatherMask + Adds` 组合实现，避免标量操作 |

**⭐ 双缓冲机制（设计亮点，防快慢 rank 竞争）**

- **问题**：快 rank 执行 combine 时，慢 rank 的 dispatch 后处理可能还没完 → 写脏对方数据、标志位被踩踏 → **精度异常 + 死锁风险**
- **方案**：`WindowsIn/Out` 各分两块；在 `WindowIn` 第一块末端（**偏移 1 MB 处**）放 `bufferChosen` 标志；Dispatch/Combine 初始化时读它决定用哪块，**结束前翻转 `bufferChosen ^= 1`**
- **正确性论证（很漂亮）**：
  1. 每张卡必须收齐**其他所有卡**发来的通信 Flag 才能结束
  2. 所以：本卡第 N 个 EP 算子未结束时，其他卡的第 N+1 个 EP 算子**必定也没结束**（收不到本卡 N+1 的 Flag）
  3. 其他卡开始第 N+2 个时，本卡第 N 个**必定已结束**
  4. → 第 N 个和第 N+2 个可以安全复用同一块缓冲

> 💡 **这条对容错设计极有启发**：用**缓冲区轮转**替代"每次算子结束做全卡同步"，**在不引入同步点的前提下保证了正确性**。这是"无同步的全卡协同"的一个范例——做容错时同样可以用序号/轮转来避免额外的全局同步。

**Combine 的四阶段**
1. **Token 重排**：基于 Dispatch 传下来的 `sendCounts` 前缀和矩阵（`(i,j)` = 专家 i 发往 rank 0..j 的累计 token 数），用差值确定每个 (专家, 目标 rank) 的 token 数与起始位置
2. **数据发送与接收同步**：同样 BatchWrite + 窗口划分 + 尾部 Flag + 分核轮询
3. **加权求和**：按 topK 权重加权。地址公式：
   `TokenAddr(i,j) = windowInGM + rankSizeOnWin × rank + expertWindowOffset(expertId) × H + expandIdx(i,j) × H`
   （`H` = 单 token 字节数，`rank` = 专家 expertId 归属的卡号）
4. **逆向 AllToAllV**：把处理后的 token 送回原始位置

### 2.2 ★ 缩容（elastic）机制：**代码就绪，但接口未开放**（2026-09-23 重要发现）

**接口本体**【核实】：`elasticInfoOptional`（可选输入）

> 官方 README 原文：**"EP通信域动态缩容信息。当某些通信卡因异常而从通信域中剔除，实际参与通信的卡数可从本参数中获取。"**

**这就是容错/缩容的接口本身。** 关键细节：

| 项 | 内容 |
|---|---|
| shape（示例代码）| `{4 + EP_WORLD_SIZE * 2}` |
| 常量定义 | `ELASTIC_INFO_OFFSET = 4`、`RANK_LIST_NUM = 2`、`EP_WORLD_SIZE_IDX = 1`、`SHARE_RANK_NUM_IDX = 2`、`MOE_NUM_IDX = 3`（`moe_distribute_v2_constant.h:57-63`）|
| **kernel 侧逻辑** | `moe_distribute_elastic.h:54-65`：当 `isScalingDownFlag_` 为真时，从 GM 读取新的 `epWorldSize` / `sharedExpertRankNum` / `moeExpertNum`，并做 **rank 重映射**：`新epRankId = elasticInfo[ELASTIC_INFO_OFFSET + 旧epRankId]`，最后**重算** `moeExpertNumPerRank = moeExpertNum / (epWorldSize - sharedExpertRankNum)` |
| **架构含义** | ⭐ **kernel 只做"按给定映射表适配"，不做任何决策** —— 决策（谁被剔除、新映射是什么）来自 host/框架。**这正是"控制平面 / 数据平面分离"的现成范例**，且那张 **RANK_LIST 就是缩容后的重分片映射表** |

**⚠️ 但是——三个硬件分支的文档都写着"不支持"：**

| 硬件 | 文档原文 |
|---|---|
| Atlas A2 | "不支持 `elasticInfoOptional`"（README:327）|
| Atlas A3 | "**`elasticInfoOptional`：当前版本不支持，传空指针即可**"（README:399）|
| Ascend 950DT | "**`elasticInfoOptional`：当前版本不支持，传空指针即可**"（README:420）|

**→ 结论：elastic 机制在 ops-transformer 里是"kernel 实现完整、aclnn 接口未开放"的状态。**

这是目前**对题目战略价值最高的一条信息**，它有两种可能，**必须问清楚**：
- **可能 A**：机制已经写好，只是还没在接口层开放 → 你的任务变成**"打通并验证"**（工作量小，但仍需补齐"谁触发缩容、新映射怎么算"的控制面）
- **可能 B**：机制有问题/未完成/硬件不支持 → 你的任务是**"重新设计这一层"**（工作量大，但正好对应"参考 NCCL 实现容错算子"）

**✅ 数据布局已完全查清（2026-09-23，四层交叉验证）**

`elasticInfo` 是一个 **int32、1 维、长度 `4 + 2 × epWorldSize`** 的数组，内容分 5 段：

| 下标 | 内容 | 谁在用 |
|---|---|---|
| `[0]` | **isScalingDownFlag**（是否处于缩容状态）| `dispatch_v2.h:376` |
| `[1]` | 新的 `epWorldSize` | `moe_distribute_elastic.h:59` |
| `[2]` | 新的 `sharedExpertRankNum` | `:60` |
| `[3]` | 新的 `moeExpertNum` | `:61` |
| `[4 … 4+N)` | **表 A：我原来的卡号 → 我的新卡号**（N = 原 epWorldSize）| `elastic.h:62`、`arch22/35` 对应行 |
| `[4+N … 4+2N)` | **表 B：我原来要发给的卡号 → 新的目标卡号** | `dispatch_v2.h:676, 769, 1046, 1219`；`arch22/35` 多行 |

→ **所以 `RANK_LIST_NUM = 2` 是对的：有两张表**（不是"布局有两套"）。表 A 解决"**我是谁**"，表 B 解决"**我的数据该发给谁**"。
→ 之前"kernel 用 stride=1 索引"的疑问是**只看了表 A 那几行**造成的误判，表 B 用的是 `ELASTIC_INFO_OFFSET + epWorldSizeOriginal + 卡号`。

**其他已确认的事实**：
- **类型是 `INT32`**（tiling `:476` 校验 `GetDataType() != DT_INT32` 就报错；aclnn 头注释也写 "数据类型int32，必须为1维"）→ **README 表格里标的 FLOAT32 是文档错误**
- **shape 必须正好是 `4 + 2 × epWorldSize`**（tiling `:1235` 校验，不符就报错）
- **代码里唯一明确的限制**：`comm_alg = hierarchy` 时不允许传 elasticInfo（tiling `:916-917` 直接报错 "Cannot support elasticInfo when comm_alg is hierarchy"）
- **infershape 有平台条件**：`infershape.cpp:190` → `if (非A2平台 && elasticInfoShape != nullptr)` 才按新公式算输出 shape

---

## 3. ops-transformer 里的目标算子地图 ★★

**所有 all2all / dispatch / combine 都在 `mc2/` 目录下。**
（`mc2` = Matmul + Communication 的融合算子集合）

### 3.1 Dispatch（分发）类

| 路径 | 说明 |
|---|---|
| `mc2/moe_distribute_dispatch/` | 基线版本（EP 域 AllToAllV），含 torch_extension |
| **`mc2/moe_distribute_dispatch_v2/`** | ⭐**主力版本**，`docs/` 里有设计文档；含 elastic 支持 |
| `mc2/moe_distribute_dispatch_v3/` | 更新版本 |
| `mc2/moe_distribute_dispatch_setup/` | setup 阶段（建链/初始化）|
| `mc2/moe_distribute_dispatch_teardown/` | teardown 阶段（拆链/清理）|
| `mc2/moe_ep_dispatch/` | EP dispatch（仅 op_api/op_host/op_kernel，无 docs/tests）|
| `mc2/moe_ep_dispatch_epilogue/` | 后处理 |

### 3.2 Combine（回收）类

| 路径 | 说明 |
|---|---|
| `mc2/moe_distribute_combine/` | 基线 |
| **`mc2/moe_distribute_combine_v2/`** | ⭐主力 |
| `mc2/moe_distribute_combine_v3/` | 更新版本 |
| `mc2/moe_distribute_combine_setup/` / `_teardown/` | 建链 / 拆链 |
| `mc2/moe_distribute_combine_add_rms_norm/` | 融合 RMSNorm 的变体 |
| `mc2/moe_ep_combine/` / `mc2/moe_ep_combine_epilogue/` | EP combine 及其后处理 |

### 3.3 名字里带 alltoall 的算子（通算融合）

| 路径 |
|---|
| `mc2/allto_all_matmul/`（+ v2） |
| `mc2/matmul_allto_all/` |
| `mc2/allto_allv_grouped_mat_mul/` |
| `mc2/allto_allv_quant_grouped_mat_mul/` |
| `mc2/quant_grouped_mat_mul_allto_allv/` |
| `mc2/grouped_mat_mul_allto_allv/` |
| `mc2/allto_all_all_gather_batch_mat_mul/` |
| `mc2/batch_mat_mul_reduce_scatter_allto_all/` |

> 这些是"通信 + 矩阵乘"融合成一个算子的产物（通算融合），用来省一次显存往返。**它们是 AllToAll 的高级用法，不是你第一轮要读的。**

### 3.4 其他 MoE 通信相关

| 路径 | 说明 |
|---|---|
| `mc2/mega_moe/` | Dispatch + Linear1 + Act + Linear2 + Combine **融合成单算子**（最激进的设计） |
| **`mc2/moe_update_expert/`** | ⭐**EPLB 落地算子**：topK 专家 → 物理卡映射、按阈值剪枝（负载均衡与冗余专家相关） |
| `mc2/distribute_barrier/`（+ `_extend/`） | 分布式 barrier |
| `mc2/attention_to_ffn/`（+ v2）、`mc2/ffn_to_attention/`（+ v2） | 层间数据搬运 |

### 3.5 ★ 容错 / 弹性相关文件（**本任务的直接落点**）

| 文件 | 内容 |
|---|---|
| **`mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_elastic.h`** | ⭐ **容错弹性实现**：`MoeDistributeElastic`、`InitElasticInfo`、`isScalingDownFlag`，**支持 EP 域缩容** |
| `mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_constant.h` | elastic info 的偏移常量：`EP_WORLD_SIZE_IDX`、`SHARE_RANK_NUM_IDX`、`RANK_LIST_NUM` |
| `mc2/moe_distribute_dispatch_v3/op_kernel/arch22|arch35/` | 新版容错路径 |
| `mc2/moe_distribute_dispatch_v2/op_kernel/arch35/moe_distribute_dispatch_v2_a5_full_mesh.h` | A5（arch35）全 mesh 实现 |
| `mc2/moe_update_expert/`（+ `op_kernel/moe_update_expert.h`、`docs/aclnnMoeUpdateExpert.md`） | 专家到物理卡的映射重排（EPLB） |

> 检索关键词：`zeroExpertNum`、`sharedExpertRankNum`、`localMoeExpertNum`（对应"冗余专家"/"零专家"概念）

**⭐ 特殊专家机制 = 冗余的工程化落地**【README 核实】

算子支持三类"特殊专家"，合法 expert ID 依次排在真实专家之后：

| 属性 | 含义 | 合法 ID 区间 |
|---|---|---|
| `zeroExpertNum` | **零专家**（不接收数据，用于占位/负载均衡） | `[moeExpertNum, moeExpertNum + zeroExpertNum)` |
| `copyExpertNum` | **拷贝专家**（很可能就是**冗余专家**机制） | `[moeExpertNum + zeroExpertNum, +copyExpertNum)` |
| `constExpertNum` | 常量专家 | 再下一段 |

> 🎯 `copyExpertNum`（同一专家的多份拷贝）**极可能就是"冗余专家"的工程实现**——这直接对应 `doc/05` §6 讲的算法层容错。**待核实**：copy 专家的数据是否在多个 rank 上冗余部署、故障时如何接管。

**⭐ 其他必须先知道的硬约束**【README 核实】

- `moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0`（整除约束，缩容后要重新满足！）
- **一个模型中该 EP 通信域只允许有 `MoeDistributeDispatchV2` 和 `CombineV2` 这两个算子，不允许有其他算子** —— 这条对"在通信域里加容错逻辑"影响很大
- `xActiveMaskOptional`：表示 **token 是否参与通信**（1D/2D）→ **这是"屏蔽"的接口，但是 token 级、不是 rank 级**
- `HCCL_BUFFSIZE` 必须按公式配置（算子用的是 HCCL 的 window buffer，见 §2.1）
- `assistInfoForCombineOut` / `epRecvCountsOut` / `expandScalesOut` 是**内部元数据，不保证跨版本稳定，业务不应依赖**（README:347）
- 硬件差异：A2 不支持共享专家；A3 单卡双 DIE（"本卡"指单 DIE）；950DT 仅支持 **UB Memory 通信**
- A2 组网：多机仅支持交换机组网，不支持双机直连

---

## 4. 昇腾算子仓库的通用目录约定

以 `mc2/moe_distribute_dispatch_v2/` 为例（**其他算子同构**，学会这个就能自己找）：

| 子目录/文件 | 作用 |
|---|---|
| `op_host/<op>_def.cpp` | 算子原型 / 输入输出 dtype 定义 |
| `op_host/<op>_infershape.cpp` | shape 推导 |
| `op_host/op_tiling/` | tiling 计算（内部还会按 arch 再分） |
| `op_host/config/<soc>/<op>_binary.json` | 各芯片型号的编译配置（`ascend910b`、`ascend910_93`、`ascend950`） |
| `op_kernel/<op>.cpp` | kernel 入口 |
| `op_kernel/arch22/` `arch35/` | **分架构实现**（arch22 ≈ A2/A3，arch35 ≈ A5） |
| `op_kernel/*_tiling.h` | tiling 常量 |
| `op_api/aclnn_*.h/.cpp` | 对外 **aclnn 接口**（框架调用入口） |
| `op_graph/*_proto.h` | GE 图模式原型 + fallback + gen_task |
| `tests/ut/{op_host,op_kernel,op_api}/`、`tests/st/` | 单测 / 系统测试 |
| `examples/test_aclnn_*.cpp` | 调用示例 |
| `docs/`、`README.md` | 接口文档、产品支持矩阵 |

**官方说明**：`docs/zh/install/dir_structure.md`；算子开发指南：`docs/zh/develop/aicore_develop_guide.md`

---

## 5. 建议的阅读顺序

1. ⭐ **`mc2/moe_distribute_dispatch_v2/docs/MoeDistributeDispatch-Combine算子设计介绍.md`** —— **最该先读的一份文档**（dispatch/combine 的设计原理）
2. `docs/README.md` + `docs/QUICKSTART.md` —— 仓库入门
3. `docs/zh/op_list.md` —— 算子全清单（建立全局感）
4. `docs/zh/install/dir_structure.md` —— 目录结构官方说明
5. `mc2/moe_distribute_dispatch_v2/` 逐层读：`op_host` → `op_kernel` → `op_api`
6. `mc2/moe_update_expert/docs/aclnnMoeUpdateExpert.md` —— EPLB（冗余专家/负载均衡，容错的算法层基础）
7. `experimental/mc2/通算融合自定义算子工程开发指南.md`

---

## 6. 待补（TODO）

**已解决（2026-09-23）**
- ✅ dispatch/combine 的数据流 —— 见 §2.1（索引计算 → BatchWrite + 分核轮询 → 后处理；含双缓冲机制）
- ✅ elastic 的触发方式 —— 见 §2.2：kernel **被动读取** host 提供的映射表，不自己做决策

**待办**
- [ ] 🔴 **最高优先**：查清 `elasticInfoOptional` 为什么标"当前版本不支持"（见 `99-问题清单.md` A5）—— 这直接决定题目是"打通"还是"重做"
- [ ] 核实 §2.2 的两个不一致：类型（FLOAT32 vs int32）、`RANK_LIST_NUM=2` 但 stride=1
- [ ] 画一张**完整的 dispatch 数据流图**（含双缓冲、Flag、分核轮询），放进 `doc/07`
- [ ] dispatch v1/v2/v3/V4 的**差异**（docs 目录里有 V2/V3/V4 三份 API 文档，共 5000+ 行）
- [ ] `setup` / `teardown` 两个阶段算子存在的意义（建链/拆链为什么要单独成算子？—— 对容错很关键，可能是重连的落点）
- [ ] `copyExpertNum`（拷贝专家）是不是冗余专家机制？故障时怎么接管？
- [ ] elastic 缩容后**丢失的数据怎么补偿**（丢弃？重路由？上层重算？）
- [ ] elastic 和 HCCL 通信子缩容**如何联动**（EP 域缩容 ≠ 通信域缩容？）
- [ ] `mega_moe` 这种"全融合单算子"对容错意味着什么（出错回滚粒度更大）
- [ ] 与 `code/hccl` 里 `HcclAlltoAll*` 的**调用关系**（这两个算子最终走不走 HCCL 的 alltoall 实现？还是只用 HCCL 的 window + BatchWrite？）
