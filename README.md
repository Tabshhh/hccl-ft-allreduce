# HCCL / FT-AllReduce 研读工作区

昇腾 HCCL 与容错集合通信方向的研读笔记、调研材料与代码走查记录。

**从这里开始： [`doc/00-入门总纲.md`](doc/00-入门总纲.md)** —— 当前目标、概念边界、阶段性产出。
**资源出处全在 [`REFS.md`](REFS.md)** —— 每个仓库/论文/文档的地址、版本、以及踩过的坑。

这个仓库的作用是**在内网和外网两台机器之间同步笔记**，所以它只装文档，不装源码。

## 目录结构

| 路径 | 内容 | 入库 |
|---|---|---|
| `doc/*.md` | 13 篇笔记：HCCL/NCCL 架构、AllToAll 与 EP、容错横评、代码库总览、问题清单等 | ✅ |
| `doc/papers/` | 2 篇 arXiv 论文 PDF | ❌ 用脚本重新下载 |
| `doc/manuals/`、`doc/notes/` | 预留目录 | ✅（空） |
| `REFS.md` | 全部外部资源的出处、版本对齐记录、网络经验 | ✅ |
| `logs/` | 上游仓库的克隆脚本（历史留存） | ✅ |
| `scripts/` | 辅助脚本 | ✅ |
| `code/` | 17 个上游源码仓的本地克隆，约 1.7G | ❌ 见下 |

### 关于 `code/`

`code/` 是 17 个**上游公开仓库**的本地克隆（CANN/HCCL 系列、vLLM、NCCL、Mooncake、DeepEP 等），
合计约 1.7G，**不纳入本仓库**：它们本来就是公开可拉的，体积也远超 git 仓库的合理范围。

需要时按 [`REFS.md`](REFS.md) §1.1 / §1.2 的地址重新克隆。注意两个高频坑：

- **官方仓在 GitCode（`gitcode.com/cann/*`），不在 GitHub**。GitHub 上的同名 `cann-hccl` / `ops-transformer`
  全是个人镜像，不要用。
- 组织名不统一：`mind-cluster` 在 `gitcode.com/Ascend/`，其余 CANN 仓在 `gitcode.com/cann/`。

完整清单和版本对齐记录见 REFS.md 的「易踩的坑」。

## 在新机器上恢复

```bash
git clone https://gitcode.com/Tabs__/hccl-ft-allreduce.git
cd hccl-ft-allreduce
bash scripts/fetch-papers.sh    # 可选：重新下载 2 篇论文 PDF
```

公开仓，clone **不需要任何凭据**。之后按需从 REFS.md 拉源码仓。

### 两个远端

| 远端名 | 地址 | 用途 |
|---|---|---|
| `origin` | `gitcode.com/Tabs__/hccl-ft-allreduce` | **主力**，内外网日常同步都走它 |
| `github` | `github.com/Tabshhh/hccl-ft-allreduce` | 备份镜像，**不参与日常流程** |

选 GitCode 做主力是因为内外网都直连可用、不用代理；GitHub 在外网要挂代理、在内网连接不稳，
不适合承担同步职责。想给 GitHub 那份留备份时手动推一次即可：

```bash
git push github main
```

## 双向同步纪律

内外网两台机器都会改这个仓，规则只有两条：**开工前先拉，收工后即推。**

```bash
git pull --rebase                                  # 开工前
# ... 编辑笔记 ...
git add -A && git commit -m "更新 02-HCCL架构 的 AllReduce 路径" && git push   # 收工后
```

`--rebase` 保持历史线性，有冲突逐条解决。

> **最容易出问题的是「两边都改了同一段却不拉」。** git 不会告诉你内容已经过期，
> 它只会静默地保留其中一份——而且往往是错的那份。所以别攒着，半天一天的推一次。

## 推送凭据（每台机器配置一次）

clone 不需要凭据，**push 需要**。GitCode 用访问令牌认证，配好后 git 自动读取，平时不用手动输入。

**1. 生成令牌**：GitCode → 头像 → 设置 → 访问令牌 → 新建，勾选 `api` 和 `write_repository`。

**2. 存到文件**：

```bash
printf '%s' '你的令牌' > ~/.gitcode_token
```

**3. 装凭据助手**（在本仓库根目录执行，`$PWD` 要展开成绝对路径）：

```bash
git config --global gitcode.username "你的GitCode用户名"
git config --global credential.https://gitcode.com.helper "!$PWD/scripts/gitcode-credential.sh"
```

这样令牌只存在于 `~/.gitcode_token` 一个文件里，**不进 `.git/config`，也不进 remote URL**。

**令牌有有效期。** 过期时 push 会报 401，重新生成一个覆盖 `~/.gitcode_token` 即可，
git 配置不用动。想省事就在生成时把有效期拉到最长；或者改用 SSH 密钥（默认不过期）。

### 外网机器：推 GitHub 备份要挂代理

本机 GitHub 直连时通时不通，本地代理 `127.0.0.1:7897` 是可靠的：

```bash
git -c http.proxy=http://127.0.0.1:7897 push github main
```

或者代理稳定常开的话，配成仅对 GitHub 生效（**不要**设全局 `http.proxy`，代理一关 git 全废）：

```bash
git config --global http.https://github.com.proxy http://127.0.0.1:7897
```

**GitCode 直连即可，不需要代理。**

### 内网机器

GitCode 直连可用，按上面「推送凭据」三步配一次即可，**不需要为 GitHub 做任何配置**。
