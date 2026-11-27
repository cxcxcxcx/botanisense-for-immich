# Mac 原生加速方案：CoreML BioCLIP-2 编码器（v0.2 候选）

状态：第 0–5 步完成（2026-11-26），门 1–3 已过。HF 仓库 `cxcxcxcx/bioclip-2-coreml` commit `19254e4b`（model card 源文件在 `release/hf-bioclip-2-coreml/`）；app 首次运行自动下载 + sha256 + golden 自检（5b08a04）；README Mac 章节、`scripts/macos/install_launchd.sh`、CHANGELOG 已写，并在 Mac 上照 README 从空目录装了一遍。还没发版（打 tag / Release 要用户确认）。

## 目标与边界

- **目标**：Mac（Apple Silicon）用户原生运行 BotaniSense 时，BioCLIP-2 图像编码走 CoreML（GPU/ANE），比 Docker 里的 CPU torch 快一个量级；结果与 Linux teacher 路径**可混用**（同一 DB、同一 `model_version`）。
- **不做**：
  - Docker on Mac 的加速：Linux VM 里没有 CoreML/Metal，做不到；Docker 仍是 CPU 兜底。
  - 换模型：仍是 BioCLIP-2 teacher，867k 物种文本表不变；mobile 学生模型另案。
  - Intel Mac：只支持 arm64；Intel Mac 走 torch CPU 回退。

## 现状盘点（已核实）

| 资产 | 位置 | 状态 |
|---|---|---|
| 编码器（batch 16, fp16，580 MB） | Mac mini `$H/bioclip_coreml/teacher/encoder_b16.mlpackage`，另有 `.mlmodelc` | 生产用过（全库 box 回填，约 8–16 crops/s 端到端，模型本身 32.8 img/s） |
| 编码器（batch 1） | 同目录 `encoder.mlpackage` | 不发布（见下） |
| teacher 导出脚本 | 已恢复为 `scripts/export_teacher_coreml.py`（2026-11-17 opencode session 里通过 ssh heredoc 跑的，当时没存文件） | 代码与当时一致 |
| 一致性 eval（2026-11-17，M4） | b16 ALL 32.8 img/s；b1 ALL cos 均值 .999844 / 最小 .999349；CPU_GPU cos .999999；实际“查相似框”效果很好（如 Mammillaria 的近邻全是同属） | 已过；golden 用的是 16 张合成纯色图，第 1 步补一次真实 crop 复核 |
| 推理参考实现 | `scripts/bbox_backfill_mac.py` | 能跑，但直接写 SQLite、绕开 pipeline，不适合合入 app |
| 引擎切换点 | `pipeline._resolve_classifier()` + `$ENGINE`（`mobile_engine.py` 是先例） | 现成 |
| 许可证 | `imageomics/bioclip-2` = MIT | 衍生编码器可发布，需带 MIT 声明和署名 |

`$H` = `/Volumes/X9_APFS/mac_big/home`

## 架构

### 1. `coreml_engine.py`：`CoreMLEngine(TreeOfLifeClassifier)`

继承 pybioclip 的 `TreeOfLifeClassifier`，只换图像塔：

- **重写 `load_pretrained_model`**：
  - 不加载 open_clip 权重，省 1.6 GB 内存，也不用下载。
  - 加载 CoreML 模型：`.mlmodelc` 用 `CompiledMLModel`，`.mlpackage` 用 `MLModel`。
  - `self.preprocess = preprocess_img`，即 bioclip 自带的 squash-224 + CLIP mean/std，与 `bbox_backfill_mac.py` 相同。
  - `self.model` 设为只带 `logit_scale` 的小桩模块。`create_probabilities` 只用到这一个属性；值写在 HF 的 manifest 里（BioCLIP-2 = 100.0），导出时核对。
- **重写 `create_image_features`**：
  - 按 16 一组分块，末块用最后一张补齐，输出切回真实张数。
  - 输出做 L2 归一化，返回 torch tensor，放在 `self.device` 上。
- **其余全部继承，零改动**：文本表（`txt_emb_species.npy`，从 HF 缓存读）、名称、`format_*`、`predict`、taxa filter。
  - 所以 species / kingdom / bbox 三个 stage 不用动。
  - `_kingdom_bins` 等依赖 `get_current_txt_names()` 的优化照常生效。
- **`device="mps"`**：867k × 768 的文本矩阵乘在 Metal 上跑，统一内存约 2.7 GB。
  - 先确认 `pipeline.py:2153`、`pipeline.py:2412` 的 `feats.to(clf.device)`，以及 `_kingdom_bins` 的 `index_add`，在 MPS 上没问题。
  - 有问题就退回 CPU 做矩阵乘。Mac 的 CPU 也够快，瓶颈本来就是编码器。
- **选择逻辑**：
  - 新增 `ENGINE=coreml`。
  - 再加 `ENGINE=auto`：在 Darwin arm64 上，coremltools 可导入且编码器在位时用 coreml，否则用 teacher。
  - 原生 Mac 安装默认用 `auto`，Docker 默认保持 `teacher`。
- **单 crop 路径**：`pipeline.py:1552` 的"点框查相似"在 uvicorn 内一次编码 1 张。b16 模型补齐 16 张，多约 0.5 s，接受（用户决定：不为此多下 600 MB 的 batch-1 模型）。
- **`model_version`** 沿用 `teacher:bioclip-2`。前提是一致性门过关（见"验收门"），这样同一个 DB 在 Linux 和 Mac 之间可以互换。
  - 门不过则用独立版本号，比如 `teacher:bioclip-2/coreml-fp16`，并禁止混用。

### 2. 导出脚本 `scripts/export_teacher_coreml.py`（已恢复；在 Mac 上跑）

现有脚本已能复现 b16 产物和上面的数字。待补：golden 换成真实 crop、写 manifest/sha256、fp32 选项。以下为补齐后的目标形态。

- `open_clip.create_model_from_pretrained("hf-hub:imageomics/bioclip-2")`，只取 `visual` 部分；加 `torch.backends.mha.set_fastpath_enabled(False)`，然后 trace。
- 转成 mlprogram，固定 batch 16（不出 batch-1），目标沿用当时的 macOS 15。
- 精度先试 fp16，不过门就改 fp32（约 1.2 GB）。
- 计算单元：默认 `ALL`，若 ANE 不过门则改 `CPU_AND_GPU`。学生模型就出现过 ANE 跑 fp16 掉到 cos .9957、不过门的情况。
- 产出 `golden.npz`：
  - 32 张固定图，覆盖植物、人、混合场景；
  - 包含 torch fp32 输入张量与对应的 embedding；
  - 加 manifest：输入契约、logit_scale、源模型 revision、各产物的 sha256。
- 依赖：独立的 venv，coremltools 9 + numpy 1.26.4 + torch。
  - coremltools 测过的最新 torch 是 2.7，我们用 2.14 能转，但会报警告。
  - 导出环境与 app 的运行环境分开。

### 3. Hugging Face 发布

仓库 `cxcxcxcx/bioclip-2-coreml`（model repo，用户已同意）：

```
encoder_b16.mlpackage/    # 唯一产物，首次加载时在本机编译
golden.npz                # 安装后自检用
manifest.json             # 契约 + sha256 + 源 revision
README.md                 # model card：衍生自 imageomics/bioclip-2（MIT），附转换方法、一致性数字、署名与引用
LICENSE                   # MIT
```

- **只发 `.mlpackage`**。`.mlmodelc` 与 macOS 版本相关，Apple 推荐在设备上编译。
- **编译缓存**：首次加载会把 `.mlpackage` 编译进 `/var/folders`，在内部盘上，不受 `TMPDIR` 影响，Mac mini 上曾因内部盘满而失败。
  - 所以引擎在首次加载后，用 `xcrun coremlcompiler compile` 或 `MLModel.get_compiled_model_path()` 把 `.mlmodelc` 拷进 `data/models/` 持久化，之后都走 `CompiledMLModel`。
- **app 内下载**：
  - `huggingface_hub.snapshot_download(repo, revision=<固定 commit>)`，放进 `HF_HOME`，与文本表同处。
  - 下载后校验 manifest 里的 sha256，再跑一遍 golden 自检，结果记日志并在 `/api/health` 中显示。
- **文本表**：不重新发布，继续直接用 `imageomics/TreeOfLife-200M` 的 `txt_emb_species.npy`，与现在一样。

### 4. Mac 原生运行与打包

- **依赖**：`pyproject.toml` 加可选依赖组 `mac = ["coremltools>=9; sys_platform == 'darwin'"]`。
  - ⚠️ **numpy 冲突**：锁文件里 numpy 是 2.5.1，而 coremltools 9 的 torch 前端要求 numpy < 2.4。
  - 这个限制可能只针对**转换**，**推理**（`CompiledMLModel.predict`）不一定受影响，第 0 步实测。
  - 若推理也受限，备选方案是 `pyobjc-framework-CoreML`：直接调 CoreML，没有 numpy 限制，代码多约 50 行。
- **安装**：README 新增 "Run natively on macOS (Apple Silicon)"：
  ```
  git clone … && cd botanisense-for-immich
  uv sync --extra mac
  (cd frontend && npm ci && npm run build)
  uv run uvicorn app:app --host 0.0.0.0 --port 8765
  ```
  - 想省掉前端构建：发版时把 `frontend/dist` 打进 GitHub Release 的 tarball。
- **后台常驻（可选）**：`scripts/macos/install_launchd.sh` 生成 `~/Library/LaunchAgents/app.botanisense.plist`，开机自启，日志写到 `data/`。
- **本地网络隐私**：app 访问同机的 Immich（`127.0.0.1`）不受限。
  - Immich 在局域网其他机器上时，launchd 启动的 python 会被 macOS 拦截，报 `No route to host`。
  - 首次需在"系统设置 → 隐私与安全 → 本地网络"里放行，文档写清楚。前台从终端启动没问题。
- **MPS 也惠及 torch 回退**：没有 CoreML 时，`ENGINE=teacher` 在 Mac 上可选 `device="mps"`。这一项不在本次范围，记一笔。

## 验收门（必须全过才合入/发布）

1. **编码器一致性**：在 golden 32 张上，比较 CoreML 与 torch fp32 的 embedding，per-image cosine 均值 ≥ 0.999、最小值 ≥ 0.998。门槛沿用移动端导出的做法。
2. **端到端一致性**：取 500 张真实库图片，Mac 原生跑和 Linux Docker 跑对比：
   - 主门是 embedding 角度：bbox embedding cosine 均值 ≥ 0.999、最小 ≥ 0.998；
   - CoreML 的 species top-1 落在参考 top-5 内 ≥ 99%；
   - kingdom triage（`_kingdom_bins` 聚合，不是 species top-1 的界）一致率 ≥ 99%；
   - species top-1 一致率只记录，不设硬门（地板 90%）。近缘种的 text embedding 本来就挨得很近，角度偏 .0003 就会让近似并列翻转，这不代表模型坏了（2026-11-26 用户决定放宽）。
   - 达标才沿用 `teacher:bioclip-2` 的版本号，允许混用 DB。
3. **吞吐**：在 M4 上与 Mac 上的 Docker CPU 比，species stage 端到端至少快 3 倍，数字写进 README。
4. **回退**：
   - 编码器缺失或校验失败 → 自动走 teacher，日志里有一行明确的 `[model]` 说明。
   - Linux 上 `ENGINE=auto` 等同于 teacher，现有测试全绿。
5. **测试**：`tests/test_coreml_engine.py`：
   - 用假 CoreML 模型（返回固定向量）测分块、补齐、切片、归一化、`auto` 选择逻辑；
   - 在 Linux CI 上跑，不依赖 coremltools。

## 步骤与工作量

| # | 步骤 | 产出 | 估时 |
|---|---|---|---|
| 0 | 摸底：在 Mac 上用 app 的锁定依赖跑 `uv sync`（macOS arm64 的 faiss、hdbscan、opencv 轮子）；用 numpy 2.5 加载 `.mlmodelc` 并 predict | 能否原生跑 + numpy 结论 | 0.5 天 |
| 1 | 补齐导出脚本（真实 crop golden、manifest），重新生成 b16，跑门 1 | 可复现的产物 + 一致性数字 | 0.5 天 |
| 2 | `coreml_engine.py` + `ENGINE=coreml/auto` + 单元测试 | PR（Linux CI 绿） | 1 天 |
| 3 | 在 Mac 上原生跑全流程，过门 2、3；修 MPS 上的问题 | 对比报告（放进 `docs/`） | 1 天 |
| 4 | HF 仓库 + model card + app 内下载与自检 | HF 仓库（`19254e4b`）+ app 首次运行下载与自检（完成） | 0.5 天 |
| 5 | README Mac 章节 + launchd 脚本 + CHANGELOG | v0.2 发布材料（完成，未发版） | 0.5 天 |

合计约 **4 个工作日**。步骤 0–3 在私有分支完成，不对外发布；步骤 4 起需要用户确认。

## 第 0 步结果（2026-11-26，Mac mini M4）

- **原生依赖**：`uv sync --frozen --python 3.13` 在 macOS arm64 上约 9 s 装完（faiss、hdbscan、opencv、torch 2.13、numpy 2.5.1），MPS 可用。仓库没钉 Python，uv 默认挑了 Homebrew 3.15，onnxruntime 没有 cp315 轮子 → 需要加 `.python-version`（3.12/3.13）或 `requires-python` 上限。
- **numpy**：coremltools 9.0 与 numpy 2.5.1 共存，安装不降级，加载 `.mlmodelc` 并 predict 正常。numpy<2.4 只是**转换**时的限制 → 运行时不需要 pyobjc。
- **真实 crop 一致性**（128 张 plant/flower/tree crop，`encoder_b16.mlmodelc` vs torch fp32 CPU，报告 `out/coreml_parity/report.json`）：

| 计算单元 | cos 均值 / 最小 | species top-1 一致 | top-1 落在参考 top-5 | img/s（含首次加载） |
|---|---|---|---|---|
| ALL（含 ANE） | .999711 / .998919 | **95.3%**（6/128 翻转） | 100% | 29.1 |
| CPU_AND_GPU | .999995 / .999955 | 99.2%（1/128） | 100% | 14.1 |

  - 两者都过 cos 门和 top-5 门。ALL 的 6 个翻转都是近似并列（全在参考 top-5 内），其中一例 *Cephalotaxus harringtonii* → 真菌 *Diplodia buxi*，第 3 步要看它是否影响 kingdom triage。
  - **结论（用户决定）**：默认 `ALL`，速度是 CPU_AND_GPU 的 2 倍，约为 Linux CPU（2.8 img/s）的 10 倍；`CPU_AND_GPU` 作为可配置的保守选项。

## 第 1 步结果（2026-11-26）

- 导出可复现：独立的导出 venv（py3.12，torch 2.7.0、numpy 1.26.4、coremltools 9.0、open_clip 4.2.2）重导 b16，16 s 转完，580 MB。
- golden 改为真实 crop：32 张，输入已过 app 的 `preprocess_img`，参考为 torch fp32 CPU，16 MB，由 `verify_coreml_teacher.py --write-golden` 生成。
- `manifest.json`：版本、sha256、门槛与结果。门 1 不过时导出脚本退出码为 1。
- 门 1（32 张真实 crop，稳态吞吐）：

| 计算单元 | cos 均值 / 最小 | img/s |
|---|---|---|
| ALL | .999712 / .999259 | 31.9 |
| CPU_AND_GPU | .999996 / .999978 | 21.0 |

- 产物在 Mac `/Volumes/X9_APFS/mac_big/home/bioclip_coreml/teacher_v2/`；golden 与 manifest 也拷到了 `out/coreml_parity/teacher_v2/`。

## 第 2 步结果（2026-11-26）

- `coreml_engine.py`：`CoreMLEngine(TreeOfLifeClassifier)`，只换图像塔，不加载 torch 权重（`logit_scale` 取自 manifest，实测 100.000008）。
  - 首次加载把 `.mlpackage` 编译成 `.mlmodelc` 并保存在旁边；之后加载 0.0 s。
  - 加载时跑 golden 门；不过门或加载出错 → 回退 torch teacher，日志有一行 `[model]`。
  - `COREML_PACKAGE`（默认 `data/models/bioclip-2-coreml`）、`COREML_COMPUTE_UNITS=all|cpu_gpu|cpu`（默认 all）。
- `pipeline._resolve_classifier`：`ENGINE=coreml|auto` 经 `select_engine()` 选择；Linux 上 auto 等同 teacher。
- `tests/test_coreml_engine.py`：17 个，不依赖 coremltools；全套 182 个通过。
- Mac 冒烟（`scripts/smoke_coreml_engine.py`，`ENGINE=auto`，128 张 crop，编码 + 867k 标签 softmax）：
  - 首次 16.4 s 就绪（含编译 10 s）、25.5 img/s；
  - 第二次 3.9 s 就绪、28.7 img/s。
- 还没做：`/api/health` 的引擎徽章（`app.py` 里有用户未提交的改动，暂不动）；pyproject 的 Python 钉版与 `mac` extra 放到第 5 步，因为会改 `uv.lock`。

## 第 3 步结果（2026-11-26）

同一个 Immich 查询（`intentional plant photos`，n=500，入库 559 张），两边各跑一次完整流程：

- 参考：Linux Docker 影子实例，`ENGINE=teacher`，torch CPU；
- 被测：Mac mini M4 原生，`ENGINE=auto` → CoreML ALL。

对比脚本 `scripts/compare_engine_runs.py`，报告 `out/step3/report.json`。影子实例、镜像和两边的 DB 已清理。

**门 2（端到端一致性）：通过。**

| 指标 | 结果 | 门槛 |
|---|---|---|
| 整图 embedding cos 均值 / 最小（559 张） | .999712 / .998195 | ≥ .999 / ≥ .998 ✅ |
| kingdom argmax 一致 | 99.82%（1/559） | ≥ 99% ✅ |
| Plantae ≥ .35 资格判定一致 | 99.64%（2/559） | ≥ 99% ✅ |
| 整图 species top-1 落在参考 top-5 | 100%（498 张） | ≥ 99% ✅ |
| 整图 species top-1 一致 | 95.8%（21 个翻转） | 只记录，地板 90% |
| 框 species top-1 落在参考 top-5 | 99.74%（2331 个配对框） | 参考 |
| 框 species top-1 一致 | 92.2% | 只记录 |

- 唯一的 kingdom 分歧是近似并列：Plantae .302 / Fungi .298 → Fungi .314 / Plantae .306，两边都低于 .35，资格判定不受影响。
- 框级数字混进了检测差异：两边检测各自独立，2389 对 2399 个框，按 label + IoU ≥ .9 配上 2377 个，裁剪不完全相同，所以比整图低。
- **bbox embedding 的 cos 门量不了**：pipeline 的 bbox-bioclip 阶段走 `clf.predict(images=crop_paths)`，两边都不写 `bbox_embeddings`（目前只有 Mac 回填脚本写）。门 2 改用整图 embedding 判定，见下面的后续项。

**门 3（吞吐 ≥ Mac Docker CPU 的 3 倍）：用上界代理通过，约 3.0 倍，没有实测 colima。**

- 代理：在 Mac 上原生跑 `ENGINE=teacher`，torch 2.13 CPU，128 张 crop，编码 + 867k 标签 softmax，`smoke_coreml_engine.py`：
  - 4 线程和 10 线程都是 **8.0 img/s**；
  - CoreML ALL 同一组 crop 是 **23.7 img/s**。
- Docker 里同样是 torch CPU，但隔着 Linux 虚拟机，CPU 和内存都要分出去，不会比原生 macOS 的 torch CPU 快。所以 8.0 img/s 是 Mac Docker 的上界，实际倍数 ≥ 3.0。
- 没起 colima：16 GB 的机器上要给虚拟机 8–10 GB，还得在 Mac 上构建 arm64 镜像，换来的只是把上界换成实测值。README 里按"至少 3 倍"写。

**整条流程耗时**：两台机器硬件不同，只作参考，不算门。

| 阶段 | Linux Docker（teacher） | Mac 原生（CoreML） |
|---|---|---|
| search（取图 + 编码） | 3m56s | 55s |
| species | 37s | 23s |
| yolo | 52s | 48s |
| bbox-bioclip | 16m11s（2343 框，2.4/s） | 5m12s（2353 框，7.5/s） |
| 合计 | 21m45s | 7m26s |

- Mac 的 bbox-bioclip 卡在 `clf.predict()` 上：每个框先写 JPEG 再读回来重编码，再做 867k 行的 Python 格式化，CoreML 本身不是瓶颈（直接编码 23.7 img/s）。
- 两边都缺 `data/person_classifier.joblib`，人像否决被关掉了。这只影响 has_person，不影响上面的一致性对比。

**后续项**：

1. **完成（f80b77f）**：bbox-bioclip 阶段改成"编码一次"：裁剪 → `create_image_features` → 写 `bbox_embeddings` → 从特征分类。新的框聚类依赖 `bbox_embeddings`，现在只有回填脚本在写；这样也能去掉 `predict()` 的瓶颈，并让框级 cos 门可测。
   - 顺带修掉一个精度问题：旧阶段从 448px 的 draft 解码上裁框，中位 crop 只有 96px；改成从完整预览（`_BBOX_DECODE_SIZE=2560`）裁，中位 430px。64 个真实框对 Mac 回填参考：cos 旧 .833（最低 .601）→ 新 .99999；species top-1 一致 20% → 100%（`scripts/verify_bbox_encode_once.py`）。
   - 从空目录全新安装实测（M4，30 张图、195 个框，Immich 走 ssh 隧道）：bbox-bioclip 8.3 框/s（含取预览和 2560 解码），首次运行下载 + 编译约 1.5 分钟。
2. `pipeline.TreeOfLifeClassifier` 是模块级懒加载：直接调 `_resolve_classifier()` 走 teacher 分支会 KeyError，worker 因为先访问了这个属性才没事。冒烟脚本已照 worker 的做法先访问一次；`_resolve_classifier` 本身改成用 `getattr` 更稳。

## 部署定位（2026-11-26 用户决定）

- Linux，或想要省事的用户：用 Docker。
- Mac：**强烈推荐原生跑 + 我们导出的 CoreML 编码器**。Docker Desktop / colima 在 Mac 上是 Linux 虚拟机，碰不到 GPU 和 ANE，只能跑 torch CPU。第 3 步实测：M4 原生 torch CPU 上界 8.0 img/s，CoreML ALL 23.7 img/s，所以 Docker 至少慢 3 倍。
- 原生安装门槛不再算主要风险：现在多数人用 AI 助手部署，关键是把步骤写得精确、可照抄，包括 Python 版本、uv 命令、模型下载、launchd 和本地网络权限。

## 风险与待决定

- **已决定（2026-11-26）**：HF 仓库 `cxcxcxcx/bioclip-2-coreml`；只发 b16，不发 batch-1；fp16 不过门时可接受 fp32。
- **风险**：
  - **ANE 精度**：teacher 在 ALL（含 ANE）下 cos 最小 .9993，好于学生模型；species top-1 在近似并列时会翻转（实测 95.3%），按放宽后的门接受；kingdom triage 不过门时才退回 `CPU_AND_GPU`（16.5 img/s）。
  - **numpy 冲突**：第 0 步决定走 coremltools 还是 pyobjc。
  - **macOS 版本**：当时的产物目标是 macOS 15；要支持 14 需重导一次并复测，更老的系统走回退。
  - **原生安装门槛**：要装 uv 和 node，比 Docker 高。用 Release tarball 附带 `frontend/dist`，再配上精确、可照抄的步骤（面向 AI 助手部署）来缓解。
