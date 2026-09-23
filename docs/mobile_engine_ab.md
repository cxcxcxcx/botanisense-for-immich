# 手机引擎 A/B 影子实例 — runbook（版本通用）

生产（`botanisense` 容器 :8770，BioCLIP-2 teacher）完全不动；本方案在**同一台机器**
跑第二个容器 `botanisense-s0`（:8771），分类引擎换成手机包（`bioclip-mobile-vN`），
检测器换成训练过的 gemini1_seg（6 类 plant/pot/person/hand/flower/tree，960
letterbox），对着一个**影子数据库**重算全库。浏览器开 :8770 与 :8771 两个窗口逐屏
对比，数值差异用 `scripts/ab_compare.py` 量化。

设计依据：`docs/web_model_ab_plan.md`（影子 DB via `DB_PATH`；hybrid kingdom）。
引擎接入面：`mobile_engine.py`（duck-type TreeOfLifeClassifier）；检测后端：
`yolo_gemini1.py`（接口与 `yolo.py` 相同，`YOLO_BACKEND=gemini1` 切换）。

## 布局与关键事实

| | 生产 :8770 | 影子 :8771 |
|---|---|---|
| 容器 | `botanisense`（deploy 目录 `~/open_source/botanisense-fresh`） | `botanisense-s0`（本仓库 `compose.s0.yml`） |
| DB | `data/state.db` | `data/ab/state_vN.db`（影子，**永不回拷**） |
| 引擎 | teacher（ENGINE 未设） | `ENGINE=mobile-vN` |
| 检测 | YOLO-World 开放词表 | `YOLO_BACKEND=gemini1` |
| kingdom 分 | teacher | **hybrid：沿用 teacher**（setup 保留 `rank='kingdom'` 行，Plantae 门/徽章标定不变） |
| person-veto 头 | 启用（teacher 空间） | `PERSON_CLF_PATH` 指向不存在路径 → 优雅禁用 |

- **每个模型版本一个影子库**（`state_v0.db` / `state_v2.db` / …）：版本一变，
  库里的 embeddings/predictions 全部作废——都是 768 维，pipeline 的 dim guard
  **分不出**两个模型。setup 脚本负责清空，换版本时**必须重跑 setup**，否则下一次
  run 会吃到旧版本的缓存（实测：v2 首跑只新算了 25 张、其余全走 v0 缓存）。
- 预处理契约（golden 对拍实锤）：**torchvision `Resize(256, BILINEAR)` + CenterCrop(256)
  + ToTensor(/255)**。manifest 旧文案写 "bicubic" 是笔误（已改 export 脚本与两份交接
  文档）；PIL-bicubic 手写实现会把 embedding cosine 从 ~.99999 拉到 ~.9989。
- gemini1 letterbox = ultralytics `LetterBox((960,960), auto=False)` 逐位复刻
  （cv2 INTER_LINEAR + 114 灰 padding）；PIL resize + 零 padding 会在 conf 阈值附近
  掉 ~15% 检测。golden 门控会抓住这类错误。
- 模型文件在 `data/models/`（gitignored，**宿主侧**——`/app/models` 是只读挂载，
  `docker cp` 进不去）：`bioclip-mobile-vN/`（包）、`gemini1_seg_best.fp16.onnx`、
  `gemini1_golden/`（golden.npz + 8 图）。导出流程见 `mobile-model-export` skill。

## Setup

```bash
# 0) 新版本：先导出 + 放到宿主侧 bind 源（见 mobile-model-export skill）
#    ~/open_source/immich_bioclip/data/models/bioclip-mobile-vN/

# 1) golden 门控（部署时一次，不进启动路径）
.venv/bin/python scripts/verify_mobile_engine.py   # cos mean ≥.999；实测 .999993
.venv/bin/python scripts/verify_yolo_gemini1.py    # 框匹配 ≥95%；实测 0.995 / conf|d| .016

# 2) compose.s0.yml 指向新包 + 新影子库（三个 env 一起改）
#      ENGINE: mobile-vN
#      MOBILE_PACKAGE: /app/models/bioclip-mobile-vN
#      DB_PATH: /app/data/ab/state_vN.db        ← 必须换新文件名

# 3) 影子库：.backup 生产 DB → 清模型侧行（保留 kingdom）→ 写 marker
.venv/bin/python scripts/setup_shadow_ab.py --shadow data/ab/state_vN.db \
    --engine mobile-vN --version bioclip-mobile-vN

# 4) 起容器（改了 Python 才要 --build；只换 env/包用 up -d）
docker compose -f compose.s0.yml up -d
curl -s localhost:8771/api/health | jq .engine    # 必须是新引擎名
```

## 跑

UI（:8771 → Run/Sync）或 API，query 与生产主 run 一致（**样本数也要一致**，
否则可比资产集变小）：

```bash
curl -s -X POST localhost:8771/api/run -H 'content-type: application/json' \
  -d '{"query":"intentional plant photos","n_samples":10000}'
```

速度参考（16 核 CPU）：species ~14-80 img/s（下载+decode 占一部分）、yolo(gemini1@960)
~9 img/s、person 全库回扫 ~9 img/s。全库首跑约 1 小时量级，之后增量 run 只算新资产。
日志在 `data/ab/pipeline-*.log`，应看到 `ENGINE=mobile-vN` / `yolo_gemini1` /
`classifier veto unavailable — disabled`。

## 数值 A/B（`scripts/ab_compare.py`）

```bash
python scripts/ab_compare.py --a data/ab/state_v0.db --b data/ab/state_vN.db
```

只读打开两库，输出：双方 engine/model_version 标记、各表行数、**共有资产上**的
top-1 物种一致率 / 属一致率 / 词表规模 / 变化最大的资产列表。

### 解读（2026-09-22 实测教训）

**不要拿手机引擎跟 :8770 生产库比。** 生产跑的是 teacher，标签空间完全不同
（手机包 12,501 条表 vs teacher 全表），一致率必然接近 0——那是标签空间差异，
不是质量信号。脚本检测到一侧是 teacher 会打警告。

**有意义的对比是 mobile-vN vs mobile-vM**（同一标签表家族，差异只来自模型）。
worked example（v0 → v2，10,000 共有资产）：

| 指标 | 值 | 读法 |
|---|---|---|
| top-1 物种一致 | 36.6% | 63% 的照片判定变了——种级有实质重排 |
| top-1 属一致 | 52.1% | 属级稳得多 → 变化集中在种级细分，不是整体漂移 |
| 物种词表 | 3,012 → 2,527 | 新模型更聚焦（训练数据里常见属更集中） |
| 词表交集 | 1,850 | A 独有 1,162 / B 独有 677 |

movers 列表要人眼扫一遍：很多"变化"其实是**异名/近义**（实测
`Nerium indicum → Nerium oleander`、`Punica protopunica → Punica granatum`），
属一致率已经把这些算作一致了。真正的质量判断要靠抽图人看或下游标注集。

## 人肉 A/B 对比面

- `/`（grid）+ cluster 卡：物种名、簇数、簇名、排序
- `/api/results`、`/api/clusters`、`/api/asset/{id}`（同参数双端对比）
- 检测框（AssetDetail）：gemini1 盒 vs YOLO-World 盒；has_person 翻转
- 数值问题一律用 `ab_compare.py` 或只读 SQL 对齐两库；**不要把影子库写回生产**。

## 版本身份

`album_meta.model_version` = `bioclip-mobile-vN/<code_rev>/<encoder md5[:8]>/<labels md5[:8]>`
（v0 实测 `bioclip-mobile-v0/ec9d011-textnce/6db560ca/labels:3b86f229`）。任何一段变了
→ 影子库作废、重跑 setup。

## 拆除

```bash
docker compose -f compose.s0.yml down        # 容器停掉；生产 :8770 无感
# 影子库归档/删除（不要 merge 回 baseline）：
#   mv data/ab/state_vN.db data/ab/state_vN.db.archived-$(date +%F)
```

镜像 `immich_bioclip-botanisense-s0` 与生产镜像互不影响；不用了 `docker rmi` 即可。
