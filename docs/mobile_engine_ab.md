# 本机 S0 引擎 A/B 实例（BioCLIP-Mobile v0 + gemini1 检测）— runbook

生产（`botanisense` 容器 :8770，BioCLIP-2 teacher）完全不动；本方案在**同一台机器**
跑第二个容器 `botanisense-s0`（:8771），分类引擎换成 S0 手机包
（`bioclip-mobile-v0`，23.6MB ONNX，768 维），检测器换成训练过的 gemini1_seg
（6 类 plant/pot/person/hand/flower/tree，960 letterbox），对着一个**影子数据库**
重算全库。浏览器开 :8770 与 :8771 两个窗口逐屏对比。

设计依据：`docs/web_model_ab_plan.md`（影子 DB via `DB_PATH`；hybrid kingdom）。
引擎接入面：`mobile_engine.py`（duck-type TreeOfLifeClassifier）；检测后端：
`yolo_gemini1.py`（接口与 `yolo.py` 相同，`YOLO_BACKEND=gemini1` 切换）。

## 布局与关键事实

| | 生产 :8770 | S0 实例 :8771 |
|---|---|---|
| 容器 | `botanisense`（deploy 目录 `~/open_source/botanisense-fresh`） | `botanisense-s0`（本仓库 `compose.s0.yml`） |
| DB | `data/state.db` | `data/ab/state.db`（影子，**永不回拷**） |
| 引擎 | teacher（ENGINE 未设） | `ENGINE=mobile-v0` |
| 检测 | YOLO-World 开放词表 | `YOLO_BACKEND=gemini1` |
| kingdom 分 | teacher | **hybrid：沿用 teacher**（setup 保留 `rank='kingdom'` 行，Plantae 门/徽章标定不变） |
| person-veto 头 | 启用（teacher 空间） | `PERSON_CLF_PATH` 指向不存在路径 → 优雅禁用 |

- **embeddings 必须清空**：teacher 与 S0 都是 768 维，pipeline 的 dim guard 分不出
  两个模型——setup 脚本负责，别跳过。
- 预处理契约（golden 对拍实锤）：**torchvision `Resize(256, BILINEAR)` + CenterCrop(256)
  + ToTensor(/255)**。manifest 旧文案写 "bicubic" 是笔误（已改 export 脚本与两份交接
  文档）；PIL-bicubic 手写实现会把 embedding cosine 从 ~.99999 拉到 ~.9989。
- gemini1 letterbox = ultralytics `LetterBox((960,960), auto=False)` 逐位复刻
  （cv2 INTER_LINEAR + 114 灰 padding）；PIL resize + 零 padding 会在 conf 阈值附近
  掉 ~15% 检测。golden 门控会抓住这类错误。
- 模型文件在 `data/models/`（gitignored）：`bioclip-mobile-v0/`（包）、
  `gemini1_seg_best.fp16.onnx`、`gemini1_golden/`（golden.npz + 8 图）。

## Setup（一次性）

```bash
# 1) golden 门控（部署时一次，不进启动路径）
.venv/bin/python scripts/verify_mobile_engine.py   # cos mean ≥.999；实测 .999993
.venv/bin/python scripts/verify_yolo_gemini1.py    # 框匹配 ≥95%；实测 0.995 / conf|d| .016

# 2) 影子库：.backup 生产 DB → 清模型侧行（保留 kingdom）→ 写 marker
.venv/bin/python scripts/setup_shadow_ab.py

# 3) 起容器（独立 image；build 会跑前端 tsc+vite）
docker compose -f compose.s0.yml up --build -d
curl -s localhost:8771/api/health | jq .engine   # {"id":"mobile-v0","yolo_backend":"gemini1",…}
```

## 跑

UI（:8771 → Run/Sync）或 API，query 与生产主 run 一致：

```bash
curl -s -X POST localhost:8771/api/run -H 'content-type: application/json' \
  -d '{"query":"intentional plant photos","n_samples":30000}'
```

速度参考（16 核 CPU）：species ~14-80 img/s（下载+decode 占一部分）、yolo(gemini1@960)
~9 img/s、person 全库回扫 ~9 img/s。全库 21.8K 资产首跑约 1 小时量级，之后增量 run
只算新资产。日志在 `data/ab/pipeline-*.log`，应看到
`ENGINE=mobile-v0` / `yolo_gemini1` / `classifier veto unavailable — disabled`。

## 对比面（人肉 A/B）

- `/`（grid）+ cluster 卡：物种名、簇数、簇名、排序
- `/api/results`、`/api/clusters`、`/api/asset/{id}`（同参数双端对比）
- 检测框（AssetDetail）：gemini1 盒 vs YOLO-World 盒；has_person 翻转
- 数值一致性问题用只读 SQL 对齐两库（`data/state.db` vs `data/ab/state.db`）；
  不要把影子库数据写回生产。

## 版本身份

`album_meta` 里 `model_version` =
`bioclip-mobile-v0/ec9d011-textnce/6db560ca/labels:3b86f229`（package/code_rev/
encoder md5[:8]/labels md5[:8]）。任何一段变了 → 影子库作废重跑 setup。

## 拆除

```bash
docker compose -f compose.s0.yml down        # 容器停掉；生产 :8770 无感
# 影子库归档/删除（不要 merge 回 baseline）：
#   mv data/ab/state.db data/ab/state.db.archived-$(date +%F)
```

镜像 `immich_bioclip-botanisense-s0` 与生产镜像互不影响；不用了
`docker rmi` 即可。
