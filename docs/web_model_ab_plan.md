# Web 端双引擎 A/B 方案 — BioCLIP-2 teacher vs BioCLIP-Mobile

结论先行：**推荐 (a) 影子 DB（`DB_PATH` 切换），不做 (b) `model_version` 列**。
A/B 期间 baseline（`data/state.db`）只读冻结，模型侧数据在新文件里重算，两边跑同一套查询做对比。
引擎接入点收敛为 pipeline 里一个 duck-type 面（`create_image_features` + 特征分类），
不改任何存储不变量（encode-once、不落原图、lazy load、模型不共存）。

## 1. 两条路线对比

代码事实：`app.py:81` `DB_PATH = os.environ.get("DB_PATH", "data/state.db")`；
所有读写从 `storage.connect(DB_PATH)` 拿连接；run 子进程经 `PIPELINE_WORKER_ARGS`
拿 `db_path`（`app.py:800`）；派生文件全部挂在 `Path(DB_PATH).parent`
（`app.py:86` near-dup pkl、`app.py:274/793` 日志与 status）。
**换一个 env var = 整个 app（server + worker + 派生文件）整体切库，零代码改动。**

| | (a) 影子 DB | (b) `model_version` 列 |
|---|---|---|
| schema 改动 | 0 列（见 §2 的 1 个 ALTER，可选） | `embeddings(asset_id)`、`predictions(asset_id,rank,position)`、`taxon_scores(asset_id,rank,taxon_name)`、`bbox_predictions(detection_id,rank,position)` 四张表 PK 全部要加一列 |
| 无法打版本的旁路状态 | 无（两套库各自独立） | `assets.cluster_id`（单列，装不下两套聚类）、`clusterer_state`（`CHECK(id=1)` 单行 pickle）、`cluster_names`/`cluster_albums`（PK=cluster_id）、`near_dup_cache.pkl`（单文件） |
| 读路径改动 | 0 | `storage.get_all_results`（~350 行 SQL）、`_plantae_score_expr`、`clusters.py` refit/assign、`app.py` 的 `_results_cache`/`_clusters_cache` key 全部要带版本，否则跨版本串缓存 |
| 同库共存两套模型 | 不支持（不需要） | 支持（A/B 用不上） |
| 风险 | 近零：baseline 物理上不可被写坏 | 大：漏改一处 reader 就静默混版本 |

`knn_predictions` 的 PK `(asset_id, index_version, k)` 是库里唯一的版本化缓存
先例——它证明了版本化可行，也标出了代价：独立建表 + 自己的 fetch/have/list/enrich
五件套。A/B 只需要"两个可对比的快照"，不需要"一个库里同时住两套"，(b) 是纯成本。
若将来真要同库共存（例如按资产灰度），再按 (b) 迁移，路径见 §6。

## 2. 推荐路径的 schema / 迁移

沿用 `storage.init_db()` 里 `ALTER TABLE ... ADD COLUMN` + 捕获
`duplicate column` 的幂等写法（`storage.py:301-321`），只加一处：

```python
# runs 表加引擎标记：让每次 A/B run 自证用的是哪个引擎，不靠人记
("ALTER TABLE runs ADD COLUMN engine TEXT NOT NULL DEFAULT ''", "engine"),
```

老行保持 `''`（teacher 时代，中性值）。版本串取 manifest 的身份键：
`package + code_rev + encoder.md5[:8]`，并**必须带上 labels 的 md5**
（`labels.f16.npy` / `labels.json` 决定了 `predictions.species` 字符串与分数
标定，表一换旧行即不可比）。落到已有的 KV 表，不建新表：

```sql
INSERT INTO album_meta(key, value_json) VALUES ('model_version', :json);
-- json = {"package":"bioclip-mobile-s2-textnce","code_rev":"...","encoder_md5":"...",
--         "labels_md5":"...","dim":768,"input":"256x256 /255 no-meanstd"}
```

然后**清空模型侧数据**（copy 自 baseline 后必须做，否则 pipeline 的显式缓存
检查会全部命中、一次都不重算）：

```sql
DELETE FROM embeddings;  DELETE FROM predictions;  DELETE FROM taxon_scores;
DELETE FROM bbox_predictions;  DELETE FROM cluster_names;
DELETE FROM clusterer_state;  DELETE FROM cluster_remap;
UPDATE assets SET cluster_id = NULL;
-- hybrid 变体：保留 rank='kingdom' 的 taxon_scores（见 §5），则改为
-- DELETE FROM taxon_scores WHERE rank <> 'kingdom';
```

外加 `rm <shadow_dir>/near_dup_cache.pkl`（路径随 `DB_PATH.parent` 派生）。

## 3. 服务端 encoder 接入点

- **位置**：`pipeline.py` 与 clf 的全部交互已经收敛成一个小面——
  `create_image_features(imgs, normalize=True)`、
  `_classify_features(clf, feats, …)`（内部只用 `create_probabilities` /
  `get_txt_embeddings` / `format_species_probs` / `format_grouped_probs`）、
  `_kingdom_bins(clf)`（用 `get_current_txt_names()`）、`_unload_classifier()`。
  写一个 `MobileEngine` 类实现同一组方法（文本表 = `labels.f16.npy`，
  名称 = `labels.json` 的 genus/species/common），在 `_resolve_classifier()`
  旁边加一个引擎选择器，各 stage（`species/kingdom/bbox-bioclip`）无需改动。
- **预处理**：复用 `pipeline._decode_preview()`（draft 解码），再按 manifest
  做 resize 短边 256 → center crop → /255（无 mean/std）。
- **不变量**：lazy —— ort 与 session 都在引擎类内部首用时加载，uvicorn 进程
  永不 import ort；encode-once —— session 输出 emb → `upsert_embedding`，
  分类走特征 × 文本表，不二次前向；不落原图不变；模型不共存 —— 引擎在
  run 开始时二选一（env，只在 worker 进程读），`_unload_classifier()` 语义
  保留，teacher 的 6 GB 与 ort 会话绝不同时驻留。
- **本机实测**（`.venv` 里 onnxruntime 1.30.0 已装——是传递依赖，
  `pyproject.toml` 未声明，正式用要显式加；16 核 CPU，`encoder.fp16.onnx`）：

  | package | 大小 | session 加载 | 吞吐 | 峰值 RSS |
  |---|---|---|---|---|
  | bioclip-mobile-v0 | 23.6 MB | 0.2 s | 78 img/s (batch 1) / 55 (b8) / 40 (b32)；预处理 5 ms/img | 1.19 GB |
  | bioclip-mobile-s2-textnce | 72 MB | 0.3 s | 13 img/s (b64) | 1.82 GB |
  | teacher 对照 | — | ~9 s | 2.6 img/s | 5.85 GB |

  RSS 拆解：ort+numpy+PIL import 52 MB，+session 307 MB，其余是 batch arena
  （减 batch 或关 arena 可再压）。估算方法即此脚本：固定
  `intra_op_num_threads=8`，跑 32/64 张 `botanisense_modeling/mobile_bioclip/
  data/bioclip_mobile_pilot/eval/`（24 类 ×30 张）图，读 `ru_maxrss`。
  附带一次接线自检（argmax vs 目录名，与 G1 协议不同，只验通路不代表质量）：
  v0 top-1 0.27 / S2 0.34；`golden.npz` 8 张 `max|Δemb|` 才是出货门控。

## 4. A/B runbook

1. **冻结 baseline**：`sqlite3 data/state.db ".backup data/state.db.baseline-YYYYMMDD"`
   （不要 `cp`，WAL 热拷会缺 `-wal`）。记 md5，此后 baseline 只读。
2. **建影子**：`mkdir -p data/ab`（独立目录是硬要求：`near_dup_cache.pkl`、
   `pipeline.log`、`pipeline-<job>.log/status.json` 全部按 `DB_PATH.parent`
   派生，混在 `data/` 会互相覆盖）；`.backup` 到 `data/ab/state.db`；执行 §2
   的 marker + 清空；直接写 SQL 后**重启 uvicorn**（见 DEVELOPMENT.md，
   `--reload-exclude=data/` 不会帮你重载）。
3. **同资产集**：copy 自带同一批 `assets`（20,323）与 `search_results`
   （294,871，Immich 检索序），因此**搜索排序天然对齐**，差异只来自
   predictions / clusters。用相同 query + 足量 `n_samples` 跑一次
   mobile run（同库不会并发，409 保护不变）。
4. **切流量**：`DB_PATH=data/ab/state.db`（+ 引擎 env）起新进程/第二容器；
   生产的 `botanisense`（:8770，deploy 目录 `~/open_source/botanisense-fresh`）
   照旧跑 teacher。不要用宿主机 uvicorn。
5. **对比面**：`/api/results`（grid + taxonomy 汇总，同样 query params）、
   `/api/clusters`（簇数、簇名、簇大小）、`/api/asset/{id}`（top-K 与
   kingdom 分）。物种 top-1 的弱真值三选一：(i) 离线 G1 24 类评测（引擎级，
   不经过 app）；(ii) 与 teacher top-1 的一致率（一致性 ≠ 正确率，只测漂移）；
   (iii) 内部 42 条人工判定（has_person；未随仓库发布）。
6. **对比脚本 outline**（`scripts/ab_compare.py`，只读）：
   两库各开 `file:...?mode=ro` 连接 → 按 `asset_id` 对齐 → 输出表：
   top-1/genus/family 一致率、Plantae 分差与绿/黄档翻转数、
   `cluster_id` 混淆矩阵（规模 + 成对 Jaccard）、`/api/results` 顺序的
   Spearman、每资产 top-1 分数分布（看标定漂移，不是只看 argmax）。

## 5. 风险与备注

- **精度**：24 类闭集 top-1 .831→.603/.619/.683（S0/S1/S2）；94 种开集
  zero-shot .621→.389（S2）。genus/family/kingdom 靠 logsumexp bin 更接近。
  预期：网格卡片名、cluster 名、物种筛选计数都会移动——这是要测的信号，不是 bug。
- **Plantae 门**：`storage._plantae_score_expr` 只认
  `taxon_scores(rank='kingdom', taxon_name='Plantae')`；mobile 标签表 12,501
  行里只有 80 Animalia + 87 Archaeplastida 负类（teacher 是 784k 全域），
  绿/黄档与非植物过滤会重新标定。便宜解法是 §2 的 hybrid： kingdom 分继续
  用 teacher（保留 kingdom 行即可，kingdom stage 见有缓存会跳过），物种层换
  mobile。
- **person-veto**：`pipeline._person_clf_p` 直接吃持久化 embedding 的
  logistic 头（`PERSON_VETO_P=0.05`）。mobile 向量与 teacher 同空间但标定
  未验证，需用那 42 条判定复测一次再信。
- **标签表耦合**：版本键必须含 `labels.*.md5`；换表后旧 `predictions.species`
  字符串不再可比，A/B 期间不换表。
- **容量 / PWA**：15K 资产 × 768 × 4 B ≈ 46 MB，相对现有 507 MB 的 state.db
  （20,323 assets / 19,864 embeddings / 75,180 predictions / 51,160
  bbox_predictions / 294,871 search_results）可忽略，影子库翻倍也无碍；
  前端对引擎零感知，纯服务端切换 → 无需 rebuild、无 SW 缓存影响。
- **永不合并回写**：影子库会覆盖 cluster_id / cluster_albums / remap，
  绝不能 `cp` 回 baseline；A/B 结束影子库直接归档删除。
- **包不进 repo**：`bioclip_mobile_packages/*` 45–305 MB，挂卷进容器；
  `onnxruntime` 需显式进 `pyproject.toml`（CPU extra，镜像约 +几十 MB）。

## 6. 何时升级到 (b)

只有当需要"同一库里、按资产同时存在两套模型结果"（灰度、逐资产回填对比）
时才做 (b)：上述四张表 PK 加 `model_version TEXT NOT NULL DEFAULT ''`（幂等
ALTER 同 §2 写法）；`assets.cluster_id` 改独立表
`asset_clusters(model_version, asset_id, cluster_id)`；`clusterer_state`
去掉 `CHECK(id=1)`、`cluster_names` / `cluster_albums` 同样加列；最后给
`get_all_results` / `_plantae_score_expr` / 两个 in-memory cache 的 key
全部补版本参数。
