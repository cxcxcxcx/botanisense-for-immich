# 接入 BotaniSense：账号登录 + Firestore 直连

> **BotaniSense** = Firebase 版官方 App（部署目录
> `/media/4T_Ext/chenxing/deployment/botaniq`，旧代码名 **BotaniQ**，
> Firebase project `plantsun-7b88e`）。
> 本仓库 = **BotaniSense for Immich**（self-host，FastAPI + React + SQLite）。

## 0. 结论

**让 BotaniSense 当身份提供方（IdP），self-host 登录成用户本人的账号。**

不是影子用户，不是服务账号，也不是让实例收集密码：

1. self-host 点“用 BotaniSense 登录”；
2. 用户在 `botanisense.app` 完成登录（Google / Apple / Email 都行；
   这是授权域名，不受 self-host 域名限制）；
3. BotaniSense 后端确认用户身份后，签发一个
   **`uid = 用户真实 uid` 的 Firebase custom token**；
4. self-host 前端 `signInWithCustomToken(auth, customToken)`；
5. 现在 `auth.currentUser.uid` 就是用户本人，**可以直接 query Firestore**，
   现有 owner rules 原样放行，不需要改读规则；
6. 写日记仍然建议走 `api.botanisense.app` 的导入接口（用同一个 Firebase
   ID token 鉴权），由服务端维护 `entryCount` / 封面 / 物种连接。

```
┌─────────────────────────────┐
│ self-host                   │
│  “用 BotaniSense 登录”       │
└──────────────┬──────────────┘
               │ 设备码 / OAuth
               ▼
┌─────────────────────────────┐
│ botanisense.app             │
│  Firebase 登录（Google/Apple/│
│  Email）+ 同意授权            │
└──────────────┬──────────────┘
               │ custom token (uid = 用户本人)
               ▼
┌─────────────────────────────┐
│ self-host 前端               │
│ signInWithCustomToken       │
│ ├─ 读：Firestore 直连        │
│ └─ 写：api.botanisense.app   │
└─────────────────────────────┘
```

---

## 1. 为什么不能直接在 self-host 放 Firebase 登录按钮

在 self-host 页面里直接 `signInWithPopup(auth, new GoogleAuthProvider())`
会被 Firebase Auth 拒绝，除非该页面的 origin 在项目的
**Authorized Domains** 列表里。

- 你自己的实例：可以把域名加一次，能跑。
- 开源用户：他们的实例在 `http://192.168.1.5:8765`、
  `https://photos.example.com`……不可能把每个域名都加到你的 Firebase 项目。
- Email/Password 不受 Authorized Domains 限制，但 Google/Apple 注册的用户
  未必有密码；让每个 self-host 页面收集密码也不是好体验。

所以正确做法是：**登录发生在 botanisense.app，登录结果用 custom token
带回 self-host。** 这样：

- 任意 self-host 域名、LAN HTTP、Docker 都能用；
- 不用给每个实例加白名单；
- 用户密码只输入给 BotaniSense；
- Google/Apple/Email 全部可用；
- 拿到的就是用户本人的 session，不是影子用户。

---

## 2. 三种登录方式对比

| 方式 | 登录页在哪 | Google/Apple | 开源可用 | 实例能看到密码 | 工作量 |
|---|---|---|---|---|---|
| A. self-host 直接 Firebase SDK | self-host | 需要加域名白名单 | ❌（除非只用 Email） | 否 | 最小 |
| **B. BotaniSense 托管登录 + custom token** | **botanisense.app** | **可以** | **✅** | **否** | **中** |
| C. 后端 email/password REST | self-host | 不涉及 | 勉强 | **是** | 最小 |

### A. self-host 直接 Firebase SDK

```ts
import { getAuth, signInWithEmailAndPassword } from 'firebase/auth';
await signInWithEmailAndPassword(auth, email, password);
```

- Email/Password 可以直接用；
- Google/Apple 需要把 self-host origin 加进 Authorized Domains；
- 适合你自己临时验证，不适合作为开源方案。

### B. BotaniSense 托管登录 + 用户本人 custom token（推荐）

- 用户在 botanisense.app 登录；
- BotaniSense 后端用 Admin SDK 签发 `create_custom_token(uid)`；
- self-host `signInWithCustomToken`；
- 这是“真的登录成他本人”，不是 bridge。

### C. 后端 email/password REST

```python
# 不要在生产第三方场景用；实例会看到密码
POST https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=<web api key>
{"email": "...", "password": "...", "returnSecureToken": true}
```

- 零 BotaniSense 改动，任意域名可用；
- 但实例拿到明文密码，Google/Apple 用户也可能没有密码；
- 只适合你自己临时用，不要做成开源功能。

---

## 3. 推荐流程：托管登录 → 用户本人 custom token

两种入口，后端逻辑一样：

- **设备码（推荐给 self-host）**：实例显示 `BCDF-2345`，用户去
  `https://botanisense.app/link.html` 登录并输入/确认码；实例轮询拿结果。
  不需要公网回调，LAN/HTTP 都能用。
- **OAuth Authorization Code + PKCE（可选）**：实例有公网 HTTPS 域名时，
  标准跳转登录，体验更顺。

### 3.1 设备码流程

```
self-host                         BotaniSense
   │ POST /api/integrations/device/code
   │ ───────────────────────────────►
   │ {device_code, user_code,
   │  verification_uri, expires_in}
   │ ◄───────────────────────────────
   │
   │ 显示 “BCDF-2345” + 链接/二维码
   │
   │                         用户在 botanisense.app/link.html
   │                         登录 + 输入码 + 同意
   │                         后端 verify_id_token → uid
   │                         记录 approval(uid)
   │
   │ POST /api/integrations/device/token
   │ {device_code}
   │ ───────────────────────────────►
   │ {status:"pending"} ← 未批准
   │ ...轮询...
   │ {customToken, uid, email,
   │  displayName, approvedAt}
   │ ◄───────────────────────────────
   │
   │ signInWithCustomToken(customToken)
   ▼
 已登录成 uid 本人
```

实现要点：

- `device_code` 32 bytes 随机，服务端只存 hash；
- `user_code` 短码（如 8 位，去掉易混字符），10 分钟过期；
- `/link.html` 页面要求 Firebase 登录态，登录后把 `user_code` 和
  Firebase ID token 一起提交；
- 后端 `auth.verify_id_token(id_token)` 得到 `uid`，**不要信任请求里
  自带的 uid**；
- 批准后写 `integrationGrants/{uid}`（记录实例名、时间、撤销状态）；
- 实例轮询到批准后，后端 `auth.create_custom_token(uid)` 签发；
- custom token 1 小时内有效，实例应立即 `signInWithCustomToken`；
- 签发后删除 device code，防重放。

### 3.2 后端签发 custom token

```python
# BotaniSense server: integrations/device_flow.py
from firebase_admin import auth

# 1) 用户在 /link 页面已登录；这里验证的是用户的 Firebase ID token
decoded = auth.verify_id_token(user_id_token)
owner_uid = decoded["uid"]          # 真实 uid，绝不从请求体拿

# 2) 记录授权（用于撤销 / 审计 / 写入 API 校验）
db.collection("integrationGrants").document(owner_uid).set({
    "uid": owner_uid,
    "provider": "immich",
    "instanceName": instance_name,
    "approvedAt": firestore.SERVER_TIMESTAMP,
    "revoked": False,
}, merge=True)

# 3) 签发用户本人的 custom token
custom_token = auth.create_custom_token(owner_uid).decode()
```

注意：

- 这里**不是** `create_custom_token("bridge_xxx")`，uid 就是用户本人；
- 不需要给用户设置 custom claims，owner rules 原样生效；
- 如果账号开了 MFA，确认 `/link.html` 页面是在完成第二因子之后才提交
  （不要把 pending MFA 的 token 当成完整登录）；
- custom token 只是“登录凭据”，不携带额外权限；权限仍由该 uid 的
  Firestore/Storage rules 决定。

### 3.3 OAuth Code + PKCE（可选）

如果实例有公网 HTTPS：

```
GET https://botanisense.app/oauth/authorize
  ?client_id=botanisense-for-immich
  &redirect_uri=https://photos.example.com/api/botanisense/callback
  &response_type=code
  &scope=openid profile
  &state=<random>
  &code_challenge=<S256>
  &code_challenge_method=S256
```

用户登录并同意后回调 `?code=...&state=...`；实例后端用
`code_verifier` 换 token。返回里包含：

```json
{
  "customToken": "<uid=用户本人的 Firebase custom token>",
  "uid": "...",
  "email": "...",
  "displayName": "..."
}
```

同样 `signInWithCustomToken`。设备码和 code flow 只是拿 custom token
的两种方式，登录结果完全一样。

### 3.4 self-host 前端登录

```ts
import { initializeApp } from 'firebase/app';
import { getAuth, signInWithCustomToken, onAuthStateChanged } from 'firebase/auth';

// 官方项目的公开 web config，不是秘密
const app = initializeApp(BOTANISENSE_FIREBASE_CONFIG);
const auth = getAuth(app);

await signInWithCustomToken(auth, customToken);

onAuthStateChanged(auth, user => {
  if (user) {
    // user.uid === 用户真实 uid
    // user.email / displayName / photoURL 都有
  }
});
```

- `signInWithCustomToken` 不检查 Authorized Domains，LAN HTTP 也能用；
- 之后 Firebase SDK 自己管理 refresh token；
- 刷新页面后仍保持登录（IndexedDB persistence）；
- 换设备需要重新走一次登录（或后端保存 refresh token，见 §6）。

---

## 4. 登录后：读 Firestore 直连

因为登录成的是用户本人，现有的 owner rules 直接放行，**不需要改任何
Firestore 读规则**：

```ts
import {
  getFirestore, collection, doc, query, orderBy, onSnapshot,
} from 'firebase/firestore';

const db = getFirestore(app);
const uid = auth.currentUser!.uid;

const unsubEntries = onSnapshot(
  query(
    collection(db, 'users', uid, 'plantDiaryEntries'),
    orderBy('timestamp', 'desc'),
  ),
  snap => setEntries(snap.docs.map(d => ({ id: d.id, ...d.data() }))),
);

const unsubTags = onSnapshot(
  doc(db, 'users', uid, 'diaryTags', 'data'),
  snap => setTags(snap.data()?.tags ?? []),
);
```

能读的：

```
users/{uid}/plantDiaryEntries
users/{uid}/diaryTags/data
users/{uid}/photos
users/{uid}/userSpeciesConnections
users/{uid}/publicProfile/data        # 本来就是 public
species/{id}                          # public read
```

这和官方 App 走的是同一套 rules、同一份数据。手机 App 里改日记，
self-host 页面通过 `onSnapshot` 实时更新。

照片显示：

- 日记 entry 自带 `photoDownloadURLs`；
- 值是 ImageKit 相对 path 时，前缀 `https://ik.imagekit.io/bpgiejhtz/`；
- 本仓库现有的图片组件可以直接渲染完整 URL。

---

## 5. 写：用 Firebase ID token 调导入 API

读可以直连，**写不建议直连**。原因不是权限，而是业务不变量：

官方 App 的 `addDiaryEntry` 会同时维护：

- `plantDiaryEntries` 文档；
- `diaryTags/data` 里的 `entryCount`、`lastEntryAt`、封面；
- `tagIds`、`photoBoxes`；
- `userSpeciesConnections`、`detectedSpeciesIds`；
- 位置类 entry 还要 recompute `currentLocationId`。

第三方直接写迟早把计数/封面写坏。所以写走一个薄 API，服务端用 Admin SDK
做唯一写者：

```
POST /api/integrations/import
Authorization: Bearer <Firebase ID token>
Content-Type: multipart/form-data

file=...
clientRef=immich:<asset_id>
capturedAt=...
latitude=...
longitude=...
detections=<PlantIdentification[] JSON>
diary=<可选：notes/plant/location/timestamp>
```

服务端：

1. `verify_token` 得到 uid（复用现有 `server/process_image.py`）；
2. 检查 `integrationGrants/{uid}` 存在且未撤销；
3. 配额 / 限流；
4. 图片校验、压到最长边 2048、JPEG/WebP q85；
5. 去重：按 `uid + sha256` 查 `originalChecksum`；
6. 上传 Storage `users/{uid}/photos/{imageId}`；
7. 写 `users/{uid}/photos/{photoId}`：
   - 有 `detections` → `status:'completed'`，
     `processingMetadata.result = detections`，**不调 Gemini**；
   - 无 `detections` → 可选走现有分析队列；
8. 如果 `diary` 有值，按官方逻辑写 entry + tag counts + cover；
9. 返回 `{photoId, entryId, url}`。

self-host 调用：

```ts
import { withIdToken } from './authToken'; // 和官方 App 一样：401 时 refresh 重试

const fd = new FormData();
fd.append('file', blob, 'photo.jpg');
fd.append('clientRef', `immich:${assetId}`);
fd.append('detections', JSON.stringify(detections));
fd.append('diary', JSON.stringify(diary));

await withIdToken(token =>
  fetch('https://api.botanisense.app/api/integrations/import', {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: fd,
  })
);
```

要点：

- API 用 Firebase ID token 鉴权，**不需要新的 API key/OAuth access token**；
- 服务端从 token 得到 uid，不接受请求体里的 uid；
- `integrationGrants/{uid}` 是撤销开关：用户在 App 里“断开 Immich”，
  删除这个 doc，之后的导入就被拒绝；
- 已登录 session 的读取仍会继续到 refresh token 失效；要立刻踢掉，
  只能 `revoke_refresh_tokens(uid)`（会把官方 App 也登出，见 §6）；
- 如果本仓库后端要做后台批量导入，它需要拿到 ID token：前端可以按请求
  传入，或后端保存 refresh token（加密）并定期换 ID token。

### 5.1 为什么不用 OAuth access token

既然 self-host 已经拿到用户本人的 Firebase session，最自然的写凭据就是
`user.getIdToken()`。BotaniSense 服务端已经有 `verify_id_token`，不需要
再造一套 OAuth token 体系。OAuth/custom token 只负责“把登录结果带回
self-host”，数据面统一用 Firebase ID token。

---

## 6. 撤销、安全边界与取舍

### 6.1 这是“真的登录”，不是“受限的 bridge”

用户登录后，self-host 拿到的是他本人的 Firebase session：

- 能读该账号能读的一切（包括私有日记、照片）；
- 能写该账号能写的一切（如果它绕过我们的 API 直接写 Firestore）；
- 在用户自己的 self-host 上，这通常可接受——这是他的服务器；
- 如果实例是多人共用/公网暴露，必须加 HTTPS + 反向代理认证。

如果你想要“只读日记、不能写、撤销不影响官方 App”的强隔离，那才需要
影子 bridge 用户（附录 A）。两者是不同产品选择：

| | 用户本人登录 | 影子 bridge |
|---|---|---|
| `uid` | 用户真实 uid | `bridge_xxx` |
| 读 | owner rules 全放行 | 只能读 rules 显式允许的集合 |
| 写 | 权限上可写（产品上只走 API） | 只能写 job |
| 撤销 | revoke 全部 refresh token，影响官方 App | 只禁用 bridge，不影响官方 App |
| 体验 | 真登录，能显示 email/头像 | 像“连接一个服务”，不是登录 |

### 6.2 撤销

- **断开集成（软撤销）**：删除 `integrationGrants/{uid}`；导入 API 立即
  拒绝新写入。已登录的读取仍有效，直到 Firebase session 失效。
- **强制下线**：`auth.revoke_refresh_tokens(uid)`；所有 refresh token
  失效（包括官方 App 和其他设备）。现有 ID token 最多再活 1 小时。
- **用户自己退出**：self-host 前端 `signOut(auth)`，清本地 session。
  但恶意实例仍可能保留 refresh token；信任边界是“用户自己的服务器”。

### 6.3 其他

- `/link.html` 同意页要显示实例名称和 origin，让用户知道授权给谁；
- device code 限流、短 TTL、一次性；
- 导入 API 按 uid/IP 限流 + 配额；
- 默认跳过 `has_person` 的照片，避免人物照片进植物日记；
- 上传重新编码、剥离不必要 EXIF；
- **`botaniq` 仓库把服务账号 JSON 提交进了 git**
  （`server/plantsun-7b88e-firebase-adminsdk-44bfq-5632bda204.json`）。
  BotaniSense 不开放源码，owner 决定把它留在**私有** git 里方便版本
  管理，这是有意的取舍。前提：仓库保持私有、只有可信的人有访问权。
  若未来改可见性、加外部协作者或 key 泄露，再在 Google Cloud 生成
  新 key、部署、删除旧 key（仅从 git 删除文件不会让旧 key 失效）。

---

## 7. 开源用户怎么用

每个用户跑自己的 self-host 实例，登录自己的 BotaniSense 账号：

1. 实例显示 `BCDF-2345` + `https://botanisense.app/link.html`；
2. 用户在自己手机/浏览器登录并同意；
3. 实例拿到用户本人的 custom token，登录；
4. 之后读自己的日记、上传自己的照片。

不需要：

- 把每个实例域名加进 Firebase Authorized Domains；
- 给用户发 API key；
- 让用户把密码输入 self-host；
- 在 BotaniSense 里为每个实例建影子用户。

需要控制的是成本：

- 导入 API 按 uid 配额（照片数/字节数/每日请求）；
- 默认压图到 2048px；
- 带 `detections` 就跳过 Gemini；
- 收费前先给硬配额；
- 以后可以加“用户自部署 BotaniSense”（Firebase config / API base 可配置），
  官方服务只做默认入口。

---

## 8. 分阶段

### Phase 1（2–3 天）：登录 + 读日记 + 上传照片

BotaniSense：

- `POST /api/integrations/device/code`
- `POST /api/integrations/device/approve`（验证 Firebase ID token）
- `POST /api/integrations/device/token`（返回 `create_custom_token(uid)`）
- `/link.html` 页面（复用现有登录态 + 同意按钮）
- `integrationGrants/{uid}`
- `POST /api/integrations/import`（先只建 photo doc，不建日记）

本仓库：

- 设置页“用 BotaniSense 登录”（设备码/二维码 + 轮询）
- 前端 Firebase SDK：`signInWithCustomToken` + `onAuthStateChanged`
- “日记”视图：`onSnapshot` 读 `plantDiaryEntries` / `diaryTags`
- 网格多选“发送到 BotaniSense”：`/thumb?size=preview` → 导入 API
- 本地 `botanisense_links` 表（asset_id ↔ photoId/status）

### Phase 2（2–3 天）：日记写入

- 导入 API 支持 `diary`：plant tag、`entryCount`、cover、`tagIds`、
  `photoBoxes`；
- self-host UI 选择“新建植物 / 已有植物 / 只入库照片” + 备注/日期；
- 幂等：`clientRef` + checksum。

### Phase 3（2–3 天）：体验与治理

- OAuth Code + PKCE（有公网域名的实例）
- “已连接的应用”管理页 + 撤销
- 配额/限流/审计
- 日记时间线/地图
- Immich asset ↔ BotaniSense entry 匹配标记

### Phase 4（可选）

- 后台批量同步（后端保存 refresh token / 定时任务）
- “Bring your own BotaniSense”：Firebase config / API base 可配置
- 社区发帖 `plantPosts`

---

## 9. 代码落点

BotaniSense：

- `server/integrations/device_flow.py` — device code + approve + token
- `server/integrations/import_api.py` — 导入 API（复用 `verify_token`、
  `process_image`、物种维护、日记写入）
- `src/pages/LinkDevicePage.tsx` + `/link`（App Web）
  或 `web_home/src/pages/link.astro` + `/link.html`（营销站静态页）— 登录/同意页
- `src/pages/Settings...` — “已连接的应用”
- `firestore.rules` — 只需给 `integrationGrants` / `deviceCodes`
  加显式 server-only（默认 deny 已覆盖；不需要改用户数据读规则）
- `storage.rules` — 不需要改（导入 API 用 Admin SDK；直传也走 owner 规则）

本仓库：

- `frontend/src/components/BotanisenseConnect.tsx` — 设备码登录
- `frontend/src/pages/Diary...` — `onSnapshot` 日记视图
- `frontend/src/components/SelectionActionBar.tsx` — “发送到 BotaniSense”
- `frontend/src/api/botanisense.ts` — 导入 API 调用 + `withIdToken`
- `storage.py` — `botanisense_links` 表（幂等/状态）
- `app.py` — 可选：代理 device code（避免 CORS/把设备码留在后端）

---

## 10. 需要拍板的点

1. 登录入口先做**设备码**，还是一步到位做 **OAuth Code + PKCE**？
   - self-host/LAN 建议设备码；有公网域名再加 PKCE。
2. 读范围：只日记+照片，还是也读 `userSpeciesConnections`？
3. 写入第一版是否建日记 entry，还是只入库照片？
4. 图片来源：`/thumb?size=preview`（快）还是 Immich 原图（质量高）？
5. 是否允许社区发帖 `plantPosts`？
6. 收费前每个账号的硬配额（照片数/字节数/日请求数）。

---

## 附录 A：影子 bridge（仅当你需要强隔离时）

如果以后要支持“只读日记、不能写、撤销不影响官方 App”的场景，
再考虑影子用户：

- `uid = bridge_xxx`，不是用户本人；
- custom claims `{bridge:true, ownerUid, scopes}`；
- rules 显式允许 bridge 读某些集合；
- 写只进 `bridgeJobs`，后端 worker 落库；
- 撤销 = 禁用 bridge + revoke 它的 refresh token，不影响用户主账号。

这和“用户本人登录”是两条不同的产品路线，不要混用。

---

## 附录 B：当前实现状态（2026-11-14，已部署）

### 已落地

BotaniSense 侧（`/media/4T_Ext/chenxing/deployment/botaniq`）：

- `server/integrations/device_flow.py`
  - `POST /api/integrations/device/code`
  - `POST /api/integrations/device/approve`
  - `POST /api/integrations/device/token` → `create_custom_token(用户真实 uid)`
  - `GET  /api/integrations/me`
  - `POST /api/integrations/revoke`
- `server/integrations/import_api.py`
  - `POST /api/integrations/import`（multipart，压图、按 checksum/clientRef 去重、
    写 photo doc、维护 species/userSpeciesConnections、跳过 Gemini）
- `server/app.py` 注册 `integrations_bp`
- `src/pages/LinkDevicePage.tsx` + `/link`（App Web）
  或 `web_home/src/pages/link.astro` + `/link.html`（营销站静态页）
  （登录 + 输入码 + 同意）

本仓库：

- `frontend/src/lib/botanisense.ts`（Firebase 初始化、custom token 登录后直连 Firestore）
- `frontend/src/api/botanisense.ts`（设备码、轮询、me、revoke、导入）
- `frontend/src/components/BotanisenseConnect.tsx`（连接/断开/设备码 UI）
- `frontend/src/components/BotanisensePage.tsx`（`onSnapshot` 日记视图）
- `Header.tsx` 新增 Diary tab（`#botanisense`）
- `SelectionActionBar.tsx` 新增“Send to BotaniSense”
- `useRoute.ts` / `App.tsx` 增加 `botanisense` 视图

### 已用真实 Firebase 验证

- 设备码创建 + pending 轮询
- 临时用户 → approve → 签发 custom token → me → revoke
- 导入一张测试图：photo doc、species doc、userSpeciesConnections、
  GPS/拍摄时间、clientRef 去重（第二次返回 `deduped:true`）
- 测试用户、photo、species、Storage 对象、grant 均已清理

### 已上线

- BotaniSense server：`/api/integrations/*` 已部署到
  `https://api.botanisense.app`（`botaniq-server.service` 已重启）。
- 授权页：`https://botanisense.app/link.html` 已发布。
- 本仓库 self-host：`http://192.168.84.113:8770`（systemd
  `botanisense-immich.service`），使用本仓库的 `data/state.db` 和 Immich
  连接；老 `botanisense` 容器已停止但保留可回滚。`8771` 的 S0
  自训练模型实例未受影响。
- 线上全链路已验证：临时邮箱用户登录 `/link.html` → 批准 → 线上 API
  签发用户本人 custom token → `/me` 返回 grant；测试用户和 grant 已清理。
- 提交：`botaniq` `5506e98 feat(integrations): Immich device-code login + photo import`。

### 尚未做

- 日记 entry 写入（Phase 2：plant tag / entryCount / cover / photoBoxes）
- OAuth Code + PKCE（有公网 HTTPS 域名时再加）
- 多实例 grant（当前 `integrationGrants/{uid}` 一个账号一条，
  多个实例共享一个撤销开关）
- 配额管理页、Audit 页面
- `botaniq` 仓库里的服务账号 JSON 按 owner 决定继续留在私有 git 中
  （方便版本管理）。BotaniSense 不开源，这不是待办；保持仓库私有即可。

### 部署检查

1. 部署 BotaniSense 服务端，使 `/api/integrations/*` 可达；
2. 构建并发布 web 端，使 `https://botanisense.app/link.html` 可达；
3. 重启/重建本仓库的 self-host 容器（`frontend/dist` 已重新 build）；
4. Firestore/Storage rules 不需要为“读”加规则：登录成用户本人后
   owner rules 原样生效；新集合 `integrationDeviceCodes` /
   `integrationUserCodes` / `integrationGrants` 由 Admin SDK 访问，
   默认 deny 已覆盖；
5. 自测：`#botanisense` → Connect → 手机打开 `/link.html` 输入码 →
   页面出现日记 → 网格选照片 → Send to BotaniSense。
