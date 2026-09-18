# 架构说明

## 分层

```text
__init__.py
  └─ 导入 handlers，完成插件注册

metadata.py / matchers.py
  ├─ 插件元数据
  └─ Matcher 的集中声明

handlers/
  ├─ 接收 NoneBot 事件并组织回复
  ├─ draw / collection / rename / statistics
  ├─ help / feature_filters / feature_query
  └─ package_painting / alias_management / lifecycle

services/
  ├─ 抽取与命名：draw_request / draw_support / painting_name
  ├─ 别名：alias_registry / game_aliases
  ├─ 打包：package_archive / package_cache / package_storage
  ├─ 模型门面：model（下载、SHA-256 校验、版本与会话管理）
  ├─ 视觉流水线：vision/{assets,runtime,types,subjects,taggers,fusion,ensemble}
  ├─ 图片特征：image_features / image_feature_store
  ├─ 特征映射：feature_aliases + config/feature_aliases.yaml
  ├─ 用户图片查询：feature_query（限量下载/临时推理）/ feature_report / people_report（标注副本）
  │  └─ feature_presentation：统一 ≥50% 展示过滤、排序及无截断换行；不改变推理阈值
  ├─ 查询人数：vision/faces（多方向去重）/ vision/census（逐脸融合与复核）
  ├─ 邻人隔离：vision/instance_regions（人脸锚点 + RTMDet-Ins 实例掩码，CPU 会话）
  ├─ 主体评分：vision/subject_salience（构图/透视尺度/色彩/明暗/光影造型/曝光/
  │  饱和度/艺术强调晋升，可解释 JSON；依据 Itti-Koch-Niebur 显著性等文献）
  ├─ 状态反馈：reactions（查询和打包共享的 282/478/479 表情）
  └─ 消息、统计、改名、帮助确认等可复用业务能力

database/
  └─ SQLite 连接池、事务边界和唯一仓储实现
```

## 依赖规则

- `__init__.py` 只负责导入处理器和导出元数据，不承载业务逻辑。
- Matcher 统一在 `matchers.py` 创建，处理器只注册回调。
- `handlers` 可以调用 `services` 和 `database`；服务不能反向导入处理器。
- 文件压缩、缓存和对象存储分别位于独立模块，避免后端协议污染指令流程。
- `model.py` 管理固定修订版模型的下载、校验和门面；`vision` 分离人物框检测、
  WD/Camie 预处理与解码、主体选择、融合规则和资源预算。融合版本变更触发重标记。
- `image_features.py` 负责启动补扫、Rich 进度、增量监控、内容去重和高置信度
  目录纠正；`image_feature_store.py` 只负责版本化特征持久化。
- `utils/__init__.py` 和 `services/painting_package.py` 是兼容门面，新代码应直接
  导入对应的细分服务。

## 运行时数据

- 游戏立绘仍保存在 `wives/` 与 `husbands/`。
- 游戏别名保存在 `utils/game_aliases.yaml`，写入采用锁、原子替换、去重和冲突检查。
- 抽取、改名等状态保存在 `database/record.db`，已停用的历史记录保留。
- 打包消息和共享对象缓存保存在 `data/painting_packages/cache.db`；临时压缩包位于
  `data/painting_packages/work/`，上传后立即删除。
- 图片特征保存在 `data/image_features.db`，记录文件签名、SHA-256、模型修订、
  主体男女置信度、主体标签、整图标签、主体状态、检测框与来源判定；不会修改图片像素。
- 可组合筛选使用图片整数 ID 关联的 `image_feature_tags` 明细索引，中文名称与
  英文标签的关系保存在 `config/feature_aliases.yaml`；多个 `-t` 特征通过索引
  自连接按 AND 关系匹配。旧的完整路径标签表会在启动时自动迁移并压缩。
- `wives/` 与 `husbands/` 启动及每 15 秒与数据库核对。缺失标签立即失效，记录保留
  24 小时以复用同内容改名，随后清理；无法访问的目录不清理，抽卡历史不改动。
- 检索标签来自同一个可信主体裁剪，只有显式人数标签使用整图属性。旧整图索引
  在升级时清空并逐步重新生成；未确认主体不索引、不在男女目录间互移。
- `Others/` 接收男女整图原始融合评分均 <30% 的图片；缺失评分不能归零判定。
  使用现有安全移动与同名处理流程，归档后按 library 隔离，不参与抽取及打包。
- `archive_policy` 统一自动移动判定：主体已确认且性别一致，主体融合标签目标分数
  严格 >70%、另一性别 ≤70%、分差 ≥10 个百分点。启动时流式复查当前模型的
  男女图库缓存，符合互移或 Others 规则者加入队列，按内容哈希复用推理结果。
  重叠或无法确认主体的多人图须安全裁剪并通过模型复核才可在男女目录间移动。
  目录监控独立于模型下载和推理循环。
- `特征查询` 接收任意用户上传图片，不查图库、不保存特征、不移库。
  `feature_query` 使用系统临时目录，返回 JPEG 字节后清理；所有推理仍通过同一个
  `ModelManager` 和串行锁。取消请求需等待原生推理结束再释放队列位置。
- `ModelManager.predict_query` 复用标签会话，延迟加载人脸模型；逐脸查询流程独立于
  `predict_paths` 的索引判定，不触发全库重标记。诊断计数保存在返回
  对象的 `analysis.census`，不持久化；最多分析 24 张脸，其余计入未确认。
- 多人主体决策（`ensemble.multi_person_subject_pass`）同时服务静默索引与特征查询：
  人脸锚点配对人物框，实例分割提供无邻人像素，显著性评分（尺度/构图/透视/
  色彩/明暗/光影造型）选出画面主体，该主体的逐人性别即图片主体性别，供男女库
  归类使用；证据不足或评分接近时弃权（ambiguous），不硬指派。`face_consensus`
  仅用于展示估计，绝不驱动移库。分割会话固定 CPU，失败自动回退人脸邻域逻辑。
- 正常查询只发送 JPEG，包含人数、编号、评分和特征；处理状态以原指令贴表情表示。
- `subject-fusion-v2` 增加几何主次门槛、局部再检测、保留躯干的避让裁剪及跨视角
  冲突检查。`analysis` 保留初始框、精修框、实际特征区域、来源视角及各模型评分。
  展示层区分未计算、计算后不确定、可信主体；未计算不再冒充零概率。

## 新功能扩展

1. 在 `matchers.py` 声明 Matcher。
2. 在 `handlers/` 添加薄处理器，只负责参数、权限、消息和异常映射。
3. 将可测试的业务逻辑放入 `services/` 的对应领域模块。
4. 需要持久化时，为 `DatabaseHandler` 增加单一方法，避免在处理器中直接写 SQL。
5. 为新增路径补充 Python 3.10 语法检查、Ruff、插件加载和领域回归测试。
