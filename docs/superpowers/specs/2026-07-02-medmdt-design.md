# MedMDT 多专家会诊医学 Agent 系统 — 设计文档

## 概述

面向临床医生的辅助诊断系统，通过多个专科 AI 专家进行多轮讨论式会诊（MDT），结合知识图谱、向量检索和关键词检索三路融合的知识库，为医生提供诊断建议和治疗方案。

### 核心特征

- **全科室覆盖**：通过 YAML 配置灵活定义任意科室专家
- **多轮讨论**：专家先独立分析，再针对分歧多轮辩论，直到达成共识或达到轮次上限
- **三路融合知识库**：Neo4j 知识图谱 + Milvus 向量检索 + Elasticsearch BM25
- **多模型可配置**：每个专家可独立配置 LLM 后端
- **离线知识提取**：独立 Agent 处理 PDF/DICOM/JPG/PNG，构建结构化知识

### 数据来源

- 医学文献：临床指南、专家共识、医学教材、论文
- 临床数据：真实病历、影像报告、检验结果、手术记录

### 部署方式

开发阶段 Docker Compose 本地部署，生产环境迁移至云服务。

---

## 架构：分层模块化

系统拆为 4 个独立模块，各自有清晰边界：

1. **知识提取服务 (Extractor)** — 离线 Agent，处理 PDF/影像/文本，输出结构化 JSON
2. **知识库服务 (Knowledge Store)** — 统一检索接口，封装三路召回与融合排序
3. **MDT 会诊引擎 (MDT Engine)** — LangGraph 编排的多 Agent 讨论系统
4. **API 网关 (Gateway)** — FastAPI 入口，请求路由与会话管理

---

## 项目结构

```
medmdt/
├── pyproject.toml
├── docker-compose.yml              # Neo4j + Milvus + Elasticsearch
├── config/
│   ├── settings.py                 # 全局配置（Pydantic Settings）
│   └── experts.yaml                # 专家定义（科室、prompt、LLM配置）
├── src/
│   └── medmdt/
│       ├── extractor/              # 模块1: 知识提取服务
│       │   ├── agent.py            # 提取Agent主逻辑（LangGraph子图）
│       │   ├── parsers/
│       │   │   ├── pdf_parser.py   # PaddleOCR-VL-1.6 API
│       │   │   ├── dicom_parser.py # pydicom + 多模态LLM
│       │   │   └── image_parser.py # 多模态LLM图像分析
│       │   ├── schemas.py          # 提取结果JSON Schema
│       │   └── ingestor.py         # 三库写入
│       ├── knowledge/              # 模块2: 知识库服务
│       │   ├── graph_store.py      # Neo4j 知识图谱
│       │   ├── vector_store.py     # Milvus 向量检索
│       │   ├── keyword_store.py    # Elasticsearch BM25
│       │   ├── retriever.py        # 三路融合检索（RRF）
│       │   └── schemas.py
│       ├── mdt/                    # 模块3: MDT会诊引擎
│       │   ├── graph.py            # LangGraph 状态图
│       │   ├── state.py            # 会诊状态定义
│       │   ├── experts/
│       │   │   ├── base.py         # 专家基类
│       │   │   └── factory.py      # 专家工厂（根据YAML配置创建）
│       │   ├── moderator.py        # 主持人Agent
│       │   └── rounds.py          # 多轮讨论逻辑
│       ├── llm/                    # LLM抽象层
│       │   ├── provider.py         # 多模型Provider工厂
│       │   └── prompts/
│       │       ├── extraction/
│       │       │   ├── entity_extraction.py
│       │       │   └── relation_extraction.py
│       │       ├── mdt/
│       │       │   ├── expert_analysis.py
│       │       │   ├── moderator_summary.py
│       │       │   └── report_generation.py
│       │       └── retrieval/
│       │           └── query_analysis.py
│       └── api/                    # 模块4: API网关
│           ├── app.py              # FastAPI应用
│           ├── routes/
│           └── models.py
├── scripts/
│   └── ingest.py                   # 批量知识导入脚本
└── tests/
```

---

## 模块1：知识提取服务 (Extractor)

离线批量运行的 Agent，将原始医学文件转化为结构化数据并写入三个数据库。

### 处理流程

```
原始文件 → 文件类型检测 → 对应Parser解析 → LLM结构化提取 → JSON Schema验证 → 写入三库
```

### Parser

| Parser | 输入格式 | 处理方式 | 输出 |
|--------|---------|---------|------|
| `pdf_parser` | PDF | PaddleOCR-VL-1.6 云端API：提交任务→轮询→获取JSONL结果，逐页输出Markdown+图片 | 逐页Markdown文本 + 提取的图片 |
| `dicom_parser` | DICOM | pydicom提取元数据（患者信息、检查类型、设备参数），Pillow转图片后送多模态LLM | 影像描述 + 元数据 |
| `image_parser` | JPG/PNG | 直接送多模态LLM进行医学图像分析 | 影像描述 + 发现列表 |

#### PDF Parser 要点

- 使用 PaddleOCR-VL-1.6 云端 API（异步任务模式：submit → poll → fetch）
- Token 从环境变量读取，不硬编码
- 可选参数（`useDocOrientationClassify`、`useDocUnwarping`、`useChartRecognition`）通过配置控制
- Markdown 中内嵌的医学图片自动送 `image_parser` 做二次分析
- 支持本地文件和 URL 两种输入模式

### 结构化提取 Schema

LLM 对解析后的内容做结构化提取，输出统一 JSON 格式：

```json
{
  "source": {"file": "xxx.pdf", "type": "clinical_guideline", "page": 12},
  "entities": [
    {
      "name": "2型糖尿病",
      "type": "disease",
      "aliases": ["T2DM", "II型糖尿病"]
    }
  ],
  "relations": [
    {
      "head": "二甲双胍",
      "relation": "first_line_treatment_for",
      "tail": "2型糖尿病",
      "evidence": "原文引用...",
      "confidence": 0.95
    }
  ],
  "chunks": [
    {
      "text": "原始文本段落...",
      "summary": "摘要...",
      "keywords": ["糖尿病", "血糖控制", "HbA1c"],
      "metadata": {"section": "治疗方案", "guideline_level": "A级推荐"}
    }
  ]
}
```

### 提取 Agent（LangGraph 子图）

1. **分类**：判断文件类型和医学领域
2. **解析**：调用对应 Parser
3. **提取**：LLM 按 Schema 提取实体、关系、文本块
4. **校验**：JSON Schema 验证 + 实体归一化（ICD-10/SNOMED CT）
5. **去重**：与已有知识库比对
6. **写入**：分别写入三个数据库

### 三库写入策略 (Ingestor)

| 目标库 | 写入内容 | 用途 |
|--------|---------|------|
| Neo4j | `entities` + `relations` → 节点和关系 | 疾病-药物-症状关系推理 |
| Milvus | `chunks.text` → embedding 向量 | 语义相似性检索 |
| Elasticsearch | `chunks` 全文 + `keywords` + `metadata` | BM25 关键词匹配 |

---

## 模块2：知识库服务 (Knowledge Store)

统一检索接口，封装三种检索引擎的调用和结果融合。

### Neo4j 知识图谱 (`graph_store.py`)

实体关系推理查询。

节点类型：`Disease`, `Symptom`, `Drug`, `Procedure`, `Anatomy`, `LabTest`

关系类型：`treated_by`, `indicates`, `contraindicated_for`, `located_in`, `diagnoses`, `causes` 等

典型查询：
- 某疾病的所有治疗方案及禁忌
- 症状→可能疾病的鉴别诊断路径

### Milvus 向量检索 (`vector_store.py`)

语义相似性检索。

- Embedding 模型可配置，默认 `bge-large-zh-v1.5`
- Collection 按知识域分区（`guideline`, `case_report`, `textbook`）
- 返回 top-K 文本块 + 相似度分数

### Elasticsearch BM25 (`keyword_store.py`)

精确关键词匹配。

- 中文分词：IK 分词器 + 医学术语词典
- 索引字段：`text`, `keywords`, `metadata.section`, `source.type`
- 支持 Bool 查询组合

### 三路融合检索 (`retriever.py`)

```
用户查询 → 意图分析(LLM)
         ├→ 关键词提取 → BM25检索
         ├→ 查询embedding → Milvus向量检索
         └→ 实体识别 → Neo4j图查询
         → Reciprocal Rank Fusion(RRF)融合排序
         → 返回Top-N结果
```

- **意图分析**：LLM 分析查询意图，动态调整三路权重
- **RRF 融合**：`score = Σ 1/(k + rank_i)`，k 默认 60，权重由意图分析动态调整

---

## 模块3：MDT 会诊引擎

LangGraph 编排的多 Agent 多轮讨论系统。

### 会诊状态

```python
class MDTState(TypedDict):
    patient_info: dict              # 患者基本信息、主诉、现病史
    medical_records: list[dict]     # 检查报告、影像描述、检验结果
    knowledge_context: list[dict]   # 知识库检索结果
    experts: list[str]              # 参与会诊的专家列表
    discussion_rounds: list[dict]   # 每轮讨论记录
    consensus: dict | None          # 共识结论
    divergences: list[dict]         # 未解决的分歧点
    final_report: str | None        # 最终会诊报告
    current_round: int              # 当前轮次
    max_rounds: int                 # 最大讨论轮数（默认3）
```

### LangGraph 状态图

```
接收患者资料 → 知识库检索 → 选择参会专家
  → 第1轮：各专家并行独立分析（互不可见）
  → 主持人汇总 + 识别分歧
  → 是否有分歧？
    → 无分歧 → 生成报告
    → 有分歧 → 第N轮：针对分歧讨论（可见前轮意见）
             → 主持人再次汇总
             → 共识或超轮次上限？
               → 是 → 生成报告（含分歧说明）
               → 否 → 继续讨论
```

### 专家 Agent

- **基类** (`base.py`)：统一接口，输入患者资料+知识上下文+前轮记录，输出 `{analysis, diagnosis, recommendation, confidence, reasoning, references}`
- **工厂** (`factory.py`)：根据 `experts.yaml` 动态创建专家实例
- 每个专家绑定独立的 LLM 配置和 system prompt

#### experts.yaml 配置示例

```yaml
experts:
  cardiologist:
    name: "心内科专家"
    knowledge_domains: ["cardiology", "internal_medicine"]
    system_prompt: |
      你是一位资深心内科主任医师...
    llm:
      provider: "openai"
      model: "gpt-4o"

  radiologist:
    name: "影像科专家"
    knowledge_domains: ["radiology"]
    llm:
      provider: "anthropic"
      model: "claude-sonnet-4-6"
```

### 主持人 Agent (`moderator.py`)

- **汇总**：提炼各专家意见的共同点和分歧点
- **追问**：向特定专家提出需要深入解释的问题
- **引导**：在分歧轮次中引导讨论聚焦于核心分歧
- **判定**：判断是否达成共识或分歧已充分讨论
- **报告**：生成最终会诊报告，含共识意见和保留意见

### 多轮讨论逻辑 (`rounds.py`)

- 第1轮：所有专家并行独立分析，互不可见
- 第2+轮：专家可见前面所有轮次的讨论记录，针对分歧发表意见
- 终止条件：共识达成（所有专家 confidence > 阈值且方向一致）或达到 max_rounds
- 超轮次时最终报告明确标注未解决的分歧及各方理由

---

## LLM 抽象层

### Provider 工厂 (`llm/provider.py`)

支持的 Provider：

| Provider | 说明 |
|----------|------|
| `openai` | GPT-4o 等 |
| `anthropic` | Claude 系列 |
| `qwen` | 通义千问 |
| `zhipu` | 智谱 GLM |
| `moonshot` | 月之暗面 Kimi |
| `deepseek` | DeepSeek |
| `ollama` | 本地部署模型 |
| `custom` | 自定义（兼容 OpenAI API 格式，配置 base_url + api_key） |

通过 LangChain `BaseChatModel` 抽象统一调用接口。配置层级：
- 全局默认：`settings.py`
- 专家覆盖：`experts.yaml` 中为每个专家单独指定
- 多模态：影像分析任务自动选用支持视觉的模型

### Custom Provider 配置示例

```yaml
experts:
  pathologist:
    name: "病理科专家"
    llm:
      provider: "custom"
      base_url: "https://your-api.example.com/v1"
      api_key_env: "CUSTOM_LLM_API_KEY"
      model: "your-model-name"
```

---

## 模块4：API 网关

FastAPI 应用，提供两类接口。

### 会诊接口

- `POST /api/v1/consultation` — 创建会诊（上传患者资料，启动 MDT）
- `GET /api/v1/consultation/{id}` — 查询会诊状态和结果
- `WebSocket /api/v1/consultation/{id}/stream` — 实时流式推送讨论过程

### 知识管理接口

- `POST /api/v1/knowledge/ingest` — 提交文件进行知识提取
- `GET /api/v1/knowledge/ingest/{job_id}` — 查询提取任务状态
- `POST /api/v1/knowledge/search` — 知识库检索

---

## 全局配置 (`config/settings.py`)

Pydantic Settings，支持环境变量覆盖：

```python
class Settings(BaseSettings):
    # 数据库
    neo4j_uri: str = "bolt://localhost:7687"
    milvus_host: str = "localhost"
    elasticsearch_url: str = "http://localhost:9200"

    # LLM 默认
    default_llm_provider: str = "openai"
    default_llm_model: str = "gpt-4o"

    # PaddleOCR
    paddleocr_token: str  # 从环境变量读取

    # MDT
    mdt_max_rounds: int = 3
    mdt_consensus_threshold: float = 0.8

    # Embedding
    embedding_model: str = "bge-large-zh-v1.5"
```

---

## 部署 (`docker-compose.yml`)

```yaml
services:
  neo4j:
    image: neo4j:5
    ports: ["7474:7474", "7687:7687"]

  milvus:
    image: milvusdb/milvus:v2.4-latest
    ports: ["19530:19530"]

  elasticsearch:
    image: elasticsearch:8.15.0
    ports: ["9200:9200"]
    # 预装 IK 中文分词插件

  medmdt:
    build: .
    depends_on: [neo4j, milvus, elasticsearch]
    ports: ["8000:8000"]
```

---

## 交付阶段

| 阶段 | 内容 | 目标 |
|------|------|------|
| P1 | 知识提取 + 知识库写入/检索 | 能导入 PDF、检索出结果 |
| P2 | MDT 引擎 + 2-3 个专家 | 能跑通完整会诊流程 |
| P3 | API 网关 + 流式输出 | 可对外提供服务 |
| P4 | 影像处理增强 + 更多专家 | 功能完善 |
