# MedMDT P2: MDT Multi-Expert Consultation Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the multi-expert consultation engine where AI specialist agents independently analyze patient cases, then engage in multi-round discussion moderated by a coordinator agent, producing a consensus diagnosis report.

**Architecture:** LangGraph StateGraph orchestrates the MDT workflow: receive patient data → retrieve knowledge → select experts → parallel expert analysis → moderator summarization → divergence-driven discussion rounds → final report. Experts are dynamically created from `config/experts.yaml` via a factory pattern; each binds its own LLM config and system prompt. The moderator coordinates discussion, detects consensus, and generates the final report.

**Tech Stack:** Python 3.12, LangGraph (StateGraph, Send for parallel fan-out), LangChain BaseChatModel, PyYAML, Pydantic v2, pytest

## Global Constraints

- Python >=3.12, managed by uv
- Source layout: `src/medmdt/` (src-layout)
- LLM access via `create_chat_model(provider, model, **kwargs) -> BaseChatModel` from `medmdt.llm.provider`
- Expert definitions in `config/experts.yaml` — YAML-driven, not hardcoded
- Settings from `medmdt.config.settings.Settings` — `mdt_max_rounds` (default 3), `mdt_consensus_threshold` (default 0.8), `default_llm_provider`, `default_llm_model`
- Knowledge retrieval via `FusionRetriever.retrieve(query, top_k) -> list[RetrievalResult]` from `medmdt.knowledge.retriever`
- Tests: pytest with mocks for LLM calls; no real API calls in unit tests
- All prompt templates in `src/medmdt/llm/prompts/mdt/`
- Package directories `src/medmdt/mdt/`, `src/medmdt/mdt/experts/`, `src/medmdt/llm/prompts/mdt/` already have empty `__init__.py` files from P1

---

### Task 1: MDT State and Expert Opinion Schema

**Files:**
- Create: `src/medmdt/mdt/__init__.py` (if not empty already, leave it)
- Create: `src/medmdt/mdt/state.py`
- Create: `src/medmdt/mdt/experts/__init__.py` (if not empty already, leave it)
- Test: `tests/test_mdt_state.py`

**Interfaces:**
- Produces:
  - `ExpertOpinion(expert_id: str, expert_name: str, analysis: str, diagnosis: str, recommendation: str, confidence: float, reasoning: str, references: list[str])` — Pydantic model
  - `DiscussionRound(round_num: int, opinions: list[ExpertOpinion], summary: str | None, divergences: list[str])` — Pydantic model
  - `MDTState(TypedDict)` with fields: `patient_info: dict`, `medical_records: list[dict]`, `knowledge_context: list[dict]`, `experts: list[str]`, `discussion_rounds: list[dict]`, `consensus: dict | None`, `divergences: list[str]`, `final_report: str | None`, `current_round: int`, `max_rounds: int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mdt_state.py
import pytest
from medmdt.mdt.state import MDTState, ExpertOpinion, DiscussionRound


def test_expert_opinion():
    op = ExpertOpinion(
        expert_id="internist",
        expert_name="内科专家",
        analysis="患者症状符合2型糖尿病表现",
        diagnosis="2型糖尿病",
        recommendation="建议口服二甲双胍",
        confidence=0.9,
        reasoning="空腹血糖>7.0, HbA1c>6.5%",
        references=["2型糖尿病诊疗指南2024"],
    )
    assert op.confidence == 0.9
    assert op.expert_id == "internist"


def test_expert_opinion_confidence_range():
    with pytest.raises(Exception):
        ExpertOpinion(
            expert_id="x", expert_name="x", analysis="x", diagnosis="x",
            recommendation="x", confidence=1.5, reasoning="x", references=[],
        )


def test_expert_opinion_defaults():
    op = ExpertOpinion(
        expert_id="test", expert_name="Test", analysis="a",
        diagnosis="d", recommendation="r", confidence=0.8, reasoning="r",
    )
    assert op.references == []


def test_discussion_round():
    opinion = ExpertOpinion(
        expert_id="internist", expert_name="内科", analysis="a",
        diagnosis="d", recommendation="r", confidence=0.8, reasoning="r",
    )
    rd = DiscussionRound(
        round_num=1,
        opinions=[opinion],
        summary="各专家意见一致",
        divergences=[],
    )
    assert rd.round_num == 1
    assert len(rd.opinions) == 1


def test_discussion_round_serialization():
    opinion = ExpertOpinion(
        expert_id="internist", expert_name="内科", analysis="a",
        diagnosis="d", recommendation="r", confidence=0.8, reasoning="r",
    )
    rd = DiscussionRound(round_num=1, opinions=[opinion])
    d = rd.model_dump()
    restored = DiscussionRound.model_validate(d)
    assert restored.round_num == rd.round_num
    assert restored.opinions[0].expert_id == "internist"


def test_mdt_state_is_typeddict():
    state: MDTState = {
        "patient_info": {"name": "张三", "chief_complaint": "反复头痛1月"},
        "medical_records": [{"type": "lab", "content": "血压150/95mmHg"}],
        "knowledge_context": [],
        "experts": ["internist", "cardiologist"],
        "discussion_rounds": [],
        "consensus": None,
        "divergences": [],
        "final_report": None,
        "current_round": 0,
        "max_rounds": 3,
    }
    assert state["max_rounds"] == 3
    assert state["consensus"] is None
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_mdt_state.py -v`
Expected: FAIL — module not found

- [ ] **Step 2: Implement state.py**

```python
# src/medmdt/mdt/state.py
from typing import TypedDict
from pydantic import BaseModel, Field


class ExpertOpinion(BaseModel):
    expert_id: str
    expert_name: str
    analysis: str
    diagnosis: str
    recommendation: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    references: list[str] = Field(default_factory=list)


class DiscussionRound(BaseModel):
    round_num: int
    opinions: list[ExpertOpinion] = Field(default_factory=list)
    summary: str | None = None
    divergences: list[str] = Field(default_factory=list)


class MDTState(TypedDict):
    patient_info: dict
    medical_records: list[dict]
    knowledge_context: list[dict]
    experts: list[str]
    discussion_rounds: list[dict]
    consensus: dict | None
    divergences: list[str]
    final_report: str | None
    current_round: int
    max_rounds: int
```

- [ ] **Step 3: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_mdt_state.py -v`
Expected: 6 passed

- [ ] **Step 4: Commit**

```bash
git add src/medmdt/mdt/ tests/test_mdt_state.py
git commit -m "feat: MDT state schema — ExpertOpinion, DiscussionRound, MDTState"
```

---

### Task 2: Expert Base Class + Analysis Prompt

**Files:**
- Create: `src/medmdt/mdt/experts/base.py`
- Create: `src/medmdt/llm/prompts/mdt/__init__.py`
- Create: `src/medmdt/llm/prompts/mdt/expert_analysis.py`
- Test: `tests/test_expert_base.py`

**Interfaces:**
- Consumes: `ExpertOpinion` from Task 1; `create_chat_model(provider, model)` from P1 Task 2; `BaseChatModel` from langchain
- Produces:
  - `BaseExpert.__init__(expert_id: str, name: str, system_prompt: str, knowledge_domains: list[str], llm: BaseChatModel)`
  - `BaseExpert.analyze(patient_info: dict, medical_records: list[dict], knowledge_context: list[dict], previous_rounds: list[dict] | None = None) -> ExpertOpinion`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_expert_base.py
from unittest.mock import MagicMock
import json
import pytest
from medmdt.mdt.experts.base import BaseExpert
from medmdt.mdt.state import ExpertOpinion


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    response = MagicMock()
    response.content = json.dumps({
        "analysis": "患者血压持续偏高",
        "diagnosis": "原发性高血压",
        "recommendation": "建议ACEI类降压药",
        "confidence": 0.85,
        "reasoning": "血压反复>140/90, 排除继发因素",
        "references": ["高血压诊疗指南"],
    })
    llm.invoke.return_value = response
    return llm


@pytest.fixture
def expert(mock_llm):
    return BaseExpert(
        expert_id="cardiologist",
        name="心内科专家",
        system_prompt="你是一位资深心内科主任医师。",
        knowledge_domains=["cardiology"],
        llm=mock_llm,
    )


def test_expert_init(expert):
    assert expert.expert_id == "cardiologist"
    assert expert.name == "心内科专家"
    assert expert.knowledge_domains == ["cardiology"]


def test_expert_analyze_returns_opinion(expert, mock_llm):
    result = expert.analyze(
        patient_info={"chief_complaint": "头痛头晕1月"},
        medical_records=[{"type": "vital", "content": "BP 160/100"}],
        knowledge_context=[{"text": "高血压诊断标准..."}],
    )
    assert isinstance(result, ExpertOpinion)
    assert result.expert_id == "cardiologist"
    assert result.expert_name == "心内科专家"
    assert result.confidence == 0.85
    mock_llm.invoke.assert_called_once()


def test_expert_analyze_includes_system_prompt(expert, mock_llm):
    expert.analyze(
        patient_info={"chief_complaint": "test"},
        medical_records=[],
        knowledge_context=[],
    )
    call_args = mock_llm.invoke.call_args[0][0]
    system_msg = call_args[0]
    assert "心内科主任医师" in system_msg.content


def test_expert_analyze_with_previous_rounds(expert, mock_llm):
    previous = [{"round_num": 1, "opinions": [], "summary": "首轮意见", "divergences": ["诊断分歧"]}]
    expert.analyze(
        patient_info={"chief_complaint": "test"},
        medical_records=[],
        knowledge_context=[],
        previous_rounds=previous,
    )
    call_args = mock_llm.invoke.call_args[0][0]
    user_msg_content = call_args[1].content
    assert "首轮意见" in user_msg_content or "诊断分歧" in user_msg_content


def test_expert_analyze_handles_json_in_markdown_fence(mock_llm):
    response = MagicMock()
    response.content = '```json\n{"analysis":"a","diagnosis":"d","recommendation":"r","confidence":0.7,"reasoning":"r","references":[]}\n```'
    mock_llm.invoke.return_value = response
    expert = BaseExpert(
        expert_id="test", name="Test", system_prompt="test",
        knowledge_domains=[], llm=mock_llm,
    )
    result = expert.analyze(
        patient_info={}, medical_records=[], knowledge_context=[],
    )
    assert isinstance(result, ExpertOpinion)
    assert result.confidence == 0.7
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_expert_base.py -v`
Expected: FAIL — module not found

- [ ] **Step 2: Create the expert analysis prompt**

```python
# src/medmdt/llm/prompts/mdt/expert_analysis.py
EXPERT_ANALYSIS_PROMPT = """请基于以下信息进行专业分析。

## 患者信息
{patient_info}

## 检查记录
{medical_records}

## 知识库参考
{knowledge_context}

{previous_rounds_section}

## 输出要求

请以JSON格式输出你的专业意见：
{{
  "analysis": "详细的病情分析",
  "diagnosis": "诊断结论",
  "recommendation": "治疗建议",
  "confidence": 0.85,
  "reasoning": "推理过程和依据",
  "references": ["参考文献或指南"]
}}

注意：
1. confidence 范围 0.0-1.0，反映你对诊断的确信程度
2. 分析应基于提供的患者资料和知识库内容
3. 只输出JSON，不要其他文字"""

PREVIOUS_ROUNDS_SECTION = """## 前轮讨论记录
以下是之前轮次的讨论记录，请结合这些信息给出你的意见：

{rounds_text}

请特别关注以下分歧点：
{divergences_text}"""

NO_PREVIOUS_ROUNDS = ""
```

- [ ] **Step 3: Implement base.py**

```python
# src/medmdt/mdt/experts/base.py
import json
import re
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from medmdt.mdt.state import ExpertOpinion
from medmdt.llm.prompts.mdt.expert_analysis import (
    EXPERT_ANALYSIS_PROMPT, PREVIOUS_ROUNDS_SECTION, NO_PREVIOUS_ROUNDS,
)


class BaseExpert:
    def __init__(
        self,
        expert_id: str,
        name: str,
        system_prompt: str,
        knowledge_domains: list[str],
        llm: BaseChatModel,
    ):
        self.expert_id = expert_id
        self.name = name
        self.system_prompt = system_prompt
        self.knowledge_domains = knowledge_domains
        self._llm = llm

    def analyze(
        self,
        patient_info: dict,
        medical_records: list[dict],
        knowledge_context: list[dict],
        previous_rounds: list[dict] | None = None,
    ) -> ExpertOpinion:
        previous_section = self._format_previous_rounds(previous_rounds)

        prompt = EXPERT_ANALYSIS_PROMPT.format(
            patient_info=json.dumps(patient_info, ensure_ascii=False, indent=2),
            medical_records=json.dumps(medical_records, ensure_ascii=False, indent=2),
            knowledge_context=json.dumps(knowledge_context, ensure_ascii=False, indent=2),
            previous_rounds_section=previous_section,
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]
        response = self._llm.invoke(messages)
        data = self._parse_json(response.content)

        return ExpertOpinion(
            expert_id=self.expert_id,
            expert_name=self.name,
            **data,
        )

    def _format_previous_rounds(self, rounds: list[dict] | None) -> str:
        if not rounds:
            return NO_PREVIOUS_ROUNDS
        rounds_text = json.dumps(rounds, ensure_ascii=False, indent=2)
        all_divergences = []
        for rd in rounds:
            all_divergences.extend(rd.get("divergences", []))
        divergences_text = "\n".join(f"- {d}" for d in all_divergences) if all_divergences else "无明确分歧"
        return PREVIOUS_ROUNDS_SECTION.format(
            rounds_text=rounds_text,
            divergences_text=divergences_text,
        )

    @staticmethod
    def _parse_json(text: str) -> dict:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        return json.loads(text.strip())
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_expert_base.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/medmdt/mdt/experts/base.py src/medmdt/llm/prompts/mdt/ tests/test_expert_base.py
git commit -m "feat: expert base class — LLM-backed medical specialist with analysis prompt"
```

---

### Task 3: Expert Factory + YAML Config

**Files:**
- Create: `src/medmdt/mdt/experts/factory.py`
- Modify: `config/experts.yaml` (add a third expert for testing variety)
- Test: `tests/test_expert_factory.py`

**Interfaces:**
- Consumes: `BaseExpert` from Task 2; `create_chat_model(provider, model, **kwargs)` from P1; `Settings` from P1 (for `default_llm_provider`, `default_llm_model`)
- Produces:
  - `ExpertConfig(expert_id: str, name: str, system_prompt: str, knowledge_domains: list[str], llm_provider: str, llm_model: str)` — Pydantic model
  - `load_expert_configs(yaml_path: str) -> list[ExpertConfig]` — parses experts.yaml
  - `create_expert(config: ExpertConfig) -> BaseExpert` — instantiates one expert
  - `create_all_experts(yaml_path: str) -> dict[str, BaseExpert]` — loads yaml and creates all experts
  - `select_experts(all_experts: dict[str, BaseExpert], patient_info: dict, llm: BaseChatModel) -> list[str]` — uses LLM to pick relevant expert IDs given patient info

- [ ] **Step 1: Write the failing test**

```python
# tests/test_expert_factory.py
from unittest.mock import patch, MagicMock
import json
import pytest
import yaml
from medmdt.mdt.experts.factory import (
    ExpertConfig, load_expert_configs, create_expert, create_all_experts,
    select_experts,
)
from medmdt.mdt.experts.base import BaseExpert


@pytest.fixture
def experts_yaml(tmp_path):
    config = {
        "defaults": {"llm": {"provider": "openai", "model": "gpt-4o"}},
        "experts": {
            "internist": {
                "name": "内科专家",
                "knowledge_domains": ["internal_medicine"],
                "system_prompt": "你是一位内科专家。",
                "llm": {"provider": "openai", "model": "gpt-4o"},
            },
            "surgeon": {
                "name": "外科专家",
                "knowledge_domains": ["surgery"],
                "system_prompt": "你是一位外科专家。",
            },
        },
    }
    yaml_file = tmp_path / "experts.yaml"
    yaml_file.write_text(yaml.dump(config, allow_unicode=True))
    return str(yaml_file)


def test_load_expert_configs(experts_yaml):
    configs = load_expert_configs(experts_yaml)
    assert len(configs) == 2
    internist = next(c for c in configs if c.expert_id == "internist")
    assert internist.name == "内科专家"
    assert internist.llm_provider == "openai"


def test_load_expert_configs_inherits_defaults(experts_yaml):
    configs = load_expert_configs(experts_yaml)
    surgeon = next(c for c in configs if c.expert_id == "surgeon")
    assert surgeon.llm_provider == "openai"
    assert surgeon.llm_model == "gpt-4o"


@patch("medmdt.mdt.experts.factory.create_chat_model")
def test_create_expert(mock_create):
    mock_create.return_value = MagicMock()
    config = ExpertConfig(
        expert_id="internist", name="内科", system_prompt="test",
        knowledge_domains=["internal_medicine"],
        llm_provider="openai", llm_model="gpt-4o",
    )
    expert = create_expert(config)
    assert isinstance(expert, BaseExpert)
    assert expert.expert_id == "internist"
    mock_create.assert_called_once_with("openai", "gpt-4o")


@patch("medmdt.mdt.experts.factory.create_chat_model")
def test_create_all_experts(mock_create, experts_yaml):
    mock_create.return_value = MagicMock()
    experts = create_all_experts(experts_yaml)
    assert len(experts) == 2
    assert "internist" in experts
    assert "surgeon" in experts
    assert isinstance(experts["internist"], BaseExpert)


def test_select_experts():
    mock_llm = MagicMock()
    response = MagicMock()
    response.content = '{"selected_experts": ["internist", "cardiologist"], "reasoning": "心血管相关"}'
    mock_llm.invoke.return_value = response

    all_experts = {
        "internist": MagicMock(spec=BaseExpert, name="内科"),
        "cardiologist": MagicMock(spec=BaseExpert, name="心内科"),
        "surgeon": MagicMock(spec=BaseExpert, name="外科"),
    }
    selected = select_experts(
        all_experts,
        patient_info={"chief_complaint": "胸闷气短"},
        llm=mock_llm,
    )
    assert "internist" in selected
    assert "cardiologist" in selected


def test_select_experts_returns_all_on_parse_failure():
    mock_llm = MagicMock()
    response = MagicMock()
    response.content = "invalid json"
    mock_llm.invoke.return_value = response

    all_experts = {"a": MagicMock(), "b": MagicMock()}
    selected = select_experts(all_experts, patient_info={}, llm=mock_llm)
    assert set(selected) == {"a", "b"}
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_expert_factory.py -v`
Expected: FAIL — module not found

- [ ] **Step 2: Implement factory.py**

```python
# src/medmdt/mdt/experts/factory.py
import json
import logging
import yaml
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from medmdt.llm.provider import create_chat_model
from medmdt.mdt.experts.base import BaseExpert

logger = logging.getLogger(__name__)


class ExpertConfig(BaseModel):
    expert_id: str
    name: str
    system_prompt: str
    knowledge_domains: list[str] = Field(default_factory=list)
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"


def load_expert_configs(yaml_path: str) -> list[ExpertConfig]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    defaults_llm = data.get("defaults", {}).get("llm", {})
    default_provider = defaults_llm.get("provider", "openai")
    default_model = defaults_llm.get("model", "gpt-4o")

    configs = []
    for expert_id, expert_data in data.get("experts", {}).items():
        llm_cfg = expert_data.get("llm", {})
        configs.append(ExpertConfig(
            expert_id=expert_id,
            name=expert_data["name"],
            system_prompt=expert_data.get("system_prompt", ""),
            knowledge_domains=expert_data.get("knowledge_domains", []),
            llm_provider=llm_cfg.get("provider", default_provider),
            llm_model=llm_cfg.get("model", default_model),
        ))
    return configs


def create_expert(config: ExpertConfig) -> BaseExpert:
    llm = create_chat_model(config.llm_provider, config.llm_model)
    return BaseExpert(
        expert_id=config.expert_id,
        name=config.name,
        system_prompt=config.system_prompt,
        knowledge_domains=config.knowledge_domains,
        llm=llm,
    )


def create_all_experts(yaml_path: str) -> dict[str, BaseExpert]:
    configs = load_expert_configs(yaml_path)
    return {cfg.expert_id: create_expert(cfg) for cfg in configs}


SELECT_EXPERTS_PROMPT = """你是一位MDT会诊协调员。根据患者信息，从以下专家中选择与病情最相关的专家参与会诊。

可用专家：
{experts_list}

患者信息：
{patient_info}

输出JSON格式：
{{"selected_experts": ["expert_id_1", "expert_id_2"], "reasoning": "选择理由"}}

只输出JSON。"""


def select_experts(
    all_experts: dict[str, BaseExpert],
    patient_info: dict,
    llm: BaseChatModel,
) -> list[str]:
    experts_list = "\n".join(
        f"- {eid}: {getattr(e, 'name', eid)}" for eid, e in all_experts.items()
    )
    prompt = SELECT_EXPERTS_PROMPT.format(
        experts_list=experts_list,
        patient_info=json.dumps(patient_info, ensure_ascii=False),
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        data = json.loads(response.content)
        selected = data.get("selected_experts", [])
        return [eid for eid in selected if eid in all_experts]
    except (json.JSONDecodeError, KeyError):
        logger.warning("Failed to parse expert selection; using all experts")
        return list(all_experts.keys())
```

- [ ] **Step 3: Update experts.yaml with a third expert**

Add a `surgeon` entry to `config/experts.yaml`:

```yaml
# config/experts.yaml
defaults:
  llm:
    provider: "openai"
    model: "gpt-4o"

experts:
  internist:
    name: "内科专家"
    knowledge_domains: ["internal_medicine"]
    system_prompt: |
      你是一位资深内科主任医师，拥有20年以上临床经验。
      你擅长内科常见病和疑难病的诊断与治疗。
      请基于提供的患者资料和医学知识，给出专业的分析和建议。
    llm:
      provider: "openai"
      model: "gpt-4o"

  radiologist:
    name: "影像科专家"
    knowledge_domains: ["radiology"]
    system_prompt: |
      你是一位资深影像科主任医师，擅长CT、MRI、X线等医学影像的判读与分析。
      请基于提供的影像描述和相关临床资料，给出专业的影像学分析和诊断建议。
    llm:
      provider: "anthropic"
      model: "claude-sonnet-4-6"

  surgeon:
    name: "外科专家"
    knowledge_domains: ["surgery", "general_surgery"]
    system_prompt: |
      你是一位资深外科主任医师，擅长普外科、腹部外科手术及围手术期管理。
      请基于提供的患者资料和医学知识，评估手术指征并给出外科治疗建议。
    llm:
      provider: "deepseek"
      model: "deepseek-chat"
```

- [ ] **Step 4: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_expert_factory.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/medmdt/mdt/experts/factory.py config/experts.yaml tests/test_expert_factory.py
git commit -m "feat: expert factory — YAML-driven expert creation with LLM-based selection"
```

---

### Task 4: Moderator Agent + Prompts

**Files:**
- Create: `src/medmdt/mdt/moderator.py`
- Create: `src/medmdt/llm/prompts/mdt/moderator_summary.py`
- Create: `src/medmdt/llm/prompts/mdt/report_generation.py`
- Test: `tests/test_moderator.py`

**Interfaces:**
- Consumes: `ExpertOpinion`, `DiscussionRound` from Task 1; `BaseChatModel` from langchain
- Produces:
  - `Moderator.__init__(llm: BaseChatModel, consensus_threshold: float = 0.8)`
  - `Moderator.summarize_round(opinions: list[ExpertOpinion]) -> DiscussionRound` — analyzes opinions, identifies consensus and divergences
  - `Moderator.check_consensus(round_result: DiscussionRound, all_opinions: list[ExpertOpinion]) -> bool` — True if consensus reached
  - `Moderator.generate_report(patient_info: dict, rounds: list[DiscussionRound], consensus: dict | None, divergences: list[str]) -> str` — generates the final MDT consultation report

- [ ] **Step 1: Write the failing test**

```python
# tests/test_moderator.py
from unittest.mock import MagicMock
import json
import pytest
from medmdt.mdt.moderator import Moderator
from medmdt.mdt.state import ExpertOpinion, DiscussionRound


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def moderator(mock_llm):
    return Moderator(llm=mock_llm, consensus_threshold=0.8)


@pytest.fixture
def agreeing_opinions():
    return [
        ExpertOpinion(
            expert_id="internist", expert_name="内科", analysis="高血压",
            diagnosis="原发性高血压", recommendation="ACEI降压",
            confidence=0.9, reasoning="BP>140/90",
        ),
        ExpertOpinion(
            expert_id="cardiologist", expert_name="心内科", analysis="高血压",
            diagnosis="原发性高血压", recommendation="ARB降压",
            confidence=0.85, reasoning="反复血压升高",
        ),
    ]


@pytest.fixture
def diverging_opinions():
    return [
        ExpertOpinion(
            expert_id="internist", expert_name="内科", analysis="肿块",
            diagnosis="良性肿瘤", recommendation="观察随访",
            confidence=0.6, reasoning="影像学特征偏良性",
        ),
        ExpertOpinion(
            expert_id="surgeon", expert_name="外科", analysis="肿块",
            diagnosis="疑似恶性肿瘤", recommendation="手术切除",
            confidence=0.7, reasoning="不排除恶性可能",
        ),
    ]


def test_moderator_init(moderator):
    assert moderator._consensus_threshold == 0.8


def test_summarize_round(moderator, mock_llm, agreeing_opinions):
    response = MagicMock()
    response.content = json.dumps({
        "summary": "两位专家均诊断为原发性高血压",
        "divergences": ["降压药物选择存在分歧：ACEI vs ARB"],
    })
    mock_llm.invoke.return_value = response

    result = moderator.summarize_round(agreeing_opinions)
    assert isinstance(result, DiscussionRound)
    assert "高血压" in result.summary
    assert len(result.divergences) == 1


def test_check_consensus_true(moderator, agreeing_opinions):
    rd = DiscussionRound(round_num=1, opinions=agreeing_opinions, divergences=[])
    assert moderator.check_consensus(rd, agreeing_opinions) is True


def test_check_consensus_false_low_confidence(moderator, diverging_opinions):
    rd = DiscussionRound(
        round_num=1, opinions=diverging_opinions,
        divergences=["诊断分歧"],
    )
    assert moderator.check_consensus(rd, diverging_opinions) is False


def test_check_consensus_false_with_divergences(moderator, agreeing_opinions):
    rd = DiscussionRound(
        round_num=1, opinions=agreeing_opinions,
        divergences=["存在分歧"],
    )
    assert moderator.check_consensus(rd, agreeing_opinions) is False


def test_generate_report(moderator, mock_llm, agreeing_opinions):
    response = MagicMock()
    response.content = "# MDT会诊报告\n\n## 诊断意见\n原发性高血压\n\n## 治疗方案\nACEI降压"
    mock_llm.invoke.return_value = response

    rd = DiscussionRound(round_num=1, opinions=agreeing_opinions, summary="共识")
    report = moderator.generate_report(
        patient_info={"chief_complaint": "头痛"},
        rounds=[rd],
        consensus={"diagnosis": "原发性高血压"},
        divergences=[],
    )
    assert isinstance(report, str)
    assert len(report) > 0
    mock_llm.invoke.assert_called_once()


def test_generate_report_with_divergences(moderator, mock_llm, diverging_opinions):
    response = MagicMock()
    response.content = "# 报告\n\n有分歧"
    mock_llm.invoke.return_value = response

    rd = DiscussionRound(round_num=1, opinions=diverging_opinions, summary="有分歧")
    report = moderator.generate_report(
        patient_info={},
        rounds=[rd],
        consensus=None,
        divergences=["诊断分歧：良性 vs 恶性"],
    )
    assert isinstance(report, str)
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_moderator.py -v`
Expected: FAIL — module not found

- [ ] **Step 2: Create moderator summary prompt**

```python
# src/medmdt/llm/prompts/mdt/moderator_summary.py
MODERATOR_SUMMARY_PROMPT = """你是一位MDT会诊主持人。请汇总以下各位专家的意见，识别共识和分歧。

## 各专家意见

{opinions_text}

## 输出要求

请以JSON格式输出汇总结果：
{{
  "summary": "简要总结各专家的核心观点和共识",
  "divergences": ["分歧点1：具体描述", "分歧点2：具体描述"]
}}

注意：
1. summary 应提炼共同点和核心观点
2. divergences 列出具体的分歧，标明哪些专家持不同意见
3. 如果各专家意见一致，divergences 为空列表
4. 只输出JSON"""
```

- [ ] **Step 3: Create report generation prompt**

```python
# src/medmdt/llm/prompts/mdt/report_generation.py
REPORT_GENERATION_PROMPT = """你是一位MDT会诊报告撰写专家。请根据以下会诊过程生成正式的MDT会诊报告。

## 患者信息
{patient_info}

## 会诊过程
{rounds_text}

## 会诊结论
{conclusion}

## 输出要求

请生成一份结构完整的MDT会诊报告，包含以下部分：
1. 患者基本信息
2. 参与会诊的专家
3. 各专家意见摘要
4. 会诊讨论要点
5. 最终诊断意见（如有共识）
6. 治疗建议
7. 未解决的分歧（如有）
8. 后续随访建议

报告应当专业、客观、完整。直接输出报告正文。"""
```

- [ ] **Step 4: Implement moderator.py**

```python
# src/medmdt/mdt/moderator.py
import json
import re
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from medmdt.mdt.state import ExpertOpinion, DiscussionRound
from medmdt.llm.prompts.mdt.moderator_summary import MODERATOR_SUMMARY_PROMPT
from medmdt.llm.prompts.mdt.report_generation import REPORT_GENERATION_PROMPT


class Moderator:
    def __init__(self, llm: BaseChatModel, consensus_threshold: float = 0.8):
        self._llm = llm
        self._consensus_threshold = consensus_threshold

    def summarize_round(self, opinions: list[ExpertOpinion]) -> DiscussionRound:
        opinions_text = "\n\n".join(
            f"### {op.expert_name} ({op.expert_id})\n"
            f"- 诊断：{op.diagnosis}\n"
            f"- 分析：{op.analysis}\n"
            f"- 建议：{op.recommendation}\n"
            f"- 置信度：{op.confidence}\n"
            f"- 推理：{op.reasoning}"
            for op in opinions
        )
        prompt = MODERATOR_SUMMARY_PROMPT.format(opinions_text=opinions_text)
        response = self._llm.invoke([
            SystemMessage(content="你是MDT会诊主持人，负责汇总和协调各专家意见。"),
            HumanMessage(content=prompt),
        ])
        data = self._parse_json(response.content)
        return DiscussionRound(
            round_num=0,
            opinions=opinions,
            summary=data.get("summary", ""),
            divergences=data.get("divergences", []),
        )

    def check_consensus(self, round_result: DiscussionRound, all_opinions: list[ExpertOpinion]) -> bool:
        if round_result.divergences:
            return False
        return all(op.confidence >= self._consensus_threshold for op in all_opinions)

    def generate_report(
        self,
        patient_info: dict,
        rounds: list[DiscussionRound],
        consensus: dict | None,
        divergences: list[str],
    ) -> str:
        rounds_text = ""
        for rd in rounds:
            rounds_text += f"\n### 第{rd.round_num}轮\n"
            if rd.summary:
                rounds_text += f"汇总：{rd.summary}\n"
            for op in rd.opinions:
                rounds_text += f"- {op.expert_name}：{op.diagnosis}（置信度{op.confidence}）\n"
            if rd.divergences:
                rounds_text += f"分歧：{', '.join(rd.divergences)}\n"

        if consensus:
            conclusion = f"达成共识：{json.dumps(consensus, ensure_ascii=False)}"
        else:
            conclusion = f"未达成完全共识。分歧点：\n" + "\n".join(f"- {d}" for d in divergences)

        prompt = REPORT_GENERATION_PROMPT.format(
            patient_info=json.dumps(patient_info, ensure_ascii=False, indent=2),
            rounds_text=rounds_text,
            conclusion=conclusion,
        )
        response = self._llm.invoke([
            SystemMessage(content="你是一位专业的医学报告撰写者。"),
            HumanMessage(content=prompt),
        ])
        return response.content

    @staticmethod
    def _parse_json(text: str) -> dict:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        return json.loads(text.strip())
```

- [ ] **Step 5: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_moderator.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add src/medmdt/mdt/moderator.py src/medmdt/llm/prompts/mdt/moderator_summary.py src/medmdt/llm/prompts/mdt/report_generation.py tests/test_moderator.py
git commit -m "feat: moderator agent — round summarization, consensus detection, report generation"
```

---

### Task 5: Multi-Round Discussion Logic

**Files:**
- Create: `src/medmdt/mdt/rounds.py`
- Test: `tests/test_rounds.py`

**Interfaces:**
- Consumes: `BaseExpert.analyze(...)` from Task 2; `Moderator.summarize_round(...)`, `Moderator.check_consensus(...)` from Task 4; `ExpertOpinion`, `DiscussionRound`, `MDTState` from Task 1
- Produces:
  - `run_expert_round(experts: dict[str, BaseExpert], expert_ids: list[str], patient_info: dict, medical_records: list[dict], knowledge_context: list[dict], previous_rounds: list[dict] | None = None) -> list[ExpertOpinion]` — runs all selected experts and returns their opinions
  - `should_continue(current_round: int, max_rounds: int, has_consensus: bool) -> bool` — returns True if another round should run

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rounds.py
from unittest.mock import MagicMock
import pytest
from medmdt.mdt.rounds import run_expert_round, should_continue
from medmdt.mdt.state import ExpertOpinion
from medmdt.mdt.experts.base import BaseExpert


@pytest.fixture
def mock_experts():
    def make_expert(expert_id, name, diagnosis, confidence):
        expert = MagicMock(spec=BaseExpert)
        expert.expert_id = expert_id
        expert.name = name
        expert.analyze.return_value = ExpertOpinion(
            expert_id=expert_id, expert_name=name,
            analysis="analysis", diagnosis=diagnosis,
            recommendation="rec", confidence=confidence,
            reasoning="reason",
        )
        return expert

    return {
        "internist": make_expert("internist", "内科", "高血压", 0.9),
        "cardiologist": make_expert("cardiologist", "心内科", "高血压", 0.85),
        "surgeon": make_expert("surgeon", "外科", "观察", 0.7),
    }


def test_run_expert_round_all_experts(mock_experts):
    opinions = run_expert_round(
        experts=mock_experts,
        expert_ids=["internist", "cardiologist"],
        patient_info={"complaint": "头痛"},
        medical_records=[],
        knowledge_context=[],
    )
    assert len(opinions) == 2
    assert all(isinstance(op, ExpertOpinion) for op in opinions)
    mock_experts["internist"].analyze.assert_called_once()
    mock_experts["cardiologist"].analyze.assert_called_once()
    mock_experts["surgeon"].analyze.assert_not_called()


def test_run_expert_round_passes_previous_rounds(mock_experts):
    previous = [{"round_num": 1, "summary": "test"}]
    run_expert_round(
        experts=mock_experts,
        expert_ids=["internist"],
        patient_info={},
        medical_records=[],
        knowledge_context=[],
        previous_rounds=previous,
    )
    call_kwargs = mock_experts["internist"].analyze.call_args[1]
    assert call_kwargs["previous_rounds"] == previous


def test_run_expert_round_first_round_no_previous(mock_experts):
    run_expert_round(
        experts=mock_experts,
        expert_ids=["internist"],
        patient_info={},
        medical_records=[],
        knowledge_context=[],
    )
    call_kwargs = mock_experts["internist"].analyze.call_args[1]
    assert call_kwargs["previous_rounds"] is None


def test_should_continue_first_round():
    assert should_continue(current_round=0, max_rounds=3, has_consensus=False) is True


def test_should_continue_no_consensus():
    assert should_continue(current_round=2, max_rounds=3, has_consensus=False) is True


def test_should_continue_consensus_reached():
    assert should_continue(current_round=1, max_rounds=3, has_consensus=True) is False


def test_should_continue_max_rounds_reached():
    assert should_continue(current_round=3, max_rounds=3, has_consensus=False) is False


def test_should_continue_past_max_rounds():
    assert should_continue(current_round=5, max_rounds=3, has_consensus=False) is False
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_rounds.py -v`
Expected: FAIL — module not found

- [ ] **Step 2: Implement rounds.py**

```python
# src/medmdt/mdt/rounds.py
from medmdt.mdt.state import ExpertOpinion
from medmdt.mdt.experts.base import BaseExpert


def run_expert_round(
    experts: dict[str, BaseExpert],
    expert_ids: list[str],
    patient_info: dict,
    medical_records: list[dict],
    knowledge_context: list[dict],
    previous_rounds: list[dict] | None = None,
) -> list[ExpertOpinion]:
    opinions = []
    for eid in expert_ids:
        expert = experts[eid]
        opinion = expert.analyze(
            patient_info=patient_info,
            medical_records=medical_records,
            knowledge_context=knowledge_context,
            previous_rounds=previous_rounds,
        )
        opinions.append(opinion)
    return opinions


def should_continue(current_round: int, max_rounds: int, has_consensus: bool) -> bool:
    if has_consensus:
        return False
    return current_round < max_rounds
```

- [ ] **Step 3: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_rounds.py -v`
Expected: 7 passed

- [ ] **Step 4: Commit**

```bash
git add src/medmdt/mdt/rounds.py tests/test_rounds.py
git commit -m "feat: multi-round discussion logic — expert round execution and termination"
```

---

### Task 6: LangGraph MDT State Graph

**Files:**
- Create: `src/medmdt/mdt/graph.py`
- Test: `tests/test_mdt_graph.py`

**Interfaces:**
- Consumes: `MDTState` from Task 1; `BaseExpert` from Task 2; `create_all_experts`, `select_experts` from Task 3; `Moderator` from Task 4; `run_expert_round`, `should_continue` from Task 5; `FusionRetriever.retrieve()` from P1
- Produces:
  - `build_mdt_graph(experts: dict[str, BaseExpert], moderator: Moderator, retriever: FusionRetriever, selector_llm: BaseChatModel) -> CompiledGraph` — creates and compiles the LangGraph MDT workflow
  - `run_consultation(graph: CompiledGraph, patient_info: dict, medical_records: list[dict], max_rounds: int = 3) -> MDTState` — convenience function to run a full consultation

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mdt_graph.py
from unittest.mock import MagicMock, patch
import json
import pytest
from medmdt.mdt.graph import build_mdt_graph, run_consultation
from medmdt.mdt.state import MDTState, ExpertOpinion, DiscussionRound
from medmdt.mdt.experts.base import BaseExpert
from medmdt.mdt.moderator import Moderator


@pytest.fixture
def mock_expert():
    def make(eid, name, diagnosis, confidence=0.9):
        expert = MagicMock(spec=BaseExpert)
        expert.expert_id = eid
        expert.name = name
        expert.knowledge_domains = ["test"]
        expert.analyze.return_value = ExpertOpinion(
            expert_id=eid, expert_name=name, analysis="分析",
            diagnosis=diagnosis, recommendation="建议",
            confidence=confidence, reasoning="推理",
        )
        return expert
    return make


@pytest.fixture
def mock_moderator():
    mod = MagicMock(spec=Moderator)
    mod.summarize_round.return_value = DiscussionRound(
        round_num=1, opinions=[], summary="意见一致", divergences=[],
    )
    mod.check_consensus.return_value = True
    mod.generate_report.return_value = "# MDT会诊报告\n\n共识诊断：高血压"
    return mod


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        MagicMock(text="高血压诊断标准", score=0.9, source="vector", metadata={}),
    ]
    return retriever


@pytest.fixture
def mock_selector_llm():
    llm = MagicMock()
    response = MagicMock()
    response.content = '{"selected_experts": ["internist", "cardiologist"], "reasoning": "心血管"}'
    llm.invoke.return_value = response
    return llm


def test_build_mdt_graph(mock_expert, mock_moderator, mock_retriever, mock_selector_llm):
    experts = {
        "internist": mock_expert("internist", "内科", "高血压"),
        "cardiologist": mock_expert("cardiologist", "心内科", "高血压"),
    }
    graph = build_mdt_graph(experts, mock_moderator, mock_retriever, mock_selector_llm)
    assert graph is not None


def test_run_consultation_reaches_consensus(
    mock_expert, mock_moderator, mock_retriever, mock_selector_llm
):
    experts = {
        "internist": mock_expert("internist", "内科", "高血压"),
        "cardiologist": mock_expert("cardiologist", "心内科", "高血压"),
    }
    graph = build_mdt_graph(experts, mock_moderator, mock_retriever, mock_selector_llm)
    result = run_consultation(
        graph,
        patient_info={"chief_complaint": "头痛头晕"},
        medical_records=[{"type": "vital", "content": "BP 160/100"}],
    )
    assert result["final_report"] is not None
    assert "会诊报告" in result["final_report"]
    assert result["current_round"] >= 1


def test_run_consultation_max_rounds(
    mock_expert, mock_moderator, mock_retriever, mock_selector_llm
):
    experts = {
        "internist": mock_expert("internist", "内科", "良性", 0.6),
        "surgeon": mock_expert("surgeon", "外科", "恶性", 0.7),
    }
    mock_moderator.check_consensus.return_value = False
    mock_moderator.summarize_round.return_value = DiscussionRound(
        round_num=1, opinions=[], summary="分歧", divergences=["诊断分歧"],
    )

    graph = build_mdt_graph(experts, mock_moderator, mock_retriever, mock_selector_llm)
    result = run_consultation(
        graph,
        patient_info={"chief_complaint": "腹部肿块"},
        medical_records=[],
        max_rounds=2,
    )
    assert result["final_report"] is not None
    assert result["current_round"] <= 2


def test_run_consultation_stores_discussion_rounds(
    mock_expert, mock_moderator, mock_retriever, mock_selector_llm
):
    experts = {
        "internist": mock_expert("internist", "内科", "高血压"),
    }
    graph = build_mdt_graph(experts, mock_moderator, mock_retriever, mock_selector_llm)
    result = run_consultation(
        graph,
        patient_info={"chief_complaint": "头痛"},
        medical_records=[],
    )
    assert len(result["discussion_rounds"]) >= 1
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_mdt_graph.py -v`
Expected: FAIL — module not found

- [ ] **Step 2: Implement graph.py**

```python
# src/medmdt/mdt/graph.py
from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel

from medmdt.mdt.state import MDTState, DiscussionRound
from medmdt.mdt.experts.base import BaseExpert
from medmdt.mdt.experts.factory import select_experts
from medmdt.mdt.moderator import Moderator
from medmdt.mdt.rounds import run_expert_round, should_continue
from medmdt.knowledge.retriever import FusionRetriever


def build_mdt_graph(
    experts: dict[str, BaseExpert],
    moderator: Moderator,
    retriever: FusionRetriever,
    selector_llm: BaseChatModel,
):
    def retrieve_knowledge(state: MDTState) -> dict:
        query = state["patient_info"].get("chief_complaint", "")
        if state["medical_records"]:
            query += " " + " ".join(
                r.get("content", "") for r in state["medical_records"]
            )
        results = retriever.retrieve(query, top_k=10)
        return {
            "knowledge_context": [
                {"text": r.text, "score": r.score, "source": r.source}
                for r in results
            ]
        }

    def select_expert_ids(state: MDTState) -> dict:
        selected = select_experts(experts, state["patient_info"], selector_llm)
        if not selected:
            selected = list(experts.keys())
        return {"experts": selected}

    def run_discussion_round(state: MDTState) -> dict:
        current = state["current_round"]
        previous = state["discussion_rounds"] if current > 0 else None

        opinions = run_expert_round(
            experts=experts,
            expert_ids=state["experts"],
            patient_info=state["patient_info"],
            medical_records=state["medical_records"],
            knowledge_context=state["knowledge_context"],
            previous_rounds=previous,
        )

        round_result = moderator.summarize_round(opinions)
        round_result.round_num = current + 1

        has_consensus = moderator.check_consensus(round_result, opinions)

        round_dict = round_result.model_dump()
        new_rounds = state["discussion_rounds"] + [round_dict]

        update = {
            "discussion_rounds": new_rounds,
            "current_round": current + 1,
            "divergences": round_result.divergences,
        }

        if has_consensus:
            diagnoses = [op.diagnosis for op in opinions]
            update["consensus"] = {
                "diagnosis": diagnoses[0] if diagnoses else "",
                "confidence": sum(op.confidence for op in opinions) / len(opinions),
                "experts_agreed": [op.expert_id for op in opinions],
            }
        else:
            update["consensus"] = None

        return update

    def generate_final_report(state: MDTState) -> dict:
        rounds = [DiscussionRound.model_validate(rd) for rd in state["discussion_rounds"]]
        report = moderator.generate_report(
            patient_info=state["patient_info"],
            rounds=rounds,
            consensus=state["consensus"],
            divergences=state["divergences"],
        )
        return {"final_report": report}

    def should_discuss_more(state: MDTState) -> str:
        has_consensus = state["consensus"] is not None
        if should_continue(state["current_round"], state["max_rounds"], has_consensus):
            return "continue"
        return "report"

    graph = StateGraph(MDTState)

    graph.add_node("retrieve_knowledge", retrieve_knowledge)
    graph.add_node("select_experts", select_expert_ids)
    graph.add_node("discussion_round", run_discussion_round)
    graph.add_node("generate_report", generate_final_report)

    graph.set_entry_point("retrieve_knowledge")
    graph.add_edge("retrieve_knowledge", "select_experts")
    graph.add_edge("select_experts", "discussion_round")
    graph.add_conditional_edges(
        "discussion_round",
        should_discuss_more,
        {"continue": "discussion_round", "report": "generate_report"},
    )
    graph.add_edge("generate_report", END)

    return graph.compile()


def run_consultation(
    graph,
    patient_info: dict,
    medical_records: list[dict],
    max_rounds: int = 3,
) -> MDTState:
    initial_state: MDTState = {
        "patient_info": patient_info,
        "medical_records": medical_records,
        "knowledge_context": [],
        "experts": [],
        "discussion_rounds": [],
        "consensus": None,
        "divergences": [],
        "final_report": None,
        "current_round": 0,
        "max_rounds": max_rounds,
    }
    result = graph.invoke(initial_state)
    return result
```

- [ ] **Step 3: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_mdt_graph.py -v`
Expected: 4 passed

- [ ] **Step 4: Run full test suite**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/ -v`
Expected: All tests pass (P1 96 + P2 new tests)

- [ ] **Step 5: Commit**

```bash
git add src/medmdt/mdt/graph.py tests/test_mdt_graph.py
git commit -m "feat: LangGraph MDT state graph — full consultation workflow with multi-round discussion"
```
