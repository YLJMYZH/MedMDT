# src/medmdt/mdt/experts/factory.py
import json
import logging
import yaml
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from medmdt.llm.network_policy import validate_base_url
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


def create_expert(config: ExpertConfig, **llm_kwargs) -> BaseExpert:
    validate_base_url(config.llm_provider, llm_kwargs.get("base_url"))
    llm = create_chat_model(config.llm_provider, config.llm_model, **llm_kwargs)
    return BaseExpert(
        expert_id=config.expert_id,
        name=config.name,
        system_prompt=config.system_prompt,
        knowledge_domains=config.knowledge_domains,
        llm=llm,
    )


def create_all_experts(yaml_path: str) -> dict[str, BaseExpert]:
    from medmdt.config.runtime import load_llm_settings, get_expert_endpoint, get_llm_kwargs

    runtime = load_llm_settings()
    configs = load_expert_configs(yaml_path)
    experts = {}
    for cfg in configs:
        endpoint = get_expert_endpoint(runtime, cfg.expert_id)
        cfg.llm_provider = endpoint.provider
        cfg.llm_model = endpoint.model
        experts[cfg.expert_id] = create_expert(cfg, **get_llm_kwargs(endpoint))
    return experts


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
