# src/medmdt/llm/prompts/extraction/entity_extraction.py
ENTITY_RELATION_PROMPT = """你是一个医学知识提取专家。从以下医学文本中提取实体和关系。

实体类型：disease（疾病）, symptom（症状）, drug（药物）, procedure（操作/手术）, anatomy（解剖部位）, lab_test（检验项目）, other（其他）

关系类型：treats（治疗）, indicates（提示/指向）, contraindicated_for（禁忌于）, located_in（位于）, diagnoses（诊断）, causes（导致）, first_line_treatment_for（一线治疗）, side_effect_of（副作用）

输出JSON格式：
{{
  "entities": [
    {{"name": "实体名称", "type": "实体类型", "aliases": ["别名1"]}}
  ],
  "relations": [
    {{
      "head": "头实体名称",
      "relation": "关系类型",
      "tail": "尾实体名称",
      "evidence": "原文中支持该关系的证据文本",
      "confidence": 0.95
    }}
  ]
}}

注意：
1. 实体名称使用标准医学术语
2. confidence 范围 0.0-1.0，反映关系的确定程度
3. evidence 必须是原文中的原始文本片段
4. 只输出JSON

医学文本：
{text}"""
