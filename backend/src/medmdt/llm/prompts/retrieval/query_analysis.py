# src/medmdt/llm/prompts/retrieval/query_analysis.py
"""Prompt template for medical query analysis.

The LLM extracts keywords, medical entities, and recommends retrieval-channel
weights so that the FusionRetriever can adapt its behaviour per query.
"""

QUERY_ANALYSIS_PROMPT = """你是一个医学查询分析器。分析用户的查询意图，提取关键信息。

输出JSON格式：
{{
  "keywords": ["关键词1", "关键词2"],
  "entities": ["医学实体1", "医学实体2"],
  "weights": {{
    "graph": 0.33,
    "vector": 0.34,
    "keyword": 0.33
  }}
}}

权重说明：
- graph: 适合实体关系查询（如"某药物的禁忌症"、"某疾病的治疗方案"）
- vector: 适合语义检索（如"如何处理老年患者的低血糖"）
- keyword: 适合精确术语匹配（如"HbA1c正常范围"）

三个权重之和必须为1.0。只输出JSON。

用户查询：{query}"""
