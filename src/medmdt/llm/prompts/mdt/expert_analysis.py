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
