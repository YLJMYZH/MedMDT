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
