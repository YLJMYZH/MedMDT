# src/medmdt/llm/prompts/extraction/relation_extraction.py
CHUNK_SUMMARY_PROMPT = """你是一个医学文本分析专家。将以下文本拆分为知识块，并为每个块生成摘要和关键词。

输出JSON格式：
{{
  "chunks": [
    {{
      "text": "原始文本段落（保持原文）",
      "summary": "一句话摘要",
      "keywords": ["关键词1", "关键词2"],
      "metadata": {{
        "section": "所属章节（如：诊断标准、治疗方案、预后评估等）",
        "guideline_level": "推荐等级（如有，如：A级推荐、专家共识等）"
      }}
    }}
  ]
}}

注意：
1. 每个chunk应该是一个语义完整的段落
2. keywords提取3-5个核心术语
3. 只输出JSON

医学文本：
{text}"""
