import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# 设置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = open("openai_api_key.txt").read().strip()
client = OpenAI()

def cal_gen(question, answers, generation):
    exp = {}

    def build_prompt(metric):
        descriptions = {
            "comprehensiveness": (
                "comprehensiveness",
                "whether the thinking considers all important aspects and is thorough",
                """Scoring Guide (0–10):
- 10: Extremely thorough, covering all relevant angles and considerations with depth.
- 8–9: Covers most key aspects clearly and thoughtfully; only minor omissions.
- 6–7: Covers some important aspects, but lacks depth or overlooks notable areas.
- 4–5: Touches on a few relevant points, but overall lacks substance or completeness.
- 1–3: Sparse or shallow treatment of the topic; misses most key aspects.
- 0: No comprehensiveness at all; completely superficial or irrelevant."""
            ),
            "knowledgeability": (
                "knowledgeability",
                "whether the thinking is rich in insightful, domain-relevant knowledge",
                """Scoring Guide (0–10):
- 10: Demonstrates exceptional depth and insight with strong domain-specific knowledge.
- 8–9: Shows clear domain knowledge with good insight; mostly accurate and relevant.
- 6–7: Displays some understanding, but lacks depth or has notable gaps.
- 4–5: Limited knowledge shown; understanding is basic or somewhat flawed.
- 1–3: Poor grasp of relevant knowledge; superficial or mostly incorrect.
- 0: No evidence of meaningful knowledge."""
            ),
            "correctness": (
                "correctness",
                "whether the reasoning and answer are logically and factually correct",
                """Scoring Guide (0–10):
- 10: Fully accurate and logically sound; no flaws in reasoning or facts.
- 8–9: Mostly correct with minor inaccuracies or small logical gaps.
- 6–7: Partially correct; some key flaws or inconsistencies present.
- 4–5: Noticeable incorrect reasoning or factual errors throughout.
- 1–3: Largely incorrect, misleading, or illogical.
- 0: Entirely wrong or nonsensical."""
            ),
            "relevance": (
                "relevance",
                "whether the reasoning and answer are highly relevant and helpful to the question",
                """Scoring Guide (0–10):
- 10: Fully focused on the question; highly relevant and helpful.
- 8–9: Mostly on point; minor digressions but overall useful.
- 6–7: Generally relevant, but includes distractions or less helpful parts.
- 4–5: Limited relevance; much of the response is off-topic or unhelpful.
- 1–3: Barely related to the question or largely unhelpful.
- 0: Entirely irrelevant."""
            ),
            "diversity": (
                "diversity",
                "whether the reasoning is thought-provoking, offering varied or novel perspectives",
                """Scoring Guide (0–10):
- 10: Exceptionally rich and original; demonstrates multiple fresh and thought-provoking ideas.
- 8–9: Contains a few novel angles or interesting perspectives.
- 6–7: Some variety, but generally safe or conventional.
- 4–5: Mostly standard thinking; minimal diversity.
- 1–3: Very predictable or monotonous.
- 0: No diversity or originality at all."""
            ),
            "logical_coherence": (
                "logical coherence",
                "whether the reasoning is internally consistent, clear, and well-structured",
                """Scoring Guide (0–10):
- 10: Highly logical, clear, and easy to follow throughout.
- 8–9: Well-structured with minor lapses in flow or clarity.
- 6–7: Some structure and logic, but a few confusing or weakly connected parts.
- 4–5: Often disorganized or unclear; logic is hard to follow.
- 1–3: Poorly structured and incoherent.
- 0: Entirely illogical or unreadable."""
            ),
            "factuality": (
                "factuality",
                "whether the reasoning and answer are based on accurate and verifiable facts",
                """Scoring Guide (0–10):
- 10: All facts are accurate and verifiable.
- 8–9: Mostly accurate; only minor factual issues.
- 6–7: Contains some factual inaccuracies or unverified claims.
- 4–5: Several significant factual errors.
- 1–3: Mostly false or misleading.
- 0: Completely fabricated or factually wrong throughout."""
            )
        }

        title, goal, rubric = descriptions[metric]

        return f"""---Role---

You are a helpful assistant evaluating the **{title}** of a generated response.

---Question---

{question}

---Golden Answers---

{str(answers)}

---Evaluation Goal---

Evaluate **{goal}** using a **0–10 integer scale**.

{rubric}

Output format:
<score>
your_score_here (an integer from 0 to 10)
</score>
<explanation>
Explain why you gave this score.
</explanation>

---Generation to be Evaluated---

{generation}
"""


    def score_extraction(metric):
        try:
            prompt = build_prompt(metric)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content
            
            score_str = content.split("<score>")[1].split("</score>")[0].strip()
            explanation = content.split("<explanation>")[1].split("</explanation>")[0].strip()
            score = int(score_str)
        except Exception:
            score = 5
            explanation = "Failed to parse GPT output. Defaulted to score=5."

        return metric, {"score": score / 10, "explanation": explanation}

    metrics = [
        "comprehensiveness", "knowledgeability", "correctness",
        "relevance", "diversity", "logical_coherence", "factuality"
    ]

    with ThreadPoolExecutor(max_workers=7) as executor:
        results = executor.map(score_extraction, metrics)

    for metric, result in results:
        exp[metric] = result

    overall_score = round(np.mean([exp[m]["score"] for m in metrics]), 4)
    return {"score": overall_score, "explanation": exp}