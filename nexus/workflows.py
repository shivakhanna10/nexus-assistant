"""Stand-alone workflows from langsmith-masterclass.

* Essay evaluator – parallel fan-out/fan-in LangGraph with structured output
  and per-dimension LangSmith tags/metadata                  (5_langgraph.py)
* Report -> summary – sequential LCEL chain                   (2_sequential_chain.py)
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from pydantic import BaseModel, Field

from .llm import get_chat_model, get_utility_model, message_text


# ========================================================= essay evaluator
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)


class EssayState(TypedDict, total=False):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]  # merges parallel branches
    avg_score: float


DIMENSIONS = {
    "language": ("language_feedback", "the language quality (grammar, vocabulary, fluency)"),
    "analysis": ("analysis_feedback", "the depth of analysis (arguments, evidence, nuance)"),
    "clarity": ("clarity_feedback", "the clarity of thought (structure, coherence, flow)"),
}


def _make_dimension_node(dimension: str):
    key, what = DIMENSIONS[dimension]

    @traceable(name=f"evaluate_{dimension}", tags=[f"dimension:{dimension}"], metadata={"dimension": dimension})
    def node(state: EssayState):
        out = get_utility_model().with_structured_output(EvaluationSchema).invoke(
            f"Evaluate {what} of the following essay. Give specific, actionable feedback "
            f"and assign a score out of 10.\n\n{state['essay']}"
        )
        return {key: out.feedback, "individual_scores": [out.score]}

    return node


@traceable(name="final_evaluation", tags=["aggregate"])
def final_evaluation(state: EssayState):
    prompt = (
        "Based on the following feedback, write a concise overall evaluation with the top 3 improvements.\n\n"
        f"Language feedback: {state.get('language_feedback', '')}\n"
        f"Depth of analysis feedback: {state.get('analysis_feedback', '')}\n"
        f"Clarity of thought feedback: {state.get('clarity_feedback', '')}\n"
    )
    overall = message_text(get_chat_model(temperature=0).invoke(prompt))
    scores = state.get("individual_scores", []) or []
    return {"overall_feedback": overall, "avg_score": (sum(scores) / len(scores)) if scores else 0.0}


def build_essay_graph():
    g = StateGraph(EssayState)
    for dim in DIMENSIONS:
        g.add_node(f"evaluate_{dim}", _make_dimension_node(dim))
        g.add_edge(START, f"evaluate_{dim}")
        g.add_edge(f"evaluate_{dim}", "final_evaluation")
    g.add_node("final_evaluation", final_evaluation)
    g.add_edge("final_evaluation", END)
    return g.compile()


def evaluate_essay(essay: str) -> EssayState:
    return build_essay_graph().invoke(
        {"essay": essay},
        config={
            "run_name": "evaluate_essay",
            "tags": ["essay", "langgraph", "evaluation"],
            "metadata": {"essay_length": len(essay), "dimensions": list(DIMENSIONS)},
        },
    )


# ======================================================= report -> summary
REPORT_PROMPT = PromptTemplate.from_template("Generate a detailed, well-structured report on {topic}.")
SUMMARY_PROMPT = PromptTemplate.from_template(
    "Generate a {points}-point summary of the following text:\n\n{text}"
)


def report_and_summary(topic: str, points: int = 5) -> dict:
    model, parser = get_chat_model(), StrOutputParser()
    report_chain = REPORT_PROMPT | model | parser
    summary_chain = SUMMARY_PROMPT | model | parser
    report = report_chain.invoke({"topic": topic}, config={"run_name": "report", "tags": ["report"]})
    summary = summary_chain.invoke(
        {"text": report, "points": points}, config={"run_name": "summary", "tags": ["report"]}
    )
    return {"report": report, "summary": summary}
