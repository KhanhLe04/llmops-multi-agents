#!/usr/bin/env python
"""
Evaluate LLM/RAG answers using RAGAS and plot average scores.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.dataset_schema import EvaluationDataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from config import Config

llm_model = ChatGoogleGenerativeAI(
    model=Config.GOOGLE_LLM_MODEL,
    temperature=Config.GOOGLE_LLM_TEMPERATURE,
    max_output_tokens=Config.GOOGLE_LLM_MAX_OUTPUT_TOKENS,
    google_api_key=Config.GOOGLE_API_KEY
)

def load_dataset(path: Path) -> EvaluationDataset:
    """Load JSON list and convert to EvaluationDataset (normalize keys)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    normalized = []
    for sample in raw:
        question = sample.get("question", "").strip()
        answer = sample.get("answer", "").strip()
        normalized.append(
            {
                "question": question,
                "user_input": question,  # required by ragas>=0.1.6
                "answer": answer,
                "response": answer,      # alias for metrics expecting 'response'
                "contexts": sample.get("contexts", []),
                "retrieved_contexts": sample.get("contexts", []),
                "ground_truth": sample.get("ground_truth", ""),
                "reference": sample.get("ground_truth", ""),
            }
        )
    return EvaluationDataset.from_list(normalized)


def plot_scores(scores):
    """Plot average scores with Matplotlib."""
    df = scores.to_pandas()
    avg = df.mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    avg.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (0-1)")
    ax.set_title("RAG Evaluation Metrics (Ragas)")
    for i, value in enumerate(avg):
        ax.text(i, value + 0.02, f"{value:.2f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("rag_evaluation_scores.png", dpi=200)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="src/evaluation/eval_dataset.json",
        help="Path to eval dataset JSON."
    )
    args = parser.parse_args()

    dataset = load_dataset(Path(args.dataset))

    ragas_llm = llm_factory(provider="google",client=llm_model,model="gemma-3-4b")
    result = evaluate(
        dataset=dataset,
        metrics=[
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall,
        ],
        llm=ragas_llm
    )

    print("=== RAGAS Evaluation Result ===")
    print(result)

    plot_scores(result)


if __name__ == "__main__":
    main()
