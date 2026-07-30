"""Run a reproducible evaluation against an already indexed document workspace.

First process your own PDF/TXT through the Streamlit app (or existing ingestion demo),
then write manually verified examples in a copy of the sample JSON dataset.
"""

import argparse

from src.evaluation.rag_evaluation import (
    build_generation_review_records,
    calculate_human_rating_averages,
    evaluate_retrieval,
    load_evaluation_dataset,
    render_retrieval_report,
    save_evaluation_results,
)
from src.rag.basic_rag import get_generation_model
from src.retrieval.embeddings import get_embedding_model
from src.retrieval.vector_store import get_vector_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an indexed document RAG workspace.")
    parser.add_argument("--dataset", required=True, help="Path to an evaluation JSON file.")
    parser.add_argument("--workspace", default="default", help="Existing Chroma workspace ID.")
    parser.add_argument("--include-generation", action="store_true", help="Also run local Ollama answers for human review.")
    parser.add_argument("--output", help="Optional path for a JSON evaluation result file.")
    args = parser.parse_args()

    examples = load_evaluation_dataset(args.dataset)
    embedding_model = get_embedding_model()
    vector_store = get_vector_store(embedding_model=embedding_model, workspace_id=args.workspace)
    report = evaluate_retrieval(examples, vector_store, embedding_model, k_values=(1, 3, 5))
    print(render_retrieval_report(report))

    generation_records = None
    if args.include_generation:
        generation_records = build_generation_review_records(
            examples,
            vector_store,
            embedding_model,
            generation_model=get_generation_model(),
        )
        print("\nGeneration review records")
        for record in generation_records:
            print(f"\nQuestion: {record.question}")
            print(f"Answer: {record.generated_answer}")
            print(f"Abstained: {record.abstained}")
            print(f"Sources: {', '.join(record.sources) or 'None'}")

    ratings = calculate_human_rating_averages(examples)
    print(f"\nHuman groundedness average: {ratings['groundedness']}")
    print(f"Human answer relevance average: {ratings['answer_relevance']}")

    if args.output:
        save_evaluation_results(args.output, report, generation_records, ratings)
        print(f"\nSaved evaluation results to {args.output}")


if __name__ == "__main__":
    main()
