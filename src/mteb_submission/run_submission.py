"""
run_submission.py
--------------------
Official MTEB submission runner, following the exact pattern from the
guidelines PDF. Runs our PrePostPipelineEncoder on the FULL AppsRetrieval
test split (not a sample) and writes appsretrieval_results.json.

Run from the repo root:
    python -m src.mteb_submission.run_submission

WARNING: This embeds the FULL corpus (~8765 docs) plus every query in the
test split. On CPU this can take around 10-15 minutes. Let it run to
completion.
"""

import json
import mteb

from src.mteb_submission.encoder import PrePostPipelineEncoder


def main():
    print("Initializing PrePostPipelineEncoder...")
    model = PrePostPipelineEncoder()

    print("Loading AppsRetrieval task...")
    task = mteb.get_task("AppsRetrieval")

    print("Running MTEB evaluation on the FULL test split (this may take a while)...")
    result = mteb.evaluate(
        model,
        [task],
        encode_kwargs={"batch_size": 64},
    )

    task_result = list(result.task_results)[0]

    output_path = "appsretrieval_results.json"
    with open(output_path, "w") as f:
        json.dump(task_result.to_dict(), f, indent=2, default=str)

    print(f"\nDone. Results written to {output_path}")
    print("\nSummary:")
    print(task_result)


if __name__ == "__main__":
    main()