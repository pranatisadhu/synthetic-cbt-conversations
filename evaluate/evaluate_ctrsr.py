"""
evaluate_ctrsr.py

Evaluate synthetic CBT therapy conversations using a CTRS-R–style rubric
and save results to a CSV file.

Usage (from repository root):

    python evaluation/evaluate_ctrsr.py \
        --input_csv data/generated/pranati_dialogs_21-30.csv \
        --output_csv data/generated/pranati_dialogs_21-30_evaluated.csv
"""

import os
import json
import argparse
from typing import Dict, Any, List, Optional

import pandas as pd
from openai import OpenAI


# ---------------------- OpenAI / LLM configuration ---------------------- #

def get_openai_client() -> OpenAI:
    """
    Initialize an OpenAI client using the OPENAI_API_KEY
    environment variable. Raises an error if it is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")
    return OpenAI(api_key=api_key)


# ---------------------- CTRS-R evaluation prompt ------------------------ #

SYSTEM_PROMPT = """
You are an expert CBT supervisor and a qualified rater using the Cognitive Therapy Rating Scale – Revised (CTRS-R).
Your task is to evaluate the quality of a CBT therapy session transcript between a therapist and a client.

You must:
1. Carefully read the full conversation.
2. Evaluate the therapist's performance on the 11 CTRS-R items:
   - Agenda
   - Feedback
   - Understanding
   - Interpersonal Effectiveness
   - Collaboration
   - Pacing and Efficient Use of Time
   - Guided Discovery
   - Focus on Key Cognitions and Behaviors
   - Strategy for Change
   - Application of CBT Technique
   - Action Plan
3. Score each item from 0 to 3:
   - 0 = Very poor / absent
   - 1 = Below standard
   - 2 = Adequate / meets standard
   - 3 = Good to excellent
4. Provide a brief, specific rationale for each item.
5. Compute the total score as the sum of all 11 item scores.

Your response MUST be valid JSON with the following structure:

{
  "ctrs_evaluation": [
    {"item_name": "Agenda", "score": 0-3, "rationale": "string"},
    {"item_name": "Feedback", "score": 0-3, "rationale": "string"},
    ...
    {"item_name": "Action Plan", "score": 0-3, "rationale": "string"}
  ],
  "total_score": integer
}

Do not include any additional keys or text. Return ONLY the JSON object.
"""


def build_user_prompt(dialog_text: str) -> str:
    """
    Build the user message prompt for the LLM, given the full dialog text.
    """
    return (
        "Below is a CBT therapy conversation between a therapist and a client.\n\n"
        "CONVERSATION:\n"
        "--------------------\n"
        f"{dialog_text}\n"
        "--------------------\n\n"
        "Please evaluate this conversation according to CTRS-R as described in the system instructions "
        "and return ONLY the JSON object."
    )


# ----------------------- LLM call and parsing --------------------------- #

def evaluate_single_dialog(
    client: OpenAI,
    dialog_text: str,
    model: str = "gpt-4o-mini"
) -> Optional[Dict[str, Any]]:
    """
    Send a single conversation to the LLM for CTRS-R evaluation
    and return the parsed JSON dictionary.

    Returns None if parsing fails.
    """
    if not dialog_text or not isinstance(dialog_text, str):
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(dialog_text)},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )

    raw_content = response.choices[0].message.content.strip()

    # Attempt to parse JSON directly
    try:
        data = json.loads(raw_content)
        # Basic validation
        if "ctrs_evaluation" in data and "total_score" in data:
            return data
    except json.JSONDecodeError:
        # Try to recover JSON if there is extra text around it
        try:
            start = raw_content.index("{")
            end = raw_content.rindex("}") + 1
            candidate = raw_content[start:end]
            data = json.loads(candidate)
            if "ctrs_evaluation" in data and "total_score" in data:
                return data
        except Exception:
            return None

    return None


# ----------------------- Batch CSV evaluation --------------------------- #

def evaluate_csv(
    input_csv: str,
    output_csv: str,
    dialog_column: str = "dialog_contents",
    model: str = "gpt-4o-mini",
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Evaluate all dialogs in a CSV file and save results to a new CSV.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV containing dialog text.
    output_csv : str
        Path where the evaluated CSV will be saved.
    dialog_column : str
        Name of the column containing the conversation text.
    model : str
        OpenAI model to use for evaluation (e.g., "gpt-4o-mini", "gpt-4o").
    max_rows : int or None
        If provided, evaluate only the first `max_rows` rows (for testing).

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'total_score' and 'evaluation_json' columns.
    """
    client = get_openai_client()

    df = pd.read_csv(input_csv)

    if dialog_column not in df.columns:
        raise ValueError(
            f"Column '{dialog_column}' not found in input CSV. "
            f"Available columns: {list(df.columns)}"
        )

    if max_rows is not None:
        df = df.head(max_rows)

    total_scores: List[Optional[int]] = []
    evaluations_json: List[Optional[str]] = []

    for idx, row in df.iterrows():
        dialog_text = row[dialog_column]

        print(f"Evaluating row {idx}...")

        eval_result = evaluate_single_dialog(client, dialog_text, model=model)

        if eval_result is None:
            print(f"  -> Evaluation failed for row {idx}.")
            total_scores.append(None)
            evaluations_json.append(None)
        else:
            total_scores.append(eval_result.get("total_score"))
            evaluations_json.append(json.dumps(eval_result, ensure_ascii=False))

    df["total_score"] = total_scores
    df["evaluation_json"] = evaluations_json

    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\nSaved evaluated results to: {output_csv}")

    return df


# ----------------------------- CLI entrypoint --------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CBT conversations using a CTRS-R–style rubric."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to the input CSV containing conversations.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to the output CSV to save evaluations.",
    )
    parser.add_argument(
        "--dialog_column",
        type=str,
        default="dialog_contents",
        help="Name of the column containing the dialog text.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use for evaluation (e.g., gpt-4o-mini, gpt-4o).",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optionally limit the number of rows to evaluate (for testing).",
    )

    args = parser.parse_args()

    evaluate_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        dialog_column=args.dialog_column,
        model=args.model,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
