import os
import io
import contextlib
import random
import re
from typing import Optional

import pandas as pd
import sdialog
from sdialog.agents import Agent
from sdialog.personas import Persona
from sdialog import Context


# ----------------------------- LLM CONFIG --------------------------------- #

def configure_llm(model: str = "openai:gpt-4o", temperature: float = 0.9) -> None:
    """
    Configure sdialog to use a specific LLM backend.

    The OPENAI_API_KEY must be set in your environment, e.g.:

        export OPENAI_API_KEY="your-key-here"        (Mac/Linux)
        setx OPENAI_API_KEY "your-key-here"          (Windows PowerShell)
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Please set it as an environment variable before running."
        )

    os.environ["OPENAI_API_KEY"] = openai_api_key
    sdialog.config.llm(model, temperature=temperature)


# --------------------------- HELPER FUNCTIONS ----------------------------- #

def extract_dialog_text(text: str) -> Optional[str]:
    """
    Extract only the dialogue portion between:
        --- Dialogue Begins ---
        --- Dialogue Ends ---

    Returns the stripped dialogue text, or None if markers are not found.
    """
    if not isinstance(text, str):
        return None

    match = re.search(
        r"---\s*Dialogue Begins\s*---(.*?)---\s*Dialogue Ends\s*---",
        text,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return None


def clean_ansi_codes(text: str) -> str:
    """
    Remove ANSI color codes from a string (if present).
    """
    if pd.isna(text):
        return text
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m|\[0m|\[31m|\[94m|\[1m|\[35m")
    return ansi_escape.sub("", str(text))


# -------------------------- CORE GENERATION LOGIC ------------------------- #

def generate_conversations(
    therapist_csv_path: str = "data/SDG_therapists.csv",
    patient_csv_path: str = "data/SDG_patients500.csv",
    start_index: int = 20,
    num_dialogs: int = 10,
    model: str = "openai:gpt-4o",
    temperature: float = 0.9,
    output_prefix: str = "pranati_dialogs",
    save_excel: bool = True,
) -> pd.DataFrame:
    """
    Generate a batch of synthetic CBT conversations and return them as a DataFrame.

    Parameters
    ----------
    therapist_csv_path : str
        Path to the therapist metadata CSV (must contain 'therapist_id', 'core_traits', 'modality').
    patient_csv_path : str
        Path to the patient metadata CSV (must contain 'Background Context' and a therapist_id column).
    start_index : int
        Index in the patient DataFrame to start from.
    num_dialogs : int
        Number of conversations to generate in this batch.
    model : str
        LLM model spec for sdialog (e.g., "openai:gpt-4o").
    temperature : float
        Sampling temperature for the LLM.
    output_prefix : str
        Prefix for the output filenames.
    save_excel : bool
        If True, also save an Excel copy in addition to CSV.

    Returns
    -------
    pd.DataFrame
        Cleaned conversations DataFrame with columns:
        ['therapist_personality', 'therapist_role', 'situation', 'dialog_index', 'dialog_contents']
    """

    # Configure LLM
    configure_llm(model=model, temperature=temperature)

    # Load therapist and patient datasets
    df_therapist = pd.read_csv(therapist_csv_path).set_index("therapist_id")
    df_patient = pd.read_csv(patient_csv_path)

    # Optional manual fix if needed
    # df_patient.iloc[1, 2] = "Recently laid off from a long-term job and has suicidal ideation"

    # Build therapist persona info for each patient
    therapist_personas = []
    therapist_roles = []

    for i in range(len(df_patient)):
        therapist_id = df_patient.iloc[i, 5]
        therapist_trait = df_therapist.loc[therapist_id, "core_traits"]
        therapist_type = f"{df_therapist.loc[therapist_id, 'modality']} therapist"
        therapist_personas.append(therapist_trait)
        therapist_roles.append(therapist_type)

    df_patient["therapist_persona"] = therapist_personas
    df_patient["therapist_role"] = therapist_roles

    # Therapist opening lines
    therapist_openings = [
        "Hello, I'm Dr. Kyle. I'm glad you came in today. How have things been feeling lately?",
        "Thank you for being here today. What would you like us to focus on?",
        "It sounds like you've been carrying a lot recently. Would you like to start by sharing what's been on your mind?",
        "Welcome, I'm here to listen and support you. What brought you to therapy this week?",
        "I appreciate your openness in meeting today. What's been most difficult for you lately?",
    ]

    background_contexts = list(df_patient["Background Context"])
    dialogs = []

    end_index = start_index + num_dialogs

    for i in range(start_index, end_index):
        try:
            random_therapist_opening = random.choice(therapist_openings)

            # Therapist persona
            p_therapist = Persona(
                name="Kyle",
                role=df_patient["therapist_role"].iloc[i],
                personality=df_patient["therapist_persona"].iloc[i],
            )

            # Patient persona
            p_patient = Persona(
                name="Lily",
                role="patient in a counseling session",
                personality=df_patient.iloc[i, 3],
                circumstances=background_contexts[i],
            )

            # Therapist agent
            therapist_agent = Agent(
                persona=p_therapist,
                first_utterance=random_therapist_opening,
                context=Context(
                    goals=df_patient.iloc[i, 5],
                    constraints=(
                        "The therapist must respond with empathy but must never agree with "
                        "or endorse harmful, delusional, or self-destructive beliefs. "
                        "Whenever such beliefs are detected, the therapist validates the "
                        "client's emotions and redirects toward safety, coping strategies, "
                        "or professional help."
                    ),
                ),
            )

            # Patient agent
            patient_agent = Agent(persona=p_patient)

            # Generate dialogue
            dialog = therapist_agent.dialog_with(patient_agent, max_turns=10)

            # Capture printed dialog from sdialog
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                dialog.print()
            output_text = buffer.getvalue()

            dialog_text = extract_dialog_text(output_text)
            if dialog_text is None:
                dialog_text = (
                    f"Parsing error for generating the {i}-th dialog (no markers found)."
                )

        except Exception as e:
            dialog_text = f"Model error for generating the {i}-th dialog: {e}"

        dialogs.append(
            {
                "therapist_personality": p_therapist.personality,
                "therapist_role": p_therapist.role,
                "situation": background_contexts[i],
                "dialog_index": i,
                "dialog_contents": dialog_text,
            }
        )

    dialog_df = pd.DataFrame(dialogs)

    # Clean ANSI color codes
    dialog_df_clean = dialog_df.copy()
    dialog_df_clean["dialog_contents"] = dialog_df_clean["dialog_contents"].apply(
        clean_ansi_codes
    )

    # Save CSV (and Excel) with batch range in the filename
    csv_filename = f"{output_prefix}_{start_index+1}-{end_index}.csv"
    dialog_df_clean.to_csv(csv_filename, index=False, encoding="utf-8")

    if save_excel:
        excel_filename = f"{output_prefix}_{start_index+1}-{end_index}.xlsx"
        dialog_df_clean.to_excel(excel_filename, index=False)

    print(f"Generated {len(dialog_df_clean)} cleaned dialogs.")
    print(f"Saved CSV to:   {csv_filename}")
    if save_excel:
        print(f"Saved Excel to: {excel_filename}")

    return dialog_df_clean


# ------------------------------- CLI ENTRY -------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic CBT conversations using LLMs and sdialog."
    )
    parser.add_argument(
        "--therapists",
        type=str,
        default="data/SDG_therapists.csv",
        help="Path to therapist CSV file.",
    )
    parser.add_argument(
        "--patients",
        type=str,
        default="data/SDG_patients500.csv",
        help="Path to patient CSV file.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=20,
        help="Start index in the patient CSV.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=10,
        help="Number of dialogs to generate.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai:gpt-4o",
        help='sdialog model spec, e.g. "openai:gpt-4o".',
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature for the LLM.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="pranati_dialogs",
        help="Output file prefix.",
    )
    parser.add_argument(
        "--no-excel",
        action="store_true",
        help="If set, do not save Excel output.",
    )

    args = parser.parse_args()

    generate_conversations(
        therapist_csv_path=args.therapists,
        patient_csv_path=args.patients,
        start_index=args.start,
        num_dialogs=args.num,
        model=args.model,
        temperature=args.temperature,
        output_prefix=args.prefix,
        save_excel=not args.no_excel,
    )
