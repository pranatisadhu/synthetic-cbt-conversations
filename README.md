# synthetic-cbt-conversations
LLM-powered pipeline that generates synthetic CBT therapy conversations and evaluates them using custom CTRS-R–based clinical quality metrics.

## Overview
This repository contains an end-to-end pipeline for generating synthetic therapist–client Cognitive Behavioral Therapy (CBT) conversations using large language models and evaluating their clinical quality based on the Cognitive Therapy Rating Scale – Revised (CTRS-R). The project addresses the challenge of data scarcity in mental-health research, where real therapy transcripts cannot be widely used due to HIPAA restrictions and privacy concerns.

The system generates structured psychotherapy dialogues grounded in CBT principles, then evaluates each conversation across 11 CTRS-R categories, producing both numerical scores and explanatory rationales. This enables early-stage research in behavioral-health AI, synthetic data generation, model assessment, and psychotherapy-tech alignment.

## Features

### 1. LLM-Powered Conversation Generation
- Developed initially using Ollama; final pipeline uses GPT-5 for improved quality.
- Generates approximately 16-turn dialogues between matched therapist and patient personas.
- Utilizes:
  - 500 predefined patient personalities
  - 20 therapist personas with distinct therapeutic styles
  - Scenario templates representing multiple mental-health concerns (depression, anxiety, OCD, insomnia, grief)
- Conversations are guided by CTRS-R expectations to promote clinically aligned interactions.

### 2. CTRS-R Evaluation System
Generated conversations are evaluated on 11 standardized CBT dimensions, including:
- Agenda setting
- Feedback
- Understanding and empathy
- Interpersonal effectiveness
- Collaboration
- Guided discovery
- Strategy for change
- Application of CBT techniques
- Action planning

Evaluation pipeline outputs include:
- Individual scores for each CTRS-R category
- Narrative rationales for each score
- Structured JSON or CSV files for downstream analysis
- Quality indicators and scoring consistency markers

### 3. Technical Stack
- Python
- GPT-5 API
- Sdialog
- Custom prompt engineering
- JSON schemas for persona, scenario, and rubric structures
- CSV/JSON outputs for analysis

## Repository Structure
generation/

    generate_conversations.py 
    # LLM pipeline for generating CBT conversations

evaluation/

    evaluate_ctrsr.py
    # Script to score conversations using CTRS-R
    CTRS-R-and-CBT-Checklist.pdf    
    # Full CTRS-R rubric used as reference
    evaluation_metrics.txt
    # Operational definitions of the 11 CTRS-R metrics

data/

    SDG_patients500.csv
    # CSV of defined pateint personas
    SDG_therapists500.csv
    # CSV of defined therapist personas

README.md

## Example Generated Conversation
[Kyle] It sounds like you've been carrying a lot recently. Would you like to start by sharing what's been on your mind?

[Lily] Yeah, it just feels like we're having the same argument over and over again, and it's getting really exhausting...

[Kyle] It's understandable to feel overwhelmed when conflicts seem circular. Let's explore what might be triggering these arguments...

[Lily] It usually starts over small things, like chores or plans, but it quickly escalates...

[Kyle] Focusing on expressing feelings without blame and listening actively may help. Would you be open to trying some techniques...

[Lily] I guess so, but it's hard not to let frustration take over...

[Kyle] One technique that helps is taking a short break to cool down before continuing the discussion...

[Lily] Yeah, I think a break could help. I'm open to strategies that keep things from escalating.

## Example CTRS-R Evaluation Output
{
 "ctrs_evaluation": [
   {"item_name": "Agenda", "score": 1,
    "rationale": "The therapist identified significant events and conducted a mood check but did not review the Action Plan or set specific goals."},

   {"item_name": "Feedback", "score": 2,
    "rationale": "The therapist ensured understanding and agreement with the treatment plan but did not consistently request feedback."},

   {"item_name": "Understanding", "score": 3,
    "rationale": "The therapist consistently demonstrated empathy and accurately reflected emotional content."},

   {"item_name": "Interpersonal Effectiveness", "score": 3,
    "rationale": "The therapist maintained professionalism, genuineness, and warmth."},

   {"item_name": "Collaboration", "score": 2,
    "rationale": "Client input was solicited at key moments but not throughout every session decision."},

   {"item_name": "Guided Discovery", "score": 2,
    "rationale": "Cognitive shifts were encouraged, but follow-up assessment was limited."},

   {"item_name": "Strategy for Change", "score": 2,
    "rationale": "The therapist used appropriate CBT strategies but did not provide alternative options."},

   {"item_name": "Application of CBT Technique", "score": 2,
    "rationale": "CBT techniques were recognizable and used with reasonable skill."},

   {"item_name": "Action Plan", "score": 2,
    "rationale": "A new plan was developed, but previous plans were not reviewed."}
 ],
 "total_score": 23
}

## Motivation
Building safe and clinically grounded mental-health AI systems requires high-quality conversational data; however, access to real therapy transcripts is limited due to HIPAA regulations and ethical restrictions. Synthetic conversations, when guided by formal psychiatric evaluation frameworks such as CTRS-R, provide an alternative dataset that can be used for research, safety evaluations, model comparison, and pipeline prototyping.

This project contributes a reproducible method for generating and evaluating synthetic CBT interactions that can support:
- Behavioral-health AI research
- Psychotherapy model alignment
- Synthetic data evaluation
- LLM safety and reliability studies

## Role and Contributions
- Designed the end-to-end pipeline architecture for synthetic CBT conversation generation.
- Authored technical requirements for the LLM generation and evaluation workflow.
- Created 500 patient personas and 20 therapist personas.
- Developed scenario templates reflecting real clinical presentation patterns.
- Wrote CTRS-R–aligned evaluation prompts and supporting documentation.
- Collaborated with the technical team on implementation, evaluation scoring, and persona mapping.
- Ensured alignment with CBT structure and ethical considerations for synthetic behavioral data.

## Results
- Generated a scalable framework capable of producing structured CBT conversations.
- Developed an evaluation system aligned with established psychotherapy standards.
- Produced example outputs demonstrating quality, rationales, and structured scoring.
- Established the foundation for future synthetic dataset expansion.

## Future Work
Planned next steps include:
- Extending the pipeline to additional therapy modalities such as ACT, DBT, and motivational interviewing.
- Building visualization tools to explore CTRS-R score distributions across populations.
- Expanding evaluator diversity through multi-model ensemble scoring.
- Integrating clinical retrieval mechanisms to ground conversations in guideline-based information.
- Adding new clinical scenarios and persona variations.
- Analyzing the probability distribution of CTRS-R scoring when the same patient–therapist pair is repeatedly regenerated, to evaluate LLM stability, variance, and performance improvements across retries.
