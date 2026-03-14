from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class IntakeQuestion:
    key: str
    prompt: str
    required: bool = True


INTAKE_QUESTIONS: tuple[IntakeQuestion, ...] = (
    IntakeQuestion("fallback_interim_root", "Path to an existing interim NLSY/NLSCYA/NLSY97 directory", required=False),
    IntakeQuestion("nlsy79_main_path", "Absolute path to the NLSY79 main extract"),
    IntakeQuestion("nlsy79_child_path", "Absolute path to the NLSY79 Child extract"),
    IntakeQuestion("nlsy79_young_adult_path", "Absolute path to the NLSY79 Young Adult extract"),
    IntakeQuestion("nlsy97_path", "Absolute path to the NLSY97 extract (if already available)", required=False),
    IntakeQuestion("codebooks_dir", "Path to codebooks / variable lists / prior extraction scripts", required=False),
    IntakeQuestion("file_formats", "File format for each source (.dta, .csv, .sav, fixed-width, other)"),
    IntakeQuestion("extract_provenance", "Whether each file is raw, cleaned, or merged from a prior project"),
    IntakeQuestion("id_columns", "Which key ID columns are present in each file"),
    IntakeQuestion("paths", "Preferred directories for raw, processed, outputs, and cache"),
    IntakeQuestion("public_sources_next", "Which public sources to add next: PSID, Add Health, FFCWS, ACS, CPS, SIPP"),
    IntakeQuestion("primary_outcomes", "Primary adult outcomes for milestone 1"),
    IntakeQuestion("first_milestone", "Preferred first milestone: NLSY-only, NLSY+PSID, or full scaffold"),
)


def render_questions_markdown(questions: Iterable[IntakeQuestion] = INTAKE_QUESTIONS) -> str:
    lines = [
        "# Intake questions",
        "",
        "Please answer these in one block before ingestion code is written:",
        "",
    ]
    for idx, question in enumerate(questions, start=1):
        required_text = "required" if question.required else "optional"
        lines.append(f"{idx}. **{question.prompt}** ({required_text})")
    return "\n".join(lines)
