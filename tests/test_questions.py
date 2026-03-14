from father_longrun.questions import INTAKE_QUESTIONS, render_questions_markdown


def test_questions_exist() -> None:
    assert len(INTAKE_QUESTIONS) >= 8


def test_render_questions_markdown() -> None:
    rendered = render_questions_markdown()
    assert "Intake questions" in rendered
    assert "NLSY79 main" in rendered
