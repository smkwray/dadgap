from __future__ import annotations

from pathlib import Path


def test_docs_pages_use_shared_results_loader() -> None:
    root = Path(__file__).resolve().parents[1]
    index_html = (root / "docs" / "index.html").read_text(encoding="utf-8")
    methods_html = (root / "docs" / "methods.html").read_text(encoding="utf-8")
    outcomes_html = (root / "docs" / "outcomes.html").read_text(encoding="utf-8")
    prevalence_html = (root / "docs" / "prevalence.html").read_text(encoding="utf-8")
    faq_html = (root / "docs" / "faq.html").read_text(encoding="utf-8")

    assert '<script src="js/results.js"></script>' in index_html
    assert '<script src="js/results.js"></script>' in methods_html
    assert '<script src="js/results.js"></script>' in outcomes_html
    assert '<script src="js/results.js"></script>' in prevalence_html
    assert '<script src="js/results.js"></script>' in faq_html
    assert "window.dgResults.load()" in index_html
    assert "window.dgResults.load()" in methods_html
    assert "window.dgResults.load()" in outcomes_html
    assert "window.dgResults.load()" in prevalence_html
    assert "window.dgResults.load()" in faq_html


def test_docs_pages_expose_provenance_and_download_links() -> None:
    root = Path(__file__).resolve().parents[1]
    index_html = (root / "docs" / "index.html").read_text(encoding="utf-8")
    methods_html = (root / "docs" / "methods.html").read_text(encoding="utf-8")

    assert "Provenance &amp; Downloads" in index_html
    assert "Artifact Provenance" in methods_html
    assert 'href="results.json"' in index_html
    assert 'href="../outputs/manifests/results.json"' in index_html
    assert 'href="../outputs/manifests/cross_cohort_synthesis_summary.csv"' in methods_html
    assert 'href="../outputs/manifests/cross_cohort_synthesis_forest_ready.csv"' in methods_html
