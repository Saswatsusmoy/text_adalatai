"""Unit tests for the legal-entity panel (synthetic pairs only)."""

from src.evaluation.entity_panel import entity_metrics, entity_panel, extract_entities


class TestExtractEntities:
    def test_hindi_probe_and_section_citation(self):
        e = extract_entities('धारा 227 के अनुसार न्यायालय का आदेश')
        assert ('probe', 'section') in e
        assert ('probe', 'court') in e
        assert ('probe', 'order') in e
        assert ('cite', 'section', 227) in e

    def test_english_citation_cross_script_key(self):
        assert ('cite', 'article', 227) in extract_entities('अनुच्छेद 227')
        assert ('cite', 'article', 227) in extract_entities('Article 227')
        assert ('cite', 'section', 227) in extract_entities('धारा 227')
        assert ('cite', 'section', 227) in extract_entities('Section 227')

    def test_case_citation_normalized(self):
        assert ('cite', 'SCR 482') in extract_entities('S.C.R. 482')
        assert ('cite', 'SCR 482') in extract_entities('SCR 482')
        assert ('cite', 'INSC 123') in extract_entities('INSC 123')

    def test_dates_cross_script(self):
        assert ('date', '2024-05-12') in extract_entities('12 मई 2024')
        assert ('date', '2024-05-12') in extract_entities('12 May 2024')
        assert ('date', '2024-05-12') in extract_entities('12/05/2024')

    def test_empty_text(self):
        assert extract_entities('') == set()


class TestEntityPanel:
    def test_perfect_match_recall_precision_one(self):
        ref = ['धारा 227 के अनुसार न्यायालय का आदेश']
        hyp = ['धारा 227 के अनुसार न्यायालय का आदेश']
        p = entity_panel(hyp, ref)
        assert p['recall'] == 1.0
        assert p['precision'] == 1.0
        assert p['f1'] == 1.0

    def test_wrong_article_section_mapping_is_a_miss(self):
        ref = ['अनुच्छेद 227']
        hyp = ['धारा 227']
        p = entity_panel(hyp, ref)
        assert p['matched'] == 0
        assert p['recall'] == 0.0

    def test_cross_script_entities_match(self):
        ref = ['अनुच्छेद 227 के अनुसार अपीलकर्ता का याचिका']
        hyp = ['Article 227 के अनुसार appellant का petition']
        p = entity_panel(hyp, ref)
        assert p['recall'] == 1.0
        assert p['precision'] == 1.0

    def test_cross_script_dates_match(self):
        ref = ['निर्णय 12 मई 2024 को पारित']
        hyp = ['निर्णय 12 May 2024 को पारित']
        p = entity_panel(hyp, ref)
        assert p['recall'] == 1.0
        assert p['precision'] == 1.0

    def test_extra_hyp_entity_hurts_precision(self):
        p = entity_panel(['न्यायालय और रिट'], ['न्यायालय'])
        assert p['recall'] == 1.0
        assert p['precision'] == 0.5
        assert p['f1'] == round(2 * 0.5 * 1.0 / 1.5, 4)

    def test_empty_inputs_safe(self):
        p = entity_panel([], [])
        assert p['n'] == 0
        assert p['f1'] == 0.0

    def test_entity_metrics_empty_ref(self):
        m = entity_metrics(set(), set())
        assert m['recall'] == 0.0
        assert m['precision'] == 0.0
