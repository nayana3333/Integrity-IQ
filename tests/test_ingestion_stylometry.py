from app.ingestion.parser import chunk_text, split_sentences
from app.stylometry import StyleProfile

SAMPLE_A = (
    "The rise of renewable energy has, however, changed the way nations think "
    "about power. Solar and wind are no longer niche technologies; they are, "
    "in many countries, the cheapest source of new electricity. Furthermore, "
    "storage costs have fallen sharply, which makes intermittent generation "
    "far more practical than it was a decade ago."
)

SAMPLE_A_SIMILAR_STYLE = (
    "The growth of electric vehicles has, however, changed the way cities plan "
    "transport. Buses and taxis are no longer a fixed cost; they are, in many "
    "regions, the fastest-depreciating asset on the books. Furthermore, battery "
    "prices have dropped sharply, which makes fleet electrification far more "
    "practical than it was a decade ago."
)

SAMPLE_B_DIFFERENT_STYLE = (
    "yo so basically evs are getting cheap af rn lol. like batteries cost way "
    "less than before!! and honestly cities dont even need gas stations "
    "anymore fr fr, its kinda wild ngl."
)


def test_split_sentences():
    sentences = split_sentences(SAMPLE_A)
    assert len(sentences) == 3


def test_chunk_text_overlap():
    chunks = chunk_text(SAMPLE_A + " " + SAMPLE_A_SIMILAR_STYLE, sentences_per_chunk=3, overlap=1)
    assert len(chunks) >= 2
    assert chunks[0].end_sentence - chunks[0].start_sentence == 2


def test_style_profile_low_drift_for_similar_style():
    profile = StyleProfile(student_id="s1")
    for _ in range(4):
        profile.update(SAMPLE_A)
    result_similar = profile.score(SAMPLE_A_SIMILAR_STYLE)
    result_different = profile.score(SAMPLE_B_DIFFERENT_STYLE)

    assert result_similar.is_reliable
    assert result_different.drift_score > result_similar.drift_score
