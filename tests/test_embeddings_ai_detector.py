import tempfile
import warnings

from app.ai_detector import AITextDetector
from app.embeddings import VectorStore
from app.ingestion.parser import chunk_text

ORIGINAL = (
    "Photosynthesis is the process by which green plants convert sunlight, "
    "water, and carbon dioxide into glucose and oxygen. It occurs primarily "
    "in the chloroplasts of plant cells, using a green pigment called "
    "chlorophyll to capture light energy."
)

PARAPHRASED = (
    "Green plants use a process called photosynthesis to transform light "
    "from the sun, along with water and carbon dioxide, into oxygen and "
    "glucose. This mostly happens inside chloroplasts, where chlorophyll, "
    "a green pigment, absorbs the light energy needed."
)

UNRELATED = (
    "The stock market experienced significant volatility this quarter as "
    "investors reacted to changing interest rate expectations from the "
    "central bank."
)


def test_vector_store_finds_paraphrase_but_not_unrelated_text():
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(persist_dir=tmp)

        original_chunks = chunk_text(ORIGINAL, sentences_per_chunk=2, overlap=0)
        store.add_submission("course-1", "sub-original", "student-a", original_chunks)

        query_chunks = chunk_text(PARAPHRASED, sentences_per_chunk=2, overlap=0)
        matches = store.find_similar(
            query_chunks, course_id="course-1", exclude_submission_id="sub-paraphrase",
            similarity_threshold=0.6,
        )
        assert len(matches) > 0, "Expected the paraphrased text to match the original"

        unrelated_chunks = chunk_text(UNRELATED, sentences_per_chunk=2, overlap=0)
        unrelated_matches = store.find_similar(
            unrelated_chunks, course_id="course-1", exclude_submission_id="sub-unrelated",
            similarity_threshold=0.6,
        )
        assert len(unrelated_matches) == 0, "Unrelated text should not match"


def test_ai_detector_runs_end_to_end():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        detector = AITextDetector(model_path="nonexistent_path_for_test.joblib")
    proba, features = detector.predict_proba(ORIGINAL)
    assert 0.0 <= proba <= 1.0
    assert features["mean_perplexity"] > 0
