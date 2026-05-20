import numpy as np

from insightface.gui.core.storage import Storage


def test_storage_people_samples_and_search(tmp_path):
    db = tmp_path / "test.db"
    storage = Storage(db)
    person_id = storage.add_person("Alice")
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    sample_id = storage.add_face_sample(person_id, emb, source_image_path="a.jpg", det_score=0.9)
    people = storage.list_people()
    assert people[0]["name"] == "Alice"
    assert people[0]["sample_count"] == 1
    samples = storage.list_face_samples(person_id)
    assert samples[0]["id"] == sample_id
    assert np.allclose(samples[0]["embedding"], emb)
    results = storage.search_embeddings(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=1, threshold=0.5)
    assert results[0].person_id == person_id
    assert results[0].status == "matched"
