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


def test_album_directories_and_results_persist(tmp_path):
    db = tmp_path / "test.db"
    storage = Storage(db)
    album_dir = tmp_path / "album"
    album_dir.mkdir()
    image_path = album_dir / "a.jpg"
    image_path.write_bytes(b"placeholder")

    storage.save_album_directories([str(album_dir)])
    assert storage.list_album_directories() == [str(album_dir)]

    media_id = storage.add_media_item(str(image_path), "image")
    face_id = storage.add_media_face(media_id, np.array([1.0, 0.0], dtype=np.float32), crop_path=str(image_path))
    cluster = {
        "id": 1,
        "label": 0,
        "name": "Album Person 1",
        "source": "album",
        "face_count": 1,
        "photo_count": 1,
        "avg_quality": 0.0,
        "thumbnail_path": str(image_path),
        "photos": [str(image_path)],
    }
    storage.save_album_results(
        [cluster],
        {1: [{"id": face_id, "media_path": str(image_path)}]},
        "HDBSCAN",
        cluster_threshold=None,
        duplicate_threshold=0.28,
        min_samples=2,
        min_face_size=80,
    )

    results = storage.load_album_results()
    assert results["algorithm"] == "HDBSCAN"
    assert results["cluster_threshold"] is None
    assert results["duplicate_threshold"] == 0.28
    assert results["min_face_size"] == 80
    assert results["clusters"][0]["face_ids"] == [face_id]
    assert storage.list_media_faces()[0]["cluster_id"] == 1

    storage.clear_album_results()
    assert storage.load_album_results() == {}
    assert storage.list_media_faces()[0]["cluster_id"] is None
