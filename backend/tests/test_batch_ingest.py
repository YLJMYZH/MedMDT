from medmdt.extractor.batch_ingest import _collect_supported_files


def test_archive_collection_includes_webp_and_gif(tmp_path):
    patient = tmp_path / "patient"
    patient.mkdir()
    (patient / "scan.webp").write_bytes(b"webp")
    (patient / "scan.gif").write_bytes(b"gif")
    (patient / "notes.txt").write_text("unsupported")

    assert [path.name for path in _collect_supported_files(patient)] == [
        "scan.gif",
        "scan.webp",
    ]
