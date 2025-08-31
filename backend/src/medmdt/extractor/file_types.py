"""Shared file-extension groups for medical document ingestion."""

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}
)
DICOM_EXTENSIONS = frozenset({".dcm", ".dicom"})
MEDICAL_FILE_EXTENSIONS = frozenset({".pdf"}) | IMAGE_EXTENSIONS | DICOM_EXTENSIONS
