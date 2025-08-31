"""Stable error categories for image-understanding calls."""


class VisionError(RuntimeError):
    """Base class for expected image-understanding failures."""


class VisionProviderNotSupported(VisionError):
    """The provider has no supported image-input transport."""


class VisionModelNotSupported(VisionError):
    """The selected model rejected image input."""


class InvalidImageError(VisionError):
    """The supplied image cannot be safely sent to a model."""


class VisionRequestError(VisionError):
    """Authentication, network, rate-limit, or provider request failure."""


class InvalidVisionResponse(VisionError):
    """The provider response contains no valid structured image analysis."""
