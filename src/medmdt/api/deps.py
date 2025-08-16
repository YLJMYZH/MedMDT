from functools import lru_cache
from medmdt.api.models import ConsultationStore


@lru_cache
def get_consultation_store() -> ConsultationStore:
    return ConsultationStore()
