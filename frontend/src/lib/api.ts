import type {
  ConsultationRequest,
  ConsultationResponse,
  ConsultationResult,
  SearchResponse,
  IngestResponse,
} from './types'

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`${res.status}: ${text || res.statusText}`)
  }
  return res.json()
}

export const api = {
  health: () => request<{ status: string }>('/health'),

  createConsultation: (body: ConsultationRequest) =>
    request<ConsultationResponse>('/api/v1/consultation', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  getConsultation: (id: string) =>
    request<ConsultationResult>(`/api/v1/consultation/${id}`),

  listConsultations: () =>
    request<ConsultationResult[]>('/api/v1/consultation'),

  searchKnowledge: (query: string, topK: number = 10) =>
    request<SearchResponse>('/api/v1/knowledge/search', {
      method: 'POST',
      body: JSON.stringify({ query, top_k: topK }),
    }),

  ingestFile: async (file: File): Promise<IngestResponse> => {
    const formData = new FormData()
    formData.append('file', file)
    const res = await fetch('/api/v1/knowledge/ingest', {
      method: 'POST',
      body: formData,
    })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`${res.status}: ${text || res.statusText}`)
    }
    return res.json()
  },

  getIngestStatus: (jobId: string) =>
    request<IngestResponse>(`/api/v1/knowledge/ingest/${jobId}`),
}
