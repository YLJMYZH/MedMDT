export interface ConsultationRequest {
  patient_info: Record<string, string>
  medical_records: Array<Record<string, string>>
  max_rounds: number
}

export interface ConsultationResponse {
  id: string
  status: string
  created_at: string
}

export interface ExpertOpinion {
  expert_id: string
  expert_name: string
  analysis: string
  diagnosis: string
  recommendation: string
  confidence: number
  reasoning: string
  references: string[]
}

export interface DiscussionRound {
  round_num: number
  opinions: ExpertOpinion[]
  summary: string
  divergences: string[]
}

export interface ConsultationResult {
  id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  final_report: string | null
  discussion_rounds: DiscussionRound[]
  consensus: Record<string, unknown> | null
  divergences: string[]
}

export interface SearchResult {
  text: string
  score: number
  source: string
  metadata: Record<string, unknown>
}

export interface SearchResponse {
  results: SearchResult[]
  total: number
}

export interface IngestResponse {
  job_id: string
  status: string
  message: string
}

export interface StreamEvent {
  type: 'status' | 'done' | 'error'
  status?: string
  message?: string
}
