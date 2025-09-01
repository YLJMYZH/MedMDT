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

export type StreamEvent =
  | { type: 'status'; status: string }
  | { type: 'phase_start'; phase: string }
  | { type: 'experts_selected'; expert_ids: string[]; expert_names: Record<string, string> }
  | { type: 'round_start'; round_num: number; total_rounds: number }
  | { type: 'expert_start'; round_num: number; expert_id: string; expert_name: string; expert_index: number; total_experts: number }
  | { type: 'expert_token'; expert_id: string; token: string }
  | { type: 'expert_end'; round_num: number; expert_id: string; opinion: ExpertOpinion }
  | { type: 'round_summary'; round_num: number; summary: string; divergences: string[]; has_consensus: boolean }
  | { type: 'report_token'; token: string }
  | { type: 'done'; status: string }
  | { type: 'error'; message: string }

export interface ExpertStreamState {
  expertId: string
  expertName: string
  streamedText: string
  opinion: ExpertOpinion | null
  isStreaming: boolean
  isComplete: boolean
}

export interface RoundState {
  roundNum: number
  totalRounds: number
  experts: ExpertStreamState[]
  summary: string | null
  divergences: string[]
  hasConsensus: boolean | null
  isComplete: boolean
}

export interface MDTStreamState {
  currentPhase: string | null
  expertNames: Record<string, string>
  selectedExpertIds: string[]
  rounds: RoundState[]
  reportBuffer: string
  isDone: boolean
  doneStatus: string | null
  error: string | null
}

export interface FolderResult {
  folder_name: string
  status: 'completed' | 'failed' | 'skipped'
  files_processed: number
  files_failed: number
  message: string
}

export interface BatchIngestResponse {
  job_id: string
  status: string
  message: string
  progress: { current: number; total: number }
  folders: FolderResult[]
}

export interface MedicalRecord {
  record_type: string
  content: string
  filename: string
  error?: string
}

export interface SaveToKnowledgeResponse {
  status: string
  message: string
  entities_count: number
  chunks_count: number
}

export type VisionStatus = 'supported' | 'unavailable' | 'unknown'
export type EmbeddingStatus = 'supported' | 'unavailable'

export interface ProviderInfo {
  key: string
  label: string
  needs_key: boolean
  needs_base_url?: boolean
  vision_status: VisionStatus
  embedding_status: EmbeddingStatus
}

export interface EndpointData {
  provider: string
  model: string
  api_key: string | null
  base_url: string | null
}

export interface EmbeddingData {
  provider: string
  model: string
  api_key: string | null
  base_url: string | null
  dim: number
}

export interface ExpertLLMData {
  provider: string
  model: string
}

export interface LLMSettingsData {
  consultation: EndpointData
  knowledge: EndpointData
  vision: EndpointData
  embedding: EmbeddingData
  experts: Record<string, ExpertLLMData>
  expert_names: Record<string, string>
  paddleocr_token: string | null
  providers: ProviderInfo[]
}
