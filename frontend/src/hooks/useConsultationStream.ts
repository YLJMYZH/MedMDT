import { useEffect, useRef, useReducer, useCallback } from 'react'
import type { StreamEvent, MDTStreamState, RoundState, ExpertStreamState, ConsultationResult } from '@/lib/types'
import { api } from '@/lib/api'

type Action =
  | { type: 'CONNECTED' }
  | { type: 'DISCONNECTED' }
  | { type: 'EVENT'; event: StreamEvent }
  | { type: 'FLUSH_TOKENS'; buffers: Record<string, string>; reportBuffer: string }
  | { type: 'RESULT'; result: ConsultationResult }
  | { type: 'WS_ERROR'; message: string }

interface FullState extends MDTStreamState {
  isConnected: boolean
  result: ConsultationResult | null
}

const initialState: FullState = {
  currentPhase: null,
  expertNames: {},
  selectedExpertIds: [],
  rounds: [],
  reportBuffer: '',
  isDone: false,
  doneStatus: null,
  error: null,
  isConnected: false,
  result: null,
}

function reducer(state: FullState, action: Action): FullState {
  switch (action.type) {
    case 'CONNECTED':
      return { ...state, isConnected: true }
    case 'DISCONNECTED':
      return { ...state, isConnected: false }
    case 'WS_ERROR':
      return { ...state, error: action.message, isConnected: false }
    case 'RESULT':
      return { ...state, result: action.result }

    case 'EVENT': {
      const ev = action.event
      switch (ev.type) {
        case 'status':
          return { ...state, currentPhase: ev.status }

        case 'phase_start':
          return { ...state, currentPhase: ev.phase }

        case 'experts_selected':
          return {
            ...state,
            expertNames: ev.expert_names,
            selectedExpertIds: ev.expert_ids,
          }

        case 'round_start': {
          const newRound: RoundState = {
            roundNum: ev.round_num,
            totalRounds: ev.total_rounds,
            experts: [],
            summary: null,
            divergences: [],
            hasConsensus: null,
            isComplete: false,
          }
          return { ...state, rounds: [...state.rounds, newRound] }
        }

        case 'expert_start': {
          const rounds = [...state.rounds]
          const last = { ...rounds[rounds.length - 1] }
          const expert: ExpertStreamState = {
            expertId: ev.expert_id,
            expertName: ev.expert_name,
            streamedText: '',
            opinion: null,
            isStreaming: true,
            isComplete: false,
          }
          last.experts = [...last.experts, expert]
          rounds[rounds.length - 1] = last
          return { ...state, rounds }
        }

        case 'expert_end': {
          const rounds = [...state.rounds]
          const last = { ...rounds[rounds.length - 1] }
          last.experts = last.experts.map(e =>
            e.expertId === ev.expert_id
              ? { ...e, opinion: ev.opinion, isStreaming: false, isComplete: true }
              : e
          )
          rounds[rounds.length - 1] = last
          return { ...state, rounds }
        }

        case 'round_summary': {
          const rounds = [...state.rounds]
          const last = { ...rounds[rounds.length - 1] }
          last.summary = ev.summary
          last.divergences = ev.divergences
          last.hasConsensus = ev.has_consensus
          last.isComplete = true
          rounds[rounds.length - 1] = last
          return { ...state, rounds }
        }

        case 'done':
          return { ...state, isDone: true, doneStatus: ev.status, isConnected: false }

        case 'error':
          return { ...state, error: ev.message, isConnected: false }

        default:
          return state
      }
    }

    case 'FLUSH_TOKENS': {
      let rounds = state.rounds
      const hasExpertTokens = Object.keys(action.buffers).length > 0

      if (hasExpertTokens) {
        rounds = [...rounds]
        const last = { ...rounds[rounds.length - 1] }
        last.experts = last.experts.map(e => {
          const buf = action.buffers[e.expertId]
          if (buf !== undefined) {
            return { ...e, streamedText: buf }
          }
          return e
        })
        rounds[rounds.length - 1] = last
      }

      return {
        ...state,
        rounds,
        reportBuffer: action.reportBuffer || state.reportBuffer,
      }
    }

    default:
      return state
  }
}

export function useConsultationStream(consultationId: string | null) {
  const [state, dispatch] = useReducer(reducer, initialState)
  const wsRef = useRef<WebSocket | null>(null)

  // Token buffers — flushed via requestAnimationFrame
  const expertBuffersRef = useRef<Record<string, string>>({})
  const reportBufferRef = useRef('')
  const rafRef = useRef<number | null>(null)
  const dirtyRef = useRef(false)

  const flushBuffers = useCallback(() => {
    if (dirtyRef.current) {
      dispatch({
        type: 'FLUSH_TOKENS',
        buffers: { ...expertBuffersRef.current },
        reportBuffer: reportBufferRef.current,
      })
      dirtyRef.current = false
    }
    rafRef.current = null
  }, [])

  const scheduleFlush = useCallback(() => {
    dirtyRef.current = true
    if (!rafRef.current) {
      rafRef.current = requestAnimationFrame(flushBuffers)
    }
  }, [flushBuffers])

  const connect = useCallback(() => {
    if (!consultationId) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const url = `${protocol}//${window.location.host}/api/v1/consultation/${consultationId}/stream`
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      dispatch({ type: 'CONNECTED' })
    }

    ws.onmessage = (e) => {
      const event: StreamEvent = JSON.parse(e.data)

      if (event.type === 'expert_token') {
        const eid = event.expert_id
        expertBuffersRef.current[eid] = (expertBuffersRef.current[eid] || '') + event.token
        scheduleFlush()
        return
      }

      if (event.type === 'report_token') {
        reportBufferRef.current += event.token
        scheduleFlush()
        return
      }

      dispatch({ type: 'EVENT', event })

      if (event.type === 'expert_start') {
        expertBuffersRef.current[event.expert_id] = ''
      }

      if (event.type === 'done') {
        // Flush any remaining tokens
        if (dirtyRef.current) {
          flushBuffers()
        }
        api.getConsultation(consultationId).then(result => {
          dispatch({ type: 'RESULT', result })
        })
        ws.close()
      }

      if (event.type === 'error') {
        ws.close()
      }
    }

    ws.onerror = () => {
      dispatch({ type: 'WS_ERROR', message: 'WebSocket connection failed' })
    }

    ws.onclose = () => {
      dispatch({ type: 'DISCONNECTED' })
    }
  }, [consultationId, scheduleFlush, flushBuffers])

  useEffect(() => {
    connect()
    return () => {
      wsRef.current?.close()
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current)
      }
    }
  }, [connect])

  return state
}
