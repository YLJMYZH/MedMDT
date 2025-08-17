import { useEffect, useRef, useState, useCallback } from 'react'
import type { StreamEvent, ConsultationResult } from '@/lib/types'
import { api } from '@/lib/api'

interface StreamState {
  events: StreamEvent[]
  status: string
  isConnected: boolean
  result: ConsultationResult | null
  error: string | null
}

export function useConsultationStream(consultationId: string | null) {
  const [state, setState] = useState<StreamState>({
    events: [],
    status: 'connecting',
    isConnected: false,
    result: null,
    error: null,
  })
  const wsRef = useRef<WebSocket | null>(null)

  const connect = useCallback(() => {
    if (!consultationId) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const url = `${protocol}//${window.location.host}/api/v1/consultation/${consultationId}/stream`
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      setState(s => ({ ...s, isConnected: true, status: 'connected' }))
    }

    ws.onmessage = (e) => {
      const event: StreamEvent = JSON.parse(e.data)
      setState(s => ({
        ...s,
        events: [...s.events, event],
        status: event.status || event.type,
      }))

      if (event.type === 'done') {
        api.getConsultation(consultationId).then(result => {
          setState(s => ({ ...s, result, isConnected: false }))
        })
        ws.close()
      }

      if (event.type === 'error') {
        setState(s => ({ ...s, error: event.message || 'Unknown error', isConnected: false }))
        ws.close()
      }
    }

    ws.onerror = () => {
      setState(s => ({ ...s, error: 'WebSocket connection failed', isConnected: false }))
    }

    ws.onclose = () => {
      setState(s => ({ ...s, isConnected: false }))
    }
  }, [consultationId])

  useEffect(() => {
    connect()
    return () => {
      wsRef.current?.close()
    }
  }, [connect])

  return state
}
