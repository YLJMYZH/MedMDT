import { useEffect, useState, useRef } from 'react'
import { api } from '@/lib/api'
import type { IngestResponse } from '@/lib/types'

export function useIngestStatus(jobId: string | null) {
  const [status, setStatus] = useState<IngestResponse | null>(null)
  const [isPolling, setIsPolling] = useState(false)
  const intervalRef = useRef<number | null>(null)

  useEffect(() => {
    if (!jobId) return

    setIsPolling(true)

    const poll = async () => {
      try {
        const res = await api.getIngestStatus(jobId)
        setStatus(res)
        if (res.status === 'completed' || res.status === 'failed') {
          setIsPolling(false)
          if (intervalRef.current) clearInterval(intervalRef.current)
        }
      } catch {
        setIsPolling(false)
        if (intervalRef.current) clearInterval(intervalRef.current)
      }
    }

    poll()
    intervalRef.current = window.setInterval(poll, 3000)

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [jobId])

  return { status, isPolling }
}
