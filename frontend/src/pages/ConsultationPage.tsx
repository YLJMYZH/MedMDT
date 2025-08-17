import { useState } from 'react'
import ConsultationForm from '@/components/consultation/ConsultationForm'
import StreamingProgress from '@/components/consultation/StreamingProgress'
import ConsultationReport from '@/components/consultation/ConsultationReport'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useConsultationStream } from '@/hooks/useConsultationStream'
import { api } from '@/lib/api'
import type { ConsultationRequest, ConsultationResult } from '@/lib/types'

type PageState = 'idle' | 'submitting' | 'streaming' | 'completed' | 'failed'

export default function ConsultationPage() {
  const [pageState, setPageState] = useState<PageState>('idle')
  const [consultationId, setConsultationId] = useState<string | null>(null)
  const [result, setResult] = useState<ConsultationResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const stream = useConsultationStream(pageState === 'streaming' ? consultationId : null)

  if (stream.result && pageState === 'streaming') {
    setResult(stream.result)
    setPageState('completed')
  }
  if (stream.error && pageState === 'streaming') {
    setError(stream.error)
    setPageState('failed')
  }

  const handleSubmit = async (data: ConsultationRequest) => {
    setPageState('submitting')
    setError(null)
    try {
      const res = await api.createConsultation(data)
      setConsultationId(res.id)
      setPageState('streaming')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to create consultation')
      setPageState('failed')
    }
  }

  const handleReset = () => {
    setPageState('idle')
    setConsultationId(null)
    setResult(null)
    setError(null)
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold">新建多专家会诊</h1>
        <p className="text-muted-foreground mt-1">输入患者信息，启动MDT多学科讨论</p>
      </div>

      {pageState === 'idle' && (
        <ConsultationForm onSubmit={handleSubmit} isSubmitting={false} />
      )}

      {pageState === 'submitting' && (
        <ConsultationForm onSubmit={handleSubmit} isSubmitting={true} />
      )}

      {pageState === 'streaming' && (
        <StreamingProgress
          events={stream.events}
          status={stream.status}
          isConnected={stream.isConnected}
        />
      )}

      {pageState === 'completed' && result && (
        <div className="space-y-4">
          <ConsultationReport result={result} />
          <Button variant="outline" onClick={handleReset}>发起新会诊</Button>
        </div>
      )}

      {pageState === 'failed' && (
        <Card className="max-w-2xl border-destructive">
          <CardContent className="p-6">
            <p className="text-destructive font-medium mb-2">会诊失败</p>
            <p className="text-sm text-muted-foreground mb-4">{error}</p>
            <Button variant="outline" onClick={handleReset}>重试</Button>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
