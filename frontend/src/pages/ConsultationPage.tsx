import { useState } from 'react'
import ConsultationForm from '@/components/consultation/ConsultationForm'
import StreamingProgress from '@/components/consultation/StreamingProgress'
import { Card, CardContent } from '@/components/ui/card'
import { useConsultationStream } from '@/hooks/useConsultationStream'
import { api } from '@/lib/api'
import type { ConsultationRequest } from '@/lib/types'

type PageState = 'idle' | 'submitting' | 'streaming'

export default function ConsultationPage() {
  const [pageState, setPageState] = useState<PageState>('idle')
  const [consultationId, setConsultationId] = useState<string | null>(null)
  const [submitError, setSubmitError] = useState<string | null>(null)

  const stream = useConsultationStream(pageState === 'streaming' ? consultationId : null)

  const handleSubmit = async (data: ConsultationRequest) => {
    setPageState('submitting')
    setSubmitError(null)
    try {
      const res = await api.createConsultation(data)
      setConsultationId(res.id)
      setPageState('streaming')
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : 'Failed to create consultation')
      setPageState('idle')
    }
  }

  const handleReset = () => {
    setPageState('idle')
    setConsultationId(null)
    setSubmitError(null)
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold">新建多专家会诊</h1>
        <p className="text-muted-foreground mt-1">输入患者信息，启动MDT多学科讨论</p>
      </div>

      {(pageState === 'idle' || pageState === 'submitting') && (
        <>
          <ConsultationForm onSubmit={handleSubmit} isSubmitting={pageState === 'submitting'} />
          {submitError && (
            <Card className="max-w-2xl mt-4 border-destructive">
              <CardContent className="p-4">
                <p className="text-destructive text-sm">{submitError}</p>
              </CardContent>
            </Card>
          )}
        </>
      )}

      {pageState === 'streaming' && (
        <StreamingProgress stream={stream} onReset={handleReset} />
      )}
    </div>
  )
}
