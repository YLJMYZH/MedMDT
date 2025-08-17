import { useEffect, useState } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import ConsultationReport from '@/components/consultation/ConsultationReport'
import { api } from '@/lib/api'
import type { ConsultationResult } from '@/lib/types'

const STATUS_FILTERS = ['all', 'completed', 'running', 'pending', 'failed'] as const

export default function HistoryPage() {
  const [consultations, setConsultations] = useState<ConsultationResult[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<string>('all')
  const [selected, setSelected] = useState<ConsultationResult | null>(null)

  useEffect(() => {
    api.listConsultations()
      .then(setConsultations)
      .catch(() => setConsultations([]))
      .finally(() => setLoading(false))
  }, [])

  const filtered = filter === 'all'
    ? consultations
    : consultations.filter(c => c.status === filter)

  if (selected) {
    return (
      <div>
        <Button variant="ghost" className="mb-4" onClick={() => setSelected(null)}>
          &larr; 返回列表
        </Button>
        <ConsultationReport result={selected} />
      </div>
    )
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold">会诊历史</h1>
        <p className="text-muted-foreground mt-1">查看所有历史会诊记录</p>
      </div>

      <div className="flex gap-2 mb-4 flex-wrap">
        {STATUS_FILTERS.map(s => (
          <Button
            key={s}
            variant={filter === s ? 'default' : 'outline'}
            size="sm"
            onClick={() => setFilter(s)}
          >
            {s === 'all' ? '全部' : s === 'completed' ? '已完成' : s === 'running' ? '进行中' : s === 'pending' ? '等待中' : '失败'}
          </Button>
        ))}
      </div>

      {loading && (
        <div className="space-y-3">
          {[1, 2, 3].map(i => <Skeleton key={i} className="h-16" />)}
        </div>
      )}

      {!loading && filtered.length === 0 && (
        <div className="text-center py-12 text-muted-foreground">
          暂无会诊记录
        </div>
      )}

      {!loading && filtered.length > 0 && (
        <div className="space-y-2">
          {filtered.map(c => (
            <Card
              key={c.id}
              className="cursor-pointer hover:shadow-md transition-shadow"
              onClick={() => c.status === 'completed' ? setSelected(c) : undefined}
            >
              <CardContent className="p-4 flex items-center justify-between">
                <div>
                  <span className="font-mono text-sm">{c.id}</span>
                  {c.final_report && (
                    <p className="text-xs text-muted-foreground mt-0.5 truncate max-w-md">
                      {c.final_report.slice(0, 80)}...
                    </p>
                  )}
                </div>
                <StatusBadge status={c.status} />
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  const variant = {
    completed: 'success' as const,
    running: 'default' as const,
    pending: 'secondary' as const,
    failed: 'destructive' as const,
  }[status] || 'secondary' as const

  const label = {
    completed: '已完成',
    running: '进行中',
    pending: '等待中',
    failed: '失败',
  }[status] || status

  return <Badge variant={variant}>{label}</Badge>
}
