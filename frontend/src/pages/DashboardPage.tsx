import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Activity, CheckCircle, AlertCircle, Stethoscope } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { api } from '@/lib/api'
import type { ConsultationResult } from '@/lib/types'

export default function DashboardPage() {
  const navigate = useNavigate()
  const [consultations, setConsultations] = useState<ConsultationResult[]>([])
  const [healthy, setHealthy] = useState<boolean | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      api.listConsultations().catch(() => []),
      api.health().then(() => true).catch(() => false),
    ]).then(([list, h]) => {
      setConsultations(list)
      setHealthy(h)
      setLoading(false)
    })
  }, [])

  const stats = {
    total: consultations.length,
    completed: consultations.filter(c => c.status === 'completed').length,
    running: consultations.filter(c => c.status === 'running' || c.status === 'pending').length,
    failed: consultations.filter(c => c.status === 'failed').length,
  }

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-48" />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map(i => <Skeleton key={i} className="h-28" />)}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">仪表盘</h1>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${healthy ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-xs text-muted-foreground">
            {healthy ? '系统正常' : '系统离线'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard title="总会诊数" value={stats.total} icon={<Stethoscope size={20} />} />
        <StatCard title="已完成" value={stats.completed} icon={<CheckCircle size={20} />} color="text-green-600" />
        <StatCard title="进行中" value={stats.running} icon={<Activity size={20} />} color="text-blue-600" />
        <StatCard title="失败" value={stats.failed} icon={<AlertCircle size={20} />} color="text-red-600" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">最近会诊</CardTitle>
          </CardHeader>
          <CardContent>
            {consultations.length === 0 ? (
              <p className="text-sm text-muted-foreground">暂无会诊记录</p>
            ) : (
              <div className="space-y-3">
                {consultations.slice(0, 5).map(c => (
                  <div key={c.id} className="flex items-center justify-between text-sm">
                    <span className="font-mono text-xs truncate max-w-[200px]">{c.id}</span>
                    <StatusBadge status={c.status} />
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">快捷操作</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Button className="w-full" onClick={() => navigate('/consultation')}>
              <Stethoscope size={16} className="mr-2" />
              发起新会诊
            </Button>
            <Button variant="outline" className="w-full" onClick={() => navigate('/knowledge')}>
              搜索知识库
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function StatCard({ title, value, icon, color }: { title: string; value: number; icon: React.ReactNode; color?: string }) {
  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-muted-foreground">{title}</p>
            <p className="text-3xl font-bold mt-1">{value}</p>
          </div>
          <div className={color || 'text-muted-foreground'}>{icon}</div>
        </div>
      </CardContent>
    </Card>
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
