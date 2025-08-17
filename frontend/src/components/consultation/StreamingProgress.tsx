import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { StreamEvent } from '@/lib/types'

interface Props {
  events: StreamEvent[]
  status: string
  isConnected: boolean
}

export default function StreamingProgress({ events, status, isConnected }: Props) {
  return (
    <Card className="max-w-2xl">
      <CardContent className="p-6">
        <div className="flex items-center gap-3 mb-6">
          {isConnected && (
            <div className="flex gap-1">
              <span className="w-2 h-2 bg-primary rounded-full animate-bounce [animation-delay:-0.3s]" />
              <span className="w-2 h-2 bg-primary rounded-full animate-bounce [animation-delay:-0.15s]" />
              <span className="w-2 h-2 bg-primary rounded-full animate-bounce" />
            </div>
          )}
          <span className="text-sm font-medium">
            {isConnected ? '会诊进行中...' : '会诊已结束'}
          </span>
          <Badge variant={isConnected ? 'default' : 'secondary'}>{status}</Badge>
        </div>

        <div className="space-y-2">
          {events.map((event, i) => (
            <div key={i} className="flex items-center gap-2 text-sm">
              <div className="w-1.5 h-1.5 rounded-full bg-primary/60" />
              <span className="text-muted-foreground">
                {event.type === 'status' && `状态: ${event.status}`}
                {event.type === 'done' && `完成: ${event.status}`}
                {event.type === 'error' && `错误: ${event.message}`}
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
