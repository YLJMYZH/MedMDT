import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { ConsultationResult } from '@/lib/types'

interface Props {
  result: ConsultationResult
}

export default function ConsultationReport({ result }: Props) {
  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center gap-3">
        <h2 className="text-xl font-bold">会诊报告</h2>
        <Badge variant={result.status === 'completed' ? 'success' : 'destructive'}>
          {result.status === 'completed' ? '已完成' : '失败'}
        </Badge>
      </div>

      {result.final_report && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">综合诊断意见</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {result.final_report}
              </ReactMarkdown>
            </div>
          </CardContent>
        </Card>
      )}

      {result.consensus && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">共识</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm">
              {Object.entries(result.consensus).map(([key, value]) => (
                <div key={key} className="flex gap-2">
                  <span className="font-medium text-muted-foreground">{key}:</span>
                  <span>{String(value)}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {result.divergences.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">分歧点</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {result.divergences.map((d, i) => (
                <li key={i} className="flex items-start gap-2 text-sm">
                  <Badge variant="warning" className="mt-0.5 shrink-0">分歧</Badge>
                  <span>{d}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {result.discussion_rounds.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">讨论记录</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {result.discussion_rounds.map((round, i) => (
                <details key={i} className="border rounded-md">
                  <summary className="px-4 py-2 cursor-pointer text-sm font-medium hover:bg-accent">
                    第 {round.round_num} 轮讨论
                    {round.summary && <span className="ml-2 text-muted-foreground font-normal">— {round.summary.slice(0, 60)}...</span>}
                  </summary>
                  <div className="px-4 py-3 border-t space-y-3">
                    {round.opinions?.map((op, j) => (
                      <div key={j} className="text-sm space-y-1">
                        <div className="font-medium text-primary">{op.expert_name}</div>
                        <div className="text-muted-foreground">{op.analysis}</div>
                        <div><span className="font-medium">诊断:</span> {op.diagnosis}</div>
                        <div><span className="font-medium">建议:</span> {op.recommendation}</div>
                        <div className="text-xs text-muted-foreground">
                          置信度: {(op.confidence * 100).toFixed(0)}%
                        </div>
                      </div>
                    ))}
                  </div>
                </details>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
