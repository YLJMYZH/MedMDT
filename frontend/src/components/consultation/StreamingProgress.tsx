import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { MDTStreamState, RoundState, ExpertStreamState, ExpertOpinion } from '@/lib/types'
import { api } from '@/lib/api'

const PHASE_LABELS: Record<string, string> = {
  running: '启动中',
  retrieve_knowledge: '知识检索',
  select_experts: '专家选择',
  discussion: '多轮讨论',
  generate_report: '生成报告',
}

const PHASES = ['retrieve_knowledge', 'select_experts', 'discussion', 'generate_report']

interface Props {
  stream: MDTStreamState & { isConnected: boolean; result: import('@/lib/types').ConsultationResult | null }
  onReset?: () => void
}

export default function StreamingProgress({ stream, onReset }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)
  const [saved, setSaved] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saveMessage, setSaveMessage] = useState('')

  useEffect(() => {
    if (autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [stream.rounds, stream.reportBuffer, stream.currentPhase, autoScroll])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const handleScroll = () => {
      const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 100
      setAutoScroll(nearBottom)
    }
    el.addEventListener('scroll', handleScroll)
    return () => el.removeEventListener('scroll', handleScroll)
  }, [])

  const handleSaveToKnowledge = async () => {
    if (!stream.result) return
    setSaving(true)
    try {
      const res = await api.saveToKnowledge(stream.result.id)
      setSaved(true)
      setSaveMessage(res.message)
    } catch (err) {
      setSaveMessage(`保存失败: ${err instanceof Error ? err.message : '未知错误'}`)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div ref={containerRef} className="space-y-4 max-w-4xl">
      <PhaseIndicator currentPhase={stream.currentPhase} isDone={stream.isDone} />

      {stream.selectedExpertIds.length > 0 && (
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-sm text-muted-foreground">参与专家:</span>
          {stream.selectedExpertIds.map(id => (
            <Badge key={id} variant="secondary">{stream.expertNames[id] || id}</Badge>
          ))}
        </div>
      )}

      {stream.rounds.map((round, i) => (
        <RoundSection key={i} round={round} />
      ))}

      {(stream.currentPhase === 'generate_report' || stream.reportBuffer) && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              会诊报告
              {!stream.isDone && <BlinkingCursor />}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {stream.reportBuffer || stream.result?.final_report || ''}
              </ReactMarkdown>
            </div>
          </CardContent>
        </Card>
      )}

      {stream.isDone && (
        <div className="flex items-center gap-3 pt-2">
          {stream.doneStatus === 'completed' && (
            <Button
              onClick={handleSaveToKnowledge}
              disabled={saved || saving}
              variant={saved ? 'outline' : 'default'}
            >
              {saving ? '正在入库...' : saved ? '已记入知识库' : '记入知识库'}
            </Button>
          )}
          {saveMessage && (
            <span className={`text-sm ${saved ? 'text-green-600' : 'text-destructive'}`}>
              {saveMessage}
            </span>
          )}
          {onReset && (
            <Button variant="outline" onClick={onReset}>发起新会诊</Button>
          )}
        </div>
      )}

      {stream.error && (
        <Card className="border-destructive">
          <CardContent className="p-4">
            <p className="text-destructive text-sm">{stream.error}</p>
          </CardContent>
        </Card>
      )}

      <div ref={bottomRef} />
    </div>
  )
}


function PhaseIndicator({ currentPhase, isDone }: { currentPhase: string | null; isDone: boolean }) {
  const currentIdx = currentPhase ? PHASES.indexOf(currentPhase) : -1

  return (
    <div className="flex items-center gap-1 overflow-x-auto pb-2">
      {PHASES.map((phase, i) => {
        const isActive = phase === currentPhase && !isDone
        const isCompleted = isDone || (currentIdx >= 0 && i < currentIdx)
        return (
          <div key={phase} className="flex items-center gap-1">
            <div
              className={`
                px-3 py-1.5 rounded-full text-xs font-medium whitespace-nowrap transition-colors
                ${isActive ? 'bg-primary text-primary-foreground' : ''}
                ${isCompleted ? 'bg-primary/20 text-primary' : ''}
                ${!isActive && !isCompleted ? 'bg-muted text-muted-foreground' : ''}
              `}
            >
              {PHASE_LABELS[phase] || phase}
            </div>
            {i < PHASES.length - 1 && (
              <span className={`text-xs ${isCompleted ? 'text-primary' : 'text-muted-foreground/40'}`}>→</span>
            )}
          </div>
        )
      })}
    </div>
  )
}


function RoundSection({ round }: { round: RoundState }) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <div className="h-px flex-1 bg-border" />
        <span className="text-xs font-medium text-muted-foreground px-2">
          第 {round.roundNum} 轮讨论 {round.totalRounds > 1 && `/ ${round.totalRounds}`}
        </span>
        <div className="h-px flex-1 bg-border" />
      </div>

      {round.experts.map(expert => (
        <ExpertCard key={expert.expertId} expert={expert} />
      ))}

      {round.isComplete && round.summary && (
        <Card className="bg-muted/50">
          <CardContent className="p-4">
            <div className="text-xs font-medium text-muted-foreground mb-2">主持人汇总</div>
            <p className="text-sm">{round.summary}</p>
            {round.divergences.length > 0 && (
              <div className="flex gap-1.5 mt-2 flex-wrap">
                {round.divergences.map((d, i) => (
                  <Badge key={i} variant="warning" className="text-[10px]">{d}</Badge>
                ))}
              </div>
            )}
            {round.hasConsensus !== null && (
              <Badge variant={round.hasConsensus ? 'success' : 'secondary'} className="mt-2">
                {round.hasConsensus ? '已达成共识' : '未达成共识'}
              </Badge>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}


function ExpertCard({ expert }: { expert: ExpertStreamState }) {
  if (expert.isComplete && expert.opinion) {
    return <CompletedExpertCard name={expert.expertName} opinion={expert.opinion} />
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
          {expert.expertName}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {expert.streamedText ? (
          <div className="text-sm text-muted-foreground whitespace-pre-wrap font-mono text-xs leading-relaxed">
            {expert.streamedText}
            <BlinkingCursor />
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">正在分析中...</p>
        )}
      </CardContent>
    </Card>
  )
}


function CompletedExpertCard({ name, opinion }: { name: string; opinion: ExpertOpinion }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-green-500" />
          {name}
          <Badge variant="outline" className="ml-auto text-[10px]">
            置信度 {(opinion.confidence * 100).toFixed(0)}%
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        <div>
          <span className="font-medium">分析: </span>
          <span className="text-muted-foreground">{opinion.analysis}</span>
        </div>
        <div>
          <span className="font-medium">诊断: </span>
          {opinion.diagnosis}
        </div>
        <div>
          <span className="font-medium">建议: </span>
          {opinion.recommendation}
        </div>
        {opinion.reasoning && (
          <details className="text-xs text-muted-foreground">
            <summary className="cursor-pointer hover:text-foreground">推理过程</summary>
            <p className="mt-1 pl-2 border-l-2 border-muted">{opinion.reasoning}</p>
          </details>
        )}
      </CardContent>
    </Card>
  )
}


function BlinkingCursor() {
  return <span className="inline-block w-1.5 h-4 bg-primary/70 animate-pulse ml-0.5 align-text-bottom" />
}
