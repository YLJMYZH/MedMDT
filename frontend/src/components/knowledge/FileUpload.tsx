import { useState, useRef } from 'react'
import { Upload, FileText, X } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { api } from '@/lib/api'
import { useIngestStatus } from '@/hooks/useIngestStatus'
import { cn } from '@/lib/utils'

const ACCEPTED = '.pdf,.dcm,.jpg,.jpeg,.png,.bmp,.tiff'
const MAX_SIZE = 100 * 1024 * 1024

interface IngestJob {
  jobId: string
  filename: string
}

export default function FileUpload() {
  const [dragOver, setDragOver] = useState(false)
  const [jobs, setJobs] = useState<IngestJob[]>([])
  const [error, setError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return
    const file = files[0]

    if (file.size > MAX_SIZE) {
      setError('文件大小超过 100MB 限制')
      return
    }

    setError(null)
    setUploading(true)
    try {
      const res = await api.ingestFile(file)
      setJobs(prev => [{ jobId: res.job_id, filename: file.name }, ...prev])
    } catch (e) {
      setError(e instanceof Error ? e.message : '上传失败')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="space-y-4">
      <div
        className={cn(
          'border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer',
          dragOver ? 'border-primary bg-primary/5' : 'border-muted-foreground/25 hover:border-primary/50'
        )}
        onDragOver={e => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={e => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files) }}
        onClick={() => inputRef.current?.click()}
      >
        <Upload className="mx-auto mb-3 text-muted-foreground" size={32} />
        <p className="text-sm font-medium">拖拽文件到此处或点击上传</p>
        <p className="text-xs text-muted-foreground mt-1">支持 PDF, DICOM, JPG, PNG (最大 100MB)</p>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED}
          className="hidden"
          onChange={e => handleFiles(e.target.files)}
        />
      </div>

      {uploading && <p className="text-sm text-muted-foreground">上传中...</p>}
      {error && (
        <div className="flex items-center gap-2 text-sm text-destructive">
          <X size={14} />
          {error}
        </div>
      )}

      {jobs.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium">处理任务</h3>
          {jobs.map(job => (
            <JobItem key={job.jobId} {...job} />
          ))}
        </div>
      )}
    </div>
  )
}

function JobItem({ jobId, filename }: IngestJob) {
  const { status, isPolling } = useIngestStatus(jobId)

  const statusVariant = () => {
    if (!status) return 'secondary'
    if (status.status === 'completed') return 'success' as const
    if (status.status === 'failed') return 'destructive' as const
    return 'default' as const
  }

  return (
    <div className="flex items-center gap-3 p-3 rounded-md border text-sm">
      <FileText size={16} className="text-muted-foreground shrink-0" />
      <span className="flex-1 truncate">{filename}</span>
      <Badge variant={statusVariant()}>
        {isPolling && '处理中...'}
        {!isPolling && status?.status === 'completed' && '已完成'}
        {!isPolling && status?.status === 'failed' && '失败'}
        {!isPolling && !status && '等待中'}
      </Badge>
    </div>
  )
}
