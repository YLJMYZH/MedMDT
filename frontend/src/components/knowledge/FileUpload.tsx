import { useState, useRef, useEffect } from 'react'
import { Upload, FolderOpen, CheckCircle, AlertCircle, SkipForward, X } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { api } from '@/lib/api'
import { cn } from '@/lib/utils'
import type { BatchIngestResponse, FolderResult } from '@/lib/types'

const ACCEPTED = '.zip,.tar,.tar.gz,.tgz,.tar.bz2,.rar,.7z'
const MAX_SIZE = 500 * 1024 * 1024

export default function FileUpload() {
  const [dragOver, setDragOver] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<BatchIngestResponse | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const pollRef = useRef<number | null>(null)

  useEffect(() => {
    if (!jobId) return

    const poll = async () => {
      try {
        const res = await api.getBatchIngestStatus(jobId)
        setJobStatus(res)
        if (res.status === 'completed' || res.status === 'failed') {
          if (pollRef.current) clearInterval(pollRef.current)
        }
      } catch {
        if (pollRef.current) clearInterval(pollRef.current)
      }
    }

    poll()
    pollRef.current = window.setInterval(poll, 2000)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [jobId])

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return
    const file = files[0]

    if (file.size > MAX_SIZE) {
      setError('文件大小超过 500MB 限制')
      return
    }

    setError(null)
    setUploading(true)
    setJobStatus(null)
    try {
      const res = await api.ingestBatch(file)
      setJobId(res.job_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : '上传失败')
    } finally {
      setUploading(false)
    }
  }

  const statusIcon = (status: string) => {
    if (status === 'completed') return <CheckCircle size={14} className="text-green-600" />
    if (status === 'skipped') return <SkipForward size={14} className="text-amber-600" />
    return <AlertCircle size={14} className="text-red-600" />
  }

  const statusVariant = (status: string) => {
    if (status === 'completed') return 'success' as const
    if (status === 'skipped') return 'warning' as const
    return 'destructive' as const
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
        <p className="text-sm font-medium">拖拽压缩包到此处或点击上传</p>
        <p className="text-xs text-muted-foreground mt-1">
          支持 ZIP、TAR.GZ、RAR、7Z 格式（最大 500MB）
        </p>
        <p className="text-xs text-muted-foreground mt-1">
          压缩包内每个文件夹代表一个病人，包含病例文件（PDF/DICOM/图片）
        </p>
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

      {jobStatus && (
        <Card>
          <CardContent className="p-4 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FolderOpen size={18} className="text-primary" />
                <span className="font-medium text-sm">批量处理</span>
              </div>
              <Badge variant={jobStatus.status === 'completed' ? 'success' : jobStatus.status === 'failed' ? 'destructive' : 'default'}>
                {jobStatus.status === 'processing' ? '处理中' : jobStatus.status === 'completed' ? '完成' : jobStatus.status === 'failed' ? '失败' : '等待中'}
              </Badge>
            </div>

            {jobStatus.progress.total > 0 && (
              <div>
                <div className="flex justify-between text-xs text-muted-foreground mb-1">
                  <span>进度</span>
                  <span>{jobStatus.progress.current} / {jobStatus.progress.total} 个文件夹</span>
                </div>
                <div className="w-full h-2 bg-secondary rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary transition-all duration-300"
                    style={{ width: `${(jobStatus.progress.current / jobStatus.progress.total) * 100}%` }}
                  />
                </div>
              </div>
            )}

            <p className="text-sm text-muted-foreground">{jobStatus.message}</p>

            {jobStatus.folders.length > 0 && (
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {jobStatus.folders.map((folder: FolderResult, i: number) => (
                  <div key={i} className="flex items-center gap-2 p-2 rounded border text-sm">
                    {statusIcon(folder.status)}
                    <span className="font-medium truncate flex-1">{folder.folder_name}</span>
                    <Badge variant={statusVariant(folder.status)} className="text-xs">
                      {folder.status === 'completed' && `${folder.files_processed} 文件`}
                      {folder.status === 'skipped' && '跳过'}
                      {folder.status === 'failed' && '失败'}
                    </Badge>
                    {folder.message && folder.status !== 'completed' && (
                      <span className="text-xs text-muted-foreground truncate max-w-[200px]" title={folder.message}>
                        {folder.message}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
