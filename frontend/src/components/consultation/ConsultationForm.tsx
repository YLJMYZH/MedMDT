import { useState, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import type { ConsultationRequest, MedicalRecord } from '@/lib/types'
import { api } from '@/lib/api'

interface Props {
  onSubmit: (data: ConsultationRequest) => void
  isSubmitting: boolean
}

export default function ConsultationForm({ onSubmit, isSubmitting }: Props) {
  const [name, setName] = useState('')
  const [age, setAge] = useState('')
  const [gender, setGender] = useState('男')
  const [chiefComplaint, setChiefComplaint] = useState('')
  const [medicalHistory, setMedicalHistory] = useState('')
  const [maxRounds, setMaxRounds] = useState(3)
  const [uploadedRecords, setUploadedRecords] = useState<MedicalRecord[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    setIsUploading(true)
    try {
      const result = await api.uploadConsultationFiles(Array.from(files))
      setUploadedRecords(prev => [...prev, ...result.records])
    } catch (err) {
      console.error('File upload failed:', err)
    } finally {
      setIsUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const removeRecord = (index: number) => {
    setUploadedRecords(prev => prev.filter((_, i) => i !== index))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const baseRecords: Array<Record<string, string>> = medicalHistory
      ? [{ record_type: 'history', content: medicalHistory }]
      : []
    const fileRecords = uploadedRecords
      .filter(r => !r.error)
      .map(r => ({ record_type: r.record_type, content: r.content }))

    onSubmit({
      patient_info: {
        name,
        age,
        gender,
        chief_complaint: chiefComplaint,
        medical_history: medicalHistory,
      },
      medical_records: [...baseRecords, ...fileRecords],
      max_rounds: maxRounds,
    })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6 max-w-2xl">
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">患者基本信息</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="text-sm font-medium mb-1.5 block">姓名</label>
              <Input
                value={name}
                onChange={e => setName(e.target.value)}
                placeholder="患者姓名"
                required
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-1.5 block">年龄</label>
              <Input
                value={age}
                onChange={e => setAge(e.target.value)}
                placeholder="如：45"
                required
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-1.5 block">性别</label>
              <select
                value={gender}
                onChange={e => setGender(e.target.value)}
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              >
                <option value="男">男</option>
                <option value="女">女</option>
              </select>
            </div>
          </div>
          <div>
            <label className="text-sm font-medium mb-1.5 block">主诉</label>
            <Textarea
              value={chiefComplaint}
              onChange={e => setChiefComplaint(e.target.value)}
              placeholder="描述患者的主要症状和持续时间..."
              rows={3}
              required
            />
          </div>
          <div>
            <label className="text-sm font-medium mb-1.5 block">病史 / 检查结果</label>
            <Textarea
              value={medicalHistory}
              onChange={e => setMedicalHistory(e.target.value)}
              placeholder="既往病史、用药情况、检查结果等..."
              rows={4}
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">上传检查报告 / 影像</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div
            className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6 text-center cursor-pointer hover:border-primary/50 transition-colors"
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.jpg,.jpeg,.png,.bmp,.tiff,.webp,.gif,.dcm,.dicom"
              onChange={handleFileUpload}
              className="hidden"
            />
            <p className="text-sm text-muted-foreground">
              {isUploading ? '正在解析文件...' : '点击或拖拽上传 PDF / 图片 / DICOM 文件'}
            </p>
            <p className="text-xs text-muted-foreground/60 mt-1">
              支持格式: PDF, JPG, PNG, BMP, TIFF, WEBP, GIF, DICOM
            </p>
          </div>

          {uploadedRecords.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {uploadedRecords.map((rec, i) => (
                <Badge
                  key={i}
                  variant={rec.error ? 'destructive' : 'secondary'}
                  className="gap-1 pr-1"
                >
                  <span className="max-w-[150px] truncate">{rec.filename}</span>
                  <span className="text-[10px] opacity-70">({rec.record_type})</span>
                  <button
                    type="button"
                    onClick={() => removeRecord(i)}
                    className="ml-1 rounded-full w-4 h-4 inline-flex items-center justify-center hover:bg-foreground/10"
                  >
                    ×
                  </button>
                </Badge>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">会诊设置</CardTitle>
        </CardHeader>
        <CardContent>
          <div>
            <label className="text-sm font-medium mb-3 block">
              讨论轮次: <span className="text-primary font-bold">{maxRounds}</span>
            </label>
            <Slider
              value={maxRounds}
              onValueChange={setMaxRounds}
              min={1}
              max={10}
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>1轮 (快速)</span>
              <span>10轮 (深入)</span>
            </div>
          </div>
        </CardContent>
      </Card>

      <Button type="submit" disabled={isSubmitting} className="w-full sm:w-auto">
        {isSubmitting ? '提交中...' : '开始多专家会诊'}
      </Button>
    </form>
  )
}
