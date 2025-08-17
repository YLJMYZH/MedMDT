import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Slider } from '@/components/ui/slider'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import type { ConsultationRequest } from '@/lib/types'

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

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit({
      patient_info: {
        name,
        age,
        gender,
        chief_complaint: chiefComplaint,
        medical_history: medicalHistory,
      },
      medical_records: medicalHistory
        ? [{ record_type: 'history', content: medicalHistory }]
        : [],
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
