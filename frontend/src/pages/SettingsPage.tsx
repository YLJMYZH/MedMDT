import { useCallback, useEffect, useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { RefreshCw, ChevronRight, ArrowLeft } from 'lucide-react'
import { api } from '@/lib/api'
import type { ProviderInfo, EndpointData, EmbeddingData, ExpertLLMData } from '@/lib/types'

interface Message {
  type: 'success' | 'error'
  text: string
}

function LLMSection({
  providers,
  endpoint,
  showTest,
  onSave,
}: {
  providers: ProviderInfo[]
  endpoint: EndpointData
  showTest?: boolean
  onSave: (ep: EndpointData) => Promise<void>
}) {
  const [provider, setProvider] = useState(endpoint.provider)
  const [model, setModel] = useState(endpoint.model)
  const [apiKey, setApiKey] = useState(endpoint.api_key || '')
  const [baseUrl, setBaseUrl] = useState(endpoint.base_url || '')
  const [models, setModels] = useState<string[]>([])
  const [loadingModels, setLoadingModels] = useState(false)
  const [modelsError, setModelsError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [message, setMessage] = useState<Message | null>(null)

  const currentProvider = providers.find(p => p.key === provider)
  const needsKey = currentProvider?.needs_key ?? true
  const needsBaseUrl = currentProvider?.needs_base_url ?? false

  const fetchModels = useCallback(async (p: string, key: string, url: string) => {
    setLoadingModels(true)
    setModelsError(null)
    try {
      const res = await api.listModels({ provider: p, api_key: key || null, base_url: url || null })
      setModels(res.models)
      if (res.models.length > 0) {
        setModel(prev => res.models.includes(prev) ? prev : res.models[0])
      }
    } catch (err) {
      setModels([])
      setModelsError(err instanceof Error ? err.message : '获取模型列表失败')
    } finally {
      setLoadingModels(false)
    }
  }, [])

  useEffect(() => {
    if (endpoint.api_key || !needsKey) {
      fetchModels(endpoint.provider, endpoint.api_key || '', endpoint.base_url || '')
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const getEp = (): EndpointData => ({ provider, model, api_key: apiKey || null, base_url: baseUrl || null })

  const handleSave = async () => {
    setSaving(true)
    setMessage(null)
    try {
      await onSave(getEp())
      setMessage({ type: 'success', text: '已保存' })
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : '保存失败' })
    } finally {
      setSaving(false)
    }
  }

  const handleTest = async () => {
    setTesting(true)
    setMessage(null)
    try {
      const res = await api.testConnection({ provider, model, api_key: apiKey || null, base_url: baseUrl || null })
      setMessage({ type: 'success', text: res.message })
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : '连接失败' })
    } finally {
      setTesting(false)
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <label className="text-sm font-medium mb-1.5 block">Provider</label>
        <select
          value={provider}
          onChange={e => {
            const p = e.target.value
            setProvider(p)
            setModels([])
            setModel('')
            fetchModels(p, apiKey, baseUrl)
          }}
          className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
        >
          {providers.map(p => (
            <option key={p.key} value={p.key}>{p.label}</option>
          ))}
        </select>
      </div>

      {needsKey && (
        <div>
          <label className="text-sm font-medium mb-1.5 block">API Key</label>
          <Input
            type="password"
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
            placeholder="sk-..."
          />
        </div>
      )}

      {(needsBaseUrl || provider === 'custom') && (
        <div>
          <label className="text-sm font-medium mb-1.5 block">Base URL</label>
          <Input
            value={baseUrl}
            onChange={e => setBaseUrl(e.target.value)}
            placeholder="https://api.example.com/v1"
          />
        </div>
      )}

      <div>
        <div className="flex items-center gap-2 mb-1.5">
          <label className="text-sm font-medium">Model</label>
          <button
            type="button"
            onClick={() => fetchModels(provider, apiKey, baseUrl)}
            disabled={loadingModels}
            className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
            title="刷新模型列表"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loadingModels ? 'animate-spin' : ''}`} />
            {loadingModels ? '加载中' : '刷新'}
          </button>
        </div>
        {models.length > 0 ? (
          <select
            value={model}
            onChange={e => setModel(e.target.value)}
            className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          >
            {models.map(m => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        ) : (
          <Input
            value={model}
            onChange={e => setModel(e.target.value)}
            placeholder="输入模型名称"
          />
        )}
        {modelsError && (
          <p className="text-xs text-muted-foreground mt-1">{modelsError}</p>
        )}
      </div>

      <div className="flex gap-3 pt-2">
        {showTest && (
          <Button onClick={handleTest} disabled={testing} variant="outline">
            {testing ? '测试中...' : '测试连接'}
          </Button>
        )}
        <Button onClick={handleSave} disabled={saving}>
          {saving ? '保存中...' : '保存'}
        </Button>
      </div>

      {message && (
        <p className={`text-sm ${message.type === 'success' ? 'text-green-600' : 'text-destructive'}`}>
          {message.text}
        </p>
      )}
    </div>
  )
}

function EmbeddingSection({
  providers,
  embeddingData,
  onSave,
}: {
  providers: ProviderInfo[]
  embeddingData: EmbeddingData
  onSave: (emb: EmbeddingData) => Promise<void>
}) {
  const [provider, setProvider] = useState(embeddingData.provider)
  const [model, setModel] = useState(embeddingData.model)
  const [apiKey, setApiKey] = useState(embeddingData.api_key || '')
  const [baseUrl, setBaseUrl] = useState(embeddingData.base_url || '')
  const [dim, setDim] = useState(embeddingData.dim?.toString() || '1024')
  const [models, setModels] = useState<string[]>([])
  const [loadingModels, setLoadingModels] = useState(false)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [message, setMessage] = useState<Message | null>(null)

  const currentProvider = providers.find(p => p.key === provider)
  const needsBaseUrl = currentProvider?.needs_base_url ?? false

  const fetchModels = useCallback(async (p: string, key: string, url: string) => {
    setLoadingModels(true)
    try {
      const res = await api.listEmbeddingModels({ provider: p, api_key: key || null, base_url: url || null })
      setModels(res.models)
      if (res.models.length > 0) {
        setModel(prev => res.models.includes(prev) ? prev : res.models[0])
      }
    } catch {
      setModels([])
    } finally {
      setLoadingModels(false)
    }
  }, [])

  useEffect(() => {
    fetchModels(embeddingData.provider, embeddingData.api_key || '', embeddingData.base_url || '')
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const getEmb = (): EmbeddingData => ({
    provider, model, api_key: apiKey || null, base_url: baseUrl || null, dim: parseInt(dim, 10) || 1024,
  })

  const handleSave = async () => {
    setSaving(true)
    setMessage(null)
    try {
      await onSave(getEmb())
      setMessage({ type: 'success', text: '已保存' })
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : '保存失败' })
    } finally {
      setSaving(false)
    }
  }

  const handleTest = async () => {
    setTesting(true)
    setMessage(null)
    try {
      const res = await api.testEmbeddingConnection({ provider, model, api_key: apiKey || null, base_url: baseUrl || null })
      setMessage({ type: 'success', text: res.message })
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : '连接失败' })
    } finally {
      setTesting(false)
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <label className="text-sm font-medium mb-1.5 block">Provider</label>
        <select
          value={provider}
          onChange={e => {
            const p = e.target.value
            setProvider(p)
            setModels([])
            setModel('')
            fetchModels(p, apiKey, baseUrl)
          }}
          className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
        >
          {providers.map(p => (
            <option key={p.key} value={p.key}>{p.label}</option>
          ))}
        </select>
      </div>

      <div>
        <label className="text-sm font-medium mb-1.5 block">API Key</label>
        <Input
          type="password"
          value={apiKey}
          onChange={e => setApiKey(e.target.value)}
          placeholder="sk-..."
        />
      </div>

      {(needsBaseUrl || provider === 'custom') && (
        <div>
          <label className="text-sm font-medium mb-1.5 block">Base URL</label>
          <Input
            value={baseUrl}
            onChange={e => setBaseUrl(e.target.value)}
            placeholder="https://api.example.com/v1"
          />
        </div>
      )}

      <div>
        <div className="flex items-center gap-2 mb-1.5">
          <label className="text-sm font-medium">Model</label>
          <button
            type="button"
            onClick={() => fetchModels(provider, apiKey, baseUrl)}
            disabled={loadingModels}
            className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
            title="刷新模型列表"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loadingModels ? 'animate-spin' : ''}`} />
            {loadingModels ? '加载中' : '刷新'}
          </button>
        </div>
        {models.length > 0 ? (
          <select
            value={model}
            onChange={e => setModel(e.target.value)}
            className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          >
            {models.map(m => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        ) : (
          <Input
            value={model}
            onChange={e => setModel(e.target.value)}
            placeholder="如 bge-large-zh-v1.5、text-embedding-3-small"
          />
        )}
      </div>

      <div>
        <label className="text-sm font-medium mb-1.5 block">维度</label>
        <Input
          type="number"
          value={dim}
          onChange={e => setDim(e.target.value)}
          placeholder="1024"
        />
      </div>

      <div className="flex gap-3 pt-2">
        <Button onClick={handleTest} disabled={testing} variant="outline">
          {testing ? '测试中...' : '测试连接'}
        </Button>
        <Button onClick={handleSave} disabled={saving}>
          {saving ? '保存中...' : '保存'}
        </Button>
      </div>

      {message && (
        <p className={`text-sm ${message.type === 'success' ? 'text-green-600' : 'text-destructive'}`}>
          {message.text}
        </p>
      )}
    </div>
  )
}

function ConsultationCard({
  endpoint,
  providers,
  experts,
  expertNames,
  onSaveEndpoint,
  onSaveExperts,
}: {
  endpoint: EndpointData
  providers: ProviderInfo[]
  experts: Record<string, ExpertLLMData>
  expertNames: Record<string, string>
  onSaveEndpoint: (ep: EndpointData) => Promise<void>
  onSaveExperts: (exp: Record<string, ExpertLLMData>) => Promise<void>
}) {
  const [expertMode, setExpertMode] = useState(false)
  const [localExperts, setLocalExperts] = useState<Record<string, ExpertLLMData>>(experts)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState<Message | null>(null)

  const expertIds = Object.keys(expertNames)

  const handleBack = () => {
    if (window.confirm('将不保存，回到统一配置')) {
      setLocalExperts(experts)
      setExpertMode(false)
      setMessage(null)
    }
  }

  const handleCancel = () => {
    if (window.confirm('将不保存，回到统一配置')) {
      setLocalExperts(experts)
      setExpertMode(false)
      setMessage(null)
    }
  }

  const handleSaveExperts = async () => {
    setSaving(true)
    setMessage(null)
    try {
      await onSaveExperts(localExperts)
      setExpertMode(false)
      setMessage(null)
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : '保存失败' })
    } finally {
      setSaving(false)
    }
  }

  if (expertMode) {
    return (
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <button
              onClick={handleBack}
              className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              <ArrowLeft className="h-4 w-4" />
              返回统一配置
            </button>
          </div>
          <CardTitle className="text-lg">专家模型单独配置</CardTitle>
          <p className="text-sm text-muted-foreground">为每位专家单独配置模型，API Key 继承会诊配置</p>
        </CardHeader>
        <CardContent className="space-y-4">
          {expertIds.map(eid => {
            const ecfg = localExperts[eid] || { provider: '', model: '' }
            return (
              <div key={eid} className="flex items-center gap-3">
                <span className="text-sm font-medium w-24 shrink-0">{expertNames[eid]}</span>
                <select
                  value={ecfg.provider}
                  onChange={e => {
                    setLocalExperts(prev => ({
                      ...prev,
                      [eid]: { ...prev[eid], provider: e.target.value },
                    }))
                  }}
                  className="h-9 rounded-md border border-input bg-background px-2 text-sm flex-1"
                >
                  <option value="">（跟随会诊）</option>
                  {providers.map(p => (
                    <option key={p.key} value={p.key}>{p.label}</option>
                  ))}
                </select>
                <Input
                  value={ecfg.model}
                  onChange={e => {
                    setLocalExperts(prev => ({
                      ...prev,
                      [eid]: { ...prev[eid], model: e.target.value },
                    }))
                  }}
                  placeholder="模型名（留空跟随会诊）"
                  className="h-9 flex-1"
                />
              </div>
            )
          })}

          <div className="flex gap-3 pt-2">
            <Button variant="outline" onClick={handleCancel}>
              取消
            </Button>
            <Button onClick={handleSaveExperts} disabled={saving}>
              {saving ? '保存中...' : '保存'}
            </Button>
          </div>

          {message && (
            <p className={`text-sm ${message.type === 'success' ? 'text-green-600' : 'text-destructive'}`}>
              {message.text}
            </p>
          )}
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">会诊</CardTitle>
        <p className="text-sm text-muted-foreground">用于专家讨论和会诊对话</p>
      </CardHeader>
      <CardContent className="space-y-4">
        <LLMSection
          providers={providers}
          endpoint={endpoint}
          showTest
          onSave={onSaveEndpoint}
        />
        <div className="border-t pt-3 mt-2">
          <button
            onClick={() => {
              setLocalExperts(experts)
              setExpertMode(true)
            }}
            className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            专家模型单独配置
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      </CardContent>
    </Card>
  )
}

export default function SettingsPage() {
  const [loading, setLoading] = useState(true)
  const [providers, setProviders] = useState<ProviderInfo[]>([])
  const [consultation, setConsultation] = useState<EndpointData>({ provider: 'deepseek', model: 'deepseek-chat', api_key: null, base_url: null })
  const [knowledge, setKnowledge] = useState<EndpointData>({ provider: 'deepseek', model: 'deepseek-chat', api_key: null, base_url: null })
  const [vision, setVision] = useState<EndpointData>({ provider: 'deepseek', model: 'deepseek-chat', api_key: null, base_url: null })
  const [embedding, setEmbedding] = useState<EmbeddingData>({ provider: '', model: '', api_key: null, base_url: null, dim: 1024 })
  const [experts, setExperts] = useState<Record<string, ExpertLLMData>>({})
  const [expertNames, setExpertNames] = useState<Record<string, string>>({})
  const [paddleocrToken, setPaddleocrToken] = useState('')
  const [savingOcr, setSavingOcr] = useState(false)
  const [ocrMessage, setOcrMessage] = useState<Message | null>(null)

  useEffect(() => {
    api.getSettings().then(data => {
      setProviders(data.providers)
      setConsultation(data.consultation)
      setKnowledge(data.knowledge)
      setVision(data.vision)
      setEmbedding(data.embedding)
      setExperts(data.experts)
      setExpertNames(data.expert_names)
      setPaddleocrToken(data.paddleocr_token || '')
    }).catch(() => {}).finally(() => setLoading(false))
  }, [])

  const handleSaveOcr = async () => {
    setSavingOcr(true)
    setOcrMessage(null)
    try {
      const res = await api.saveSettings({ paddleocr_token: paddleocrToken || null })
      setOcrMessage({ type: 'success', text: res.message })
    } catch (err) {
      setOcrMessage({ type: 'error', text: err instanceof Error ? err.message : '保存失败' })
    } finally {
      setSavingOcr(false)
    }
  }

  if (loading) {
    return (
      <div>
        <h1 className="text-2xl font-bold mb-6">设置</h1>
        <Card className="max-w-2xl"><CardContent className="p-6 text-muted-foreground">加载中...</CardContent></Card>
      </div>
    )
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold">设置</h1>
        <p className="text-muted-foreground mt-1">配置模型和 API 密钥</p>
      </div>

      <div className="space-y-6 max-w-2xl">
        <ConsultationCard
          endpoint={consultation}
          providers={providers}
          experts={experts}
          expertNames={expertNames}
          onSaveEndpoint={async (ep) => {
            await api.saveSettings({ consultation: ep })
            setConsultation(ep)
          }}
          onSaveExperts={async (exp) => {
            await api.saveSettings({ experts: exp })
            setExperts(exp)
          }}
        />

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">知识检索/提取</CardTitle>
            <p className="text-sm text-muted-foreground">用于知识库检索和文档信息提取</p>
          </CardHeader>
          <CardContent>
            <h4 className="text-sm font-semibold mb-4">大语言模型</h4>
            <LLMSection
              providers={providers}
              endpoint={knowledge}
              showTest
              onSave={async (ep) => {
                await api.saveSettings({ knowledge: ep })
                setKnowledge(ep)
              }}
            />
            <hr className="my-8 border-border" />
            <h4 className="text-sm font-semibold mb-4">Embedding 模型</h4>
            <EmbeddingSection
              providers={providers}
              embeddingData={embedding}
              onSave={async (emb) => {
                await api.saveSettings({ embedding: emb })
                setEmbedding(emb)
              }}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">图片分析（多模态）</CardTitle>
            <p className="text-sm text-muted-foreground">用于医学影像和 DICOM 图片分析，需支持多模态</p>
          </CardHeader>
          <CardContent>
            <LLMSection
              providers={providers}
              endpoint={vision}
              showTest
              onSave={async (ep) => {
                await api.saveSettings({ vision: ep })
                setVision(ep)
              }}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">PaddleOCR</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm font-medium mb-1.5 block">Token</label>
              <Input
                type="password"
                value={paddleocrToken}
                onChange={e => setPaddleocrToken(e.target.value)}
                placeholder="PaddleOCR API Token"
              />
              <p className="text-xs text-muted-foreground mt-1">用于 PDF 文档 OCR 解析</p>
            </div>
            <Button onClick={handleSaveOcr} disabled={savingOcr}>
              {savingOcr ? '保存中...' : '保存'}
            </Button>
            {ocrMessage && (
              <p className={`text-sm ${ocrMessage.type === 'success' ? 'text-green-600' : 'text-destructive'}`}>
                {ocrMessage.text}
              </p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
