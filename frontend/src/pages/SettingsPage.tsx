import { useCallback, useEffect, useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { RefreshCw } from 'lucide-react'
import { api } from '@/lib/api'
import type { ProviderInfo } from '@/lib/types'

export default function SettingsPage() {
  const [provider, setProvider] = useState('openai')
  const [model, setModel] = useState('gpt-4o')
  const [apiKey, setApiKey] = useState('')
  const [baseUrl, setBaseUrl] = useState('')
  const [paddleocrToken, setPaddleocrToken] = useState('')
  const [providers, setProviders] = useState<ProviderInfo[]>([])
  const [models, setModels] = useState<string[]>([])
  const [loadingModels, setLoadingModels] = useState(false)
  const [modelsError, setModelsError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  const fetchModels = useCallback(async (p: string, key: string, url: string) => {
    setLoadingModels(true)
    setModelsError(null)
    try {
      const res = await api.listModels({
        provider: p,
        api_key: key || null,
        base_url: url || null,
      })
      setModels(res.models)
    } catch (err) {
      setModels([])
      setModelsError(err instanceof Error ? err.message : '获取模型列表失败')
    } finally {
      setLoadingModels(false)
    }
  }, [])

  useEffect(() => {
    api.getSettings().then(data => {
      setProvider(data.provider)
      setModel(data.model)
      setApiKey(data.api_key || '')
      setBaseUrl(data.base_url || '')
      setPaddleocrToken(data.paddleocr_token || '')
      setProviders(data.providers)
      fetchModels(data.provider, data.api_key || '', data.base_url || '')
    }).catch(() => {}).finally(() => setLoading(false))
  }, [fetchModels])

  const currentProvider = providers.find(p => p.key === provider)
  const needsKey = currentProvider?.needs_key ?? true
  const needsBaseUrl = currentProvider?.needs_base_url ?? false

  const handleSave = async () => {
    setSaving(true)
    setMessage(null)
    try {
      const res = await api.saveSettings({
        provider,
        model,
        api_key: apiKey || null,
        base_url: baseUrl || null,
        paddleocr_token: paddleocrToken || null,
      })
      setMessage({ type: 'success', text: res.message })
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
      const res = await api.testConnection({
        provider,
        model,
        api_key: apiKey || null,
        base_url: baseUrl || null,
      })
      setMessage({ type: 'success', text: res.message })
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : '连接失败' })
    } finally {
      setTesting(false)
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
        <p className="text-muted-foreground mt-1">配置 LLM 模型和 API 密钥</p>
      </div>

      <div className="space-y-6 max-w-2xl">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">LLM 配置</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
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
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">PaddleOCR 配置</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm font-medium mb-1.5 block">PaddleOCR Token</label>
              <Input
                type="password"
                value={paddleocrToken}
                onChange={e => setPaddleocrToken(e.target.value)}
                placeholder="PaddleOCR API Token"
              />
              <p className="text-xs text-muted-foreground mt-1">用于 PDF 文档 OCR 解析</p>
            </div>
          </CardContent>
        </Card>

      </div>
    </div>
  )
}
