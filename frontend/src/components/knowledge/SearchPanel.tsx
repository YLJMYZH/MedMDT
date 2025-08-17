import { useState } from 'react'
import { Search } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { api } from '@/lib/api'
import type { SearchResult } from '@/lib/types'

export default function SearchPanel() {
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(10)
  const [results, setResults] = useState<SearchResult[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [searched, setSearched] = useState(false)

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    try {
      const res = await api.searchKnowledge(query, topK)
      setResults(res.results)
      setTotal(res.total)
      setSearched(true)
    } catch {
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  const scoreBadge = (score: number) => {
    if (score > 0.8) return 'success'
    if (score > 0.6) return 'warning'
    return 'secondary'
  }

  return (
    <div className="space-y-4">
      <form onSubmit={handleSearch} className="flex gap-2">
        <div className="relative flex-1">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="搜索医学知识..."
            className="pl-9"
          />
        </div>
        <select
          value={topK}
          onChange={e => setTopK(Number(e.target.value))}
          className="h-10 rounded-md border border-input bg-background px-3 text-sm"
        >
          <option value={5}>Top 5</option>
          <option value={10}>Top 10</option>
          <option value={20}>Top 20</option>
        </select>
        <Button type="submit" disabled={loading}>
          {loading ? '搜索中...' : '搜索'}
        </Button>
      </form>

      {searched && results.length === 0 && (
        <div className="text-center py-12 text-muted-foreground">
          未找到相关结果
        </div>
      )}

      {results.length > 0 && (
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground">找到 {total} 条结果</p>
          {results.map((r, i) => (
            <Card key={i} className="hover:shadow-md transition-shadow">
              <CardContent className="p-4">
                <div className="flex items-start justify-between gap-3">
                  <p className="text-sm flex-1">{r.text.slice(0, 300)}{r.text.length > 300 ? '...' : ''}</p>
                  <Badge variant={scoreBadge(r.score)} className="shrink-0">
                    {(r.score * 100).toFixed(0)}%
                  </Badge>
                </div>
                <div className="mt-2 text-xs text-muted-foreground">
                  来源: {r.source || '未知'}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
