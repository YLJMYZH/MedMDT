import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import SearchPanel from '@/components/knowledge/SearchPanel'
import FileUpload from '@/components/knowledge/FileUpload'

export default function KnowledgePage() {
  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold">知识库管理</h1>
        <p className="text-muted-foreground mt-1">搜索医学知识或上传文件进行知识提取</p>
      </div>

      <Tabs defaultValue="search">
        <TabsList>
          <TabsTrigger value="search">知识搜索</TabsTrigger>
          <TabsTrigger value="upload">文件上传</TabsTrigger>
        </TabsList>
        <TabsContent value="search">
          <SearchPanel />
        </TabsContent>
        <TabsContent value="upload">
          <FileUpload />
        </TabsContent>
      </Tabs>
    </div>
  )
}
