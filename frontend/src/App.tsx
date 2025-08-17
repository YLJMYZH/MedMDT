import { Routes, Route } from 'react-router-dom'
import { ThemeProvider } from '@/components/layout/ThemeProvider'
import AppLayout from '@/components/layout/AppLayout'
import DashboardPage from '@/pages/DashboardPage'
import ConsultationPage from '@/pages/ConsultationPage'
import KnowledgePage from '@/pages/KnowledgePage'
import HistoryPage from '@/pages/HistoryPage'

export default function App() {
  return (
    <ThemeProvider>
      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/consultation" element={<ConsultationPage />} />
          <Route path="/knowledge" element={<KnowledgePage />} />
          <Route path="/history" element={<HistoryPage />} />
        </Route>
      </Routes>
    </ThemeProvider>
  )
}
