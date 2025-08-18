import { Routes, Route } from 'react-router-dom'
import { ThemeProvider } from '@/components/layout/ThemeProvider'
import AppLayout from '@/components/layout/AppLayout'
import DashboardPage from '@/pages/DashboardPage'
import ConsultationPage from '@/pages/ConsultationPage'
import KnowledgePage from '@/pages/KnowledgePage'
import HistoryPage from '@/pages/HistoryPage'
import SettingsPage from '@/pages/SettingsPage'

export default function App() {
  return (
    <ThemeProvider>
      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/consultation" element={<ConsultationPage />} />
          <Route path="/knowledge" element={<KnowledgePage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Route>
      </Routes>
    </ThemeProvider>
  )
}
