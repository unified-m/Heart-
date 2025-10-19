import { useState } from 'react';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { AuthForm } from './components/AuthForm';
import { Dashboard } from './pages/Dashboard';
import { RecordingPage } from './pages/RecordingPage';
import { AnalysisPage } from './pages/AnalysisPage';

type Page = 'dashboard' | 'record' | 'analysis';

function AppContent() {
  const { user, loading } = useAuth();
  const [currentPage, setCurrentPage] = useState<Page>('dashboard');
  const [selectedRecordingId, setSelectedRecordingId] = useState<string | null>(null);

  const handleNavigate = (page: Page, recordingId?: string) => {
    setCurrentPage(page);
    if (recordingId) {
      setSelectedRecordingId(recordingId);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return <AuthForm />;
  }

  if (currentPage === 'record') {
    return <RecordingPage onNavigate={handleNavigate} />;
  }

  if (currentPage === 'analysis' && selectedRecordingId) {
    return <AnalysisPage recordingId={selectedRecordingId} onNavigate={handleNavigate} />;
  }

  return <Dashboard onNavigate={handleNavigate} />;
}

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App;
