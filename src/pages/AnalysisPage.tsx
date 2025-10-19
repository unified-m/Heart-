import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { HealthMonitoringAPI } from '../services/api';
import { VideoRecording, HeartRateData, RiskPrediction } from '../lib/supabase';
import { HeartRateChart } from '../components/HeartRateChart';
import { RiskAssessment } from '../components/RiskAssessment';
import { ArrowLeft, Loader } from 'lucide-react';

interface AnalysisPageProps {
  recordingId: string;
  onNavigate: (page: 'dashboard' | 'record' | 'analysis', recordingId?: string) => void;
}

export function AnalysisPage({ recordingId, onNavigate }: AnalysisPageProps) {
  useAuth();
  const [recording, setRecording] = useState<VideoRecording | null>(null);
  const [heartRateData, setHeartRateData] = useState<HeartRateData[]>([]);
  const [riskPrediction, setRiskPrediction] = useState<RiskPrediction | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadAnalysisData();
  }, [recordingId]);

  const loadAnalysisData = async () => {
    try {
      setLoading(true);

      const recordingData = await HealthMonitoringAPI.getRecordingById(recordingId);
      setRecording(recordingData);

      const heartRateFromDb = await HealthMonitoringAPI.getHeartRateData(recordingId);
      setHeartRateData(heartRateFromDb);

      const riskFromDb = await HealthMonitoringAPI.getRiskPrediction(recordingId);
      setRiskPrediction(riskFromDb);
    } catch (error) {
      console.error('Failed to load analysis data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Loader className="w-12 h-12 text-blue-600 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading analysis...</p>
        </div>
      </div>
    );
  }

  if (!recording) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-600 mb-4">Recording not found</p>
          <button
            onClick={() => onNavigate('dashboard')}
            className="text-blue-600 hover:text-blue-700 font-medium"
          >
            Return to Dashboard
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm border-b border-gray-200 mb-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center h-16">
            <button
              onClick={() => onNavigate('dashboard')}
              className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              <span className="font-medium">Back to Dashboard</span>
            </button>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Heart Rate Analysis</h2>
          <p className="text-gray-600">
            Recorded on {new Date(recording.recording_date).toLocaleDateString()} at{' '}
            {new Date(recording.recording_date).toLocaleTimeString()}
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <HeartRateChart data={heartRateData} />

            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-xl font-semibold text-gray-900 mb-4">Analysis Details</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-sm text-gray-600 mb-1">Recording Duration</div>
                  <div className="text-lg font-semibold text-gray-900">
                    {recording.duration_seconds} seconds
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-600 mb-1">Data Points</div>
                  <div className="text-lg font-semibold text-gray-900">
                    {heartRateData.length} samples
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-600 mb-1">Processing Status</div>
                  <div className="text-lg font-semibold text-green-600 capitalize">
                    {recording.processing_status}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-600 mb-1">Average Confidence</div>
                  <div className="text-lg font-semibold text-gray-900">
                    {(
                      (heartRateData.reduce((sum, d) => sum + d.confidence_score, 0) /
                        heartRateData.length) *
                      100
                    ).toFixed(1)}
                    %
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="lg:col-span-1">
            {riskPrediction && <RiskAssessment prediction={riskPrediction} />}
          </div>
        </div>

        <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h4 className="font-semibold text-blue-900 mb-2">About This Analysis</h4>
          <p className="text-sm text-blue-800 mb-3">
            This analysis uses advanced computer vision and signal processing techniques to extract
            heart rate information from subtle color changes in your facial video. The technology is
            based on photoplethysmography (PPG) principles, which detect blood volume changes in
            facial blood vessels.
          </p>
          <p className="text-sm text-blue-800">
            This tool is designed for informational and educational purposes. It is not a medical
            device and should not be used as a substitute for professional medical advice,
            diagnosis, or treatment. Always consult with a qualified healthcare provider for any
            health concerns.
          </p>
        </div>
      </div>
    </div>
  );
}
