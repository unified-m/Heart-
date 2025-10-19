import { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { HealthMonitoringAPI } from '../services/api';
import { VideoRecorder } from '../components/VideoRecorder';
import { ArrowLeft, Upload, CheckCircle } from 'lucide-react';

interface RecordingPageProps {
  onNavigate: (page: 'dashboard' | 'record' | 'analysis', recordingId?: string) => void;
}

export function RecordingPage({ onNavigate }: RecordingPageProps) {
  const { user } = useAuth();
  const [videoBlob, setVideoBlob] = useState<Blob | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadComplete, setUploadComplete] = useState(false);

  const handleRecordingComplete = (blob: Blob) => {
    setVideoBlob(blob);
  };

  const handleUpload = async () => {
    if (!videoBlob || !user) return;

    try {
      setUploading(true);

      await HealthMonitoringAPI.ensureUserProfile(user.id, user.email || '');

      const videoUrl = `recording-${Date.now()}.webm`;

      const recording = await HealthMonitoringAPI.createRecording(
        user.id,
        videoUrl,
        60
      );

      await HealthMonitoringAPI.processVideoWithML(recording.id, videoUrl);

      setUploadComplete(true);

      setTimeout(() => {
        onNavigate('analysis', recording.id);
      }, 2000);
    } catch (error) {
      console.error('Upload failed:', error);
      alert(`Failed to upload video: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setUploading(false);
    }
  };

  const handleRetake = () => {
    setVideoBlob(null);
    setUploadComplete(false);
  };

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
        <div className="mb-8 text-center">
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Record Heart Rate Video</h2>
          <p className="text-gray-600">
            Record a 60-second video of your face for AI-powered heart rate analysis
          </p>
        </div>

        {!videoBlob && !uploadComplete && (
          <VideoRecorder onRecordingComplete={handleRecordingComplete} maxDuration={60} />
        )}

        {videoBlob && !uploadComplete && (
          <div className="max-w-2xl mx-auto">
            <div className="bg-white rounded-xl shadow-lg overflow-hidden mb-6">
              <div className="relative aspect-video bg-gray-900">
                <video
                  src={URL.createObjectURL(videoBlob)}
                  controls
                  className="w-full h-full object-cover"
                />
              </div>

              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="font-semibold text-gray-900 mb-1">Recording Complete</h3>
                    <p className="text-sm text-gray-600">
                      Video size: {(videoBlob.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                  <CheckCircle className="w-8 h-8 text-green-600" />
                </div>

                <div className="flex space-x-4">
                  <button
                    onClick={handleRetake}
                    disabled={uploading}
                    className="flex-1 bg-gray-200 text-gray-700 py-3 px-6 rounded-lg font-semibold hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    Retake Video
                  </button>
                  <button
                    onClick={handleUpload}
                    disabled={uploading}
                    className="flex-1 bg-blue-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center space-x-2"
                  >
                    {uploading ? (
                      <>
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                        <span>Processing...</span>
                      </>
                    ) : (
                      <>
                        <Upload className="w-5 h-5" />
                        <span>Analyze Video</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {uploadComplete && (
          <div className="max-w-2xl mx-auto">
            <div className="bg-white rounded-xl shadow-lg p-12 text-center">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <CheckCircle className="w-10 h-10 text-green-600" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-2">Analysis Complete</h3>
              <p className="text-gray-600 mb-6">
                Your heart rate data has been analyzed successfully
              </p>
              <div className="animate-pulse text-blue-600">
                Redirecting to results...
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
