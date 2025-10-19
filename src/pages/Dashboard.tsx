import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { HealthMonitoringAPI } from '../services/api';
import { VideoRecording } from '../lib/supabase';
import { Heart, Video, LogOut, Clock, TrendingUp, Activity } from 'lucide-react';

interface DashboardProps {
  onNavigate: (page: 'dashboard' | 'record' | 'analysis', recordingId?: string) => void;
}

export function Dashboard({ onNavigate }: DashboardProps) {
  const { user, signOut } = useAuth();
  const [recordings, setRecordings] = useState<VideoRecording[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadRecordings();
  }, [user]);

  const loadRecordings = async () => {
    if (!user) return;

    try {
      setLoading(true);
      await HealthMonitoringAPI.ensureUserProfile(user.id, user.email || '');
      const data = await HealthMonitoringAPI.getRecordings(user.id);
      setRecordings(data);
    } catch (error) {
      console.error('Failed to load recordings:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'processing':
        return 'bg-blue-100 text-blue-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusLabel = (status: string) => {
    return status.charAt(0).toUpperCase() + status.slice(1);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-red-600 rounded-lg flex items-center justify-center">
                <Heart className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">AI Health Monitor</h1>
                <p className="text-xs text-gray-500">Heart Rate Analysis</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-right">
                <p className="text-sm font-medium text-gray-900">{user?.email}</p>
                <p className="text-xs text-gray-500">Logged in</p>
              </div>
              <button
                onClick={signOut}
                className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
                title="Sign out"
              >
                <LogOut className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Dashboard</h2>
          <p className="text-gray-600">Manage your heart rate recordings and analysis</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-blue-100 rounded-lg">
                <Video className="w-6 h-6 text-blue-600" />
              </div>
            </div>
            <h3 className="text-2xl font-bold text-gray-900">{recordings.length}</h3>
            <p className="text-gray-600 text-sm">Total Recordings</p>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-green-100 rounded-lg">
                <Activity className="w-6 h-6 text-green-600" />
              </div>
            </div>
            <h3 className="text-2xl font-bold text-gray-900">
              {recordings.filter((r) => r.processing_status === 'completed').length}
            </h3>
            <p className="text-gray-600 text-sm">Completed Analysis</p>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-red-100 rounded-lg">
                <TrendingUp className="w-6 h-6 text-red-600" />
              </div>
            </div>
            <h3 className="text-2xl font-bold text-gray-900">
              {recordings.filter((r) => r.processing_status === 'processing').length}
            </h3>
            <p className="text-gray-600 text-sm">Processing</p>
          </div>
        </div>

        <div className="flex justify-between items-center mb-6">
          <h3 className="text-xl font-semibold text-gray-900">Recent Recordings</h3>
          <button
            onClick={() => onNavigate('record')}
            className="bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors flex items-center space-x-2"
          >
            <Video className="w-5 h-5" />
            <span>New Recording</span>
          </button>
        </div>

        {loading ? (
          <div className="bg-white rounded-lg shadow-md p-12 text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading recordings...</p>
          </div>
        ) : recordings.length === 0 ? (
          <div className="bg-white rounded-lg shadow-md p-12 text-center">
            <Video className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-gray-900 mb-2">No recordings yet</h3>
            <p className="text-gray-600 mb-6">
              Start by recording your first 60-second facial video for heart rate analysis
            </p>
            <button
              onClick={() => onNavigate('record')}
              className="bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors inline-flex items-center space-x-2"
            >
              <Video className="w-5 h-5" />
              <span>Record Now</span>
            </button>
          </div>
        ) : (
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50 border-b border-gray-200">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      Date
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      Duration
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-semibold text-gray-600 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {recordings.map((recording) => (
                    <tr key={recording.id} className="hover:bg-gray-50 transition-colors">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center space-x-3">
                          <Clock className="w-5 h-5 text-gray-400" />
                          <div>
                            <div className="text-sm font-medium text-gray-900">
                              {new Date(recording.recording_date).toLocaleDateString()}
                            </div>
                            <div className="text-xs text-gray-500">
                              {new Date(recording.recording_date).toLocaleTimeString()}
                            </div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                        {recording.duration_seconds} seconds
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span
                          className={`px-3 py-1 text-xs font-semibold rounded-full ${getStatusColor(
                            recording.processing_status
                          )}`}
                        >
                          {getStatusLabel(recording.processing_status)}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm">
                        {recording.processing_status === 'completed' && (
                          <button
                            onClick={() => onNavigate('analysis', recording.id)}
                            className="text-blue-600 hover:text-blue-700 font-medium"
                          >
                            View Analysis
                          </button>
                        )}
                        {recording.processing_status === 'processing' && (
                          <span className="text-gray-500">Processing...</span>
                        )}
                        {recording.processing_status === 'failed' && (
                          <span className="text-red-600">Failed</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
