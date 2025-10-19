import { supabase, VideoRecording, HeartRateData, RiskPrediction } from '../lib/supabase';

export class HealthMonitoringAPI {
  static async ensureUserProfile(userId: string, email: string): Promise<void> {
    const { data: existingUser } = await supabase
      .from('users')
      .select('id')
      .eq('id', userId)
      .maybeSingle();

    if (!existingUser) {
      const { error } = await supabase
        .from('users')
        .insert({
          id: userId,
          email: email,
          full_name: email.split('@')[0],
        });

      if (error && !error.message.includes('duplicate')) {
        throw error;
      }
    }
  }

  static async uploadVideo(userId: string, videoBlob: Blob): Promise<string> {
    const fileName = `${userId}/${Date.now()}.webm`;

    const { data, error } = await supabase.storage
      .from('health-videos')
      .upload(fileName, videoBlob, {
        contentType: 'video/webm',
        upsert: false,
      });

    if (error) throw error;

    const { data: { publicUrl } } = supabase.storage
      .from('health-videos')
      .getPublicUrl(data.path);

    return publicUrl;
  }

  static async createRecording(
    userId: string,
    videoUrl: string,
    durationSeconds: number
  ): Promise<VideoRecording> {
    const { data, error } = await supabase
      .from('video_recordings')
      .insert({
        user_id: userId,
        video_url: videoUrl,
        duration_seconds: durationSeconds,
        processing_status: 'pending',
      })
      .select()
      .single();

    if (error) throw error;
    return data;
  }

  static async processVideoWithML(recordingId: string, videoUrl: string): Promise<void> {
    const response = await fetch(
      `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/process-heart-rate`,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${import.meta.env.VITE_SUPABASE_ANON_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          recording_id: recordingId,
          video_url: videoUrl,
        }),
      }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to process video');
    }

    return await response.json();
  }

  static async saveHeartRateData(heartRateData: Omit<HeartRateData, 'id' | 'created_at'>[]): Promise<void> {
    const { error } = await supabase
      .from('heart_rate_data')
      .insert(heartRateData);

    if (error) throw error;
  }

  static async saveRiskPrediction(riskPrediction: Omit<RiskPrediction, 'id' | 'created_at'>): Promise<void> {
    const { error } = await supabase
      .from('risk_predictions')
      .insert(riskPrediction);

    if (error) throw error;
  }

  static async getRecordings(userId: string): Promise<VideoRecording[]> {
    const { data, error } = await supabase
      .from('video_recordings')
      .select('*')
      .eq('user_id', userId)
      .order('recording_date', { ascending: false });

    if (error) throw error;
    return data || [];
  }

  static async getRecordingById(recordingId: string): Promise<VideoRecording | null> {
    const { data, error } = await supabase
      .from('video_recordings')
      .select('*')
      .eq('id', recordingId)
      .maybeSingle();

    if (error) throw error;
    return data;
  }

  static async getHeartRateData(recordingId: string): Promise<HeartRateData[]> {
    const { data, error } = await supabase
      .from('heart_rate_data')
      .select('*')
      .eq('recording_id', recordingId)
      .order('timestamp_ms', { ascending: true });

    if (error) throw error;
    return data || [];
  }

  static async getRiskPrediction(recordingId: string): Promise<RiskPrediction | null> {
    const { data, error } = await supabase
      .from('risk_predictions')
      .select('*')
      .eq('recording_id', recordingId)
      .maybeSingle();

    if (error) throw error;
    return data;
  }

  static async processVideo(recordingId: string): Promise<void> {
    await supabase
      .from('video_recordings')
      .update({ processing_status: 'processing' })
      .eq('id', recordingId);

    const response = await fetch(
      `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/process-heart-rate`,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${import.meta.env.VITE_SUPABASE_ANON_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ recording_id: recordingId }),
      }
    );

    if (!response.ok) {
      throw new Error('Failed to process video');
    }
  }

  static simulateHeartRateData(recordingId: string, durationSeconds: number): HeartRateData[] {
    const data: HeartRateData[] = [];
    const samplesPerSecond = 4;
    const totalSamples = durationSeconds * samplesPerSecond;

    let baseHeartRate = 70 + Math.random() * 20;

    for (let i = 0; i < totalSamples; i++) {
      const timestamp = (i / samplesPerSecond) * 1000;
      const variation = Math.sin(i / 10) * 5 + (Math.random() - 0.5) * 3;
      const heartRate = Math.max(50, Math.min(120, baseHeartRate + variation));

      data.push({
        id: `sim-${recordingId}-${i}`,
        recording_id: recordingId,
        timestamp_ms: Math.round(timestamp),
        heart_rate_bpm: Math.round(heartRate * 100) / 100,
        confidence_score: 0.85 + Math.random() * 0.15,
        created_at: new Date().toISOString(),
      });
    }

    return data;
  }

  static simulateRiskPrediction(
    recordingId: string,
    heartRateData: HeartRateData[]
  ): RiskPrediction {
    const avgHeartRate = heartRateData.reduce((sum, d) => sum + d.heart_rate_bpm, 0) / heartRateData.length;
    const maxHeartRate = Math.max(...heartRateData.map(d => d.heart_rate_bpm));
    const minHeartRate = Math.min(...heartRateData.map(d => d.heart_rate_bpm));
    const variability = maxHeartRate - minHeartRate;

    let riskLevel: 'low' | 'medium' | 'high';
    let riskScore: number;
    const recommendations: string[] = [];
    const anomalies: string[] = [];

    if (avgHeartRate < 60) {
      riskLevel = 'medium';
      riskScore = 45 + Math.random() * 10;
      anomalies.push('Resting heart rate below normal range detected');
      recommendations.push('Consider consulting with a healthcare provider about bradycardia');
    } else if (avgHeartRate > 100) {
      riskLevel = 'high';
      riskScore = 70 + Math.random() * 20;
      anomalies.push('Elevated resting heart rate detected');
      recommendations.push('Elevated heart rate may indicate stress or cardiovascular concerns');
      recommendations.push('Schedule an appointment with your doctor');
    } else {
      riskLevel = 'low';
      riskScore = 15 + Math.random() * 20;
      recommendations.push('Your heart rate appears within normal range');
      recommendations.push('Continue regular physical activity and healthy lifestyle');
    }

    if (variability > 30) {
      if (riskLevel === 'low') riskLevel = 'medium';
      riskScore += 15;
      anomalies.push('High heart rate variability detected during measurement');
      recommendations.push('Monitor stress levels and ensure adequate rest');
    }

    return {
      id: `sim-risk-${recordingId}`,
      recording_id: recordingId,
      risk_level: riskLevel,
      risk_score: Math.min(100, Math.round(riskScore)),
      insights: {
        variability: `Heart rate ranged from ${Math.round(minHeartRate)} to ${Math.round(maxHeartRate)} BPM`,
        trend: avgHeartRate > 80 ? 'Slightly elevated' : 'Normal',
        recommendations,
        anomalies,
      },
      predicted_at: new Date().toISOString(),
      created_at: new Date().toISOString(),
    };
  }
}
