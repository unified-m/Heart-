import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables');
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export interface User {
  id: string;
  email: string;
  full_name: string;
  date_of_birth?: string;
  created_at: string;
  updated_at: string;
}

export interface VideoRecording {
  id: string;
  user_id: string;
  video_url: string;
  duration_seconds: number;
  recording_date: string;
  processing_status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
  updated_at: string;
}

export interface HeartRateData {
  id: string;
  recording_id: string;
  timestamp_ms: number;
  heart_rate_bpm: number;
  confidence_score: number;
  created_at: string;
}

export interface RiskPrediction {
  id: string;
  recording_id: string;
  risk_level: 'low' | 'medium' | 'high';
  risk_score: number;
  insights: {
    variability?: string;
    trend?: string;
    recommendations?: string[];
    anomalies?: string[];
  };
  predicted_at: string;
  created_at: string;
}
