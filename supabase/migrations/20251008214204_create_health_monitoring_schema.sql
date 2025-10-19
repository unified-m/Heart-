/*
  # Health Monitoring System Database Schema

  ## Overview
  This migration creates the database structure for an AI-powered health monitoring system
  that analyzes facial videos to extract heart rate data and predict heart-related risks.

  ## 1. New Tables

  ### `users`
  Custom user profile table with health monitoring specific fields:
  - `id` (uuid, primary key) - References auth.users
  - `email` (text, unique) - User email address
  - `full_name` (text) - User's full name
  - `date_of_birth` (date) - For age-based risk analysis
  - `created_at` (timestamptz) - Account creation timestamp
  - `updated_at` (timestamptz) - Last profile update

  ### `video_recordings`
  Stores metadata and analysis results for recorded videos:
  - `id` (uuid, primary key) - Unique recording identifier
  - `user_id` (uuid, foreign key) - References users table
  - `video_url` (text) - Storage path to video file
  - `duration_seconds` (integer) - Video length (should be ~60s)
  - `recording_date` (timestamptz) - When video was captured
  - `processing_status` (text) - Status: pending, processing, completed, failed
  - `created_at` (timestamptz) - Record creation timestamp
  - `updated_at` (timestamptz) - Last update timestamp

  ### `heart_rate_data`
  Time-series heart rate measurements extracted from videos:
  - `id` (uuid, primary key) - Unique measurement identifier
  - `recording_id` (uuid, foreign key) - References video_recordings
  - `timestamp_ms` (integer) - Milliseconds from video start
  - `heart_rate_bpm` (numeric) - Heart rate in beats per minute
  - `confidence_score` (numeric) - Confidence level (0-1)
  - `created_at` (timestamptz) - Record creation timestamp

  ### `risk_predictions`
  AI-generated health risk assessments:
  - `id` (uuid, primary key) - Unique prediction identifier
  - `recording_id` (uuid, foreign key) - References video_recordings
  - `risk_level` (text) - Risk category: low, medium, high
  - `risk_score` (numeric) - Numerical risk score (0-100)
  - `insights` (jsonb) - Detailed analysis and recommendations
  - `predicted_at` (timestamptz) - When prediction was generated
  - `created_at` (timestamptz) - Record creation timestamp

  ## 2. Indexes
  - Performance indexes on frequently queried columns
  - Foreign key indexes for join optimization
  - Time-based indexes for historical queries

  ## 3. Security
  - Row Level Security (RLS) enabled on all tables
  - Users can only access their own data
  - Authenticated access required for all operations
  - Service role bypass for backend processing

  ## 4. Important Notes
  - All timestamps use timestamptz for timezone awareness
  - Foreign key constraints ensure referential integrity
  - Cascade deletes maintain data consistency
  - Default values prevent null-related issues
*/

CREATE TABLE IF NOT EXISTS users (
  id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email text UNIQUE NOT NULL,
  full_name text DEFAULT '',
  date_of_birth date,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS video_recordings (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  video_url text NOT NULL,
  duration_seconds integer DEFAULT 60,
  recording_date timestamptz DEFAULT now(),
  processing_status text DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS heart_rate_data (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  recording_id uuid NOT NULL REFERENCES video_recordings(id) ON DELETE CASCADE,
  timestamp_ms integer NOT NULL,
  heart_rate_bpm numeric(5,2) NOT NULL,
  confidence_score numeric(3,2) DEFAULT 0.0 CHECK (confidence_score >= 0 AND confidence_score <= 1),
  created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS risk_predictions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  recording_id uuid NOT NULL REFERENCES video_recordings(id) ON DELETE CASCADE,
  risk_level text NOT NULL CHECK (risk_level IN ('low', 'medium', 'high')),
  risk_score numeric(5,2) DEFAULT 0.0 CHECK (risk_score >= 0 AND risk_score <= 100),
  insights jsonb DEFAULT '{}',
  predicted_at timestamptz DEFAULT now(),
  created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_video_recordings_user_id ON video_recordings(user_id);
CREATE INDEX IF NOT EXISTS idx_video_recordings_status ON video_recordings(processing_status);
CREATE INDEX IF NOT EXISTS idx_video_recordings_date ON video_recordings(recording_date DESC);
CREATE INDEX IF NOT EXISTS idx_heart_rate_data_recording_id ON heart_rate_data(recording_id);
CREATE INDEX IF NOT EXISTS idx_heart_rate_data_timestamp ON heart_rate_data(recording_id, timestamp_ms);
CREATE INDEX IF NOT EXISTS idx_risk_predictions_recording_id ON risk_predictions(recording_id);

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE video_recordings ENABLE ROW LEVEL SECURITY;
ALTER TABLE heart_rate_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_predictions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own profile"
  ON users FOR SELECT
  TO authenticated
  USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
  ON users FOR UPDATE
  TO authenticated
  USING (auth.uid() = id)
  WITH CHECK (auth.uid() = id);

CREATE POLICY "Users can insert own profile"
  ON users FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = id);

CREATE POLICY "Users can view own recordings"
  ON video_recordings FOR SELECT
  TO authenticated
  USING (user_id = auth.uid());

CREATE POLICY "Users can insert own recordings"
  ON video_recordings FOR INSERT
  TO authenticated
  WITH CHECK (user_id = auth.uid());

CREATE POLICY "Users can update own recordings"
  ON video_recordings FOR UPDATE
  TO authenticated
  USING (user_id = auth.uid())
  WITH CHECK (user_id = auth.uid());

CREATE POLICY "Users can delete own recordings"
  ON video_recordings FOR DELETE
  TO authenticated
  USING (user_id = auth.uid());

CREATE POLICY "Users can view own heart rate data"
  ON heart_rate_data FOR SELECT
  TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM video_recordings
      WHERE video_recordings.id = heart_rate_data.recording_id
      AND video_recordings.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can insert own heart rate data"
  ON heart_rate_data FOR INSERT
  TO authenticated
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM video_recordings
      WHERE video_recordings.id = heart_rate_data.recording_id
      AND video_recordings.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can view own risk predictions"
  ON risk_predictions FOR SELECT
  TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM video_recordings
      WHERE video_recordings.id = risk_predictions.recording_id
      AND video_recordings.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can insert own risk predictions"
  ON risk_predictions FOR INSERT
  TO authenticated
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM video_recordings
      WHERE video_recordings.id = risk_predictions.recording_id
      AND video_recordings.user_id = auth.uid()
    )
  );

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_updated_at
  BEFORE UPDATE ON users
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_video_recordings_updated_at
  BEFORE UPDATE ON video_recordings
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();