-- GRI RAG Database Initialization
-- This script runs automatically on first PostgreSQL container start

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==========================================================================
-- Sessions Table
-- ==========================================================================
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(36) UNIQUE NOT NULL,
    memory_data JSONB NOT NULL DEFAULT '{"turns": []}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL
);

-- Indexes for session operations
CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);

-- ==========================================================================
-- Feedback Table
-- ==========================================================================
CREATE TABLE IF NOT EXISTS feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id VARCHAR(36) NOT NULL,
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    comment TEXT,
    incorrect_info TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Indexes for feedback analysis
CREATE INDEX IF NOT EXISTS idx_feedback_query_id ON feedback(query_id);
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating);
CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at);

-- ==========================================================================
-- Query Logs Table (for analytics)
-- ==========================================================================
CREATE TABLE IF NOT EXISTS query_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id VARCHAR(36) UNIQUE NOT NULL,
    query_text TEXT NOT NULL,
    intent VARCHAR(50),
    cycle VARCHAR(10),
    latency_ms FLOAT,
    iterations INTEGER,
    tool_calls JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_query_logs_created_at ON query_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_query_logs_intent ON query_logs(intent);

-- ==========================================================================
-- Trigger for updated_at
-- ==========================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to sessions table
DROP TRIGGER IF EXISTS update_sessions_updated_at ON sessions;
CREATE TRIGGER update_sessions_updated_at
    BEFORE UPDATE ON sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ==========================================================================
-- Helper Views
-- ==========================================================================
-- Feedback summary by rating
CREATE OR REPLACE VIEW feedback_summary AS
SELECT
    rating,
    COUNT(*) as count,
    DATE_TRUNC('day', created_at) as day
FROM feedback
GROUP BY rating, DATE_TRUNC('day', created_at)
ORDER BY day DESC, rating;

-- Active sessions count
CREATE OR REPLACE VIEW active_sessions AS
SELECT
    COUNT(*) as active_count,
    MAX(updated_at) as last_activity
FROM sessions
WHERE expires_at > CURRENT_TIMESTAMP;
