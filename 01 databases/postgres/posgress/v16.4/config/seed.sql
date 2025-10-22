-- Sample seeding script for initial data
-- This runs after extensions are created

-- Create a sample table
CREATE TABLE IF NOT EXISTS sample_data (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data
INSERT INTO sample_data (name) VALUES ('Sample Entry 1'), ('Sample Entry 2'), ('Sample Entry 3');