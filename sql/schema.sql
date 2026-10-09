CREATE TABLE IF NOT EXISTS customers (customer_id BIGINT PRIMARY KEY, signup_date DATE, plan TEXT, region TEXT, tenure_days INT);
CREATE TABLE IF NOT EXISTS events (event_id BIGINT PRIMARY KEY, customer_id BIGINT REFERENCES customers(customer_id), event_ts TIMESTAMP, event_type TEXT);
CREATE TABLE IF NOT EXISTS transactions (transaction_id BIGINT PRIMARY KEY, customer_id BIGINT REFERENCES customers(customer_id), transaction_ts TIMESTAMP, amount NUMERIC(12,2));
CREATE TABLE IF NOT EXISTS experiments (experiment_id TEXT, customer_id BIGINT REFERENCES customers(customer_id), variant TEXT, converted INT, segment_hint TEXT, PRIMARY KEY(experiment_id,customer_id));
CREATE TABLE IF NOT EXISTS predictions (customer_id BIGINT, scored_at TIMESTAMP DEFAULT NOW(), churn_probability DOUBLE PRECISION, predicted_value DOUBLE PRECISION, segment INT);
