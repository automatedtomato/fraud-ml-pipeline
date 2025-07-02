-- STEP 1: Add timestamp and age to base table
WITH base_with_age AS (
    SELECT
        *,
        (trans_date || ' ' || trans_time)::timestamp AS trans_ts,
        -- Age of customer as of transaction
        EXTRACT(YEAR FROM AGE((trans_date || ' ' || trans_time)::timestamp, dob::date)) AS age_at_tx
        EXTRACT(HOUR FROM (trans_date || ' ' || trans_time)::timestamp) AS hour_of_day,
        EXTRACT(DOW FROM (trans_date || ' ' || trans_time)::timestamp) AS day_of_week
    FROM
        transactions
),

-- STEP 2: Add distance to merchant and customer location
base_with_distance AS (
    SELECT
        *,
        -- Spherical distance btw customer and merchant (km)
        (
            6371 * ACOS(
                COS(RADIANS(lat)) * COS(RADIANS(merch_lat)) * 
                COS(RADIANS(merch_long) - RADIANS(long)) +
                SIN(RADIANS(lat)) * SIN(RADIANS(merch_lat))
            )
        ) AS distance_from_home_km
    FROM
        base_with_age
),

-- STEP 3: Recency, Frequency and Monetary features
all_features AS (
    SELECT
        *,
        -- Recency
        EXTRACT(EPOCH FROM (trans_ts - LAG(trans_ts, 1) OVER w_customer)) AS time_since_last_tx_sec,

        -- Frequency in last 24 hours / 7 days
        COUNT(*) OVER (w_customer RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW) AS tx_in_last_24h,
        COUNT(*) OVER (w_customer RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW) AS tx_in_last_7d,

        -- Monetary: Average amount in last 24 hours and 7 days (velocity)
        AVG(amt) OVER (w_customer RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW) AS avg_amt_in_last_24h,
        AVG(amt) OVER (w_customer RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW) AS avg_amt_in_last_7d,
        -- Ratio to avg amount in last 24 hours and 7 days
        amt / NULLIF(AVG(amt) OVER (w_customer RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW), 0) AS ratio_to_avg_amt_in_last_7d,
        amt / NULLIF(AVG(amt) OVER (w_customer RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW), 0) AS ratio_to_avg_amt_in_last_24h,

        -- Customer Behavior features
        -- Average amount and ratio to overall average amount
        AVG(amt) OVER(w_customer ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS avg_amt_historical,
        amt / NULLIF(AVG(amt) OVER(w_customer ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS amt_ratio_to_avg,
        -- Category transaction count
        COUNT(*) OVER (w_cus_cat ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS category_tx_count,
        -- Merchant velocity
        COUNT(*) OVER(w_cus_merch RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW) AS tx_with_same_merch_in_last_1h
    
    FROM
        base_with_distance

    WINDOW
        w_customer AS (PARTITION BY cc_num ORDER BY trans_ts),
        w_cus_cat AS (PARTITION BY cc_num, category ORDER BY trans_ts),
        w_cus_merch AS (PARTITION BY cc_num, merchant ORDER BY trans_ts)        
)

-- STEP 4: Concatenate all features into a single table
SELECT * FROM all_features;