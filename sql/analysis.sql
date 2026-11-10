-- Reporting / analytics queries (SQLite dialect). Run by `python -m src.sql_analysis`.
-- Each block starts with `-- name:`; results are saved to reports/sql_results/<name>.csv.
-- Requires the `customers` and `segments` tables created by the earlier pipeline steps.

-- name: overview
-- desc: Headline KPIs for the whole customer base.
SELECT COUNT(*)                          AS customers,
       ROUND(AVG(income), 0)             AS avg_income,
       ROUND(AVG(age), 1)                AS avg_age,
       ROUND(AVG(total_spend), 1)        AS avg_spend,
       ROUND(AVG(total_purchases), 1)    AS avg_purchases,
       ROUND(100.0 * AVG(response), 1)   AS response_rate_pct
FROM customers;

-- name: segment_profile
-- desc: Size, value and campaign response of each customer segment (JOIN + GROUP BY + scalar sub-queries).
SELECT s.segment_id,
       s.segment_name,
       COUNT(*)                                                                      AS customers,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM customers), 1)                 AS pct_customers,
       ROUND(AVG(c.income), 0)                                                       AS avg_income,
       ROUND(AVG(c.age), 1)                                                          AS avg_age,
       ROUND(AVG(c.children), 2)                                                     AS avg_children,
       ROUND(AVG(c.total_spend), 1)                                                  AS avg_spend,
       ROUND(100.0 * SUM(c.total_spend) / (SELECT SUM(total_spend) FROM customers), 1) AS pct_of_total_spend,
       ROUND(100.0 * AVG(c.response), 1)                                             AS response_rate_pct
FROM customers AS c
JOIN segments  AS s ON s.customer_id = c.customer_id
GROUP BY s.segment_id, s.segment_name
ORDER BY avg_spend DESC;

-- name: channel_mix_by_segment
-- desc: Average purchases per sales channel and web visits per segment.
SELECT s.segment_name,
       ROUND(AVG(c.web_purchases), 2)      AS avg_web,
       ROUND(AVG(c.catalog_purchases), 2)  AS avg_catalog,
       ROUND(AVG(c.store_purchases), 2)    AS avg_store,
       ROUND(AVG(c.deal_purchases), 2)     AS avg_deal,
       ROUND(AVG(c.web_visits_month), 2)   AS avg_web_visits_month
FROM customers AS c
JOIN segments  AS s ON s.customer_id = c.customer_id
GROUP BY s.segment_name
ORDER BY s.segment_name;

-- name: category_spend_by_segment
-- desc: Average spend per product category for each segment.
SELECT s.segment_name,
       ROUND(AVG(c.spend_wines), 1)   AS wines,
       ROUND(AVG(c.spend_meat), 1)    AS meat,
       ROUND(AVG(c.spend_fish), 1)    AS fish,
       ROUND(AVG(c.spend_fruits), 1)  AS fruits,
       ROUND(AVG(c.spend_sweets), 1)  AS sweets,
       ROUND(AVG(c.spend_gold), 1)    AS gold
FROM customers AS c
JOIN segments  AS s ON s.customer_id = c.customer_id
GROUP BY s.segment_name
ORDER BY s.segment_name;

-- name: income_band_analysis
-- desc: Spend and campaign response by income band (CASE expression inside a CTE).
WITH banded AS (
    SELECT CASE WHEN income < 30000 THEN '1. under 30k'
                WHEN income < 50000 THEN '2. 30k-50k'
                WHEN income < 70000 THEN '3. 50k-70k'
                WHEN income < 90000 THEN '4. 70k-90k'
                ELSE '5. 90k+' END AS income_band,
           total_spend,
           response
    FROM customers
)
SELECT income_band,
       COUNT(*)                          AS customers,
       ROUND(AVG(total_spend), 1)        AS avg_spend,
       ROUND(100.0 * AVG(response), 1)   AS response_rate_pct
FROM banded
GROUP BY income_band
ORDER BY income_band;

-- name: education_and_household
-- desc: Spend by education level and household type, ignoring tiny groups (HAVING).
SELECT education_level,
       living_with,
       COUNT(*)                     AS customers,
       ROUND(AVG(total_spend), 1)   AS avg_spend,
       ROUND(AVG(income), 0)        AS avg_income
FROM customers
GROUP BY education_level, living_with
HAVING COUNT(*) >= 10
ORDER BY avg_spend DESC;

-- name: spend_concentration_by_decile
-- desc: Pareto view - how much of total spend comes from each customer decile (NTILE + running total window).
WITH ranked AS (
    SELECT customer_id,
           total_spend,
           NTILE(10) OVER (ORDER BY total_spend DESC) AS spend_decile
    FROM customers
)
SELECT spend_decile,
       COUNT(*)                                                                         AS customers,
       ROUND(SUM(total_spend), 0)                                                       AS spend_sum,
       ROUND(100.0 * SUM(total_spend) / (SELECT SUM(total_spend) FROM customers), 1)    AS pct_of_total_spend,
       ROUND(100.0 * SUM(SUM(total_spend)) OVER (ORDER BY spend_decile)
                   / (SELECT SUM(total_spend) FROM customers), 1)                       AS cumulative_pct
FROM ranked
GROUP BY spend_decile
ORDER BY spend_decile;

-- name: campaign_response_by_prior_acceptance
-- desc: Does accepting earlier campaigns predict responding to the latest one?
SELECT campaigns_accepted,
       COUNT(*)                          AS customers,
       ROUND(100.0 * AVG(response), 1)   AS response_rate_pct
FROM customers
GROUP BY campaigns_accepted
ORDER BY campaigns_accepted;

-- name: prior_campaign_engagement
-- desc: Response rate of customers who never accepted a past campaign versus those who accepted at least one.
SELECT CASE WHEN campaigns_accepted = 0 THEN 'None' ELSE '1 or more' END AS prior_campaigns_accepted,
       COUNT(*)                                                         AS customers,
       ROUND(100.0 * AVG(response), 1)                                  AS response_rate_pct
FROM customers
GROUP BY prior_campaigns_accepted
ORDER BY prior_campaigns_accepted;

-- name: recency_vs_response
-- desc: Campaign response rate by days since the customer's last purchase.
WITH banded AS (
    SELECT CASE WHEN recency_days <= 15 THEN '1. 0-15 days'
                WHEN recency_days <= 30 THEN '2. 16-30 days'
                WHEN recency_days <= 60 THEN '3. 31-60 days'
                WHEN recency_days <= 90 THEN '4. 61-90 days'
                ELSE '5. over 90 days' END AS recency_band,
           response
    FROM customers
)
SELECT recency_band,
       COUNT(*)                          AS customers,
       ROUND(100.0 * AVG(response), 1)   AS response_rate_pct
FROM banded
GROUP BY recency_band
ORDER BY recency_band;

-- name: age_group_channel_preference
-- desc: Average purchases per channel by age group.
WITH grouped AS (
    SELECT CASE WHEN age < 30 THEN '1. under 30'
                WHEN age < 45 THEN '2. 30-44'
                WHEN age < 60 THEN '3. 45-59'
                ELSE '4. 60+' END AS age_group,
           web_purchases, catalog_purchases, store_purchases, total_spend
    FROM customers
)
SELECT age_group,
       COUNT(*)                           AS customers,
       ROUND(AVG(web_purchases), 2)       AS avg_web,
       ROUND(AVG(catalog_purchases), 2)   AS avg_catalog,
       ROUND(AVG(store_purchases), 2)     AS avg_store,
       ROUND(AVG(total_spend), 1)         AS avg_spend
FROM grouped
GROUP BY age_group
ORDER BY age_group;

-- name: high_value_at_risk_by_segment
-- desc: Top-quartile spenders who have not purchased for 60+ days - retention targets, summarised per segment.
WITH ranked AS (
    SELECT customer_id, total_spend, recency_days,
           PERCENT_RANK() OVER (ORDER BY total_spend) AS spend_pct_rank
    FROM customers
)
SELECT s.segment_name,
       COUNT(*)                      AS at_risk_customers,
       ROUND(SUM(r.total_spend), 0)  AS spend_at_risk,
       ROUND(AVG(r.recency_days), 1) AS avg_days_since_purchase
FROM ranked AS r
JOIN segments AS s ON s.customer_id = r.customer_id
WHERE r.spend_pct_rank >= 0.75
  AND r.recency_days > 60
GROUP BY s.segment_name
ORDER BY spend_at_risk DESC;

-- name: high_value_at_risk_top20
-- desc: The 20 highest-spending customers who have gone quiet for 60+ days.
WITH ranked AS (
    SELECT customer_id, total_spend, recency_days, income,
           PERCENT_RANK() OVER (ORDER BY total_spend) AS spend_pct_rank
    FROM customers
)
SELECT r.customer_id,
       s.segment_name,
       r.total_spend,
       r.recency_days,
       ROUND(r.income, 0) AS income
FROM ranked AS r
JOIN segments AS s ON s.customer_id = r.customer_id
WHERE r.spend_pct_rank >= 0.75
  AND r.recency_days > 60
ORDER BY r.total_spend DESC
LIMIT 20;
