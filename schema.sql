CREATE DATABASE IF NOT EXISTS `monkeybeat`;

CREATE TABLE IF NOT EXISTS `monkeybeat`.`prices` (
    `date` Date,
    `tradingsymbol` String,
    `category` LowCardinality(String),
    `segment` LowCardinality(String),
    `close` Float64
) ENGINE = ReplacingMergeTree()
ORDER BY (segment,category,tradingsymbol,toYYYYMMDD(date));

CREATE TABLE IF NOT EXISTS `monkeybeat`.`link` (
    `timestamp` Datetime('Asia/Kolkata'),
    uuid UUID,
    `portfolio` String
) ENGINE = MergeTree()
ORDER BY (toYYYYMMDD(timestamp))
TTL timestamp + INTERVAL 1 WEEK;
