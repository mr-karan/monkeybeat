CREATE DATABASE IF NOT EXISTS `monkeybeat`;

CREATE TABLE IF NOT EXISTS `monkeybeat`.`prices` (
    `date` Date,
    `tradingsymbol` String,
    `segment` LowCardinality(String),
    `close` Float64
) ENGINE = ReplacingMergeTree()
ORDER BY (tradingsymbol, toYYYYMMDD(date));
