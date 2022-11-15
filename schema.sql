CREATE DATABASE IF NOT EXISTS `stocks`;

CREATE TABLE IF NOT EXISTS `stocks`.`prices` (
    `date` Datetime('Asia/Kolkata'),
    `tradingsymbol` String,
    `close` Float64
) ENGINE = ReplacingMergeTree()
ORDER BY (tradingsymbol, toYYYYMMDD(date))

