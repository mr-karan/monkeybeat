#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

cd "$(dirname "$0")"

TODAY_DATE="$(date --iso-8601)"

# From and To date to backfill the data for seeding DB.
FROM_DATE="${MONKEY_BEAT_FROM_DATE:-2019-01-01}"
TO_DATE="${MONKEY_BEAT_TO_DATE:-${TODAY_DATE}}"

# Directory to temporary store the bhav copy files.
OUTPUT_DIR="data"

main() {
    # Create the output directory to store the files.
    # rm -rf ${OUTPUT_DIR}
    mkdir -p ${OUTPUT_DIR}

    echo "📥 Downloading data for period ${FROM_DATE} - ${TO_DATE}"
    python seed.py --from "${FROM_DATE}" --to "${TO_DATE}" --output "${OUTPUT_DIR}"

    # Load the relevant data in DB
    cat ${OUTPUT_DIR}/"ticker.csv" | clickhouse-client --query="INSERT INTO stocks.prices SELECT toDate(Date) AS date,ticker as tradingsymbol,Close AS close FROM input('Date Date,Open Float64,High Float64,Low Float64,Close Float64,Volume Float64,ticker String') FORMAT CSVWithNames"

    # echo "✅ Inserted data for ${fetch_date}"
    # rm ${OUTPUT_DIR}/"${FILENAME}"
}

main "$@"
