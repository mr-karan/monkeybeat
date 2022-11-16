#!/usr/bin/env bash

# NOTE: This is UNUSED.
# Since bhavcopy doesn't adjust the close price to accomodate corporate actions.
# this data source is not used.
# I've just kept the script for reference/future use if any.
# Check `load_data.sh` which is the current version in use.

set -o errexit
set -o nounset
set -o pipefail

cd "$(dirname "$0")"

# Number of days to backfill the data for seeding DB.
DAYS_AGO=400
# Directory to temporary store the bhav copy files.
OUTPUT_DIR="data"

main() {
    # Create the output directory to store the files.
    rm -rf ${OUTPUT_DIR}
    mkdir -p ${OUTPUT_DIR}

    for ((i=DAYS_AGO; i>=1; i--))
    do
        # Format the dates.
        fetch_date=$(date --date "${i} days ago" +"%d%b%Y")
        fetch_year=$(date --date "${i} days ago" +"%Y")
        fetch_month=$(date --date "${i} days ago" +"%b")
        
        # Format the URL.
        FILENAME=cm${fetch_date^^}bhav.csv
        BHAV_FILE_URL="https://www1.nseindia.com/content/historical/EQUITIES/${fetch_year}/${fetch_month^^}/${FILENAME}.zip"
        OUTPUT_FILE=${OUTPUT_DIR}/${fetch_date}.zip
        
        echo "📥 Downloading ${BHAV_FILE_URL}"
        CODE=$(curl -sSL -w '%{http_code}' -o data/"${fetch_date}".zip "${BHAV_FILE_URL}")
        # cURL gives an exit code 22 if the HTTP status code is 404 (https://daniel.haxx.se/blog/2021/02/11/curl-fail-with-body/)
        if [[ "$CODE" = 404 ]]; then
            echo "👻 Skipping for date $fetch_date"
            rm -rf data/"${fetch_date}".zip
            continue
        fi

        # Extract the CSV
        unzip -d ${OUTPUT_DIR} "${OUTPUT_FILE}" && rm "${OUTPUT_FILE}"

        # Load the relevant data in DB
        cat ${OUTPUT_DIR}/"${FILENAME}" | clickhouse-client --query="INSERT INTO monkeybeat.stocks SELECT toDate(parseDateTimeBestEffort(TIMESTAMP)) AS date,SYMBOL as tradingsymbol,CLOSE AS close FROM input('SYMBOL String,SERIES String,OPEN Float64,HIGH Float64,LOW Float64,CLOSE Float64,LAST Float64,PREVCLOSE Float64,TOTTRDQTY Float64,TOTTRDVAL Float64,TIMESTAMP String,TOTALTRADES Float64,ISIN String') WHERE SERIES='EQ' FORMAT CSVWithNames"
        
        echo "✅ Inserted data for ${fetch_date}"
        rm ${OUTPUT_DIR}/"${FILENAME}"

        # Sleep for a bit to be gentle on the poor servers.
        sleep 1

    done
}

main "$@"
