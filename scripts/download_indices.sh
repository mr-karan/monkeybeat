#!/usr/bin/env bash
#
# Download NSE index constituent CSVs from niftyindices.com
# Usage: ./download_indices.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="https://www.niftyindices.com/IndexConstituent"
USER_AGENT="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

# Index definitions: "display_name|filename"
INDICES=(
    "NIFTY 50|ind_nifty50list.csv"
    "NIFTY Next 50|ind_niftynext50list.csv"
    "NIFTY 100|ind_nifty100list.csv"
    "NIFTY 200|ind_nifty200list.csv"
    "NIFTY 500|ind_nifty500list.csv"
    "NIFTY Midcap 100|ind_niftymidcap100list.csv"
    "NIFTY Midcap 150|ind_niftymidcap150list.csv"
    "NIFTY Smallcap 100|ind_niftysmallcap100list.csv"
    "NIFTY Smallcap 250|ind_niftysmallcap250list.csv"
)

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

error() {
    echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2
}

download_index() {
    local name="$1"
    local filename="$2"
    local url="${BASE_URL}/${filename}"
    local output="${SCRIPT_DIR}/${filename}"
    local temp_file
    
    temp_file=$(mktemp)
    # shellcheck disable=SC2064
    trap "rm -f '$temp_file'" RETURN
    
    log "Downloading ${name}..."
    
    if ! curl -fsSL -A "${USER_AGENT}" -o "$temp_file" "$url"; then
        error "Failed to download ${name} from ${url}"
        return 1
    fi
    
    # Validate CSV has expected header
    if ! head -1 "$temp_file" | grep -q "Company Name"; then
        error "Invalid CSV format for ${name}"
        return 1
    fi
    
    # Count stocks (excluding header)
    local count
    count=$(($(wc -l < "$temp_file") - 1))
    
    if [[ $count -lt 10 ]]; then
        error "Too few stocks in ${name}: ${count}"
        return 1
    fi
    
    mv "$temp_file" "$output"
    log "  ✓ ${name}: ${count} stocks"
}

main() {
    log "Starting index download..."
    log "Output directory: ${SCRIPT_DIR}"
    echo
    
    local failed=0
    local success=0
    
    for entry in "${INDICES[@]}"; do
        local name="${entry%%|*}"
        local filename="${entry##*|}"
        
        if download_index "$name" "$filename"; then
            ((success++)) || true
        else
            ((failed++)) || true
        fi
    done
    
    echo
    log "Download complete: ${success} succeeded, ${failed} failed"
    
    if [[ $failed -gt 0 ]]; then
        exit 1
    fi
}

main "$@"
