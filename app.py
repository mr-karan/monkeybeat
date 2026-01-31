#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["fastapi>=0.115", "uvicorn>=0.34", "jinja2>=3.1", "duckdb>=1.2", "yfinance>=1.1", "pandas>=2.2"]
# ///
"""
MonkeyBeat - Can a monkey beat your fund manager?

A single-file FastAPI application that generates random stock portfolios
and compares their performance against market indices.
"""

import json
import os
import random
import urllib.parse
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# =============================================================================
# Constants
# =============================================================================

STOCKS_COUNT = 10
NORMALIZATION_FACTOR = 100
PORTFOLIO_AMOUNT = 10000
RETURN_PERIODS = [30, 180, 365, 1095, 1825]
DOMAIN = "http://localhost:7777"

INDICES = {
    "NIFTY50": {
        "name": "NIFTY 50",
        "symbol": "^NSEI",
        "suffix": ".NS",
        "csv": "scripts/ind_nifty50list.csv",
    },
    "NIFTYNEXT50": {
        "name": "NIFTY Next 50",
        "symbol": "^NSMIDCP",
        "suffix": ".NS",
        "csv": "scripts/ind_niftynext50list.csv",
    },
    "NIFTY100": {
        "name": "NIFTY 100",
        "symbol": "^CNX100",
        "suffix": ".NS",
        "csv": "scripts/ind_nifty100list.csv",
    },
    "NIFTY200": {
        "name": "NIFTY 200",
        "symbol": "^CNX200",
        "suffix": ".NS",
        "csv": "scripts/ind_nifty200list.csv",
    },
    "NIFTY500": {
        "name": "NIFTY 500",
        "symbol": "^CRSLDX",
        "suffix": ".NS",
        "csv": "scripts/ind_nifty500list.csv",
    },
    "NIFTYMIDCAP100": {
        "name": "NIFTY Midcap 100",
        "symbol": "NIFTYMIDCAP150.NS",  # Use Midcap 150 as proxy
        "suffix": ".NS",
        "csv": "scripts/ind_niftymidcap100list.csv",
    },
    "NIFTYMIDCAP150": {
        "name": "NIFTY Midcap 150",
        "symbol": "NIFTYMIDCAP150.NS",
        "suffix": ".NS",
        "csv": "scripts/ind_niftymidcap150list.csv",
    },
    "NIFTYSMALLCAP100": {
        "name": "NIFTY Smallcap 100",
        "symbol": "NIFTYSMLCAP250.NS",  # Use Smallcap 250 as proxy
        "suffix": ".NS",
        "csv": "scripts/ind_niftysmallcap100list.csv",
    },
    "NIFTYSMALLCAP250": {
        "name": "NIFTY Smallcap 250",
        "symbol": "NIFTYSMLCAP250.NS",
        "suffix": ".NS",
        "csv": "scripts/ind_niftysmallcap250list.csv",
    },
}

VALID_INDICES = list(INDICES.keys())

DB_PATH = Path("data/stocks.duckdb")
TEMPLATES_DIR = Path("templates")

# Word lists for human-readable IDs (adjective-adjective-noun pattern)
# ~100 words each = 100^3 = 1,000,000 possible combinations
ADJECTIVES = [
    "able",
    "bold",
    "calm",
    "dark",
    "easy",
    "fair",
    "glad",
    "good",
    "half",
    "keen",
    "kind",
    "late",
    "lean",
    "left",
    "live",
    "long",
    "loud",
    "main",
    "mild",
    "near",
    "neat",
    "next",
    "nice",
    "null",
    "okay",
    "only",
    "open",
    "pale",
    "past",
    "pink",
    "pure",
    "rare",
    "real",
    "rich",
    "ripe",
    "safe",
    "same",
    "slow",
    "soft",
    "sure",
    "tall",
    "thin",
    "tidy",
    "tiny",
    "true",
    "vast",
    "warm",
    "weak",
    "wide",
    "wild",
    "wise",
    "zero",
    "blue",
    "cool",
    "deep",
    "dual",
    "dull",
    "each",
    "epic",
    "even",
    "fast",
    "fine",
    "firm",
    "flat",
    "free",
    "full",
    "gold",
    "gray",
    "hard",
    "high",
    "holy",
    "huge",
    "iron",
    "just",
    "last",
    "lazy",
    "less",
    "like",
    "lost",
    "mega",
    "more",
    "most",
    "much",
    "bare",
    "best",
    "both",
    "busy",
    "cozy",
    "cute",
    "dear",
    "done",
    "down",
    "east",
    "evil",
    "fake",
    "away",
    "back",
    "base",
    "home",
    "west",
]

NOUNS = [
    "ape",
    "ant",
    "bat",
    "bee",
    "bug",
    "cat",
    "cow",
    "dog",
    "elk",
    "emu",
    "fly",
    "fox",
    "gnu",
    "hen",
    "jay",
    "koi",
    "owl",
    "pig",
    "ram",
    "rat",
    "yak",
    "ace",
    "arc",
    "art",
    "axe",
    "bar",
    "bay",
    "bed",
    "bow",
    "box",
    "bud",
    "cap",
    "car",
    "cup",
    "day",
    "den",
    "dot",
    "dew",
    "fan",
    "fig",
    "fin",
    "fog",
    "gem",
    "hat",
    "hut",
    "ice",
    "ink",
    "ivy",
    "jam",
    "jar",
    "jet",
    "jug",
    "key",
    "kit",
    "lab",
    "lap",
    "log",
    "map",
    "mat",
    "mix",
    "mop",
    "mud",
    "net",
    "nod",
    "nut",
    "oak",
    "orb",
    "ore",
    "pad",
    "pan",
    "pea",
    "pen",
    "pie",
    "pin",
    "pit",
    "pod",
    "pot",
    "pup",
    "ray",
    "rib",
    "rod",
    "rug",
    "rye",
    "sap",
    "saw",
    "sky",
    "sun",
    "tap",
    "tea",
    "tin",
    "toe",
    "top",
    "toy",
    "tub",
    "urn",
    "van",
    "vat",
    "wax",
    "web",
    "wig",
]


def generate_human_id() -> str:
    """Generate a human-readable ID like 'bold-calm-fox'."""
    adj1 = random.choice(ADJECTIVES)
    adj2 = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    return f"{adj1}-{adj2}-{noun}"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DailyReturn:
    date: str
    return_percent: float
    current_invested: float


@dataclass
class SimulationResults:
    num_simulations: int
    win_count: int
    win_rate: float
    user_rank: int
    user_percentile: float
    min_return: float
    max_return: float
    median_return: float
    index_return: float
    all_returns: list[float]  # For distribution chart


@dataclass
class PortfolioData:
    share_id: str
    category: str
    stocks: list[str]
    avg_portfolio_returns: dict[int, float]
    avg_index_returns: dict[int, float]
    avg_stock_returns: dict[str, dict[int, float]]
    daily_portfolio_returns: list[DailyReturn]
    daily_index_returns: list[DailyReturn]
    current_portfolio_amount: float
    current_index_amount: float
    simulation: SimulationResults | None = None


# =============================================================================
# Database Functions
# =============================================================================


def get_db() -> duckdb.DuckDBPyConnection:
    """Get a database connection."""
    return duckdb.connect(str(DB_PATH))


def init_db() -> None:
    """Initialize the database schema."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                date DATE,
                tradingsymbol VARCHAR,
                category VARCHAR,
                segment VARCHAR,
                close DOUBLE
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_prices 
            ON prices(segment, category, tradingsymbol, date)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS link (
                timestamp TIMESTAMP,
                uuid VARCHAR PRIMARY KEY,
                portfolio JSON
            )
        """)


def is_data_stale(category: str) -> bool:
    """Check if data needs to be refreshed (last date < yesterday)."""
    yesterday = date.today() - timedelta(days=1)

    with get_db() as conn:
        result = conn.execute(
            """
            SELECT MAX(date) FROM prices WHERE category = ?
        """,
            [category],
        ).fetchone()

        if result is None or result[0] is None:
            return True

        last_date = result[0]
        if isinstance(last_date, str):
            last_date = datetime.strptime(last_date, "%Y-%m-%d").date()

        return last_date < yesterday


def load_stock_symbols(csv_path: str) -> list[str]:
    """Load stock symbols from CSV file."""
    df = pd.read_csv(csv_path)
    return df["Symbol"].tolist()


def fetch_and_store_data(category: str) -> None:
    """Fetch stock data from yfinance and store in DuckDB."""
    config = INDICES[category]
    symbols = load_stock_symbols(config["csv"])
    suffix = config["suffix"]
    index_symbol = config["symbol"]

    # Add suffix to stock symbols
    tickers = [f"{sym}{suffix}" for sym in symbols]
    # Add index symbol
    tickers.append(index_symbol)

    # Calculate date range (6 years back)
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * 6)

    print(
        f"Fetching data for {category}: {len(tickers)} tickers from {start_date} to {end_date}"
    )

    # Fetch data in batch
    data = yf.download(
        tickers,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        progress=True,
        group_by="ticker",
        auto_adjust=True,
    )

    if data.empty:
        print(f"No data fetched for {category}")
        return

    # Prepare records for insertion
    records = []

    with get_db() as conn:
        # Clear existing data for this category
        conn.execute("DELETE FROM prices WHERE category = ?", [category])

        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    ticker_data = data
                else:
                    ticker_data = data[ticker]

                if ticker_data.empty:
                    continue

                # Determine segment and trading symbol
                if ticker == index_symbol:
                    segment = "INDEX"
                    tradingsymbol = index_symbol
                else:
                    segment = "EQ"
                    # Remove suffix to get clean symbol
                    tradingsymbol = ticker.replace(suffix, "")

                for idx, row in ticker_data.iterrows():
                    if pd.notna(row.get("Close")):
                        records.append(
                            (
                                idx.date() if hasattr(idx, "date") else idx,
                                tradingsymbol,
                                category,
                                segment,
                                float(row["Close"]),
                            )
                        )
            except (KeyError, TypeError) as e:
                print(f"Error processing {ticker}: {e}")
                continue

        if records:
            conn.executemany(
                """
                INSERT INTO prices (date, tradingsymbol, category, segment, close)
                VALUES (?, ?, ?, ?, ?)
            """,
                records,
            )
            print(f"Inserted {len(records)} records for {category}")


def ensure_data_fresh(category: str) -> None:
    """Ensure data is fresh, fetching if stale."""
    if is_data_stale(category):
        print(f"Data for {category} is stale, fetching fresh data...")
        fetch_and_store_data(category)


# =============================================================================
# Portfolio Functions
# =============================================================================


def get_random_stocks(category: str) -> list[str]:
    """Get N random stocks from the specified category."""
    with get_db() as conn:
        result = conn.execute(
            """
            SELECT tradingsymbol 
            FROM prices 
            WHERE segment = 'EQ' AND category = ?
            GROUP BY tradingsymbol 
            ORDER BY random() 
            LIMIT ?
        """,
            [category, STOCKS_COUNT],
        ).fetchall()

        return [row[0] for row in result]


def get_returns(
    symbols: list[str], category: str, segment: str, days: int
) -> dict[str, float]:
    """Calculate returns for given symbols over N days - BATCH query."""
    start_date = date.today() - timedelta(days=days)

    with get_db() as conn:
        # Single batch query for all symbols
        result = conn.execute(
            """
            WITH start_prices AS (
                SELECT tradingsymbol, close as start_close
                FROM (
                    SELECT tradingsymbol, close,
                           ROW_NUMBER() OVER (PARTITION BY tradingsymbol ORDER BY date) as rn
                    FROM prices
                    WHERE category = ? AND segment = ? AND date >= ?
                    AND tradingsymbol IN (SELECT UNNEST(?::VARCHAR[]))
                )
                WHERE rn = 1
            ),
            end_prices AS (
                SELECT tradingsymbol, close as end_close
                FROM (
                    SELECT tradingsymbol, close,
                           ROW_NUMBER() OVER (PARTITION BY tradingsymbol ORDER BY date DESC) as rn
                    FROM prices
                    WHERE category = ? AND segment = ?
                    AND tradingsymbol IN (SELECT UNNEST(?::VARCHAR[]))
                )
                WHERE rn = 1
            )
            SELECT s.tradingsymbol, 
                   ((e.end_close / s.start_close) - 1) * 100 as return_pct
            FROM start_prices s
            JOIN end_prices e ON s.tradingsymbol = e.tradingsymbol
            """,
            [category, segment, start_date, symbols, category, segment, symbols],
        ).fetchall()

        returns = {sym: 0.0 for sym in symbols}
        for row in result:
            returns[row[0]] = row[1] if row[1] is not None else 0.0

    return returns


def get_daily_values(
    symbols: list[str], category: str, segment: str
) -> list[DailyReturn]:
    """Calculate daily normalized portfolio values - BATCH query."""
    with get_db() as conn:
        # Single query: get all data, compute in SQL
        result = conn.execute(
            """
            WITH initial_prices AS (
                SELECT tradingsymbol, close as initial_close
                FROM (
                    SELECT tradingsymbol, close,
                           ROW_NUMBER() OVER (PARTITION BY tradingsymbol ORDER BY date) as rn
                    FROM prices
                    WHERE category = ? AND segment = ?
                    AND tradingsymbol IN (SELECT UNNEST(?::VARCHAR[]))
                )
                WHERE rn = 1
            ),
            daily_normalized AS (
                SELECT p.date,
                       SUM((? / i.initial_close) * p.close) as normalized_close,
                       COUNT(*) as stock_count
                FROM prices p
                JOIN initial_prices i ON p.tradingsymbol = i.tradingsymbol
                WHERE p.category = ? AND p.segment = ?
                AND p.tradingsymbol IN (SELECT UNNEST(?::VARCHAR[]))
                GROUP BY p.date
                ORDER BY p.date
            )
            SELECT date,
                   ((normalized_close / (? * stock_count)) - 1) * 100 as return_percent,
                   ? + (((normalized_close / (? * stock_count)) - 1) * ?) as current_invested
            FROM daily_normalized
            """,
            [
                category,
                segment,
                symbols,  # initial_prices CTE
                NORMALIZATION_FACTOR,  # multiplier calc
                category,
                segment,
                symbols,  # daily_normalized CTE
                NORMALIZATION_FACTOR,  # baseline calc
                PORTFOLIO_AMOUNT,
                NORMALIZATION_FACTOR,
                PORTFOLIO_AMOUNT,  # final select
            ],
        ).fetchall()

        return [
            DailyReturn(
                date=row[0].isoformat()
                if hasattr(row[0], "isoformat")
                else str(row[0]),
                return_percent=round(row[1], 2) if row[1] else 0.0,
                current_invested=round(row[2], 2) if row[2] else PORTFOLIO_AMOUNT,
            )
            for row in result
        ]


def get_index_symbol(category: str) -> str:
    return INDICES[category]["symbol"]


def get_all_stock_returns(category: str, days: int = 1825) -> dict[str, float]:
    """Get 5-year returns for ALL stocks in category - single batch query."""
    start_date = date.today() - timedelta(days=days)

    with get_db() as conn:
        result = conn.execute(
            """
            WITH start_prices AS (
                SELECT tradingsymbol, close as start_close
                FROM (
                    SELECT tradingsymbol, close,
                           ROW_NUMBER() OVER (PARTITION BY tradingsymbol ORDER BY date) as rn
                    FROM prices
                    WHERE category = ? AND segment = 'EQ' AND date >= ?
                )
                WHERE rn = 1
            ),
            end_prices AS (
                SELECT tradingsymbol, close as end_close
                FROM (
                    SELECT tradingsymbol, close,
                           ROW_NUMBER() OVER (PARTITION BY tradingsymbol ORDER BY date DESC) as rn
                    FROM prices
                    WHERE category = ? AND segment = 'EQ'
                )
                WHERE rn = 1
            )
            SELECT s.tradingsymbol, 
                   ((e.end_close / s.start_close) - 1) * 100 as return_pct
            FROM start_prices s
            JOIN end_prices e ON s.tradingsymbol = e.tradingsymbol
            WHERE s.start_close > 0
            """,
            [category, start_date, category],
        ).fetchall()

        return {row[0]: row[1] for row in result if row[1] is not None}


def run_monte_carlo(
    category: str, user_return: float, index_return: float, num_simulations: int = 100
) -> SimulationResults:
    """Run Monte Carlo simulation - pick random portfolios and compare."""
    import random

    all_returns = get_all_stock_returns(category)
    stocks = list(all_returns.keys())

    if len(stocks) < STOCKS_COUNT:
        return SimulationResults(
            num_simulations=0,
            win_count=0,
            win_rate=0,
            user_rank=1,
            user_percentile=100,
            min_return=user_return,
            max_return=user_return,
            median_return=user_return,
            index_return=index_return,
            all_returns=[user_return],
        )

    sim_returns = []
    win_count = 0

    for _ in range(num_simulations):
        picked = random.sample(stocks, STOCKS_COUNT)
        portfolio_return = sum(all_returns[s] for s in picked) / STOCKS_COUNT
        sim_returns.append(portfolio_return)
        if portfolio_return > index_return:
            win_count += 1

    sim_returns.sort()
    user_rank = sum(1 for r in sim_returns if r > user_return) + 1
    median_return = sim_returns[len(sim_returns) // 2]

    return SimulationResults(
        num_simulations=num_simulations,
        win_count=win_count,
        win_rate=round(win_count / num_simulations * 100, 1),
        user_rank=user_rank,
        user_percentile=round((1 - user_rank / num_simulations) * 100, 1),
        min_return=round(min(sim_returns), 2),
        max_return=round(max(sim_returns), 2),
        median_return=round(median_return, 2),
        index_return=round(index_return, 2),
        all_returns=[round(r, 1) for r in sim_returns],
    )


def calculate_portfolio(
    category: str, stocks: list[str] | None = None
) -> PortfolioData:
    """Calculate complete portfolio data."""
    if stocks is None:
        stocks = get_random_stocks(category)

    index_symbol = get_index_symbol(category)

    # Calculate returns for all periods
    avg_portfolio_returns = {}
    avg_index_returns = {}
    avg_stock_returns: dict[str, dict[int, float]] = {stock: {} for stock in stocks}

    for period in RETURN_PERIODS:
        # Portfolio returns (individual stocks)
        stock_returns = get_returns(stocks, category, "EQ", period)
        avg_portfolio_returns[period] = (
            sum(stock_returns.values()) / len(stock_returns) if stock_returns else 0
        )

        # Store individual stock returns
        for stock, ret in stock_returns.items():
            avg_stock_returns[stock][period] = ret

        # Index returns
        index_returns = get_returns([index_symbol], category, "INDEX", period)
        avg_index_returns[period] = index_returns.get(index_symbol, 0)

    # Calculate daily values for charting
    daily_portfolio_returns = get_daily_values(stocks, category, "EQ")
    daily_index_returns = get_daily_values([index_symbol], category, "INDEX")

    five_year_portfolio_return = avg_portfolio_returns.get(1825, 0)
    five_year_index_return = avg_index_returns.get(1825, 0)

    current_portfolio_amount = round(
        PORTFOLIO_AMOUNT * (1 + five_year_portfolio_return / 100), 2
    )
    current_index_amount = round(
        PORTFOLIO_AMOUNT * (1 + five_year_index_return / 100), 2
    )

    simulation = run_monte_carlo(
        category, five_year_portfolio_return, five_year_index_return
    )

    share_id = generate_human_id()

    return PortfolioData(
        share_id=share_id,
        category=category,
        stocks=stocks,
        avg_portfolio_returns=avg_portfolio_returns,
        avg_index_returns=avg_index_returns,
        avg_stock_returns=avg_stock_returns,
        daily_portfolio_returns=daily_portfolio_returns,
        daily_index_returns=daily_index_returns,
        current_portfolio_amount=current_portfolio_amount,
        current_index_amount=current_index_amount,
        simulation=simulation,
    )


# =============================================================================
# Share / Link Functions
# =============================================================================


def save_portfolio(portfolio: PortfolioData) -> str:
    """Save portfolio to database and return UUID."""
    portfolio_json = json.dumps(
        {
            "category": portfolio.category,
            "stocks": portfolio.stocks,
        }
    )

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO link (timestamp, uuid, portfolio)
            VALUES (?, ?, ?)
        """,
            [datetime.now(), portfolio.share_id, portfolio_json],
        )

    return portfolio.share_id


def get_portfolio_by_id(share_id: str) -> PortfolioData | None:
    """Retrieve portfolio by UUID."""
    with get_db() as conn:
        result = conn.execute(
            """
            SELECT portfolio FROM link WHERE uuid = ?
        """,
            [share_id],
        ).fetchone()

        if not result:
            return None

        data = json.loads(result[0])
        category = data["category"]
        stocks = data["stocks"]

        # Recalculate portfolio with stored stocks
        return calculate_portfolio(category, stocks)


def cleanup_old_links() -> None:
    """Delete links older than 7 days."""
    cutoff = datetime.now() - timedelta(days=7)

    with get_db() as conn:
        conn.execute(
            """
            DELETE FROM link WHERE timestamp < ?
        """,
            [cutoff],
        )


# =============================================================================
# Template Filters
# =============================================================================


def format_number(value: float) -> str:
    """Format number as percentage with +/- prefix."""
    if value >= 0:
        return f"+{value:.2f}%"
    return f"{value:.2f}%"


def colorize(value: float) -> str:
    """Wrap value in colored span based on +/-."""
    formatted = format_number(value)
    if value >= 0:
        return f'<span class="text-neon-green">{formatted}</span>'
    return f'<span class="text-neon-red">{formatted}</span>'


def x_share(share_id: str) -> str:
    """Generate Twitter/X share URL."""
    url = f"{DOMAIN}/portfolio/{share_id}"
    text = "Check out my MonkeyBeat portfolio! Can a monkey beat your fund manager?"
    return f"https://twitter.com/intent/tweet?url={urllib.parse.quote(url)}&text={urllib.parse.quote(text)}"


def whatsapp_share(share_id: str) -> str:
    """Generate WhatsApp share URL."""
    url = f"{DOMAIN}/portfolio/{share_id}"
    text = f"Check out my MonkeyBeat portfolio! {url}"
    return f"https://wa.me/?text={urllib.parse.quote(text)}"


def stock_link(ticker: str, category: str) -> str:
    """Generate Zerodha market URL for stock."""
    # Remove .NS suffix if present for the URL
    symbol = ticker.replace(".NS", "")
    return f"https://zerodha.com/markets/stocks/NSE/{symbol}/"


# =============================================================================
# FastAPI Application
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("Initializing database...")
    init_db()
    print("Cleaning up old links...")
    cleanup_old_links()

    # Pre-fetch data for all indices
    for index_name in VALID_INDICES:
        print(f"Checking data freshness for {index_name}...")
        ensure_data_fresh(index_name)

    print("MonkeyBeat ready!")

    yield

    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title="MonkeyBeat",
    description="Can a monkey beat your fund manager?",
    lifespan=lifespan,
)

# Mount static files
static_path = Path("static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Setup templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Register custom filters
templates.env.filters["format_number"] = format_number
templates.env.filters["colorize"] = colorize
templates.env.filters["x_share"] = x_share
templates.env.filters["whatsapp_share"] = whatsapp_share
templates.env.filters["yfinance_link"] = lambda t, c: stock_link(t, c)


def portfolio_to_context(portfolio: PortfolioData) -> dict[str, Any]:
    """Convert PortfolioData to template context dict."""
    # Convert category key to display name
    display_name = INDICES.get(portfolio.category, {}).get("name", portfolio.category)

    ctx = {
        "share_id": portfolio.share_id,
        "category": display_name,
        "stocks": portfolio.stocks,
        "avg_portfolio_returns": portfolio.avg_portfolio_returns,
        "avg_index_returns": portfolio.avg_index_returns,
        "avg_stock_returns": portfolio.avg_stock_returns,
        "daily_portfolio_returns": portfolio.daily_portfolio_returns,
        "daily_index_returns": portfolio.daily_index_returns,
        "current_portfolio_amount": portfolio.current_portfolio_amount,
        "current_index_amount": portfolio.current_index_amount,
    }
    if portfolio.simulation:
        ctx["simulation"] = {
            "num_simulations": portfolio.simulation.num_simulations,
            "win_count": portfolio.simulation.win_count,
            "win_rate": portfolio.simulation.win_rate,
            "user_rank": portfolio.simulation.user_rank,
            "user_percentile": portfolio.simulation.user_percentile,
            "min_return": portfolio.simulation.min_return,
            "max_return": portfolio.simulation.max_return,
            "median_return": portfolio.simulation.median_return,
            "index_return": portfolio.simulation.index_return,
            "all_returns": portfolio.simulation.all_returns,
        }
    return ctx


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render homepage."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/portfolio", response_class=HTMLResponse)
async def generate_portfolio(
    request: Request,
    index: str = Query(..., description="Index to use"),
):
    """Generate a random portfolio for the specified index (HTMX partial)."""
    if index not in VALID_INDICES:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "error": f"Invalid index: {index}. Valid options: {VALID_INDICES}",
            },
            status_code=400,
        )

    try:
        # Ensure data is fresh
        ensure_data_fresh(index)

        # Calculate portfolio
        portfolio = calculate_portfolio(index)

        # Save to database
        save_portfolio(portfolio)

        # Return partial HTML (for HTMX)
        return templates.TemplateResponse(
            "portfolio.html",
            {"request": request, "partial": True, **portfolio_to_context(portfolio)},
        )
    except Exception as e:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": str(e)},
            status_code=500,
        )


@app.get("/portfolio/{share_id}", response_class=HTMLResponse)
async def get_shared_portfolio(request: Request, share_id: str):
    """Retrieve a saved portfolio by UUID (full page)."""
    portfolio = get_portfolio_by_id(share_id)

    if portfolio is None:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": "Portfolio not found or has expired."},
            status_code=404,
        )

    # Return full page (not HTMX partial)
    return templates.TemplateResponse(
        "portfolio.html",
        {"request": request, "partial": False, **portfolio_to_context(portfolio)},
    )


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7777)
