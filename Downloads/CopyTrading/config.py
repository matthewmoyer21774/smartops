"""
Configuration for Polymarket Trader Discovery & Analysis
"""

# API Base URLs
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
DATA_API_BASE = "https://data-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"

# Rate Limiting
REQUESTS_PER_MINUTE = 30  # Conservative rate limit
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # Base delay in seconds

# Discovery Settings
DEFAULT_MARKETS_TO_SCAN = 50
DEFAULT_HOLDERS_PER_MARKET = 20
DEFAULT_LEADERBOARD_LIMIT = 100

# Platform wallets to exclude (market makers, etc.)
EXCLUDED_WALLETS = {
    '0xc5d563a36ae78145c45a50134d48a1215220f80a',
    '0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e',
}

# Data directories
DATA_DIR = "data"
TRADERS_DIR = "data/traders"
EXPORTS_DIR = "data/exports"

# Market Categories for scanning
# Each category has keywords to match against market slugs/titles
MARKET_CATEGORIES = {
    'nba': ['nba', 'basketball', 'lakers', 'celtics', 'warriors', 'nets', 'knicks', 'bulls', 'heat', 'bucks'],
    'nfl': ['nfl', 'super-bowl', 'superbowl', 'patriots', 'bills', 'chiefs', 'cowboys', 'eagles', 'packers', '49ers', 'ravens', 'steelers', 'broncos', 'seahawks', 'rams', 'bears', 'texans', 'nfc', 'afc'],
    'nhl': ['nhl', 'hockey', 'stanley-cup'],
    'mlb': ['mlb', 'baseball', 'world-series'],
    'ufc': ['ufc', 'mma', 'fight-night'],
    'soccer': ['soccer', 'premier-league', 'champions-league', 'world-cup', 'epl', 'manchester', 'chelsea', 'arsenal', 'liverpool'],
    'crypto': ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'solana', 'sol', 'xrp', 'doge'],
    'politics': ['trump', 'biden', 'election', 'president', 'senate', 'congress', 'governor', 'republican', 'democrat', 'deport', 'tariff'],
    'economy': ['fed', 'interest-rate', 'inflation', 'gdp', 'unemployment', 'recession', 'revenue', 'spending', 'federal'],
    'tech': ['apple', 'google', 'microsoft', 'openai', 'chatgpt', 'tesla', 'spacex', 'meta', 'gta', 'ai'],
    'entertainment': ['oscar', 'grammy', 'emmy', 'movie', 'netflix', 'spotify'],
}

# Filter Presets for trader discovery
FILTER_PRESETS = {
    "smart_money": {
        "name": "Smart Money",
        "description": "Profitable traders with proven track record",
        "criteria": {
            "min_pnl": 1000,
            "min_roi": 5,
            "min_win_rate": 55,
            "min_trades": 10
        }
    },
    "day_trader": {
        "name": "Day Trader",
        "description": "High activity traders with frequent trades",
        "criteria": {
            "min_trades": 50,
            "sort_by": "total_trades"
        }
    },
    "conservative": {
        "name": "Conservative Hodler",
        "description": "Low risk, steady returns",
        "criteria": {
            "min_win_rate": 60,
            "min_roi": 10
        }
    },
    "high_volume": {
        "name": "High Volume",
        "description": "Traders with significant capital deployed",
        "criteria": {
            "min_volume": 100000,
            "min_trades": 20
        }
    },
    "whale": {
        "name": "Whale",
        "description": "Very large traders by P&L",
        "criteria": {
            "min_pnl": 100000,
            "min_trades": 10
        }
    },
    "consistent_winner": {
        "name": "Consistent Winner",
        "description": "High win rate with decent volume",
        "criteria": {
            "min_win_rate": 65,
            "min_trades": 20,
            "min_pnl": 500
        }
    }
}
