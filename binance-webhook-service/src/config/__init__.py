"""
Configuration module for Binance Webhook Service
Handles all environment variables and configuration settings
"""
import os

# Load environment variables from .env file (if present)
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
    try:
        load_dotenv(env_path)
    except (PermissionError, IOError, FileNotFoundError):
        pass
except ImportError:
    pass

# Binance API Configuration
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
BINANCE_TESTNET = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
BINANCE_SUB_ACCOUNT_EMAIL = os.getenv('BINANCE_SUB_ACCOUNT_EMAIL', '')

# Webhook Security
WEBHOOK_TOKEN = os.getenv('WEBHOOK_TOKEN', 'CHANGE_ME')

# Slack Notifications
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL', '')
SLACK_SIGNAL_WEBHOOK_URL = os.getenv('SLACK_SIGNAL_WEBHOOK_URL', '')

# AI Validation Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
ENABLE_AI_VALIDATION = os.getenv('ENABLE_AI_VALIDATION', 'true').lower() == 'true'
AI_VALIDATION_MIN_CONFIDENCE = float(os.getenv('AI_VALIDATION_MIN_CONFIDENCE', '55'))
ENABLE_AI_PRICE_SUGGESTIONS = os.getenv('ENABLE_AI_PRICE_SUGGESTIONS', 'true').lower() == 'true'

# Trading Configuration
ENTRY_SIZE_USD = float(os.getenv('ENTRY_SIZE_USD', '10.0'))
LEVERAGE = int(os.getenv('LEVERAGE', '20'))
TOTAL_ENTRIES = 2

# TP Configuration
TP1_PERCENT = float(os.getenv('TP1_PERCENT', '4.0'))
TP1_SPLIT = float(os.getenv('TP1_SPLIT', '70.0'))
TP2_SPLIT = float(os.getenv('TP2_SPLIT', '30.0'))
TP_HIGH_CONFIDENCE_THRESHOLD = float(os.getenv('TP_HIGH_CONFIDENCE_THRESHOLD', '90.0'))

# Risk Management Configuration
ENABLE_RISK_VALIDATION = os.getenv('ENABLE_RISK_VALIDATION', 'true').lower() == 'true'
MAX_RISK_PERCENT = float(os.getenv('MAX_RISK_PERCENT', '20.0'))
ENABLE_TRAILING_STOP_LOSS = os.getenv('ENABLE_TRAILING_STOP_LOSS', 'true').lower() == 'true'
TRAILING_SL_BREAKEVEN_PERCENT = float(os.getenv('TRAILING_SL_BREAKEVEN_PERCENT', '0.5'))

# Cache Configuration
BALANCE_CACHE_TTL = 60
VALIDATION_CACHE_TTL = 600
ORDER_COOLDOWN = 60
EXIT_COOLDOWN = 30

# Gemini Model Names (Free tier models)
GEMINI_MODEL_NAMES = [
    'gemini-1.5-flash-latest',
    'gemini-1.5-flash',
    'gemini-1.0-pro-latest',
    'gemini-1.0-pro',
    'gemini-1.5-pro-latest',
    'gemini-1.5-pro',
    'gemini-2.5-flash-lite',
    'gemini-3-flash',
    'gemini-2.5-flash',
    'gemini-2.5-flash-latest',
]

