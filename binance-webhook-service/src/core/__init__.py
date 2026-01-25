"""
Core module for Binance Webhook Service
Handles Flask app initialization, logging, and Binance client setup
"""
import os
import logging
from flask import Flask
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Try to import Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from ..config import (
    BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TESTNET, BINANCE_SUB_ACCOUNT_EMAIL,
    GEMINI_API_KEY, ENABLE_AI_VALIDATION, GEMINI_MODEL_NAMES
)

# Configure logging
LOG_DIR = os.getenv('LOG_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs'))
LOG_FILE = os.path.join(LOG_DIR, 'webhook_service.log')

try:
    os.makedirs(LOG_DIR, exist_ok=True)
except Exception:
    try:
        LOG_DIR = os.path.dirname(os.path.abspath(__file__))
        LOG_FILE = os.path.join(LOG_DIR, 'webhook_service.log')
        os.makedirs(LOG_DIR, exist_ok=True)
    except Exception:
        LOG_FILE = None

handlers = [logging.StreamHandler()]
if LOG_FILE:
    try:
        handlers.append(logging.FileHandler(LOG_FILE))
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger(__name__)

if not GEMINI_AVAILABLE:
    logger.warning("google-generativeai not available. AI validation will be disabled. Install with: pip install google-generativeai")

# Initialize Flask app
app = Flask(__name__)

# Initialize Binance client
client = None
if BINANCE_API_KEY and BINANCE_API_SECRET:
    try:
        if BINANCE_TESTNET:
            client = Client(
                api_key=BINANCE_API_KEY,
                api_secret=BINANCE_API_SECRET,
                testnet=True
            )
            logger.info("✅ Binance Testnet client initialized")
        else:
            client = Client(
                api_key=BINANCE_API_KEY,
                api_secret=BINANCE_API_SECRET
            )
            logger.info("✅ Binance Production client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Binance client: {e}")
        client = None
else:
    logger.warning("BINANCE_API_KEY or BINANCE_API_SECRET not set - Binance client not initialized")

# Initialize Gemini client
gemini_client = None
gemini_model_name = None

if GEMINI_AVAILABLE and GEMINI_API_KEY and ENABLE_AI_VALIDATION:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        available_models = []
        try:
            models = genai.list_models()
            all_models = [m.name.split('/')[-1] for m in models if 'generateContent' in m.supported_generation_methods]
            paid_model_keywords = ['2.5-pro', '2.0-pro', 'ultra', '2.5-pro-exp']
            available_models = [m for m in all_models if not any(paid in m.lower() for paid in paid_model_keywords)]
            logger.info(f"Found {len(available_models)} FREE TIER Gemini models (excluded {len(all_models) - len(available_models)} paid models): {', '.join(available_models[:5])}...")
        except Exception as e:
            logger.debug(f"Could not list available models: {e}. Will try common model names.")
        
        user_model = os.getenv('GEMINI_MODEL', None)
        if user_model:
            model_names = [user_model]
        elif available_models:
            paid_models = ['2.5-pro', '2.0-pro', 'ultra']
            free_tier_models = [m for m in available_models if not any(paid in m.lower() for paid in paid_models)]
            
            if not free_tier_models:
                logger.warning("⚠️ No free tier models found in available models. Using fallback list.")
                model_names = GEMINI_MODEL_NAMES
            else:
                high_limit_models = [m for m in free_tier_models if ('1.5-flash' in m.lower() or '1.0-pro' in m.lower()) and '2.5' not in m.lower() and '3' not in m.lower()]
                medium_limit_models = [m for m in free_tier_models if '1.5-pro' in m.lower() and m not in high_limit_models]
                separate_quota_models = [m for m in free_tier_models if ('2.5-flash-lite' in m.lower() or '3-flash' in m.lower())]
                low_limit_models = [m for m in free_tier_models if '2.5-flash' in m.lower() and 'lite' not in m.lower()]
                other_free_models = [m for m in free_tier_models if m not in high_limit_models and m not in medium_limit_models and m not in low_limit_models and m not in separate_quota_models]
                model_names = (high_limit_models[:3] + medium_limit_models[:2] + separate_quota_models[:2] + other_free_models[:2] + low_limit_models[:1])[:10]
                logger.info(f"✅ Selected {len(model_names)} free tier models (excluded paid models like gemini-2.5-pro)")
                if separate_quota_models:
                    logger.info(f"   📊 Models with SEPARATE quotas (can use even if gemini-2.5-flash quota exceeded): {', '.join(separate_quota_models[:2])}")
        else:
            model_names = GEMINI_MODEL_NAMES
        
        for model_name in model_names:
            if not model_name:
                continue
            try:
                gemini_client = genai.GenerativeModel(model_name)
                gemini_model_name = model_name
                logger.info(f"✅ Gemini API initialized successfully (using {model_name})")
                break
            except Exception as e:
                logger.debug(f"Failed to initialize {model_name}: {e}")
                continue
        
        if not gemini_client:
            logger.info("Trying fallback model names...")
            for model_name in GEMINI_MODEL_NAMES:
                if model_name in model_names:
                    continue
                try:
                    gemini_client = genai.GenerativeModel(model_name)
                    gemini_model_name = model_name
                    logger.info(f"✅ Gemini API initialized successfully (using fallback {model_name})")
                    break
                except Exception as e:
                    logger.debug(f"Failed to initialize {model_name}: {e}")
                    continue
        
        if not gemini_client:
            logger.warning(f"Failed to initialize any Gemini model. Tried: {', '.join([m for m in model_names if m])}")
            logger.warning("Tip: Check your API key and available models at https://aistudio.google.com/app/apikey")
    except Exception as e:
        logger.warning(f"Failed to configure Gemini API: {e}. AI validation will be disabled.")
        gemini_client = None
elif ENABLE_AI_VALIDATION and not GEMINI_API_KEY:
    logger.warning("ENABLE_AI_VALIDATION is true but GEMINI_API_KEY is not set. AI validation will be disabled.")
elif ENABLE_AI_VALIDATION and not GEMINI_AVAILABLE:
    logger.warning("AI validation is enabled but google-generativeai package is not installed. Install it with: pip install google-generativeai")

# Note: Global state is now managed in models.state module
# Import them here for backward compatibility
from ..models.state import (
    active_trades, recent_orders, recent_exits,
    account_balance_cache, validation_cache
)

