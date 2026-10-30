"""Settings, read once from the environment (.env). Defaults are the v1 choices from Phase 0."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODES = ("paper", "demo", "live")

BINANCE_URLS = {
    "public": "https://api.binance.com",
    "demo": "https://demo-api.binance.com",
    "live": "https://api.binance.com",
}


def _env(name, default):
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    if isinstance(default, bool):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return type(default)(val)


@dataclass
class Settings:
    mode: str = field(default_factory=lambda: _env("MODE", "paper"))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "data"))))

    # Universe (section 3)
    universe_size: int = field(default_factory=lambda: _env("UNIVERSE_SIZE", 30))
    cg_top: int = field(default_factory=lambda: _env("CG_TOP", 40))
    min_volume_usdt: float = field(default_factory=lambda: _env("MIN_VOLUME_USDT", 20_000_000.0))
    # Universe v2. A single quiet day used to eject established coins on a 24h snapshot: on
    # 4 Oct 2026 the list fell from 19 names to 14 because ADA, LINK, TRX and XLM each printed one
    # thin day. These three settings make the list judge a coin on its recent habit, not one
    # reading, and make leaving harder than joining.
    volume_window_days: int = field(default_factory=lambda: _env("VOLUME_WINDOW_DAYS", 7))
    volume_exit_ratio: float = field(default_factory=lambda: _env("VOLUME_EXIT_RATIO", 0.7))
    universe_min_names: int = field(default_factory=lambda: _env("UNIVERSE_MIN_NAMES", 12))
    universe_refresh_hours: float = field(default_factory=lambda: _env("UNIVERSE_REFRESH_HOURS", 24.0))
    min_history_days: int = field(default_factory=lambda: _env("MIN_HISTORY_DAYS", 90))
    exclude_memes: bool = field(default_factory=lambda: _env("EXCLUDE_MEMES", True))

    # Strategy (section 6)
    timeframe: str = "4h"
    # How often the bot decides. "4h" is the validated strategy. "1h" keeps every horizon the same
    # in wall-clock terms but checks four times as often; measured as worse (see RESEARCH_LOG.md),
    # and offered because the choice is yours.
    decision_tf: str = field(default_factory=lambda: _env("DECISION_TF", "4h"))
    # "live": the levels still come from closed 4h bars, but the triggers are checked against the
    # live price every LIVE_POLL_SECONDS, so an entry or exit can happen at any moment.
    live_poll_seconds: float = field(default_factory=lambda: _env("LIVE_POLL_SECONDS", 15.0))
    # Which closed candle the levels are rebuilt from. Empty means "follow DECISION_TF" (4h).
    # LEVEL_TF=1h rebuilds the channel, the ranking and the filters every hour, so in the live
    # cycle nothing waits on a 4-hour boundary at all.
    level_tf: str = field(default_factory=lambda: _env("LEVEL_TF", ""))
    donchian_bars: int = field(default_factory=lambda: _env("DONCHIAN_BARS", 20))
    exit_rank: int = field(default_factory=lambda: _env("EXIT_RANK", 10))
    candidates: int = field(default_factory=lambda: _env("CANDIDATES", 5))

    # Risk (section 6 / phase 6)
    start_equity: float = field(default_factory=lambda: _env("START_EQUITY", 10_000.0))
    risk_pct: float = field(default_factory=lambda: _env("RISK_PCT", 0.01))
    max_names: int = field(default_factory=lambda: _env("MAX_NAMES", 3))
    max_weight: float = field(default_factory=lambda: _env("MAX_WEIGHT", 0.30))
    daily_loss_halt: float = field(default_factory=lambda: _env("DAILY_LOSS_HALT", 0.03))
    kill_drawdown: float = field(default_factory=lambda: _env("KILL_DRAWDOWN", 0.08))
    fee_rate: float = field(default_factory=lambda: _env("FEE_RATE", 0.001))
    slippage_bps: float = field(default_factory=lambda: _env("SLIPPAGE_BPS", 7.0))

    # Approval gate (section 5.1 / phase 8)
    signal_expiry_hours: float = field(default_factory=lambda: _env("SIGNAL_EXPIRY_HOURS", 2.0))
    live_max_names: int = field(default_factory=lambda: _env("LIVE_MAX_NAMES", 1))
    live_budget_usdt: float = field(default_factory=lambda: _env("LIVE_BUDGET_USDT", 100.0))
    live_confirmed: bool = field(default_factory=lambda: _env("LIVE_TRADING_CONFIRMED", False))
    # AUTO_TRADE=true removes the human approval step in live mode: the bot enters by itself
    # instead of waiting for a TRADE reply. Telegram still reports every fill. Paper and demo
    # already trade by themselves, so this changes nothing outside live mode.
    auto_trade: bool = field(default_factory=lambda: _env("AUTO_TRADE", False))

    # A second account (ACCOUNT=paper100) shares the first one's price files instead of
    # downloading its own, so the two never write the same file at the same time.
    shared_data: bool = field(default_factory=lambda: _env("SHARED_DATA", False))
    # Set by the fleet so several accounts can run in one process. Empty means "use ACCOUNT, or
    # fall back to the mode name", which is how a single-account bot has always behaved.
    account_name: str = ""

    # Secrets: only ever read from .env, never from source.
    # Binance runs two separate test environments and a key from one is rejected by the other:
    #   demo-api.binance.com      - the Demo Trading product inside your Binance account
    #   testnet.binance.vision    - the Spot Test Network, signed up for with GitHub
    # Set DEMO_API_URL to whichever issued your key.
    demo_api_url: str = field(default_factory=lambda: _env("DEMO_API_URL", BINANCE_URLS["demo"]))

    binance_key: str = field(default_factory=lambda: _env("BINANCE_API_KEY", ""))
    binance_secret: str = field(default_factory=lambda: _env("BINANCE_API_SECRET", ""))
    telegram_token: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: _env("TELEGRAM_CHAT_ID", ""))

    def __post_init__(self):
        if self.level_tf not in ("", "1h", "4h"):
            raise ValueError(f"LEVEL_TF must be 1h, 4h or empty, got {self.level_tf!r}")
        if self.decision_tf not in ("1h", "4h", "live"):
            raise ValueError(f"DECISION_TF must be 1h, 4h or live, got {self.decision_tf!r}")
        if self.mode not in MODES:
            raise ValueError(f"MODE must be one of {MODES}, got {self.mode!r}")
        if self.mode == "live" and not self.live_confirmed:
            raise RuntimeError(
                "MODE=live needs LIVE_TRADING_CONFIRMED=true in .env as a separate opt-in. "
                "Finish Phase 10 (a boring demo week) first."
            )
        self.data_dir = Path(self.data_dir)
        (self.data_dir / "klines").mkdir(parents=True, exist_ok=True)

    @property
    def bar_tf(self) -> str:
        """The candle the levels are built from.

        LEVEL_TF wins when set, so the live cycle can refresh its levels hourly while still never
        reading an unfinished candle. Otherwise it follows DECISION_TF.
        """
        if self.level_tf in ("1h", "4h"):
            return self.level_tf
        return "4h" if self.decision_tf in ("4h", "live") else self.decision_tf

    @property
    def bars_per_day(self) -> int:
        return {"1h": 24, "4h": 6}[self.bar_tf]

    @property
    def donchian_for_tf(self) -> int:
        """The same 3.33-day channel, expressed in bars of the chosen timeframe."""
        return self.donchian_bars * (self.bars_per_day // 6)

    @property
    def live_cycle(self) -> bool:
        return self.decision_tf == "live"

    @property
    def klines_dir(self) -> Path:
        return self.data_dir / "klines"

    @property
    def account(self) -> str:
        """Name of this account's state file. Lets a second, smaller paper account run alongside
        the main one on the same prices (ACCOUNT=paper100 START_EQUITY=100)."""
        return self.account_name or _env("ACCOUNT", self.mode)

    @property
    def db_path(self) -> Path:
        return self.data_dir / f"state_{self.account}.sqlite"

    @property
    def heartbeat_path(self) -> Path:
        return self.data_dir / (f"heartbeat_{self.account}" if self.account != self.mode else "heartbeat")

    @property
    def universe_path(self) -> Path:
        return self.data_dir / "universe.json"

    @property
    def kill_switch_path(self) -> Path:
        return self.data_dir / "KILL"

    @property
    def slippage(self) -> float:
        return self.slippage_bps / 10_000


def load() -> Settings:
    return Settings()
