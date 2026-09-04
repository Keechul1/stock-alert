# -*- coding: utf-8 -*-
"""
주가 데이터 + 기술적 지표 기반 매수/매도 시그널 시각화
====================================================

구조
----
1. fetch_price()          : yfinance로 일자별 OHLCV 수집
2. compute_XXX(df) 함수들  : 지표 계산 + '시그널' 컬럼(1=매수, -1=매도, 0=없음) 생성
3. INDICATORS 리스트       : 등록된 모든 지표. 여기 하나 추가하면
                             (a) 현재가 차트 위 오버레이
                             (b) 해당 지표 서브플롯의 매수/매도 마킹
                             양쪽에 자동 반영됨.
4. plot_all()              : 위 결과를 하나의 Figure로 렌더링

새 지표 추가하는 법
-------------------
    def compute_my_indicator(df: pd.DataFrame) -> pd.DataFrame:
        df["MyInd"] = ...                     # 지표값
        df["MyInd_signal"] = 0
        df.loc[매수조건, "MyInd_signal"] = 1
        df.loc[매도조건, "MyInd_signal"] = -1
        return df

    INDICATORS.append({
        "name": "MyInd",
        "compute": compute_my_indicator,
        "signal_col": "MyInd_signal",
        "panel": "MyInd",          # None이면 서브플롯 없이 가격차트에만 표시
    })
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm


def _set_korean_font() -> None:
    """
    matplotlib 기본 폰트(DejaVu Sans)에는 한글이 없어서 라벨이 네모(□)로 깨진다.
    시스템에 설치된 한글 폰트를 찾아 자동으로 지정한다.
    - Windows: 맑은 고딕(Malgun Gothic)이 기본 내장
    - macOS: AppleGothic 기본 내장
    - Linux: 나눔고딕 등을 별도 설치해야 하는 경우가 많음
      (Ubuntu/Debian: sudo apt install fonts-nanum 후 재실행)
    """
    candidates = [
        "Malgun Gothic", "AppleGothic", "NanumGothic",
        "NanumBarunGothic", "Noto Sans CJK KR", "Noto Sans KR",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    else:
        print(
            "[안내] 한글 폰트를 찾지 못해 그래프의 한글이 깨질 수 있습니다.\n"
            "  Windows/맥에서는 보통 자동으로 해결되며, 리눅스라면 나눔고딕 설치 후 "
            "다시 실행해보세요. (예: sudo apt install fonts-nanum)"
        )
    plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호(-)도 깨지는 것 방지


_set_korean_font()


# ---------------------------------------------------------------------------
# 1. 데이터 수집
# ---------------------------------------------------------------------------
def fetch_price(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    일자별 OHLCV 수집.
    한국 종목은 티커 뒤에 .KS(코스피)/.KQ(코스닥) 붙여서 사용
      예) 삼성전자 = '005930.KS', KOSPI200 지수 = '^KS200'
    미국 종목/지수는 그대로 (예: 'SPY', 'SOXX', '^GSPC')
    """
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"'{ticker}' 데이터를 가져오지 못했습니다. 티커/기간을 확인하세요.")

    # yfinance가 멀티인덱스 컬럼을 줄 때가 있어 정리
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.title)  # Open/High/Low/Close/Volume
    df.index.name = "Date"
    df.attrs["ticker"] = ticker  # MA_rank 지표에서 한국/미국 이평 세트를 구분하는 데 사용
    return df


# ---------------------------------------------------------------------------
# 2. 지표별 계산 함수 (지표값 + 시그널)
# ---------------------------------------------------------------------------
def compute_ma_cross(df: pd.DataFrame, fast: int = 20, slow: int = 60) -> pd.DataFrame:
    """단기/장기 이동평균 교차. 골든크로스=매수, 데드크로스=매도."""
    df[f"MA{fast}"] = df["Close"].rolling(fast).mean()
    df[f"MA{slow}"] = df["Close"].rolling(slow).mean()

    diff = df[f"MA{fast}"] - df[f"MA{slow}"]
    cross = np.sign(diff).diff()  # +2: 골든크로스 발생, -2: 데드크로스 발생

    df["MA_signal"] = 0
    df.loc[cross == 2, "MA_signal"] = 1
    df.loc[cross == -2, "MA_signal"] = -1
    return df


def compute_rsi(df: pd.DataFrame, period: int = 14, lower: int = 30, upper: int = 70) -> pd.DataFrame:
    """RSI가 과매도선을 상향돌파=매수, 과매수선을 하향돌파=매도."""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    prev = df["RSI"].shift(1)
    df["RSI_signal"] = 0
    df.loc[(prev < lower) & (df["RSI"] >= lower), "RSI_signal"] = 1
    df.loc[(prev > upper) & (df["RSI"] <= upper), "RSI_signal"] = -1
    return df


def compute_stochastic(df: pd.DataFrame, k_period: int = 14, smooth_k: int = 3,
                        d_period: int = 3, lower: int = 20, upper: int = 80) -> pd.DataFrame:
    """
    슬로우 스토캐스틱(%K, %D). RSI와 같은 평균회귀 방식:
    %K가 과매도선(20) 아래->위로 벗어나면 매수, 과매수선(80) 위->아래로 벗어나면 매도.
    """
    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    fast_k = 100 * (df["Close"] - low_min) / (high_max - low_min)

    slow_k = fast_k.rolling(smooth_k).mean()   # 흔히 쓰는 완만화(smoothed) %K
    d = slow_k.rolling(d_period).mean()        # %D = %K의 신호선

    df["Stoch_K"] = slow_k
    df["Stoch_D"] = d

    prev_k = slow_k.shift(1)
    df["Stoch_signal"] = 0
    df.loc[(prev_k < lower) & (slow_k >= lower), "Stoch_signal"] = 1
    df.loc[(prev_k > upper) & (slow_k <= upper), "Stoch_signal"] = -1
    return df


def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD선이 시그널선을 상향돌파=매수, 하향돌파=매도."""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal_line"] = df["MACD"].ewm(span=signal, adjust=False).mean()

    diff = df["MACD"] - df["MACD_signal_line"]
    cross = np.sign(diff).diff()

    df["MACD_signal"] = 0
    df.loc[cross == 2, "MACD_signal"] = 1
    df.loc[cross == -2, "MACD_signal"] = -1
    return df


def compute_bollinger(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """종가가 하단밴드 하회 후 재진입=매수, 상단밴드 상회 후 재진입=매도."""
    mid = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()

    df["BB_mid"] = mid
    df["BB_upper"] = mid + num_std * std
    df["BB_lower"] = mid - num_std * std

    prev_close = df["Close"].shift(1)
    prev_lower = df["BB_lower"].shift(1)
    prev_upper = df["BB_upper"].shift(1)

    df["BB_signal"] = 0
    df.loc[(prev_close < prev_lower) & (df["Close"] >= df["BB_lower"]), "BB_signal"] = 1
    df.loc[(prev_close > prev_upper) & (df["Close"] <= df["BB_upper"]), "BB_signal"] = -1
    return df


def compute_ma5_slope(df: pd.DataFrame, ma_period: int = 5, slope_avg_period: int = 5) -> pd.DataFrame:
    """
    MA5(5일 이동평균)의 기울기(전일 대비 변화량)와, 그 기울기의 5일 평균을 비교.
    - 매수: 기울기가 양수(상승 중)이거나, 자신의 5일 평균보다 높아지는 순간 (둘 중 하나만 충족돼도 매수)
    - 매도: 기울기가 자신의 5일 평균보다 낮으면서 동시에 음수로 바뀌는 순간
      (그냥 평균보다 낮기만 해서는 매도로 보지 않음 — 여전히 우상향 중일 수 있어서
       '방향 자체가 꺾인(기울기<0)' 경우만 매도로 잡음)
    """
    ma = df["Close"].rolling(ma_period).mean()
    slope = ma.diff()
    slope_avg = slope.rolling(slope_avg_period).mean()

    df["MA5"] = ma
    df["MA5_slope"] = slope
    df["MA5_slope_avg"] = slope_avg

    # 매수: 기울기가 양수(상승 중)이거나, 자신의 5일 평균보다 높아졌을 때 (둘 중 하나만 충족해도 매수)
    buy_cond = (slope > 0) | (slope > slope_avg)
    sell_cond = (slope < slope_avg) & (slope < 0)

    prev_buy = buy_cond.shift(1).fillna(False)
    prev_sell = sell_cond.shift(1).fillna(False)

    df["MA5_slope_signal"] = 0
    df.loc[buy_cond & ~prev_buy, "MA5_slope_signal"] = 1     # 조건이 새로 성립하는 전환점에서만 신호
    df.loc[sell_cond & ~prev_sell, "MA5_slope_signal"] = -1
    return df


def compute_ma_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    이동평균 여러 개(한국: 5/20/60/120/240, 미국: 5/20/60/100/200) 중에서
    MA5가 몇 등(순위)인지를 매일 계산.
    - 매수: MA5의 순위가 전일보다 올라감 (= 다른 이평 하나 이상을 상향 돌파)
    - 매도: MA5의 순위가 전일보다 내려감 (= 하향 돌파)
    - 순위 변동 없음: 신호 없음(0) -> 이후 상태 유지 로직(ffill)에서 이전 매수/매도 상태 그대로 유지
    """
    ticker = df.attrs.get("ticker", "")
    is_kr = ticker.upper().endswith((".KS", ".KQ")) or ticker.upper().startswith("^KS")
    periods = [5, 20, 60, 120, 240] if is_kr else [5, 20, 60, 100, 200]
    base_p = periods[0]

    ma_cols = []
    for p in periods:
        col = f"MA{p}"
        if col not in df.columns:  # MA20/MA60은 compute_ma_cross가 이미 만들어놨을 수 있음 -> 재사용
            df[col] = df["Close"].rolling(p).mean()
        ma_cols.append(col)
    base_col = f"MA{base_p}"

    ma_frame = df[ma_cols]
    valid = ma_frame.notna().all(axis=1)  # 모든 이평이 계산 가능한 시점부터만 순위 판단
    rank = ma_frame.rank(axis=1, method="min")[base_col].where(valid)  # 1=제일 낮음 ~ 5=제일 높음
    rank_prev = rank.shift(1)

    df["MA_rank"] = rank
    df["MA_rank_signal"] = 0
    df.loc[rank > rank_prev, "MA_rank_signal"] = 1
    df.loc[rank < rank_prev, "MA_rank_signal"] = -1
    return df


# ---------------------------------------------------------------------------
# 3. 지표 등록 — 여기에 추가하면 차트에 자동 반영
# ---------------------------------------------------------------------------
# group: "trend"     = 추세추종형 (방향이 정해지면 그 방향 유지 -> 지속 상승장에서 계속 매수)
#        "reversion"  = 평균회귀형 (과매수/과매도에서 반전 기대 -> 지속 상승장에서도 매도가 자주 뜸)
INDICATORS = [
    {"name": "MA Cross(20/60)", "compute": compute_ma_cross, "signal_col": "MA_signal", "panel": None, "group": "trend"},
    {"name": "RSI(14)", "compute": compute_rsi, "signal_col": "RSI_signal", "panel": "RSI", "group": "reversion"},
    {"name": "MACD", "compute": compute_macd, "signal_col": "MACD_signal", "panel": "MACD", "group": "trend"},
    {"name": "Bollinger Band", "compute": compute_bollinger, "signal_col": "BB_signal", "panel": None, "group": "reversion"},
    {"name": "MA5 기울기", "compute": compute_ma5_slope, "signal_col": "MA5_slope_signal", "panel": "MA5_slope", "group": "trend"},
    {"name": "MA 순위(5이평 기준)", "compute": compute_ma_rank, "signal_col": "MA_rank_signal", "panel": "MA_rank", "group": "trend"},
    {"name": "Stochastic(14,3,3)", "compute": compute_stochastic, "signal_col": "Stoch_signal", "panel": "Stoch", "group": "reversion"},
]


def build_dataframe(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = fetch_price(ticker, period=period)
    for ind in INDICATORS:
        df = ind["compute"](df)
    return df


def compute_indicator_states(df: pd.DataFrame) -> pd.DataFrame:
    """
    지표별 '현재 상태'를 계산.
    - 매수 시그널(1)이 뜨면 그 이후로는 매도 시그널(-1)이 뜨기 전까지 +1 상태 유지
    - 매도 시그널이 뜨면 그 이후로는 매수 시그널이 뜨기 전까지 -1 상태 유지
    - 아직 한 번도 시그널이 없었으면 0(중립)

    group별로 따로 합산:
      trend_score     = (추세추종형 지표 중 매수상태 수) - (매도상태 수)
      reversion_score = (평균회귀형 지표 중 매수상태 수) - (매도상태 수)
    두 계열을 섞지 않고 따로 두는 이유: 강한 한방향 추세에서는 평균회귀형이
    계속 매도 상태에 갇히는 경향이 있어, 합쳐버리면 추세추종 신호가 가려짐.
    """
    states = pd.DataFrame(index=df.index)
    for ind in INDICATORS:
        col = ind["signal_col"]
        # 신호 없는 날(0)은 NaN으로 바꿔서 직전 상태를 그대로 이어받게(ffill) 함
        state = df[col].replace(0, np.nan).ffill().fillna(0)
        states[ind["name"]] = state

    trend_cols = [i["name"] for i in INDICATORS if i["group"] == "trend"]
    reversion_cols = [i["name"] for i in INDICATORS if i["group"] == "reversion"]
    states["trend_score"] = states[trend_cols].sum(axis=1) if trend_cols else 0.0
    states["reversion_score"] = states[reversion_cols].sum(axis=1) if reversion_cols else 0.0
    return states


def add_sentiment_strip(ax: plt.Axes, dates: pd.DatetimeIndex, scores: pd.Series,
                         n_indicators: int, label: str):
    """
    scores(-n_indicators ~ +n_indicators)를 파랑(매도 우세)-흰색(중립)-빨강(매수 우세)로
    매핑해서, 가격 차트 바로 아래에 얇은 색 띠(heatmap strip)로 표시한다.
    컬러바는 여기서 붙이지 않고 im 객체만 반환한다 — plt.colorbar(im, ax=ax)를
    바로 쓰면 이 축의 폭이 줄어들어 다른 패널과 x축 매핑이 어긋나기 때문에,
    전체 레이아웃이 확정된 뒤 add_group_colorbars()에서 별도 축으로 붙인다.
    """
    cmap = plt.cm.bwr
    norm_val = np.clip(scores.values / max(n_indicators, 1), -1, 1).reshape(1, -1)

    x0 = mdates.date2num(dates[0])
    x1 = mdates.date2num(dates[-1])
    im = ax.imshow(
        norm_val, aspect="auto", cmap=cmap, vmin=-1, vmax=1,
        extent=[x0, x1, 0, 1], interpolation="nearest",
    )
    ax.set_yticks([])
    ax.set_xlim(x0, x1)
    ax.set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center", labelpad=30)
    return im


def add_group_colorbars(fig: plt.Figure, entries: list) -> None:
    """
    entries: [(ax, im, n_indicators), ...]
    각 축 옆에 '그 축의 폭을 줄이지 않는' 별도 컬러바 축을 그림 오른쪽 여백에 붙인다.
    반드시 fig.tight_layout(rect=[...])로 전체 레이아웃(오른쪽 여백 포함)을
    확정한 뒤에 호출해야 모든 패널의 x축 폭이 서로 일치한다.
    """
    if not entries:
        return
    right_edge = max(ax.get_position().x1 for ax, _, _ in entries)
    gap, width = 0.012, 0.016
    for ax, im, n in entries:
        pos = ax.get_position()
        cax = fig.add_axes((right_edge + gap, pos.y0, width, pos.height))
        cbar = fig.colorbar(im, cax=cax)
        ticks = np.arange(-n, n + 1)
        cbar.set_ticks(ticks / max(n, 1))
        cbar.set_ticklabels([str(int(t)) for t in ticks])
        cbar.ax.tick_params(labelsize=7)


def add_crosshair(fig: plt.Figure, axes, df: pd.DataFrame, states: pd.DataFrame):
    """
    마우스를 올리면 모든 서브플롯을 관통하는 수직선이 같이 움직이고,
    가격 패널 상단 중앙에 그 날짜의 값(종가/추세추종·평균회귀 점수/RSI/MACD)을 표시한다.
    plt.show()로 연 인터랙티브 창에서만 동작하며, 저장된 PNG에는 나타나지 않는다.

    matplotlib 내장 MultiCursor 대신 axvline을 직접 만들어 우리가 직접 관리한다.
    - MultiCursor는 내부적으로 축마다 axhline도 함께 만들어서 y축 자동범위를 틀어지게
      하거나(값 범위가 서로 다른 패널들 사이에서), 확대/축소 후 갱신이 꼬이는 경우가 있었음.
    - axvline은 x=데이터, y=축 기준 0~1(blended transform)이라 y축 확대/축소와 무관하게
      항상 패널 전체 높이를 관통하고, 확대해서 보는 중에도 mouse move 이벤트마다
      새로 계산한 x위치로 다시 그려주므로 어긋나지 않는다.
    """
    axes_list = list(axes)
    lines = [
        ax.axvline(df.index[0], color="gray", lw=0.8, linestyle="--", visible=False, zorder=6)
        for ax in axes_list
    ]
    fig._crosshair_lines = lines  # 참조 유지(가비지컬렉션 방지)

    info_text = axes_list[0].text(
        0.5, 0.97, "", transform=axes_list[0].transAxes, fontsize=9,
        va="top", ha="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    dates_num = mdates.date2num(df.index.to_pydatetime())

    def _hide():
        for ln in lines:
            ln.set_visible(False)
        info_text.set_text("")
        fig.canvas.draw_idle()

    def on_move(event):
        # 확대/축소 중이라도(toolbar 줌 포함) 이 콜백은 마우스가 실제로
        # 서브플롯 위에 있을 때마다 매번 새로 호출되어 x위치를 다시 계산한다.
        if event.inaxes not in axes_list or event.xdata is None:
            _hide()
            return

        idx = int(np.clip(np.searchsorted(dates_num, event.xdata), 0, len(df) - 1))
        x_val = df.index[idx]
        for ln in lines:
            ln.set_xdata([x_val, x_val])
            ln.set_visible(True)

        row, st = df.iloc[idx], states.iloc[idx]
        text_lines = [df.index[idx].strftime("%Y-%m-%d"), f"종가: {row['Close']:.2f}"]
        text_lines.append(f"추세추종: {int(st['trend_score']):+d}   평균회귀: {int(st['reversion_score']):+d}")
        if "RSI" in df.columns:
            text_lines.append(f"RSI: {row['RSI']:.1f}")
        if "MACD" in df.columns:
            text_lines.append(f"MACD: {row['MACD']:.2f} / Signal: {row['MACD_signal_line']:.2f}")
        info_text.set_text("\n".join(text_lines))
        fig.canvas.draw_idle()

    def on_leave(event):
        _hide()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("axes_leave_event", on_leave)
    fig.canvas.mpl_connect("figure_leave_event", on_leave)
    return lines


# ---------------------------------------------------------------------------
# 4. 시각화
# ---------------------------------------------------------------------------
def plot_all(df: pd.DataFrame, ticker: str) -> plt.Figure:
    panel_indicators = [i for i in INDICATORS if i["panel"]]
    trend_n = sum(1 for i in INDICATORS if i["group"] == "trend")
    reversion_n = sum(1 for i in INDICATORS if i["group"] == "reversion")

    # 가격패널(3) + 추세띠(0.35) + 반전띠(0.35) + 지표별 서브플롯(1씩)
    n_panels = 3 + len(panel_indicators)
    height_ratios = [3, 0.35, 0.35] + [1] * len(panel_indicators)

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(13, 4.5 + 2.2 * len(panel_indicators)),
        sharex=True, gridspec_kw={"height_ratios": height_ratios},
    )

    ax_price, ax_trend_strip, ax_reversion_strip, *panel_axes = axes

    # --- (a) 현재가(종가) 차트 ---
    ax_price.plot(df.index, df["Close"], color="black", linewidth=1.3, label="Close", zorder=4)

    if "MA20" in df.columns:
        ax_price.plot(df.index, df["MA20"], linewidth=0.9, label="MA20", zorder=3)
        ax_price.plot(df.index, df["MA60"], linewidth=0.9, label="MA60", zorder=3)
    if "BB_upper" in df.columns:
        ax_price.plot(df.index, df["BB_upper"], linestyle="--", linewidth=0.7, color="gray", zorder=3)
        ax_price.plot(df.index, df["BB_lower"], linestyle="--", linewidth=0.7, color="gray",
                      label="BB(20,2)", zorder=3)

    ax_price.set_title(f"{ticker} — 현재가 / 추세추종 지표 합의 / 평균회귀 지표 합의")
    ax_price.legend(loc="upper left", fontsize=8, ncol=2)
    ax_price.grid(alpha=0.3)

    # --- (b) 추세추종형 / 평균회귀형 합의 상태를 별도 색 띠로 분리 표시 ---
    states = compute_indicator_states(df)
    im_trend = add_sentiment_strip(ax_trend_strip, df.index, states["trend_score"], trend_n, "추세추종")
    im_reversion = add_sentiment_strip(ax_reversion_strip, df.index, states["reversion_score"],
                                        reversion_n, "평균회귀")

    # --- (c) 지표별 서브플롯: 지표 라인 + 해당 지표 위에 시그널 표시 ---
    for ax, ind in zip(panel_axes, panel_indicators):
        name = ind["panel"]
        col = ind["signal_col"]

        if name == "RSI":
            ax.plot(df.index, df["RSI"], color="purple", linewidth=1)
            ax.axhline(70, color="red", linestyle="--", linewidth=0.7)
            ax.axhline(30, color="green", linestyle="--", linewidth=0.7)
            value_col = "RSI"
        elif name == "MACD":
            ax.plot(df.index, df["MACD"], color="blue", linewidth=1, label="MACD")
            ax.plot(df.index, df["MACD_signal_line"], color="orange", linewidth=1, label="Signal")
            ax.axhline(0, color="gray", linewidth=0.6)
            ax.legend(loc="upper left", fontsize=8)
            value_col = "MACD"
        elif name == "MA5_slope":
            ax.plot(df.index, df["MA5_slope"], color="teal", linewidth=1, label="MA5 기울기")
            ax.plot(df.index, df["MA5_slope_avg"], color="brown", linewidth=1,
                    linestyle="--", label="기울기 5일평균")
            ax.axhline(0, color="gray", linewidth=0.6)
            ax.legend(loc="upper left", fontsize=8)
            value_col = "MA5_slope"
        elif name == "MA_rank":
            ax.step(df.index, df["MA_rank"], where="post", color="darkgreen", linewidth=1.1)
            ax.set_yticks([1, 2, 3, 4, 5])
            ax.set_ylim(0.5, 5.5)
            value_col = "MA_rank"
        elif name == "Stoch":
            ax.plot(df.index, df["Stoch_K"], color="blue", linewidth=1, label="%K")
            ax.plot(df.index, df["Stoch_D"], color="orange", linewidth=1, label="%D")
            ax.axhline(80, color="red", linestyle="--", linewidth=0.7)
            ax.axhline(20, color="green", linestyle="--", linewidth=0.7)
            ax.legend(loc="upper left", fontsize=8)
            value_col = "Stoch_K"
        else:
            continue

        buy = df[df[col] == 1]
        sell = df[df[col] == -1]
        ax.scatter(buy.index, buy[value_col], marker="^", s=60, color="green", zorder=5)
        ax.scatter(sell.index, sell[value_col], marker="v", s=60, color="red", zorder=5)
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)

    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))

    # rect 오른쪽을 0.90으로 고정 -> 모든 패널(가격/띠/RSI/MACD)의 폭이 동일하게 맞춰짐.
    # 컬러바는 이 레이아웃이 확정된 '뒤'에 남은 오른쪽 여백(0.90~1.0)에 별도로 붙인다
    # (먼저 붙이면 그 축만 좁아져서 커서 수직선이 어긋나는 원인이 됨).
    fig.tight_layout(rect=(0.0, 0.0, 0.90, 1.0))
    add_group_colorbars(fig, [
        (ax_trend_strip, im_trend, trend_n),
        (ax_reversion_strip, im_reversion, reversion_n),
    ])

    add_crosshair(fig, axes, df, states)
    return fig


def print_recent_signals(df: pd.DataFrame, n: int = 10) -> None:
    """가장 최근 시그널들을 텍스트로도 출력 (콘솔 확인용)."""
    rows = []
    for ind in INDICATORS:
        col = ind["signal_col"]
        hits = df[df[col] != 0][["Close", col]].copy()
        hits["지표"] = ind["name"]
        hits["구분"] = hits[col].map({1: "매수", -1: "매도"})
        rows.append(hits[["지표", "구분", "Close"]])

    if not rows:
        return
    all_hits = pd.concat(rows).sort_index()
    print(f"\n=== 최근 시그널 {n}건 ===")
    print(all_hits.tail(n).to_string())


# ---------------------------------------------------------------------------
# 실행 예시
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    TICKER = "090430.KS"          # 예: '005930.KS'(삼성전자), '^KS200'(KOSPI200), 'SOXX'
    PERIOD = "2y"

    data = build_dataframe(TICKER, period=PERIOD)
    print_recent_signals(data)

    fig = plot_all(data, TICKER)
    fig.savefig(f"{TICKER.replace('^', '').replace('.', '_')}_signals.png", dpi=150)
    plt.show()
