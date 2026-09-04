# -*- coding: utf-8 -*-
r"""
자동 주가 모니터링 + 매수/매도 알림 시스템
==========================================

지난 대화에서 그린 시스템 구조를 그대로 코드로 옮긴 것입니다. 각 함수/블록 위에
그 구조도의 몇 번 단계에 해당하는지 [번호]로 표시해뒀습니다.

  [1] 스케줄러      -> 이 스크립트 전체를 매일 정해진 시각에 실행 (아래 맨 밑 설명 참고)
  [2] 데이터 수집    -> build_dataframe() (stock_indicator_signals.py 재사용)
  [3] 지표 계산      -> compute_indicator_states() (역시 재사용)
  [4] 상태 비교      -> DB에 저장된 '어제 상태'와 '오늘 상태'를 diff
  [5] 알림 발송      -> 변화가 있을 때만 텔레그램으로 메시지 전송
  [6] DB 저장       -> SQLite에 오늘자 상태를 이력으로 기록 (+ [4]가 읽어갈 '어제 상태'도 여기서 조회)

필요 패키지: yfinance, pandas, numpy, matplotlib, requests (requirements.txt 참고)

같은 폴더에 stock_indicator_signals.py가 있어야 합니다 (지표 계산 로직을 import해서 재사용).

실행 방식 두 가지:
  (A) 로컬 PC (Windows 작업 스케줄러) — DB_PATH를 환경변수로 F:\AIData\... 로 지정
  (B) GitHub Actions (클라우드, PC 꺼져 있어도 동작) — DB를 리포지토리 안 data/ 폴더에 두고
      워크플로가 실행 끝날 때마다 변경사항을 커밋 (.github/workflows/daily_check.yml 참고)
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# stock_indicator_signals.py가 같은 폴더에 있어야 함 (지표 계산 로직 재사용 -> [2],[3] 담당)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_indicator_signals import build_dataframe, compute_indicator_states, INDICATORS


# ---------------------------------------------------------------------------
# 설정 — 여기만 바꾸면 감시 종목/저장 위치/알림 채널이 바뀜
# ---------------------------------------------------------------------------
WATCHLIST = ["SPY", "SOXX", "005930.KS", "^KS200"]     # 감시할 티커 목록

# DB 경로: 기본값은 리포지토리 안의 data/stock_alert.db (GitHub Actions에서 이 경로를
# 그대로 커밋해서 다음 실행 때 이어받는 방식 — [6] 참고).
# 로컬 Windows에서 기존 F:\AIData 폴더를 쓰고 싶으면 환경변수로 덮어쓰면 됨:
#   setx STOCK_ALERT_DB_PATH "F:\AIData\stock_alert.db"
DB_PATH = os.environ.get("STOCK_ALERT_DB_PATH", "data/stock_alert.db")

# 텔레그램 봇 토큰/채팅ID는 환경변수로 넣는 걸 권장 (코드에 직접 안 적는 게 안전)
#   Windows: setx TELEGRAM_BOT_TOKEN "123456:ABC..."  /  setx TELEGRAM_CHAT_ID "987654321"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# ---------------------------------------------------------------------------
# [6] DB 저장 — 스키마 정의 + 읽기/쓰기
#     읽기 함수(get_last_state)가 여기 있는 이유: [4] 상태 비교가 필요로 하는
#     '어제 상태'의 출처가 바로 이 DB이기 때문 (파일 안에서 로직이 서로 맞물려 있음)
# ---------------------------------------------------------------------------
def init_db(db_path: str) -> None:
    """최초 실행 시 테이블이 없으면 생성. [1]이 스크립트를 처음 돌릴 때 자동으로 호출됨."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signal_history (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            close REAL,
            trend_score INTEGER,
            reversion_score INTEGER,
            indicator_states TEXT,   -- {"지표이름": 상태값, ...} 형태의 JSON 문자열
            recorded_at TEXT NOT NULL,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.commit()
    conn.close()


def get_last_state(db_path: str, ticker: str) -> dict | None:
    """이 티커에 대해 가장 최근에 저장된 상태를 읽어온다. 저장된 게 없으면 None(최초 실행)."""
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT date, trend_score, reversion_score, indicator_states "
        "FROM signal_history WHERE ticker = ? ORDER BY date DESC LIMIT 1",
        (ticker,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    date_str, trend_score, reversion_score, states_json = row
    return {
        "date": date_str,
        "trend_score": trend_score,
        "reversion_score": reversion_score,
        "states": json.loads(states_json) if states_json else {},
    }


def save_state(db_path: str, ticker: str, today: dict) -> None:
    """오늘자 상태를 이력 테이블에 기록. 변화가 있었든 없었든 매일 저장 (백테스트용 이력)."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO signal_history "
        "(ticker, date, close, trend_score, reversion_score, indicator_states, recorded_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            ticker, today["date"], today["close"],
            today["trend_score"], today["reversion_score"],
            json.dumps(today["states"], ensure_ascii=False),
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# [2] 데이터 수집 + [3] 지표 계산
#     실제 계산은 build_dataframe()/compute_indicator_states()에 이미 다 구현돼
#     있으므로, 여기서는 '오늘자 마지막 행에서 필요한 값만 뽑아 정리'하는 역할만 한다.
# ---------------------------------------------------------------------------
def compute_today_state(ticker: str) -> dict:
    df = build_dataframe(ticker, period="2y")   # [2] 데이터 수집 (내부에서 yfinance 호출)
    states = compute_indicator_states(df)        # [3] 지표별 상태 + trend/reversion 점수 계산

    latest_date = df.index[-1]
    latest = states.iloc[-1]
    per_indicator = {ind["name"]: int(latest[ind["name"]]) for ind in INDICATORS}

    return {
        "date": latest_date.strftime("%Y-%m-%d"),
        "close": float(df["Close"].iloc[-1]),
        "trend_score": int(latest["trend_score"]),
        "reversion_score": int(latest["reversion_score"]),
        "states": per_indicator,
    }


# ---------------------------------------------------------------------------
# [4] 상태 비교 — 어제 vs 오늘, 실제로 바뀐 지표만 골라내기
#     ('신호가 몇 점이냐'가 아니라 '어제와 달라졌냐'를 보는 게 핵심 — 매일 같은
#      내용을 알려주는 스팸을 막아준다)
# ---------------------------------------------------------------------------
def diff_states(prev: dict | None, today: dict) -> list[str]:
    """상태가 바뀐 지표들을 사람이 읽을 문장 리스트로 반환. 최초 실행(prev=None)이면 빈 리스트."""
    if prev is None:
        return []

    changes = []
    for name, today_val in today["states"].items():
        prev_val = int(prev["states"].get(name, 0))
        if today_val != prev_val:
            direction = "매수" if today_val > prev_val else "매도"
            changes.append(f"· {name}: {direction} 전환 ({prev_val:+d} → {today_val:+d})")
    return changes


# ---------------------------------------------------------------------------
# [5] 알림 발송 — 텔레그램 봇 API
#     토큰/채팅ID가 설정 안 돼 있으면 콘솔에만 출력 (로컬 테스트할 때 편하도록)
# ---------------------------------------------------------------------------
def send_telegram(message: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[알림 미설정 - 콘솔 출력으로 대체]\n" + message)
        return

    import requests  # 알림을 실제로 쓸 때만 필요하므로 지역 import

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
        resp.raise_for_status()
    except Exception as e:  # 네트워크 문제로 알림이 실패해도 나머지 종목 처리는 계속 진행
        print(f"[텔레그램 발송 실패] {e}\n메시지 내용:\n{message}")


# ---------------------------------------------------------------------------
# [1] 스케줄러가 매일 호출하는 진입점
#     -> 감시 종목마다 [2]~[6]을 순서대로 실행
# ---------------------------------------------------------------------------
def run_daily_check() -> None:
    init_db(DB_PATH)  # [6] 최초 1회 테이블 생성

    for ticker in WATCHLIST:
        try:
            today = compute_today_state(ticker)          # [2] 데이터 수집 + [3] 지표 계산
        except Exception as e:
            # 공휴일이라 데이터가 안 갱신됐거나, API가 그날 응답을 안 주는 경우 등
            print(f"[{ticker}] 데이터 수집/계산 실패: {e}")
            continue

        prev = get_last_state(DB_PATH, ticker)             # [4]에서 쓸 '어제 상태' 조회 ([6] DB에서)
        changes = diff_states(prev, today)                 # [4] 비교

        if changes:
            msg = (
                f"[{ticker}] {today['date']} 신호 변화 감지\n"
                f"종가: {today['close']:.2f}\n"
                f"추세추종: {today['trend_score']:+d}   평균회귀: {today['reversion_score']:+d}\n"
                + "\n".join(changes)
            )
            send_telegram(msg)                             # [5] 변화가 있을 때만 알림
        else:
            print(f"[{ticker}] {today['date']} 변화 없음 (알림 생략)")

        save_state(DB_PATH, ticker, today)                 # [6] 오늘자 상태는 변화 여부와 무관하게 항상 저장


if __name__ == "__main__":
    run_daily_check()

# ---------------------------------------------------------------------------
# [1] 스케줄러 등록 방법
# ---------------------------------------------------------------------------
# (A) 로컬 PC — Windows 작업 스케줄러, 명령 프롬프트에서 한 줄로 등록:
#
#   schtasks /create /tn "StockAlertDaily" /tr "python C:\path\to\stock_alert_system.py" ^
#            /sc daily /st 07:00
#
#   삭제하려면:  schtasks /delete /tn "StockAlertDaily" /f
#
# (B) GitHub Actions — PC를 안 켜둬도 클라우드에서 매일 자동 실행됨.
#     .github/workflows/daily_check.yml 파일이 스케줄러 역할을 대신함 (cron 설정 참고).
