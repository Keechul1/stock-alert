# stock-alert

주가 지표 기반 매수/매도 상태를 매일 자동으로 체크하고, 어제와 달라졌을 때만
텔레그램으로 알려주는 시스템. (구조 설명은 대화 참고 — 각 코드 블록에 [1]~[6] 주석으로
어느 단계인지 표시돼 있음)

## 처음 설정할 때 (한 번만)

1. **Settings → Secrets and variables → Actions → New repository secret** 에서 아래 두 개 등록
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
2. `stock_alert_system.py` 맨 위쪽 `WATCHLIST` 를 원하는 종목으로 수정
3. **Actions** 탭 → `Daily Stock Alert` 워크플로 → **Run workflow** 로 한 번 수동 실행해서
   정상 동작하는지 확인 (첫 실행은 "이전 상태 없음"이라 알림 없이 기준값만 저장됨)

## 이후

`.github/workflows/daily_check.yml`에 등록된 시각(기본: 매일 한국시간 오전 7시)에 자동 실행됩니다.
실행할 때마다 `data/stock_alert.db`가 갱신되고 그 변경사항이 자동으로 커밋됩니다 —
그래야 다음 실행이 "어제 상태"를 이어받아 비교할 수 있습니다.

참고: GitHub Actions의 cron은 정확한 시각 보장이 아니라 몇 분~몇십 분 늦게 시작될 수 있습니다.
또한 리포지토리에 60일 이상 활동이 없으면 스케줄이 자동으로 꺼지니, 가끔 커밋 한 번씩 해주거나
Actions 탭에서 상태를 확인해주세요.
