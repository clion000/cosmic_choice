import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from datetime import date

# ==========================================
# 1. 페이지 및 헬퍼 함수 설정
# ==========================================
st.set_page_config(page_title="기본 QRNG 시뮬레이터", page_icon="📈", layout="wide")

def format_price(value, currency):
    """표 데이터 찌꺼기를 제거하고 순수 실수(float) 2자리로 변환"""
    try:
        val = float(value.iloc[-1]) if isinstance(value, (pd.Series, pd.DataFrame)) else float(value)
    except:
        val = float(value)
    return f"{currency}{val:,.2f}"

st.title("📈 기본 양자 난수(QRNG) 주가 네비게이터")
st.markdown("정규분포(Box-Muller)로 변환된 양자 난수를 기반으로 기본적인 기하 브라운 운동(GBM) 시뮬레이션을 수행합니다.")

# ==========================================
# 2. 사이드바 - 파라미터 및 파일 업로드
# ==========================================
st.sidebar.header("⚙️ 설정 및 데이터 입력")

ticker_input = st.sidebar.text_input("종목 코드 (Ticker)", value="AAPL")
TICKER = ticker_input.strip().upper()
currency_symbol = "₩" if TICKER.endswith(".KS") or TICKER.endswith(".KQ") else "$"

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("난수 데이터 파일 업로드 (.bin)", type=['bin'])

STEPS = 252              
NUM_PATHS = 1000         

# ==========================================
# 3. 메인 로직 연산
# ==========================================
if uploaded_file is None:
    st.info("👈 좌측 사이드바에서 `.bin` 파일을 업로드하면 시뮬레이션이 시작됩니다.")
else:
    with st.spinner("데이터 무결성 검사 및 시뮬레이션 연산 중..."):
        try:
            # --- A. 주가 데이터 로드 (yfinance 버그 완벽 회피 방식) ---
            # download 대신 Ticker.history 사용 (표 구조가 꼬이지 않는 가장 안전한 방법)
            ticker_obj = yf.Ticker(TICKER)
            data = ticker_obj.history(period="1y")
            
            if data.empty:
                st.error(f"❌ '{TICKER}' 종목의 데이터를 불러오지 못했습니다.")
                st.stop()
                
            close_prices = data['Close'].dropna()
            
            if len(close_prices) < 5:
                st.error("과거 주가 데이터가 너무 적습니다.")
                st.stop()

            # 수익률 계산
            returns = np.log(close_prices / close_prices.shift(1)).dropna()

            # 파라미터 강제 실수(float) 변환
            sigma_annual = float(np.std(returns)) * np.sqrt(252)
            mu_annual = (float(np.mean(returns)) * 252) + (0.5 * sigma_annual**2)

            mu_daily = mu_annual / 252
            sigma_daily = sigma_annual / np.sqrt(252)
            
            S0 = float(close_prices.iloc[-1])
            last_date = close_prices.index[-1]

            # 🚨 [안전 장치 1] 주가 데이터 결측치 스캐너
            if np.isnan(S0) or np.isnan(mu_daily) or np.isnan(sigma_daily):
                st.error(f"주가 연산 실패: S0={S0}, mu={mu_daily}, sigma={sigma_daily}")
                st.stop()

            # --- B. QRNG 파일 로드 및 Box-Muller 정규화 ---
            raw_data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
            u_data = raw_data.astype(np.float32) / 255.0
            u_data[u_data == 0] = np.finfo(float).eps

            half = len(u_data) // 2
            u1 = u_data[:half]
            u2 = u_data[half:half*2]

            z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)

            # 🚨 [안전 장치 2] 양자 난수 결측치 스캐너
            if np.isnan(z).any():
                st.error("업로드된 난수 데이터 변환 중 오류(NaN)가 발생했습니다. 다른 .bin 파일을 사용해 보세요.")
                st.stop()

            required_z = STEPS * NUM_PATHS
            if len(z) < required_z:
                st.error(f"❌ 난수 데이터가 부족합니다! 필요: {required_z}개, 보유: {len(z)}개")
                st.stop()

            # --- C. 시뮬레이션 연산 ---
            results = np.zeros((STEPS, NUM_PATHS))
            results[0] = S0

            for t in range(1, STEPS):
                z_t = z[(t-1)*NUM_PATHS : t*NUM_PATHS]
                results[t] = results[t-1] * np.exp((mu_daily - 0.5 * sigma_daily**2) + sigma_daily * z_t)

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
            st.stop()

        # ==========================================
        # 4. 결과 출력 및 시각화
        # ==========================================
        st.success("시뮬레이션 완료! (데이터 무결성 검증 통과)")
        
        final_prices = results[-1, :]
        expected_avg = np.mean(final_prices)
        expected_median = np.median(final_prices)
        max_price = np.max(final_prices)
        min_price = np.min(final_prices)
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=STEPS)
        final_date_str = future_dates[-1].strftime('%Y-%m-%d')

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("현재 주가", format_price(S0, currency_symbol))
        col2.metric("중앙값 (Median)", format_price(expected_median, currency_symbol))
        col3.metric("최고 예상 주가 (Max)", format_price(max_price, currency_symbol))
        col4.metric("최저 예상 주가 (Min)", format_price(min_price, currency_symbol))

        tab1, tab2, tab3 = st.tabs(["📊 확률 원뿔 (Probability Cone)", "📈 경로 샘플", "📉 주가 분포"])

        with tab1:
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            quantiles = np.percentile(results, [5, 25, 50, 75, 95], axis=1)
            
            ax1.fill_between(future_dates, quantiles[0], quantiles[4], color='blue', alpha=0.1, label='5% - 95% Range')
            ax1.fill_between(future_dates, quantiles[1], quantiles[3], color='blue', alpha=0.3, label='25% - 75% Range')
            ax1.plot(future_dates, quantiles[2], color='navy', linewidth=2, label='Median Path (50%)')
            
            ax1.set_title(f"[{TICKER}] 1-Year Future Price Projection", fontsize=15, fontweight='bold')
            ax1.set_ylabel(f"Predicted Price ({currency_symbol})")
            ax1.legend(loc='upper left')
            ax1.grid(True, linestyle='--', alpha=0.6)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            st.pyplot(fig1)
            plt.close(fig1)

        with tab2:
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(future_dates, results[:, :50], alpha=0.6)
            
            ax2.set_title(f"[{TICKER}] 50 Random Future Path Scenarios", fontsize=15, fontweight='bold')
            ax2.set_ylabel(f"Predicted Price ({currency_symbol})")
            ax2.grid(True, linestyle='--', alpha=0.6)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            st.pyplot(fig2)
            plt.close(fig2)

        with tab3:
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            ax3.hist(final_prices, bins=50, color='skyblue', edgecolor='black', alpha=0.8)

            ax3.axvline(S0, color='red', linestyle='dashed', linewidth=2, label=f"Current: {format_price(S0, currency_symbol)}")
            ax3.axvline(expected_avg, color='green', linestyle='dashed', linewidth=2, label=f"Average: {format_price(expected_avg, currency_symbol)}")
            ax3.axvline(expected_median, color='navy', linestyle='dashed', linewidth=2, label=f"Median: {format_price(expected_median, currency_symbol)}")

            ax3.set_title(f"[{TICKER}] Expected Price Distribution on {final_date_str}", fontsize=15, fontweight='bold')
            ax3.set_xlabel(f"Final Price ({currency_symbol})")
            ax3.set_ylabel("Frequency")
            ax3.legend()
            ax3.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig3)
            plt.close(fig3)