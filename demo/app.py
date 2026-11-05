import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from fredapi import Fred
from google import genai

# 頁面基本配置
st.set_page_config(page_title="總體經濟與金融即時儀表板", layout="wide")
st.title("🌐 全球總體經濟與金融指標即時儀表板")

# 側邊欄：輸入 API Key 與刷新控制
st.sidebar.header("⚙️ 設定與金鑰")

# 1. FRED API Key
if "FRED_API_KEY" in st.secrets:
    fred_key = st.secrets["FRED_API_KEY"]
    st.sidebar.success("已自動讀取 FRED API Key！")
else:
    fred_key = st.sidebar.text_input("請輸入你的 FRED API Key:", type="password")

# 2. Gemini API Key
if "GEMINI_API_KEY" in st.secrets:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("已自動讀取 Gemini API Key！")
else:
    gemini_key = st.sidebar.text_input("請輸入你的 Gemini API Key:", type="password")

if st.sidebar.button("手動刷新數據"):
    st.cache_data.clear()
    st.rerun()

# 抓取 FRED 數據 (快取 1 小時)
@st.cache_data(ttl=3600)
def get_fred_data(api_key):
    fred = Fred(api_key=api_key)
    
    cpi = fred.get_series("CPIAUCSL")
    dgs10 = fred.get_series("DGS10")
    dgs2 = fred.get_series("DGS2")
    unrate = fred.get_series("UNRATE")
    fedfunds = fred.get_series("FEDFUNDS")
    jgb10 = fred.get_series("IRLTLT01JPM156N")
    
    cpi_yoy = cpi.pct_change(12) * 100
    yield_spread = dgs10 - dgs2
    
    series_dict = {
        "CPI 物價指數": cpi,
        "CPI 年增率 (%)": cpi_yoy,
        "美國10年期美債殖利率 (%)": dgs10,
        "美國2年期美債殖利率 (%)": dgs2,
        "10Y-2Y美債利差 (%)": yield_spread,
        "日本10年期國債殖利率 (%)": jgb10,
        "美國失業率 (%)": unrate,
        "聯邦基金利率 (%)": fedfunds
    }
    
    df = pd.DataFrame(series_dict).sort_index()
    df = df.ffill().dropna(how="all").tail(730)
    return df

# 抓取 Market 數據 (快取 10 分鐘)
@st.cache_data(ttl=600)
def get_market_data():
    tickers = {
        "S&P 500": "^GSPC",
        "台灣加權指數": "^TWII",
        "布蘭特原油": "BZ=F",
        "黃金期貨/現貨 (USD/oz)": "GC=
