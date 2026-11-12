import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from fredapi import Fred
import feedparser
import requests
import time
from urllib.parse import quote
from google import genai
from google.genai.errors import APIError

# ==========================================
# 1. 頁面基本配置與側邊欄設定
# ==========================================
st.set_page_config(page_title="全球總體經濟、金融與 AI 情緒儀表板", layout="wide")
st.title("🌐 全球總體經濟、金融指標與 AI 輿論情緒即時儀表板")

st.sidebar.header("⚙️ 設定與金鑰")

# 1.1 FRED API Key
if "FRED_API_KEY" in st.secrets:
    fred_key = st.secrets["FRED_API_KEY"]
    st.sidebar.success("已自動讀取 FRED API Key！")
else:
    fred_key = st.sidebar.text_input("請輸入你的 FRED API Key:", type="password")

# 1.2 Gemini API Key
if "GEMINI_API_KEY" in st.secrets:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("已自動讀取 Gemini API Key！")
else:
    gemini_key = st.sidebar.text_input("請輸入你的 Gemini API Key:", type="password")

if st.sidebar.button("🔄 手動刷新所有數據"):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 2. 數據抓取函式 (FRED, yfinance, Google News RSS)
# ==========================================
@st.cache_data(ttl=3600)
def get_fred_data(api_key):
    fred = Fred(api_key=api_key)
    
    # 抓取原始指標並去除空值
    cpi = fred.get_series("CPIAUCSL").dropna()
    dgs10 = fred.get_series("DGS10").dropna()
    dgs2 = fred.get_series("DGS2").dropna()               # 2 年期美債殖利率
    unrate = fred.get_series("UNRATE").dropna()
    fedfunds = fred.get_series("FEDFUNDS").dropna()
    jgb10 = fred.get_series("IRLTLT01JPM156N").dropna()   # 日本 10 年期公債殖利率
    
    # 計算衍生指標（保留原始頻率，避免 ffill 稀釋變化量）
    cpi_yoy = (cpi.pct_change(12) * 100).dropna()
    yield_spread = (dgs10 - dgs2).dropna()
    
    raw_series = {
        "CPI 物價指數": cpi,
        "CPI 年增率 (%)": cpi_yoy,
        "美國10年期美債殖利率 (%)": dgs10,
        "美國2年期美債殖利率 (%)": dgs2,
        "10Y-2Y美債利差 (%)": yield_spread,
        "日本10年期國債殖利率 (%)": jgb10,
        "美國失業率 (%)": unrate,
        "聯邦基金利率 (%)": fedfunds
    }
    
    # 合併時間序列供圖表繪製（保留近兩年）
    df = pd.DataFrame(raw_series).sort_index()
    df = df.ffill().dropna(how="all").tail(730)
    return df, raw_series

@st.cache_data(ttl=600)
def get_market_data():
    tickers = {
        "S&P 500": "^GSPC",
        "台灣加權指數": "^TWII",
        "布蘭特原油": "BZ=F",
        "黃金期貨/現貨 (USD/oz)": "GC=F",
        "美元/台幣匯率": "TWD=X"
    }
    raw_df = yf.download(list(tickers.values()), period="1y", progress=False)
    df = raw_df["Close"].copy() if "Close" in raw_df.columns else raw_df.copy()
    df.rename(columns={v: k for k, v in tickers.items()}, inplace=True)
    # 填補跨國市場休市日落差，避免折線圖斷裂
    return df.dropna(how="all").ffill()

@st.cache_data(ttl=1800)
def fetch_google_news(keyword="Federal Reserve Inflation"):
    safe_keyword = quote(keyword)
    url = f"https://news.google.com/rss/search?q={safe_keyword}&hl=en-US&gl=US&ceid=US:en"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        feed = feedparser.parse(resp.content)
        return [
            {
                "title": entry.title,
                "link": entry.link,
                "published": entry.get("published", "")
            }
            for entry in feed.entries[:8]
        ]
    except Exception:
        return []

# ==========================================
# 3. Gemini AI 呼叫核心（含容錯重試機制）
# ==========================================
def call_gemini_with_retry(api_key, prompt):
    client = genai.Client(api_key=api_key)
    models_to_try = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    last_error = None

    for model_name in models_to_try:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text
            except APIError as e:
                last_error = e
                if e.code == 404:
                    break
                if e.code == 503:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise e
            except Exception as e:
                last_error = e
                time.sleep(1)
                continue

    raise last_error

def generate_macro_summary(api_key, cpi_val, unemp_val, fed_val, spread_val, gold_val):
    spread_status = "殖利率倒掛中" if spread_val < 0 else "利差正常"
    prompt = f"""
你是一位專業總體經濟與金融市場分析師。請根據以下最新公布之核心指標數據，產出一份約 200 字以內的精練總體經濟短評，重點剖析當前通膨壓力、聯準會貨幣政策路徑，以及市場衰退風險（利差信號）：

【最新指標數據】
- 美國 CPI 年增率：{cpi_val:.2f}%
- 美國失業率：{unemp_val:.1f}%
- 美國聯邦基金利率：{fed_val:.2f}%
- 10Y-2Y 美債殖利率利差：{spread_val:.2f}%（{spread_status}）
- 黃金現貨價格：\${gold_val:.1f} 美元/盎司

請以繁體中文撰寫，言簡意賅、切中要點。
"""
    return call_gemini_with_retry(api_key, prompt)

def analyze_news_sentiment(api_key, news_items):
    news_text = "\n".join([f"- {item['title']} ({item['published']})" for item in news_items])
    prompt = f"""你是一位專業總體經濟分析師。請分析以下 {len(news_items)} 條最新財經新聞標題：
{news_text}

請以繁體中文輸出結構化報告：
1. **整體市場情緒分數 (0~100)**（0 為極度恐慌/悲觀，50 為中性，100 為極度樂觀）
2. **貨幣政策立場判定**（鷹派 / 鴿派 / 中立，並簡述理由）
3. **重點分析與逐條評分表格**（以 Markdown 表格呈現：新聞摘要、鷹鴿傾向、情緒影響）"""
    return call_gemini_with_retry(api_key, prompt)

# ==========================================
# 4. 頁面主體渲染
# ==========================================
if fred_key:
    try:
        macro_df, raw_series = get_fred_data(fred_key)
        market_df = get_market_data()

        # 第一區塊：核心 KPI 卡片
