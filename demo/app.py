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

if "FRED_API_KEY" in st.secrets:
    fred_key = st.secrets["FRED_API_KEY"]
    st.sidebar.success("已自動讀取 FRED API Key！")
else:
    fred_key = st.sidebar.text_input("請輸入你的 FRED API Key:", type="password")

if "GEMINI_API_KEY" in st.secrets:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("已自動讀取 Gemini API Key！")
else:
    gemini_key = st.sidebar.text_input("請輸入你的 Gemini API Key:", type="password")

if st.sidebar.button("🔄 手動刷新所有數據"):
    st.cache_data.clear()
    st.rerun()


# ==========================================
# 2. 數據抓取函式
# ==========================================
@st.cache_data(ttl=3600)
def get_fred_data(api_key):
    fred = Fred(api_key=api_key)
    cpi = fred.get_series("CPIAUCSL").dropna()
    dgs10 = fred.get_series("DGS10").dropna()
    dgs2 = fred.get_series("DGS2").dropna()
    unrate = fred.get_series("UNRATE").dropna()
    fedfunds = fred.get_series("FEDFUNDS").dropna()
    jgb10 = fred.get_series("IRLTLT01JPM156N").dropna()

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
        "聯邦基金利率 (%)": fedfunds,
    }

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
        "美元/台幣匯率": "TWD=X",
    }
    raw_df = yf.download(list(tickers.values()), period="1y", progress=False)
    df = raw_df["Close"].copy() if "Close" in raw_df.columns else raw_df.copy()
    df.rename(columns={v: k for k, v in tickers.items()}, inplace=True)
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
                "published": entry.get("published", ""),
            }
            for entry in feed.entries[:8]
        ]
    except Exception:
        return []


# ==========================================
# 3. Gemini AI 呼叫核心（支援 3.5 系列與自動備援）
# ==========================================
def call_gemini_with_retry(api_key, prompt):
    client = genai.Client(api_key=api_key)
    models_to_try = [
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3-flash-preview",
    ]
    last_error = None

    for model_name in models_to_try:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
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

    # 最後備援：自動從當前 API Key 可用模型中挑選 Flash 模型
    try:
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if "flash" in m.name.lower() and "generateContent" in actions:
                response = client.models.generate_content(
                    model=m.name,
                    contents=prompt,
                )
                return response.text
    except Exception:
        pass

    raise last_error


def generate_macro_summary(api_key, cpi_val, unemp_val, fed_val, spread_val, gold_val):
    spread_status = "殖利率倒掛中" if spread_val < 0 else "利差正常"
    prompt = (
        "你是一位專業總體經濟與金融市場分析師。請根據以下最新公布之核心指標數據，"
        "產出一份約 200 字以內的精練總體經濟短評，重點剖析當前通膨壓力、聯準會貨幣政策路徑，以及市場衰退風險（利差信號）：\n\n"
        "【最新指標數據】\n"
        f"- 美國 CPI 年增率：{cpi_val:.2f}%\n"
        f"- 美國失業率：{unemp_val:.1f}%\n"
        f"- 美國聯邦基金利率：{fed_val:.2f}%\n"
        f"- 10Y-2Y 美債殖利率利差：{spread_val:.2f}%（{spread_status}）\n"
        f"- 黃金現貨價格：${gold_val:.1f} 美元/盎司\n\n"
        "請以繁體中文撰寫，言簡意賅、切中要點。"
    )
    return call_gemini_with_retry(api_key, prompt)


def analyze_news_sentiment(api_key, news_items):
    news_text = "\n".join([f"- {item['title']} ({item['published']})" for item in news_items])
    prompt = (
        f"你是一位專業總體經濟分析師。請分析以下 {len(news_items)} 條最新財經新聞標題：\n"
        f"{news_text}\n\n"
        "請以繁體中文輸出結構化報告：\n"
        "1. **整體市場情緒分數 (0~100)**（0 為極度恐慌/悲觀，50 為中性，100 為極度樂觀）\n"
        "2. **貨幣政策立場判定**（鷹派 / 鴿派 / 中立，並簡述理由）\n"
        "3. **重點分析與逐條評分表格**（以 Markdown 表格呈現：新聞摘要、鷹鴿傾向、情緒影響）"
    )
    return call_gemini_with_retry(api_key, prompt)


# ==========================================
# 4. 儀表板主畫面渲染函式
# ==========================================
def render_dashboard(macro_df, raw_series, market_df, gemini_api_key):
    # 第一區塊：核心 KPI 卡片 (5 欄)
    st.subheader("📊 關鍵總體經濟與資產價格概覽")
    col1, col2, col3, col4, col5 = st.columns(5)

    cpi_series = raw_series["CPI 年增率 (%)"]
    latest_cpi = cpi_series.iloc[-1] if not cpi_series.empty else 0.0
    if len(cpi_series) >= 2:
        col1.metric("CPI 年增率", f"{latest_cpi:.2f}%", f"{latest_cpi - cpi_series.iloc[-2]:+.2f}% MoM")

    us10_series = raw_series["美國10年期美債殖利率 (%)"]
    if len(us10_series) >= 2:
        col2.metric("美債 10年期", f"{us10_series.iloc[-1]:.2f}%", f"{us10_series.iloc[-1] - us10_series.iloc[-2]:+.2f}%")

    unemp_series = raw_series["美國失業率 (%)"]
    latest_unemp = unemp_series.iloc[-1] if not unemp_series.empty else 0.0
    if len(unemp_series) >= 2:
        col3.metric("美國失業率", f"{latest_unemp:.1f}%", f"{latest_unemp - unemp_series.iloc[-2]:+.1f}% MoM")

    spread_series = raw_series["10Y-2Y美債利差 (%)"]
    latest_spread = spread_series.iloc[-1] if not spread_series.empty else 0.0
    if len(spread_series) >= 2:
        col4.metric("10Y-2Y 利差", f"{latest_spread:.2f}%", f"{latest_spread - spread_series.iloc[-2]:+.2f}%")

    gold_series = (
        market_df["黃金期貨/現貨 (USD/oz)"].dropna()
        if "黃金期貨/現貨 (USD/oz)" in market_df.columns
        else pd.Series(dtype=float)
    )
    latest_gold = gold_series.iloc[-1] if not gold_series.empty else 0.0
    if len(gold_series) >= 2:
        col5.metric("黃金價格 (oz)", f"${latest_gold:.1f}", f"{latest_gold - gold_series.iloc[-2]:+.1f} USD")

    fed_series = raw_series["聯邦基金利率 (%)"]
    latest_fed = fed_series.iloc[-1] if not fed_series.empty else 0.0

    # 核心警示區塊：殖利率曲線狀態
    st.write("---")
    if not spread_series.empty:
        if latest_spread < 0:
            st.error(
                f"⚠️ **殖利率曲線倒掛警示（Inverted Yield Curve）**｜目前 10Y-2Y 利差為 **{latest_spread:.2f}%**（小於 0%）。"
                "歷史上長短期殖利率倒掛常被視為經濟衰退的先行訊號，請留意市場波動風險！"
            )
        else:
            st.success(
                f"✅ **殖利率曲線正常（Normal Yield Curve）**｜目前 10Y-2Y 利差為 **{latest_spread:.2f}%**（大於等於 0%），未處於倒掛狀態。"
            )

    # 第二區塊：多功能分頁儀表板 (4 個分頁)
    st.subheader("📈 市場走勢、總經指標與 AI 輿論分析")
    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 美債 10Y-2Y 利差走勢",
        "📈 金融市場走勢 (yfinance)",
        "🏛️ 總體經濟與公債趨勢 (FRED)",
        "📰 NLP 市場新聞與鷹鴿情緒分析",
    ])

    with tab1:
        st.markdown("#### 10 年期減 2 年期美債殖利率利差歷史走勢")
        recent_spread = macro_df["10Y-2Y美債利差 (%)"].dropna()
        fig_spread = go.Figure()
        fig_spread.add_trace(go.Scatter(
            x=recent_spread.index,
            y=recent_spread.values,
            mode="lines",
            name="10Y - 2Y 利差",
            line=dict(color="#1f77b4", width=2),
        ))
        fig_spread.add_hline(
            y=0,
            line_dash="dash",
            line_color="red",
            annotation_text="倒掛門檻 (0%)",
            annotation_position="bottom right",
        )
        fig_spread.update_layout(
            hovermode="x unified",
            xaxis_title="日期",
            yaxis_title="利差百分比 (%)",
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_spread, width="stretch")

    with tab2:
        market_choice = st.selectbox("選擇金融市場標的：", market_df.columns)
        fig_market = px.line(
            market_df,
            y=market_choice,
            title=f"{market_choice} 近一年走勢",
            labels={"index": "日期", market_choice: "價格 / 點數"},
        )
        fig_market.update_layout(hovermode="x unified")
        st.plotly_chart(fig_market, width="stretch")

    with tab3:
        macro_choice = st.selectbox("選擇總體經濟指標：", macro_df.columns)
        clean_macro = macro_df[macro_choice].dropna()
        fig_macro = px.line(
            clean_macro,
            title=f"{macro_choice} 近年趨勢",
            labels={"index": "日期", "value": macro_choice},
        )
        fig_macro.update_layout(hovermode="x unified", showlegend=False)
        st.plotly_chart(fig_macro, width="stretch")

    with tab4:
        st.markdown("#### 📰 即時財經新聞與 LLM 鷹鴿派情緒分析")
        search_kw = st.text_input("🔍 輸入新聞關鍵字 (預設: Federal Reserve Inflation):", value="Federal Reserve Inflation")
        news_items = fetch_google_news(search_kw)

        nlp_col1, nlp_col2 = st.columns([1, 1])

        with nlp_col1:
            st.markdown("##### 📌 最新財經新聞標題 (Google News RSS)")
            if news_items:
                for i, item in enumerate(news_items, 1):
                    st.markdown(f"**{i}.** [{item['title']}]({item['link']})")
            else:
                st.warning("查無相關新聞標題，請嘗試更換關鍵字。")

        with nlp_col2:
            st.markdown("##### 🤖 Gemini AI 鷹鴿派與情緒打分")
            if gemini_api_key:
                if st.button("🚀 開始分析新聞市場情緒", type="primary"):
                    if news_items:
                        with st.spinner("Gemini 正在剖析文本立場與鷹鴿情緒中..."):
                            try:
                                report = analyze_news_sentiment(gemini_api_key, news_items)
                                st.markdown(report)
                            except Exception as ai_err:
                                st.error(f"AI 分析時發生錯誤: {ai_err}")
                    else:
                        st.warning("尚無新聞數據可供分析。")
            else:
                st.warning("👈 請在側邊欄輸入 Gemini API Key 以啟用 AI 文本情緒分析功能。")

    # 第三區塊：AI 總經指標短評
    st.write("---")
    st.subheader("🤖 AI 總體經濟指標即時解讀")

    if st.button("✨ 產生 AI 總體經濟摘要", type="primary"):
        if not gemini_api_key:
            st.warning("⚠️ 請先在側邊欄填入 Gemini API Key（或於 Secrets 配置 GEMINI_API_KEY）以啟用 AI 摘要功能。")
        else:
            with st.spinner("AI 正在分析最新總經指標並撰寫短評..."):
                try:
                    summary_text = generate_macro_summary(
                        api_key=gemini_api_key,
                        cpi_val=latest_cpi,
                        unemp_val=latest_unemp,
                        fed_val=latest_fed,
                        spread_val=latest_spread,
                        gold_val=latest_gold,
                    )
                    st.markdown(f"> **📌 AI 總經短評：**  \n> {summary_text}")
                except Exception as ai_err:
                    st.error(f"調用 Gemini API 時發生錯誤: {ai_err}")


# ==========================================
# 5. 主程式進入點
# ==========================================
if fred_key:
    try:
        macro_data, raw_data = get_fred_data(fred_key)
        market_data = get_market_data()
        render_dashboard(macro_data, raw_data, market_data, gemini_key)
    except Exception as e:
        st.error(f"數據讀取失敗，請確認 FRED API Key 是否正確。錯誤訊息: {e}")
else:
    st.info("👈 請在左側輸入您的 FRED API Key 以開啟儀表板。")*
