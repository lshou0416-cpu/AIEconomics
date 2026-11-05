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
        "黃金期貨/現貨 (USD/oz)": "GC=F",
        "美元/台幣匯率": "TWD=X"
    }
    raw_df = yf.download(list(tickers.values()), period="1y", progress=False)
    
    if "Close" in raw_df.columns:
        df = raw_df["Close"].copy()
    else:
        df = raw_df.copy()
        
    df.rename(columns={v: k for k, v in tickers.items()}, inplace=True)
    return df.dropna(how="all")

# 調用 google-genai 產生評論
def generate_macro_summary(api_key, cpi_val, unemp_val, fed_val, spread_val, gold_val):
    client = genai.Client(api_key=api_key)
    prompt = f"""
你是一位專業總體經濟與金融市場分析師。請根據以下最新公布之核心指標數據，產出一份約 200 字以內的精練總體經濟短評，重點剖析當前通膨壓力、聯準會貨幣政策路徑，以及市場衰退風險（利差信號）：

【最新指標數據】
- 美國 CPI 年增率：{cpi_val:.2f}%
- 美國失業率：{unemp_val:.1f}%
- 美國聯邦基金利率：{fed_val:.2f}%
- 10Y-2Y 美債殖利率利差：{spread_val:.2f}%（{'殖利率倒掛中' if spread_val < 0 else '利差正常'}）
- 黃金現貨價格：${gold_val:.1f} 美元/盎司

請以繁體中文撰寫，言簡意賅、切中要點。
"""
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )
    return response.text

# 頁面主體渲染
if fred_key:
    try:
        macro_df = get_fred_data(fred_key)
        market_df = get_market_data()

        # 第一區塊：核心 KPI 卡片
        st.subheader("📊 關鍵總體經濟與資產價格概覽")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # 1. CPI
        cpi_series = macro_df["CPI 年增率 (%)"].dropna()
        latest_cpi = cpi_series.iloc[-1] if not cpi_series.empty else 0.0
        if len(cpi_series) >= 2:
            col1.metric("CPI 年增率", f"{latest_cpi:.2f}%", f"{latest_cpi - cpi_series.iloc[-2]:+.2f}% MoM")
        
        # 2. 10Y 美債
        us10_series = macro_df["美國10年期美債殖利率 (%)"].dropna()
        if len(us10_series) >= 2:
            col2.metric("美債 10年期", f"{us10_series.iloc[-1]:.2f}%", f"{us10_series.iloc[-1] - us10_series.iloc[-2]:+.2f}%")
        
        # 3. 失業率
        unemp_series = macro_df["美國失業率 (%)"].dropna()
        latest_unemp = unemp_series.iloc[-1] if not unemp_series.empty else 0.0
        if len(unemp_series) >= 2:
            col3.metric("美國失業率", f"{latest_unemp:.1f}%", f"{latest_unemp - unemp_series.iloc[-2]:+.1f}%")
        
        # 4. 10Y - 2Y 利差
        spread_series = macro_df["10Y-2Y美債利差 (%)"].dropna()
        latest_spread = spread_series.iloc[-1] if not spread_series.empty else 0.0
        if len(spread_series) >= 2:
            col4.metric("10Y-2Y 利差", f"{latest_spread:.2f}%", f"{latest_spread - spread_series.iloc[-2]:+.2f}%")

        # 5. 黃金報價
        gold_series = market_df["黃金期貨/現貨 (USD/oz)"].dropna() if "黃金期貨/現貨 (USD/oz)" in market_df.columns else pd.Series(dtype=float)
        latest_gold = gold_series.iloc[-1] if not gold_series.empty else 0.0
        if len(gold_series) >= 2:
            col5.metric("黃金價格 (oz)", f"${latest_gold:.1f}", f"{latest_gold - gold_series.iloc[-2]:+.1f} USD")

        fed_series = macro_df["聯邦基金利率 (%)"].dropna()
        latest_fed = fed_series.iloc[-1] if not fed_series.empty else 0.0

        # 倒掛警示區塊
        st.write("---")
        if not spread_series.empty:
            if latest_spread < 0:
                st.error(f"⚠️ **殖利率曲線倒掛警示**｜目前 10Y-2Y 利差為 **{latest_spread:.2f}%**，歷史上常視為經濟衰退的領先信號。")
            else:
                st.success(f"✅ **殖利率曲線正常**｜目前 10Y-2Y 利差為 **{latest_spread:.2f}%**，未處於倒掛狀態。")

        # 第二區塊：互動式圖表
        st.subheader("📈 走勢圖表")
        tab1, tab2 = st.tabs(["📉 美債 10Y-2Y 利差走勢", "📈 市場與總經走勢"])
        
        with tab1:
            fig_spread = go.Figure()
            fig_spread.add_trace(go.Scatter(
                x=spread_series.index, y=spread_series.values,
                mode="lines", name="10Y - 2Y 利差", line=dict(color="#1f77b4", width=2)
            ))
            fig_spread.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="0% 門檻")
            fig_spread.update_layout(hovermode="x unified", xaxis_title="日期", yaxis_title="利差 (%)", height=400)
            st.plotly_chart(fig_spread, use_container_width=True)
            
        with tab2:
            choice = st.selectbox("選擇指標：", list(market_df.columns) + list(macro_df.columns))
            plot_series = market_df[choice] if choice in market_df.columns else macro_df[choice]
            fig = px.line(plot_series.dropna(), title=f"{choice} 近期趨勢", labels={"index": "日期", "value": choice})
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        # 第三區塊：AI 總體經濟短評按鈕
        st.write("---")
        st.subheader("🤖 AI 總體經濟即時解讀")
        
        if st.button("✨ 產生 AI 總體經濟摘要", type="primary"):
            if not gemini_key:
                st.warning("⚠️ 請先在側邊欄填入 Gemini API Key（或於 Secrets 配置 GEMINI_API_KEY）以啟用 AI 摘要功能。")
            else:
                with st.spinner("AI 正在分析最新指標並撰寫短評..."):
                    try:
                        summary_text = generate_macro_summary(
                            api_key=gemini_key,
                            cpi_val=latest_cpi,
                            unemp_val=latest_unemp,
                            fed_val=latest_fed,
                            spread_val=latest_spread,
                            gold_val=latest_gold
                        )
                        st.markdown(
                            f"""
                            > **📌 AI 總經短評：**  
                            > {summary_text}
                            """
                        )
                    except Exception as ai_err:
                        st.error(f"調用 Gemini API 時發生錯誤: {ai_err}")

    except Exception as e:
        st.error(f"數據讀取失敗，請確認 FRED API Key 是否正確。錯誤訊息: {e}")
else:
    st.info("👈 請在左側輸入您的 FRED API Key 以開啟儀表板。")
