import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from fredapi import Fred

# 頁面基本配置
st.set_page_config(page_title="總體經濟與金融即時儀表板", layout="wide")
st.title("🌐 全球總體經濟與金融指標即時儀表板")

# 側邊欄：輸入 API Key 與刷新控制
st.sidebar.header("⚙️ 設定與金鑰")

# 優先使用 Streamlit Secrets，若無則讓使用者在側邊欄手動輸入
if "FRED_API_KEY" in st.secrets:
    fred_key = st.secrets["FRED_API_KEY"]
    st.sidebar.success("已自動讀取 Secrets 中的 API Key！")
else:
    fred_key = st.sidebar.text_input("請輸入你的 FRED API Key:", type="password")

if st.sidebar.button("手動刷新數據"):
    st.cache_data.clear()
    st.rerun()

# 1. 抓取 FRED 數據 (快取 1 小時)
@st.cache_data(ttl=3600)
def get_fred_data(api_key):
    fred = Fred(api_key=api_key)
    
    # 抓取指標數據
    cpi = fred.get_series("CPIAUCSL")
    dgs10 = fred.get_series("DGS10")
    unrate = fred.get_series("UNRATE")
    fedfunds = fred.get_series("FEDFUNDS")
    jgb10 = fred.get_series("IRLTLT01JPM156N")  # 日本 10 年期政府公債殖利率 (月度)
    
    # 計算 CPI 年增率 (YoY Inflation Rate)
    cpi_yoy = cpi.pct_change(12) * 100
    
    series_dict = {
        "CPI 物價指數": cpi,
        "CPI 年增率 (%)": cpi_yoy,
        "美國10年期美債殖利率 (%)": dgs10,
        "日本10年期國債殖利率 (%)": jgb10,
        "美國失業率 (%)": unrate,
        "聯邦基金利率 (%)": fedfunds
    }
    
    # 合併時間序列並取近 3 年覆蓋足夠數據
    df = pd.DataFrame(series_dict).sort_index()
    # 填補日與月資料的頻率差
    df = df.ffill().dropna(how="all").tail(730)
    return df

# 2. 抓取 Market 數據 (快取 10 分鐘)
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
    
    # 確保抓取 Close 欄位並重命名
    if "Close" in raw_df.columns:
        df = raw_df["Close"].copy()
    else:
        df = raw_df.copy()
        
    df.rename(columns={v: k for k, v in tickers.items()}, inplace=True)
    return df.dropna(how="all")

# 頁面渲染邏輯
if fred_key:
    try:
        macro_df = get_fred_data(fred_key)
        market_df = get_market_data()

        # 第一區塊：核心 KPI 卡片 (5 欄均分)
        st.subheader("📊 關鍵總體經濟與資產價格概覽")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # 1. CPI 年增率卡片
        cpi_series = macro_df["CPI 年增率 (%)"].dropna()
        if len(cpi_series) >= 2:
            latest_cpi_yoy = cpi_series.iloc[-1]
            prev_cpi_yoy = cpi_series.iloc[-2]
            col1.metric(
                "CPI 年增率", 
                f"{latest_cpi_yoy:.2f}%", 
                f"{latest_cpi_yoy - prev_cpi_yoy:+.2f}% MoM"
            )
        
        # 2. 美國 10 年期美債殖利率
        us_bond_series = macro_df["美國10年期美債殖利率 (%)"].dropna()
        if len(us_bond_series) >= 2:
            latest_us_bond = us_bond_series.iloc[-1]
            prev_us_bond = us_bond_series.iloc[-2]
            col2.metric(
                "美國10年公債", 
                f"{latest_us_bond:.2f}%", 
                f"{latest_us_bond - prev_us_bond:+.2f}%"
            )
        
        # 3. 日本 10 年期國債殖利率
        jp_bond_series = macro_df["日本10年期國債殖利率 (%)"].dropna()
        if len(jp_bond_series) >= 2:
            latest_jp_bond = jp_bond_series.iloc[-1]
            prev_jp_bond = jp_bond_series.iloc[-2]
            col3.metric(
                "日本10年公債", 
                f"{latest_jp_bond:.2f}%", 
                f"{latest_jp_bond - prev_jp_bond:+.2f}%"
            )
        
        # 4. 聯邦基金利率
        fed_series = macro_df["聯邦基金利率 (%)"].dropna()
        if len(fed_series) >= 2:
            latest_fed = fed_series.iloc[-1]
            prev_fed = fed_series.iloc[-2]
            col4.metric(
                "聯邦基金利率", 
                f"{latest_fed:.2f}%", 
                f"{latest_fed - prev_fed:+.2f}%"
            )

        # 5. 黃金最新報價
        if "黃金期貨/現貨 (USD/oz)" in market_df.columns:
            gold_series = market_df["黃金期貨/現貨 (USD/oz)"].dropna()
            if len(gold_series) >= 2:
                latest_gold = gold_series.iloc[-1]
                prev_gold = gold_series.iloc[-2]
                col5.metric(
                    "黃金價格 (oz)", 
                    f"${latest_gold:.1f}", 
                    f"{latest_gold - prev_gold:+.1f} USD"
                )

        # 第二區塊：互動式圖表
        st.write("---")
        st.subheader("📈 市場走勢與總體指標折線圖")
        
        tab1, tab2 = st.tabs(["金融市場走勢 (yfinance)", "總體經濟與公債走勢 (FRED)"])
        
        with tab1:
            market_choice = st.selectbox("選擇金融市場標的：", market_df.columns)
            fig1 = px.line(
                market_df, 
                y=market_choice, 
                title=f"{market_choice} 近一年走勢",
                labels={"index": "日期", market_choice: "價格 / 點數"}
            )
            fig1.update_layout(hovermode="x unified")
            st.plotly_chart(fig1, use_container_width=True)
            
        with tab2:
            macro_choice = st.selectbox("選擇總體經濟與公債指標：", macro_df.columns)
            # 取非空值繪圖，避免月度數據在折線圖上出現階梯斷層
            clean_macro = macro_df[macro_choice].dropna()
            fig2 = px.line(
                clean_macro, 
                title=f"{macro_choice} 近年趨勢",
                labels={"index": "日期", "value": macro_choice}
            )
            fig2.update_layout(hovermode="x unified", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"數據讀取失敗，請確認 FRED API Key 是否正確。錯誤訊息: {e}")
else:
    st.info("👈 請在左側輸入您的 FRED API Key 以開啟儀表板。")
