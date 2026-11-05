import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from fredapi import Fred

# 頁面基本配置
st.set_page_config(page_title="總體經濟與金融即時儀表板", layout="wide")
st.title("🌐 全球總體經濟與金融指標即時儀表板")

# 側邊欄：輸入 API Key 與刷新控制
st.sidebar.header("⚙️ 設定與金鑰")

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
    dgs2 = fred.get_series("DGS2")               # 新增：2 年期美債殖利率
    unrate = fred.get_series("UNRATE")
    fedfunds = fred.get_series("FEDFUNDS")
    jgb10 = fred.get_series("IRLTLT01JPM156N")    # 日本 10 年期公債殖利率
    
    # 計算指標
    cpi_yoy = cpi.pct_change(12) * 100
    yield_spread = dgs10 - dgs2                  # 計算 10Y-2Y 美債利差
    
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
            col1.metric("CPI 年增率", f"{latest_cpi_yoy:.2f}%", f"{latest_cpi_yoy - prev_cpi_yoy:+.2f}% MoM")
        
        # 2. 美國 10 年期美債殖利率
        us_bond_series = macro_df["美國10年期美債殖利率 (%)"].dropna()
        if len(us_bond_series) >= 2:
            latest_us_bond = us_bond_series.iloc[-1]
            prev_us_bond = us_bond_series.iloc[-2]
            col2.metric("美債 10年期", f"{latest_us_bond:.2f}%", f"{latest_us_bond - prev_us_bond:+.2f}%")
        
        # 3. 美國 2 年期美債殖利率
        us2_series = macro_df["美國2年期美債殖利率 (%)"].dropna()
        if len(us2_series) >= 2:
            latest_us2 = us2_series.iloc[-1]
            prev_us2 = us2_series.iloc[-2]
            col3.metric("美債 2年期", f"{latest_us2:.2f}%", f"{latest_us2 - prev_us2:+.2f}%")
        
        # 4. 10Y - 2Y 美債利差
        spread_series = macro_df["10Y-2Y美債利差 (%)"].dropna()
        if len(spread_series) >= 2:
            latest_spread = spread_series.iloc[-1]
            prev_spread = spread_series.iloc[-2]
            col4.metric("10Y-2Y 利差", f"{latest_spread:.2f}%", f"{latest_spread - prev_spread:+.2f}%")

        # 5. 黃金最新報價
        if "黃金期貨/現貨 (USD/oz)" in market_df.columns:
            gold_series = market_df["黃金期貨/現貨 (USD/oz)"].dropna()
            if len(gold_series) >= 2:
                latest_gold = gold_series.iloc[-1]
                prev_gold = gold_series.iloc[-2]
                col5.metric("黃金價格 (oz)", f"${latest_gold:.1f}", f"{latest_gold - prev_gold:+.1f} USD")

        # 核心亮點：殖利率曲線警示區塊
        st.write("---")
        spread_clean = macro_df["10Y-2Y美債利差 (%)"].dropna()
        
        if not spread_clean.empty:
            curr_spread = spread_clean.iloc[-1]
            
            # 若利差小於 0 顯示紅色警告，否則顯示綠色安全提示
            if curr_spread < 0:
                st.error(
                    f"⚠️ **殖利率曲線倒掛警示（Inverted Yield Curve）**｜目前 10Y-2Y 利差為 **{curr_spread:.2f}%**（小於 0%）。"
                    "歷史上長短期殖利率倒掛常被視為經濟衰退的先行訊號，請留意市場波動風險！"
                )
            else:
                st.success(
                    f"✅ **殖利率曲線正常（Normal Yield Curve）**｜目前 10Y-2Y 利差為 **{curr_spread:.2f}%**（大於等於 0%），未處於倒掛狀態。"
                )

        # 第二區塊：互動式圖表
        st.subheader("📈 市場走勢與總體指標折線圖")
        
        tab1, tab2, tab3 = st.tabs([
            "📉 美債 10Y-2Y 利差專題", 
            "📈 金融市場走勢 (yfinance)", 
            "🏛️ 總體經濟與公債趨勢 (FRED)"
        ])
        
        # Tab 1：專題利差折線圖（含基準 0 線與填色警示）
        with tab1:
            st.markdown("#### 10 年期減 2 年期美債殖利率利差歷史走勢")
            
            fig_spread = go.Figure()
            
            # 利差線
            fig_spread.add_trace(go.Scatter(
                x=spread_clean.index,
                y=spread_clean.values,
                mode="lines",
                name="10Y - 2Y 利差",
                line=dict(color="#1f77b4", width=2)
            ))
            
            # 零軸基準線 (Dash Line)
            fig_spread.add_hline(
                y=0, 
                line_dash="dash", 
                line_color="red", 
                annotation_text="倒掛門檻 (0%)", 
                annotation_position="bottom right"
            )
            
            fig_spread.update_layout(
                hovermode="x unified",
                xaxis_title="日期",
                yaxis_title="利差百分比 (%)",
                height=450,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_spread, use_container_width=True)

        with tab2:
            market_choice = st.selectbox("選擇金融市場標的：", market_df.columns)
            fig_market = px.line(
                market_df, 
                y=market_choice, 
                title=f"{market_choice} 近一年走勢",
                labels={"index": "日期", market_choice: "價格 / 點數"}
            )
            fig_market.update_layout(hovermode="x unified")
            st.plotly_chart(fig_market, use_container_width=True)
            
        with tab3:
            macro_choice = st.selectbox("選擇總體經濟指標：", macro_df.columns)
            clean_macro = macro_df[macro_choice].dropna()
            fig_macro = px.line(
                clean_macro, 
                title=f"{macro_choice} 近年趨勢",
                labels={"index": "日期", "value": macro_choice}
            )
            fig_macro.update_layout(hovermode="x unified", showlegend=False)
            st.plotly_chart(fig_macro, use_container_width=True)

    except Exception as e:
        st.error(f"數據讀取失敗，請確認 FRED API Key 是否正確。錯誤訊息: {e}")
else:
    st.info("👈 請在左側輸入您的 FRED API Key 以開啟儀表板。")
