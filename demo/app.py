import json
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import feedparser
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from fredapi import Fred
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field

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
    st.session_state.pop("macro_summary", None)
    st.session_state.pop("news_analysis_data", None)
    st.rerun()


# ==========================================
# 2. 結構化輸出 Schema 定義（供 NLP 視覺化使用）
# ==========================================
class NewsItemSentiment(BaseModel):
    index: int = Field(description="新聞編號，從 1 開始")
    short_title: str = Field(description="繁體中文精簡新聞摘要（約15-25字）")
    stance: str = Field(description="貨幣政策傾向，僅限填入：'鷹派'、'鴿派' 或 '中立'")
    sentiment_score: int = Field(description="該則新聞的市場情緒分數 (0~100，50為中性)")
    impact_summary: str = Field(description="繁體中文簡評該新聞對市場的具體情緒影響")


class MarketSentimentReport(BaseModel):
    overall_score: int = Field(description="整體市場情緒分數 (0~100，0為極度恐慌，50為中性，100為極度樂觀)")
    overall_stance: str = Field(description="整體貨幣政策立場判定，僅限填入：'鷹派'、'鴿派' 或 '中立'")
    stance_reason: str = Field(description="整體貨幣政策立場判定的繁體中文核心理由（約50-100字）")
    items: list[NewsItemSentiment] = Field(description="逐條新聞分析清單")


# ==========================================
# 3. 數據抓取函式
# ==========================================
@st.cache_data(ttl=3600)
def get_fred_data(api_key):
    fred = Fred(api_key=api_key)
    series_ids = {
        "cpi": "CPIAUCSL",
        "dgs10": "DGS10",
        "dgs2": "DGS2",
        "unrate": "UNRATE",
        "fedfunds": "FEDFUNDS",
        "jgb10": "IRLTLT01JPM156N",
    }

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {k: executor.submit(fred.get_series, sid) for k, sid in series_ids.items()}
        results = {k: f.result().dropna() for k, f in futures.items()}

    cpi = results["cpi"]
    dgs10 = results["dgs10"]
    dgs2 = results["dgs2"]
    unrate = results["unrate"]
    fedfunds = results["fedfunds"]
    jgb10 = results["jgb10"]

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
def fetch_google_news(keyword="Federal Reserve Inflation", limit=8):
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
            for entry in feed.entries[:limit]
        ]
    except Exception:
        return []


# ==========================================
# 4. Gemini AI 呼叫核心（v1beta + 強化版自動備援）
# ==========================================
def call_gemini_with_retry(api_key, prompt, response_schema=None):
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})

    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
    ]
    last_error = None

    config = None
    if response_schema is not None:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.2,
        )

    for model_name in models_to_try:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
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

    # 最後備援：直接從你的 API Key 可用模型清單中抓支援的 flash 模型
    try:
        available_models = [
            m.name for m in client.models.list()
            if "flash" in m.name.lower()
            and "tts" not in m.name.lower()
            and "image" not in m.name.lower()
            and "lite" not in m.name.lower()
        ]
        for candidate in available_models:
            try:
                response = client.models.generate_content(
                    model=candidate,
                    contents=prompt,
                    config=config,
                )
                return response.text
            except Exception as inner_e:
                last_error = inner_e
                continue
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


def analyze_news_sentiment_structured(api_key, news_items):
    news_text = "\n".join(
        [f"{idx}. {item['title']} ({item['published']})" for idx, item in enumerate(news_items, 1)]
    )
    prompt = (
        f"你是一位專業總體經濟與金融市場分析師。請分析以下 {len(news_items)} 條最新財經新聞標題：\n"
        f"{news_text}\n\n"
        "請評估：\n"
        "1. 整體市場情緒分數 (0~100，0為極度恐慌/悲觀，50為中性，100為極度樂觀)\n"
        "2. 整體貨幣政策立場判定（僅限：鷹派 / 鴿派 / 中立）及核心理由\n"
        "3. 針對每條新聞（編號 1 到 N），產出繁體中文精簡摘要、鷹鴿傾向（鷹派/鴿派/中立）、"
        "單篇情緒分數 (0~100) 與具體市場影響說明。"
    )
    raw_json = call_gemini_with_retry(api_key, prompt, response_schema=MarketSentimentReport)
    parsed = json.loads(raw_json)
    for item_obj in parsed.get("items", []):
        idx = item_obj.get("index", 1) - 1
        if 0 <= idx < len(news_items):
            item_obj["original_title"] = news_items[idx]["title"]
            item_obj["link"] = news_items[idx]["link"]
            item_obj["published"] = news_items[idx]["published"]
    return parsed


# ==========================================
# 5. NLP 視覺化圖表繪製輔助函式
# ==========================================
def render_sentiment_gauge(score):
    if score >= 60:
        bar_color = "#2ecc71"
        label = "偏多樂觀 (Bullish)"
    elif score <= 40:
        bar_color = "#e74c3c"
        label = "偏空恐慌 (Bearish)"
    else:
        bar_color = "#f39c12"
        label = "中性觀望 (Neutral)"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            delta={"reference": 50, "increasing": {"color": "#2ecc71"}, "decreasing": {"color": "#e74c3c"}},
            title={"text": f"整體市場情緒指數<br><span style='font-size:0.8em;color:gray'>{label}</span>"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": bar_color},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, 35], "color": "rgba(231, 76, 60, 0.15)"},
                    {"range": [35, 65], "color": "rgba(241, 196, 15, 0.15)"},
                    {"range": [65, 100], "color": "rgba(46, 204, 113, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def render_stance_donut(df_items):
    stance_counts = df_items["stance"].value_counts().reset_index()
    stance_counts.columns = ["立場", "篇數"]
    color_map = {"鷹派": "#e74c3c", "鴿派": "#2ecc71", "中立": "#95a5a6"}
    fig = px.pie(
        stance_counts,
        names="立場",
        values="篇數",
        hole=0.55,
        color="立場",
        color_discrete_map=color_map,
        title="鷹派 / 鴿派 / 中立 立場佔比",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=45, b=10), showlegend=False)
    return fig


def render_news_score_bar(df_items):
    df_sorted = df_items.sort_values("sentiment_score", ascending=True).copy()
    df_sorted["label"] = df_sorted.apply(lambda r: f"#{r['index']} {r['short_title']}", axis=1)
    color_map = {"鷹派": "#e74c3c", "鴿派": "#2ecc71", "中立": "#95a5a6"}

    fig = px.bar(
        df_sorted,
        x="sentiment_score",
        y="label",
        orientation="h",
        color="stance",
        color_discrete_map=color_map,
        text="sentiment_score",
        title="各則新聞情緒分數與貨幣政策傾向排行",
        labels={"sentiment_score": "情緒分數 (0-100)", "label": "新聞摘要", "stance": "政策傾向"},
    )
    fig.add_vline(x=50, line_dash="dash", line_color="gray", annotation_text="中性線 (50)")
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis=dict(range=[0, 110]),
        height=max(320, len(df_sorted) * 42),
        margin=dict(l=10, r=20, t=45, b=20),
    )
    return fig


# ==========================================
# 6. 儀表板主畫面渲染函式
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
        st.plotly_chart(fig_spread, use_container_width=True)

    with tab2:
        market_choice = st.selectbox("選擇金融市場標的：", market_df.columns)
        fig_market = px.line(
            market_df,
            y=market_choice,
            title=f"{market_choice} 近一年走勢",
            labels={"index": "日期", market_choice: "價格 / 點數"},
        )
        fig_market.update_layout(hovermode="x unified")
        st.plotly_chart(fig_market, use_container_width=True)

    with tab3:
        macro_choice = st.selectbox("選擇總體經濟指標：", macro_df.columns)
        clean_macro = macro_df[macro_choice].dropna()
        fig_macro = px.line(
            clean_macro,
            title=f"{macro_choice} 近年趨勢",
            labels={"index": "日期", "value": macro_choice},
        )
        fig_macro.update_layout(hovermode="x unified", showlegend=False)
        st.plotly_chart(fig_macro, use_container_width=True)

    # ==========================================
    # Tab 4：升級版 NLP 視覺化與互動介面
    # ==========================================
    with tab4:
        st.markdown("#### 📰 即時財經新聞與 LLM 鷹鴿派情緒分析")

        quick_topics = {
            "自訂關鍵字": None,
            "🇺🇸 聯準會與通膨": "Federal Reserve Inflation",
            "💹 美債殖利率與降息": "US Treasury Yield Rate Cut",
            "🛢️ 原油與地緣政治": "Crude Oil Geopolitics Economy",
            "🇹🇼 台灣半導體與股市": "Taiwan Semiconductor Stock Market",
            "🥇 黃金避險與美元": "Gold Price US Dollar Safe Haven",
        }
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 2, 0.8])
        with ctrl_col1:
            topic_preset = st.selectbox("⚡ 快速主題選擇：", list(quick_topics.keys()), index=1)
        with ctrl_col2:
            default_kw = quick_topics[topic_preset] if quick_topics[topic_preset] else "Federal Reserve Inflation"
            search_kw = st.text_input("🔍 輸入新聞關鍵字：", value=default_kw)
        with ctrl_col3:
            news_limit = st.slider("抓取新聞則數：", min_value=5, max_value=15, value=8)

        news_items = fetch_google_news(search_kw, limit=news_limit)

        action_col1, action_col2 = st.columns([1, 3])
        with action_col1:
            analyze_clicked = st.button("🚀 開始 AI 情緒視覺化分析", type="primary", use_container_width=True)
        with action_col2:
            if not gemini_api_key:
                st.warning("👈 請在側邊欄輸入 Gemini API Key 以啟用 AI 視覺化情緒分析功能。")

        if analyze_clicked and gemini_api_key:
            if news_items:
                with st.spinner("Gemini 正在量化新聞情緒分數與鷹鴿政策傾向..."):
                    try:
                        st.session_state["news_analysis_data"] = analyze_news_sentiment_structured(
                            gemini_api_key, news_items
                        )
                        st.session_state["news_analyzed_kw"] = search_kw
                    except Exception as ai_err:
                        st.error(f"AI 分析時發生錯誤: {ai_err}")
            else:
                st.warning("尚無新聞數據可供分析，請更換關鍵字。")

        if "news_analysis_data" in st.session_state and st.session_state["news_analysis_data"]:
            report_data = st.session_state["news_analysis_data"]
            items_list = report_data.get("items", [])
            df_items = pd.DataFrame(items_list)

            st.write("---")
            st.markdown(f"##### 📊 AI 情緒與貨幣政策總覽（分析關鍵字：`{st.session_state.get('news_analyzed_kw', search_kw)}`）")

            vis_col1, vis_col2, vis_col3 = st.columns([1.1, 1.2, 1])

            with vis_col1:
                st.plotly_chart(
                    render_sentiment_gauge(report_data.get("overall_score", 50)),
                    use_container_width=True,
                )

            with vis_col2:
                overall_stance = report_data.get("overall_stance", "中立")
                stance_badge = {
                    "鷹派": "🦅 **鷹派 (Hawkish)** — 傾向緊縮／維持高利率",
                    "鴿派": "🕊️ **鴿派 (Dovish)** — 傾向寬鬆／釋放降息訊號",
                    "中立": "⚖️ **中立 (Neutral)** — 觀望經濟數據與政策平衡",
                }.get(overall_stance, f"📌 **{overall_stance}**")

                st.markdown("###### 🏛️ 整體貨幣政策立場判定")
                if overall_stance == "鷹派":
                    st.error(stance_badge)
                elif overall_stance == "鴿派":
                    st.success(stance_badge)
                else:
                    st.info(stance_badge)

                st.markdown(f"**核心判讀理由：**\n{report_data.get('stance_reason', '')}")

            with vis_col3:
                if not df_items.empty:
                    st.plotly_chart(render_stance_donut(df_items), use_container_width=True)

            if not df_items.empty:
                st.write("---")
                st.markdown("##### 🎛️ 互動式新聞篩選與單篇情緒排行")

                filter_col1, filter_col2 = st.columns([1, 1])
                with filter_col1:
                    selected_stances = st.multiselect(
                        "篩選政策立場：",
                        options=["鷹派", "鴿派", "中立"],
                        default=["鷹派", "鴿派", "中立"],
                    )
                with filter_col2:
                    score_range = st.slider(
                        "篩選情緒分數區間：",
                        min_value=0,
                        max_value=100,
                        value=(0, 100),
                    )

                filtered_df = df_items[
                    (df_items["stance"].isin(selected_stances))
                    & (df_items["sentiment_score"] >= score_range[0])
                    & (df_items["sentiment_score"] <= score_range[1])
                ]

                if not filtered_df.empty:
                    st.plotly_chart(render_news_score_bar(filtered_df), use_container_width=True)

                    sub_tab_cards, sub_tab_table = st.tabs(["🗂️ 互動新聞解析卡片", "📋 完整逐條評分表格"])

                    with sub_tab_cards:
                        for _, row in filtered_df.iterrows():
                            stance_icon = {"鷹派": "🦅", "鴿派": "🕊️", "中立": "⚖️"}.get(row["stance"], "📰")
                            header_title = (
                                f"{stance_icon} [分數: {row['sentiment_score']} | {row['stance']}] "
                                f"#{row['index']} {row['short_title']}"
                            )
                            with st.expander(header_title, expanded=False):
                                st.markdown(f"**原文標題：** [{row.get('original_title', row['short_title'])}]({row.get('link', '#')})")
                                if row.get("published"):
                                    st.caption(f"發布時間：{row['published']}")
                                st.markdown(f"**AI 市場情緒影響解析：** {row['impact_summary']}")

                    with sub_tab_table:
                        display_df = filtered_df[["index", "short_title", "stance", "sentiment_score", "impact_summary"]].rename(
                            columns={
                                "index": "編號",
                                "short_title": "新聞摘要",
                                "stance": "鷹鴿傾向",
                                "sentiment_score": "情緒分數",
                                "impact_summary": "情緒影響",
                            }
                        )
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                else:
                    st.info("目前篩選條件下無符合的新聞，請放寬立場或分數區間。")

        st.write("---")
        with st.expander("📌 檢視最新抓取之原始財經新聞標題 (Google News RSS)", expanded=("news_analysis_data" not in st.session_state)):
            if news_items:
                for i, item in enumerate(news_items, 1):
                    pub_str = f" — *{item['published']}*" if item["published"] else ""
                    st.markdown(f"**{i}.** [{item['title']}]({item['link']}){pub_str}")
            else:
                st.warning("查無相關新聞標題，請嘗試更換關鍵字。")

    # 第三區塊：AI 總經指標短評
    st.write("---")
    st.subheader("🤖 AI 總體經濟指標即時解讀")

    if st.button("✨ 產生 AI 總體經濟摘要", type="primary"):
        if not gemini_api_key:
            st.warning("⚠️ 請先在側邊欄填入 Gemini API Key（或於 Secrets 配置 GEMINI_API_KEY）以啟用 AI 摘要功能。")
        else:
            with st.spinner("AI 正在分析最新總經指標並撰寫短評..."):
                try:
                    st.session_state["macro_summary"] = generate_macro_summary(
                        api_key=gemini_api_key,
                        cpi_val=latest_cpi,
                        unemp_val=latest_unemp,
                        fed_val=latest_fed,
                        spread_val=latest_spread,
                        gold_val=latest_gold,
                    )
                except Exception as ai_err:
                    st.error(f"調用 Gemini API 時發生錯誤: {ai_err}")

    if "macro_summary" in st.session_state:
        st.markdown(f"> **📌 AI 總經短評：**  \n> {st.session_state['macro_summary']}")


# ==========================================
# 7. 主程式進入點
# ==========================================
if fred_key:
    try:
        macro_data, raw_data = get_fred_data(fred_key)
        market_data = get_market_data()
        render_dashboard(macro_data, raw_data, market_data, gemini_key)
    except Exception as e:
        st.error(f"數據讀取失敗，請確認 FRED API Key 是否正確。錯誤訊息: {e}")
else:
    st.info("👈 請在左側輸入您的 FRED API Key 以開啟儀表板。")
