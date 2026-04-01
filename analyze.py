import glob

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

# ─── VERİ YÜKLEMESİ ──────────────────────────────────────────────────────────

def load_data():
    files = sorted(glob.glob("ookla_turkey_*.parquet"))
    if not files:
        raise FileNotFoundError(
            "Parquet dosyası bulunamadı. Önce index.py'yi çalıştırın."
        )
    latest = files[-1]
    df = pd.read_parquet(latest)
    df["download_mbps"]    = (df["avg_d_kbps"] / 1000).round(2)
    df["upload_mbps"]      = (df["avg_u_kbps"] / 1000).round(2)
    df["latency_ms"]       = df["avg_lat_ms"].astype(float)
    df["lat_down_ms"]      = pd.to_numeric(df["avg_lat_down_ms"], errors="coerce")
    df["lat_up_ms"]        = pd.to_numeric(df["avg_lat_up_ms"],   errors="coerce")
    # Coğrafi bölge etiketi (0.5° grid)
    df["lon_bin"] = (df["tile_x"] // 2 * 2).astype(int)
    df["lat_bin"] = (df["tile_y"] // 2 * 2 + 0.5).round(1)
    return df, latest


df, DATA_FILE = load_data()

# Harita için örneklenmiş veri (scatter modunda performans)
DF_SAMPLE = df.sample(min(len(df), 20_000), random_state=42)

# ─── STİL SABİTLERİ ──────────────────────────────────────────────────────────

BG_BASE    = "#0d0d1a"
BG_CARD    = "#16162a"
BG_HEADER  = "#12122a"
BORDER_CLR = "#2a2a42"
TEXT_PRI   = "#e8e8f0"
TEXT_MUT   = "#8888aa"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG_CARD,
    plot_bgcolor=BG_CARD,
    font=dict(color=TEXT_PRI, family="Inter, system-ui, sans-serif", size=11),
    margin=dict(l=12, r=12, t=36, b=12),
    xaxis=dict(gridcolor=BORDER_CLR, zerolinecolor=BORDER_CLR, linecolor=BORDER_CLR),
    yaxis=dict(gridcolor=BORDER_CLR, zerolinecolor=BORDER_CLR, linecolor=BORDER_CLR),
    title_font=dict(size=13, color=TEXT_PRI),
)

# Metrik konfigürasyonları
METRICS = {
    "download_mbps": {
        "label":    "📥 Download (Mbps)",
        "short":    "Download",
        "unit":     "Mbps",
        "color":    "#00d4aa",
        "scale":    "Plasma",
        "cap":      500,
    },
    "upload_mbps": {
        "label":    "📤 Upload (Mbps)",
        "short":    "Upload",
        "unit":     "Mbps",
        "color":    "#4facfe",
        "scale":    "Viridis",
        "cap":      200,
    },
    "latency_ms": {
        "label":    "⏱ Gecikme (ms)",
        "short":    "Gecikme",
        "unit":     "ms",
        "color":    "#f6d860",
        "scale":    "RdYlGn_r",
        "cap":      200,
    },
    "lat_down_ms": {
        "label":    "🔽 Yük-altı DL Gecikme (ms)",
        "short":    "DL Gecikme",
        "unit":     "ms",
        "color":    "#fb923c",
        "scale":    "RdYlGn_r",
        "cap":      200,
    },
    "lat_up_ms": {
        "label":    "🔼 Yük-altı UL Gecikme (ms)",
        "short":    "UL Gecikme",
        "unit":     "ms",
        "color":    "#c084fc",
        "scale":    "RdYlGn_r",
        "cap":      200,
    },
    "tests": {
        "label":    "🧪 Test Sayısı",
        "short":    "Test",
        "unit":     "adet",
        "color":    "#a78bfa",
        "scale":    "Blues",
        "cap":      None,
    },
}

# ─── BİLEŞEN YARDIMCILARI ────────────────────────────────────────────────────

def stat_card(icon, title, value, unit, color, card_id=None):
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.Span(icon, style={"fontSize": "1.5rem"}),
                    html.Div([
                        html.P(title, className="mb-0",
                               style={"fontSize": "0.7rem", "color": TEXT_MUT,
                                      "textTransform": "uppercase", "letterSpacing": "0.08em"}),
                        html.H4(
                            id=card_id or f"stat-{title}",
                            children=f"{value} {unit}",
                            className="mb-0 fw-bold",
                            style={"color": color, "fontSize": "1.3rem"},
                        ),
                    ], className="ms-3"),
                ], className="d-flex align-items-center"),
            ]),
            className="border-0 h-100",
            style={"background": BG_CARD, "borderLeft": f"3px solid {color} !important",
                   "borderRadius": "10px"},
        ),
        md=3, className="mb-3",
    )


def section_header(title):
    return html.Div(
        html.H6(title, className="mb-0 fw-semibold",
                style={"color": TEXT_MUT, "fontSize": "0.75rem",
                       "textTransform": "uppercase", "letterSpacing": "0.1em"}),
        className="mb-3 pb-2",
        style={"borderBottom": f"1px solid {BORDER_CLR}"},
    )


# ─── LAYOUT ──────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Ookla TR Analytics"

app.layout = dbc.Container(
    [
        # ── HEADER ────────────────────────────────────────────────────────────
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.Div([
                        html.H4("🇹🇷 Ookla Speedtest — Türkiye Analiz Paneli",
                                className="mb-0 fw-bold",
                                style={"color": TEXT_PRI, "fontSize": "1.1rem"}),
                        html.Small(
                            f"{DATA_FILE}  •  {len(df):,} tile  •  "
                            "Kaynak: Ookla Open Data (CC BY-NC-SA 4.0)",
                            style={"color": TEXT_MUT},
                        ),
                    ]),
                    html.Span("LIVE", className="badge rounded-pill",
                              style={"background": "#00d4aa22", "color": "#00d4aa",
                                     "border": "1px solid #00d4aa55", "padding": "4px 10px"}),
                ], className="d-flex justify-content-between align-items-center py-3"),
            ),
        ),

        html.Hr(style={"borderColor": BORDER_CLR, "margin": "0 0 20px 0"}),

        # ── STAT KARTLARI ─────────────────────────────────────────────────────
        dbc.Row(id="stat-row", children=[
            stat_card("⬇️", "Ort. Download", f"{df.download_mbps.mean():.1f}", "Mbps", "#00d4aa", "stat-dl"),
            stat_card("⬆️", "Ort. Upload",   f"{df.upload_mbps.mean():.1f}",   "Mbps", "#4facfe", "stat-ul"),
            stat_card("🏓", "Ort. Gecikme",  f"{df.latency_ms.mean():.0f}",    "ms",   "#f6d860", "stat-lat"),
            stat_card("📍", "Toplam Tile",   f"{len(df):,}",                   "adet", "#a78bfa", "stat-tiles"),
        ], className="g-3 mb-3"),

        # ── YÜK-ALTI GECİKME ÖZET KARTLARI ───────────────────────────────────
        dbc.Row([
            stat_card("🔽", "Yük-altı DL Gecikme",
                      f"{df.lat_down_ms.mean():.0f}", "ms", "#fb923c", "stat-lat-dl"),
            stat_card("🔼", "Yük-altı UL Gecikme",
                      f"{df.lat_up_ms.mean():.0f}",   "ms", "#c084fc", "stat-lat-ul"),
            stat_card("🧪", "Toplam Test",
                      f"{df.tests.sum():,}",           "test", "#38bdf8", "stat-tests"),
            stat_card("📱", "Toplam Cihaz",
                      f"{df.devices.sum():,}",         "cihaz", "#34d399", "stat-devices"),
        ], className="g-3 mb-4"),

        # ── HARİTA + FİLTRELER ────────────────────────────────────────────────
        dbc.Row([
            # Sol: kontroller
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        section_header("Harita Ayarları"),

                        html.Label("Gösterilecek Metrik", className="text-muted mb-1",
                                   style={"fontSize": "0.8rem"}),
                        dcc.Dropdown(
                            id="metric-select",
                            options=[{"label": v["label"], "value": k}
                                     for k, v in METRICS.items()],
                            value="download_mbps",
                            clearable=False,
                            className="mb-3",
                            style={"fontSize": "0.85rem"},
                        ),

                        html.Label("Harita Modu", className="text-muted mb-1",
                                   style={"fontSize": "0.8rem"}),
                        dbc.RadioItems(
                            id="map-type",
                            options=[
                                {"label": " Yoğunluk (Heatmap)", "value": "density"},
                                {"label": " Nokta (Scatter)",     "value": "scatter"},
                            ],
                            value="density",
                            className="mb-3",
                            inputClassName="me-1",
                        ),

                        html.Hr(style={"borderColor": BORDER_CLR}),
                        section_header("Filtrele"),

                        html.Label("Download Hızı (Mbps)", className="text-muted mb-1",
                                   style={"fontSize": "0.8rem"}),
                        dcc.RangeSlider(
                            id="dl-range",
                            min=0, max=500, step=10,
                            value=[0, 500],
                            marks={0: "0", 100: "100", 250: "250", 500: "500+"},
                            tooltip={"placement": "bottom", "always_visible": False},
                            className="mb-3",
                        ),

                        html.Label("Gecikme (ms)", className="text-muted mb-1",
                                   style={"fontSize": "0.8rem"}),
                        dcc.RangeSlider(
                            id="lat-range",
                            min=0, max=200, step=5,
                            value=[0, 200],
                            marks={0: "0", 50: "50", 100: "100", 200: "200+"},
                            tooltip={"placement": "bottom", "always_visible": False},
                            className="mb-3",
                        ),

                        html.Hr(style={"borderColor": BORDER_CLR}),

                        html.Div(id="filter-info",
                                 className="text-muted",
                                 style={"fontSize": "0.78rem"}),
                    ])
                ], className="border-0 h-100",
                   style={"background": BG_CARD, "borderRadius": "10px"}),
            ], md=3),

            # Sağ: Harita
            dbc.Col([
                dbc.Card([
                    dbc.CardBody(
                        dcc.Graph(
                            id="map-graph",
                            style={"height": "470px"},
                            config={"scrollZoom": True, "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
                        ),
                        className="p-1",
                    )
                ], className="border-0",
                   style={"background": BG_CARD, "borderRadius": "10px"}),
            ], md=9),
        ], className="g-3 mb-4"),

        # ── FOOTER ────────────────────────────────────────────────────────────
        html.Hr(style={"borderColor": BORDER_CLR}),
        html.P(
            "Speedtest® by Ookla® — CC BY-NC-SA 4.0  |  Türkiye Mobil Ağ Performansı",
            className="text-center pb-3",
            style={"color": TEXT_MUT, "fontSize": "0.75rem"},
        ),
    ],
    fluid=True,
    style={
        "background": BG_BASE,
        "minHeight": "100vh",
        "fontFamily": "Inter, system-ui, sans-serif",
        "padding": "0 24px",
    },
)

# ─── CALLBACKS ───────────────────────────────────────────────────────────────

def apply_filters(dl_range, lat_range):
    mask = (
        (df["download_mbps"] >= dl_range[0]) & (df["download_mbps"] <= dl_range[1]) &
        (df["latency_ms"]    >= lat_range[0]) & (df["latency_ms"]    <= lat_range[1])
    )
    return df[mask]


# Filtre bilgisi
@app.callback(
    Output("filter-info", "children"),
    Input("dl-range", "value"),
    Input("lat-range", "value"),
)
def update_filter_info(dl_range, lat_range):
    dff = apply_filters(dl_range, lat_range)
    pct = len(dff) / len(df) * 100
    return [
        html.Span(f"Filtrelenmiş: ", style={"color": TEXT_MUT}),
        html.Span(f"{len(dff):,} tile ", style={"color": TEXT_PRI, "fontWeight": "600"}),
        html.Span(f"({pct:.1f}%)", style={"color": TEXT_MUT}),
    ]


# Harita
@app.callback(
    Output("map-graph", "figure"),
    Input("metric-select", "value"),
    Input("map-type", "value"),
    Input("dl-range", "value"),
    Input("lat-range", "value"),
)
def update_map(metric, map_type, dl_range, lat_range):
    dff = apply_filters(dl_range, lat_range)
    m = METRICS[metric]

    colorbar = dict(
        bgcolor=BG_CARD,
        tickfont=dict(color=TEXT_PRI, size=10),
        title=dict(text=m["unit"], font=dict(color=TEXT_MUT, size=10)),
        thickness=12,
        len=0.75,
    )

    if map_type == "density":
        fig = px.density_mapbox(
            dff, lat="tile_y", lon="tile_x", z=metric,
            radius=7, zoom=5,
            center={"lat": 39.0, "lon": 35.5},
            mapbox_style="carto-darkmatter",
            color_continuous_scale=m["scale"],
            opacity=0.85,
        )
    else:
        sample = dff.sample(min(len(dff), 20_000), random_state=42)
        fig = px.scatter_mapbox(
            sample, lat="tile_y", lon="tile_x",
            color=metric,
            zoom=5,
            center={"lat": 39.0, "lon": 35.5},
            mapbox_style="carto-darkmatter",
            color_continuous_scale=m["scale"],
            opacity=0.75,
        )
        fig.update_traces(marker=dict(size=4))

    fig.update_layout(
        paper_bgcolor=BG_CARD,
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=colorbar,
        uirevision="map",  # zoom/pan korunur
    )
    return fig


# (grafik bölümleri kaldırıldı)
def update_charts(dl_range, lat_range):  # artık çağrılmıyor
    dff = apply_filters(dl_range, lat_range)

    dl_cap  = dff["download_mbps"].quantile(0.99)
    ul_cap  = dff["upload_mbps"].quantile(0.99)
    lat_cap = min(dff["latency_ms"].quantile(0.99), 200)

    # ── Download histogram ────────────────────────────────────────────────────
    hist_dl = px.histogram(
        dff[dff["download_mbps"] <= dl_cap], x="download_mbps", nbins=60,
        color_discrete_sequence=["#00d4aa"],
        title="📥 Download Dağılımı",
        labels={"download_mbps": "Mbps"},
    )
    hist_dl.update_traces(marker_line_width=0, opacity=0.85)
    hist_dl.add_vline(
        x=dff["download_mbps"].median(), line_dash="dash", line_color="#ffffff55",
        annotation_text=f"Med: {dff['download_mbps'].median():.0f}",
        annotation_font_color=TEXT_PRI, annotation_font_size=10,
    )
    hist_dl.update_layout(**PLOTLY_LAYOUT, yaxis_title="Tile Sayısı", height=240)

    # ── Upload histogram ──────────────────────────────────────────────────────
    hist_ul = px.histogram(
        dff[dff["upload_mbps"] <= ul_cap], x="upload_mbps", nbins=60,
        color_discrete_sequence=["#4facfe"],
        title="📤 Upload Dağılımı",
        labels={"upload_mbps": "Mbps"},
    )
    hist_ul.update_traces(marker_line_width=0, opacity=0.85)
    hist_ul.add_vline(
        x=dff["upload_mbps"].median(), line_dash="dash", line_color="#ffffff55",
        annotation_text=f"Med: {dff['upload_mbps'].median():.0f}",
        annotation_font_color=TEXT_PRI, annotation_font_size=10,
    )
    hist_ul.update_layout(**PLOTLY_LAYOUT, yaxis_title="Tile Sayısı", height=240)

    # ── Gecikme histogram ─────────────────────────────────────────────────────
    hist_lat = px.histogram(
        dff[dff["latency_ms"] <= lat_cap], x="latency_ms", nbins=50,
        color_discrete_sequence=["#f6d860"],
        title="⏱ Gecikme Dağılımı",
        labels={"latency_ms": "ms"},
    )
    hist_lat.update_traces(marker_line_width=0, opacity=0.85)
    hist_lat.add_vline(
        x=dff["latency_ms"].median(), line_dash="dash", line_color="#ffffff55",
        annotation_text=f"Med: {dff['latency_ms'].median():.0f}ms",
        annotation_font_color=TEXT_PRI, annotation_font_size=10,
    )
    hist_lat.update_layout(**PLOTLY_LAYOUT, yaxis_title="Tile Sayısı", height=240)

    # ── Download vs Upload scatter ────────────────────────────────────────────
    sample = dff.sample(min(len(dff), 6_000), random_state=42)
    scatter = px.scatter(
        sample, x="download_mbps", y="upload_mbps", color="latency_ms",
        title="📊 Download vs Upload (gecikmeyle renklendirilmiş)",
        labels={"download_mbps": "Download (Mbps)", "upload_mbps": "Upload (Mbps)", "latency_ms": "Gecikme (ms)"},
        color_continuous_scale="RdYlGn_r",
        range_x=[0, dl_cap], range_y=[0, ul_cap],
        opacity=0.55,
    )
    scatter.update_traces(marker=dict(size=3))
    scatter.update_layout(
        **PLOTLY_LAYOUT, height=280,
        coloraxis_colorbar=dict(bgcolor=BG_CARD,
            tickfont=dict(color=TEXT_PRI, size=9),
            title=dict(text="ms", font=dict(color=TEXT_MUT, size=10)),
            thickness=10, len=0.8),
    )

    # ── Box plot: DL / UL hız dağılımı ───────────────────────────────────────
    box_data = pd.DataFrame({
        "Mbps": pd.concat([
            dff.loc[dff["download_mbps"] <= dl_cap, "download_mbps"],
            dff.loc[dff["upload_mbps"]   <= ul_cap, "upload_mbps"],
        ], ignore_index=True),
        "Tür": (["Download"] * (dff["download_mbps"] <= dl_cap).sum() +
                ["Upload"]   * (dff["upload_mbps"]   <= ul_cap).sum()),
    })
    box = px.box(
        box_data, x="Tür", y="Mbps", color="Tür",
        title="📦 Hız Dağılımı (Box Plot)",
        color_discrete_map={"Download": "#00d4aa", "Upload": "#4facfe"},
        points=False,
    )
    box.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False)

    # ════════════════════════════════════════════════════════════════════════════
    # YÜK-ALTI GECİKME SEKMESI
    # ════════════════════════════════════════════════════════════════════════════
    dff_ld = dff.dropna(subset=["lat_down_ms", "lat_up_ms"])
    ld_cap = min(dff_ld["lat_down_ms"].quantile(0.99), 200)
    lu_cap = min(dff_ld["lat_up_ms"].quantile(0.99), 200)

    hist_lat_dl = px.histogram(
        dff_ld[dff_ld["lat_down_ms"] <= ld_cap], x="lat_down_ms", nbins=50,
        color_discrete_sequence=["#fb923c"],
        title="🔽 Yük-altı İndirme Gecikmesi",
        labels={"lat_down_ms": "ms"},
    )
    hist_lat_dl.update_traces(marker_line_width=0, opacity=0.85)
    hist_lat_dl.add_vline(
        x=dff_ld["lat_down_ms"].median(), line_dash="dash", line_color="#ffffff55",
        annotation_text=f"Med: {dff_ld['lat_down_ms'].median():.0f}ms",
        annotation_font_color=TEXT_PRI, annotation_font_size=10,
    )
    hist_lat_dl.update_layout(**PLOTLY_LAYOUT, yaxis_title="Tile", height=260)

    hist_lat_ul = px.histogram(
        dff_ld[dff_ld["lat_up_ms"] <= lu_cap], x="lat_up_ms", nbins=50,
        color_discrete_sequence=["#c084fc"],
        title="🔼 Yük-altı Yükleme Gecikmesi",
        labels={"lat_up_ms": "ms"},
    )
    hist_lat_ul.update_traces(marker_line_width=0, opacity=0.85)
    hist_lat_ul.add_vline(
        x=dff_ld["lat_up_ms"].median(), line_dash="dash", line_color="#ffffff55",
        annotation_text=f"Med: {dff_ld['lat_up_ms'].median():.0f}ms",
        annotation_font_color=TEXT_PRI, annotation_font_size=10,
    )
    hist_lat_ul.update_layout(**PLOTLY_LAYOUT, yaxis_title="Tile", height=260)

    # İndirme vs Yükleme yük-altı gecikme scatter
    samp_ld = dff_ld.sample(min(len(dff_ld), 5_000), random_state=42)
    scatter_lat = px.scatter(
        samp_ld,
        x="lat_down_ms", y="lat_up_ms",
        color="latency_ms",
        opacity=0.5,
        title="🔀 DL vs UL Yük-altı Gecikme",
        labels={"lat_down_ms": "DL Gecikme (ms)", "lat_up_ms": "UL Gecikme (ms)", "latency_ms": "Boşta (ms)"},
        color_continuous_scale="RdYlGn_r",
        range_x=[0, ld_cap], range_y=[0, lu_cap],
    )
    scatter_lat.update_traces(marker=dict(size=3))
    scatter_lat.update_layout(
        **PLOTLY_LAYOUT, height=260,
        coloraxis_colorbar=dict(bgcolor=BG_CARD,
            tickfont=dict(color=TEXT_PRI, size=9),
            title=dict(text="ms", font=dict(color=TEXT_MUT, size=10)),
            thickness=10, len=0.8),
    )

    # Tüm gecikme türleri box plot
    box_lat_data = pd.DataFrame({
        "ms": pd.concat([
            dff.loc[dff["latency_ms"] <= lat_cap, "latency_ms"],
            dff_ld.loc[dff_ld["lat_down_ms"] <= ld_cap, "lat_down_ms"],
            dff_ld.loc[dff_ld["lat_up_ms"]   <= lu_cap, "lat_up_ms"],
        ], ignore_index=True),
        "Tür": (
            ["Boşta (idle)"]     * (dff["latency_ms"]        <= lat_cap).sum() +
            ["Yük-altı İndirme"] * (dff_ld["lat_down_ms"]   <= ld_cap).sum() +
            ["Yük-altı Yükleme"] * (dff_ld["lat_up_ms"]     <= lu_cap).sum()
        ),
    })
    box_lat = px.box(
        box_lat_data, x="Tür", y="ms", color="Tür",
        title="📦 Gecikme Türleri Karşılaştırması",
        color_discrete_map={
            "Boşta (idle)":     "#f6d860",
            "Yük-altı İndirme": "#fb923c",
            "Yük-altı Yükleme": "#c084fc",
        },
        points=False,
    )
    box_lat.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False)

    # ════════════════════════════════════════════════════════════════════════════
    # BÖLGE ANALİZİ SEKMESI
    # ════════════════════════════════════════════════════════════════════════════
    # 2°x2° grid hücrelerine göre ortalama
    region = (
        dff.groupby(["lon_bin", "lat_bin"])
        .agg(
            download_mbps=("download_mbps", "mean"),
            upload_mbps=("upload_mbps",   "mean"),
            latency_ms=("latency_ms",     "mean"),
            tile_count=("download_mbps",  "count"),
        )
        .reset_index()
    )
    region["label"] = region.apply(
        lambda r: f"{r.lat_bin:.1f}°N / {r.lon_bin}°E", axis=1
    )

    # Isı haritası: lon vs lat bazında download ortalaması
    heatmap_pivot = region.pivot_table(
        index="lat_bin", columns="lon_bin", values="download_mbps"
    )
    fig_hm = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns.tolist(),
        y=heatmap_pivot.index.tolist(),
        colorscale="Plasma",
        colorbar=dict(
            bgcolor=BG_CARD,
            tickfont=dict(color=TEXT_PRI, size=9),
            title=dict(text="Mbps", font=dict(color=TEXT_MUT, size=10)),
            thickness=12,
        ),
        hoverongaps=False,
        hovertemplate="Lon: %{x}°E<br>Lat: %{y}°N<br>Download: %{z:.1f} Mbps<extra></extra>",
    ))
    fig_hm.update_layout(
        **PLOTLY_LAYOUT,
        title="🗺️ Bölgesel Download Ortalaması (2°×2° grid)",
        xaxis_title="Boylam (°E)",
        yaxis_title="Enlem (°N)",
        height=350,
    )

    # En iyi/kötü 15 bölge bar chart
    top15 = region.nlargest(15, "download_mbps")
    bot15 = region.nsmallest(15, "download_mbps")
    bar_data = pd.concat([top15.assign(Grup="En Hızlı 15"),
                          bot15.assign(Grup="En Yavaş 15")])
    fig_bar = px.bar(
        bar_data, x="download_mbps", y="label",
        color="Grup",
        orientation="h",
        title="🏆 Bölge Bazlı Download (En Hızlı / En Yavaş)",
        labels={"download_mbps": "Download (Mbps)", "label": ""},
        color_discrete_map={"En Hızlı 15": "#00d4aa", "En Yavaş 15": "#f87171"},
        barmode="group",
    )
    fig_bar.update_layout(**PLOTLY_LAYOUT, height=280)
    fig_bar.update_yaxes(tickfont=dict(size=9), gridcolor=BORDER_CLR)

    # Download vs Gecikme bölge scatter
    fig_reg_sc = px.scatter(
        region, x="download_mbps", y="latency_ms",
        size="tile_count", color="upload_mbps",
        title="📍 Bölge: Download vs Gecikme (boyut=tile, renk=upload)",
        labels={
            "download_mbps": "Download (Mbps)",
            "latency_ms":    "Gecikme (ms)",
            "upload_mbps":   "Upload (Mbps)",
            "tile_count":    "Tile Sayısı",
        },
        color_continuous_scale="Viridis",
        hover_name="label",
        opacity=0.8,
    )
    fig_reg_sc.update_layout(
        **PLOTLY_LAYOUT, height=280,
        coloraxis_colorbar=dict(bgcolor=BG_CARD,
            tickfont=dict(color=TEXT_PRI, size=9),
            title=dict(text="UL Mbps", font=dict(color=TEXT_MUT, size=10)),
            thickness=10, len=0.8),
    )

    return (hist_dl, hist_ul, hist_lat, scatter, box,
            hist_lat_dl, hist_lat_ul, scatter_lat, box_lat,
            fig_hm, fig_bar, fig_reg_sc)


# ─── BAŞLAT ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("Dashboard baslatiliyor -> http://127.0.0.1:8050")
    app.run(debug=False, port=8050)
