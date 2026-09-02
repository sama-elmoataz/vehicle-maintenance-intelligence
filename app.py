import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix,classification_report

import model as ml


st.set_page_config(
    page_title="Vehicle Maintenance Intelligence",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
:root{
    --navy:#0F172A;
    --navy2:#111C35;
    --blue:#2563EB;
    --cyan:#38BDF8;
    --bg:#F4F7FB;
    --card:#FFFFFF;
    --text:#172033;
    --muted:#667085;
    --border:#E7ECF3;
    --green:#22C55E;
    --amber:#F59E0B;
    --red:#EF4444;
}

.stApp{
    background:linear-gradient(180deg,#F8FAFC 0%,#F3F6FB 100%);
    color:var(--text);
}

[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0B1224 0%,#111C35 100%);
    border-right:1px solid rgba(255,255,255,.08);
}

[data-testid="stSidebar"] *{
    color:#EAF0FF;
}

[data-testid="stSidebar"] .stRadio label{
    padding:.35rem 0;
}

.block-container{
    max-width:1450px;
    padding-top:2rem;
    padding-bottom:3rem;
}

h1,h2,h3{
    letter-spacing:-.025em;
}

.hero{
    position:relative;
    overflow:hidden;
    padding:42px 44px;
    border-radius:28px;
    color:white;
    background:
        radial-gradient(circle at 85% 15%,rgba(56,189,248,.35),transparent 28%),
        radial-gradient(circle at 72% 100%,rgba(37,99,235,.42),transparent 35%),
        linear-gradient(135deg,#0B1224 0%,#112247 55%,#153B82 100%);
    box-shadow:0 22px 60px rgba(15,23,42,.18);
    margin-bottom:24px;
}

.hero-kicker{
    display:inline-block;
    font-size:.78rem;
    font-weight:800;
    letter-spacing:.13em;
    text-transform:uppercase;
    color:#9BDCFB;
    margin-bottom:14px;
}

.hero-title{
    font-size:3rem;
    line-height:1.03;
    font-weight:800;
    max-width:760px;
    margin:0 0 16px 0;
}

.hero-copy{
    max-width:690px;
    font-size:1.05rem;
    line-height:1.7;
    color:#CDD9EF;
    margin:0;
}

.hero-orb{
    position:absolute;
    right:-25px;
    bottom:-55px;
    width:300px;
    height:300px;
    border-radius:50%;
    border:1px solid rgba(255,255,255,.12);
    box-shadow:inset 0 0 70px rgba(56,189,248,.12);
}

.hero-car{
    position:absolute;
    right:70px;
    top:54px;
    font-size:6.7rem;
    filter:drop-shadow(0 15px 18px rgba(0,0,0,.25));
}

.section-label{
    color:#2563EB;
    font-size:.77rem;
    font-weight:800;
    letter-spacing:.12em;
    text-transform:uppercase;
    margin-bottom:6px;
}

.section-title{
    font-size:1.75rem;
    font-weight:800;
    margin-bottom:6px;
}

.section-copy{
    color:#667085;
    margin-bottom:20px;
}

.card{
    background:#FFFFFF;
    border:1px solid #E7ECF3;
    border-radius:20px;
    padding:22px;
    box-shadow:0 9px 26px rgba(15,23,42,.055);
    height:100%;
}

.kpi-card{
    background:#FFFFFF;
    border:1px solid #E7ECF3;
    border-radius:18px;
    padding:20px 21px;
    box-shadow:0 8px 25px rgba(15,23,42,.05);
}

.kpi-label{
    color:#7A8599;
    font-size:.78rem;
    font-weight:700;
    text-transform:uppercase;
    letter-spacing:.07em;
}

.kpi-value{
    color:#101828;
    font-size:1.82rem;
    font-weight:800;
    margin-top:4px;
}

.kpi-sub{
    color:#98A2B3;
    font-size:.8rem;
    margin-top:3px;
}

.example-box{
    border:1px solid #D9E4F5;
    background:linear-gradient(180deg,#FFFFFF 0%,#F8FBFF 100%);
    border-radius:18px;
    padding:18px 20px 7px 20px;
    margin-bottom:18px;
}

.result-normal{
    background:linear-gradient(135deg,#ECFDF3,#F6FFFA);
    border:1px solid #BBF7D0;
    color:#166534;
    border-radius:22px;
    padding:26px;
}

.result-minor{
    background:linear-gradient(135deg,#FFFBEB,#FFF7DA);
    border:1px solid #FDE68A;
    color:#92400E;
    border-radius:22px;
    padding:26px;
}

.result-major{
    background:linear-gradient(135deg,#FEF2F2,#FFF7F7);
    border:1px solid #FECACA;
    color:#991B1B;
    border-radius:22px;
    padding:26px;
}

.result-label{
    font-size:.78rem;
    font-weight:800;
    letter-spacing:.1em;
    text-transform:uppercase;
    opacity:.72;
}

.result-title{
    font-size:2rem;
    font-weight:850;
    margin-top:4px;
}

.result-sub{
    margin-top:6px;
    opacity:.82;
}

div[data-testid="stMetric"]{
    background:#FFFFFF;
    border:1px solid #E7ECF3;
    padding:17px 18px;
    border-radius:17px;
    box-shadow:0 8px 22px rgba(15,23,42,.04);
}

.stButton>button{
    border-radius:12px;
    min-height:44px;
    font-weight:750;
    border:1px solid #D7E1EF;
    transition:.15s ease;
}

.stButton>button:hover{
    border-color:#2563EB;
    color:#2563EB;
    transform:translateY(-1px);
}

div[data-testid="stForm"]{
    border:1px solid #E2E8F0;
    border-radius:22px;
    padding:22px;
    background:#FFFFFF;
    box-shadow:0 9px 28px rgba(15,23,42,.045);
}

[data-testid="stDataFrame"]{
    border:1px solid #E7ECF3;
    border-radius:16px;
    overflow:hidden;
}

hr{
    border:none;
    border-top:1px solid #E7ECF3;
    margin:1.5rem 0;
}

.footer{
    margin-top:34px;
    padding-top:18px;
    border-top:1px solid #E5EAF2;
    color:#98A2B3;
    font-size:.78rem;
}

/* ---------- Native Streamlit visibility fixes ---------- */
html, body, [class*="css"]{
    color-scheme:light !important;
}

/* Main page stays light even if the browser/Streamlit uses dark mode */
[data-testid="stAppViewContainer"]{
    background:linear-gradient(180deg,#F8FAFC 0%,#F3F6FB 100%) !important;
}

/* Form/input labels */
[data-testid="stMain"] .stSelectbox label,
[data-testid="stMain"] .stNumberInput label,
[data-testid="stMain"] .stDateInput label,
[data-testid="stMain"] .stTextInput label,
[data-testid="stMain"] .stSlider label,
[data-testid="stMain"] .stMultiSelect label{
    color:#344054 !important;
    font-weight:700 !important;
}

/* Select boxes */
[data-testid="stMain"] div[data-baseweb="select"] > div{
    background:#FFFFFF !important;
    color:#101828 !important;
    border:1px solid #D7DFEA !important;
    border-radius:12px !important;
}

[data-testid="stMain"] div[data-baseweb="select"] span{
    color:#101828 !important;
}

/* Text / numeric / date inputs */
[data-testid="stMain"] input{
    background:#FFFFFF !important;
    color:#101828 !important;
    -webkit-text-fill-color:#101828 !important;
}

[data-testid="stMain"] [data-baseweb="input"]{
    background:#FFFFFF !important;
    border-color:#D7DFEA !important;
    border-radius:12px !important;
}

/* Number input +/- buttons */
[data-testid="stMain"] .stNumberInput button{
    background:#F8FAFC !important;
    color:#475467 !important;
    border:none !important;
}

/* All normal buttons: visible text at all times */
[data-testid="stMain"] .stButton > button{
    background:#0F172A !important;
    color:#FFFFFF !important;
    border:1px solid #0F172A !important;
}

[data-testid="stMain"] .stButton > button p{
    color:#FFFFFF !important;
}

[data-testid="stMain"] .stButton > button:hover{
    background:#1D4ED8 !important;
    color:#FFFFFF !important;
    border-color:#1D4ED8 !important;
}

[data-testid="stMain"] .stButton > button:hover p{
    color:#FFFFFF !important;
}

/* Main prediction button */
[data-testid="stFormSubmitButton"] > button{
    background:linear-gradient(90deg,#2563EB,#1D4ED8) !important;
    color:#FFFFFF !important;
    border:none !important;
    min-height:50px !important;
    font-size:1rem !important;
    box-shadow:0 8px 18px rgba(37,99,235,.22) !important;
}

[data-testid="stFormSubmitButton"] > button p{
    color:#FFFFFF !important;
}

/* Info box text */
[data-testid="stAlert"] p{
    color:#344054 !important;
}

/* Remove dark Streamlit top header strip */
header[data-testid="stHeader"]{
    background:transparent !important;
}

[data-testid="stToolbar"]{
    visibility:visible !important;
}


/* ---------- Final visual overrides ---------- */

/* Keep labels/headings dark and readable on the light page */
[data-testid="stMain"] label,
[data-testid="stMain"] label p,
[data-testid="stMain"] .stMarkdown p,
[data-testid="stMain"] .stMarkdown span{
    color:#344054;
}

/* Restore dark input fields */
[data-testid="stMain"] div[data-baseweb="select"] > div{
    background:#272832 !important;
    color:#FFFFFF !important;
    border:1px solid #353746 !important;
    border-radius:12px !important;
}

[data-testid="stMain"] div[data-baseweb="select"] span,
[data-testid="stMain"] div[data-baseweb="select"] svg{
    color:#FFFFFF !important;
    fill:#FFFFFF !important;
}

[data-testid="stMain"] [data-baseweb="input"],
[data-testid="stMain"] [data-baseweb="base-input"]{
    background:#272832 !important;
    border-color:#353746 !important;
    border-radius:12px !important;
}

[data-testid="stMain"] input{
    background:#272832 !important;
    color:#FFFFFF !important;
    -webkit-text-fill-color:#FFFFFF !important;
    caret-color:#FFFFFF !important;
}

/* Number input stepper buttons */
[data-testid="stMain"] .stNumberInput button{
    background:#272832 !important;
    color:#DDE3EE !important;
    border:none !important;
}

[data-testid="stMain"] .stNumberInput button svg{
    color:#DDE3EE !important;
    fill:#DDE3EE !important;
}

/* Date input calendar icon/text */
[data-testid="stMain"] .stDateInput svg{
    color:#FFFFFF !important;
    fill:#FFFFFF !important;
}

/* Make metric cards readable */
div[data-testid="stMetric"]{
    background:#FFFFFF !important;
    border:1px solid #E7ECF3 !important;
    box-shadow:0 8px 22px rgba(15,23,42,.05) !important;
}

div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"] p{
    color:#667085 !important;
    opacity:1 !important;
    font-weight:700 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] > div{
    color:#101828 !important;
    opacity:1 !important;
    font-weight:800 !important;
}

/* Dropdown menu readability */
div[data-baseweb="popover"]{
    color:#101828 !important;
}

div[data-baseweb="popover"] li{
    color:#101828 !important;
}





/* ---------- Native Streamlit navigation controls ---------- */
header[data-testid="stHeader"]{
    background:rgba(248,250,252,.94) !important;
    border-bottom:1px solid #E7ECF3 !important;
}

[data-testid="stToolbar"]{
    visibility:visible !important;
}

[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"]{
    visibility:visible !important;
    opacity:1 !important;
}


/* ---------- Permanent top navigation ---------- */
section[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"]{
    display:none !important;
}

div[role="radiogroup"]{
    background:#FFFFFF !important;
    border:1px solid #E4EAF2 !important;
    border-radius:15px !important;
    padding:5px !important;
    gap:5px !important;
    box-shadow:0 5px 16px rgba(15,23,42,.04) !important;
}

div[role="radiogroup"] label{
    background:transparent !important;
    border-radius:10px !important;
    padding:7px 13px !important;
}

div[role="radiogroup"] label:has(input:checked){
    background:#0F172A !important;
}

div[role="radiogroup"] label:has(input:checked) p{
    color:#FFFFFF !important;
}

div[role="radiogroup"] label p{
    color:#475467 !important;
    font-weight:750 !important;
}

/* Hide radio circles in the top nav */
div[role="radiogroup"] label > div:first-child{
    display:none !important;
}


/* ---------- KPI cards consistency ---------- */
.kpi-card{
    min-height:152px !important;
    height:152px !important;
    display:flex !important;
    flex-direction:column !important;
    justify-content:flex-start !important;
    box-sizing:border-box !important;
}

.kpi-label{
    min-height:20px !important;
}

.kpi-value{
    font-size:1.78rem !important;
    line-height:1.15 !important;
    min-height:42px !important;
    display:flex !important;
    align-items:center !important;
    white-space:nowrap !important;
}

.kpi-sub{
    margin-top:auto !important;
    padding-top:8px !important;
}

/* ---------- Remove Streamlit top bar ---------- */
header[data-testid="stHeader"]{
    display:none !important;
}

[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"]{
    display:none !important;
}

.block-container{
    padding-top:1.15rem !important;
}


/* ---------- Professional prediction intelligence ---------- */
.intel-card{
    background:#FFFFFF;
    border:1px solid #E5EAF2;
    border-radius:22px;
    padding:23px;
    box-shadow:0 10px 28px rgba(15,23,42,.05);
    height:100%;
}

.intel-eyebrow{
    color:#2563EB;
    text-transform:uppercase;
    letter-spacing:.1em;
    font-size:.72rem;
    font-weight:850;
}

.intel-title{
    color:#101828;
    font-size:1.35rem;
    font-weight:850;
    margin-top:5px;
}

.intel-copy{
    color:#667085;
    font-size:.87rem;
    line-height:1.6;
    margin-top:7px;
}

.risk-pill{
    display:inline-flex;
    align-items:center;
    gap:7px;
    border-radius:999px;
    padding:7px 11px;
    font-weight:800;
    font-size:.77rem;
}

.compare-wrap{
    display:grid;
    grid-template-columns:1fr auto 1fr;
    align-items:center;
    gap:16px;
    margin-top:14px;
}

.compare-card{
    background:#FFFFFF;
    border:1px solid #E5EAF2;
    border-radius:18px;
    padding:20px;
}

.compare-kicker{
    color:#98A2B3;
    font-size:.72rem;
    font-weight:800;
    text-transform:uppercase;
    letter-spacing:.08em;
}

.compare-result{
    color:#101828;
    font-size:1.55rem;
    font-weight:900;
    margin-top:5px;
}

.compare-score{
    color:#667085;
    font-size:.82rem;
    margin-top:5px;
}

.compare-arrow{
    color:#2563EB;
    font-size:1.8rem;
    font-weight:900;
}

.simulator-note{
    background:#F8FAFC;
    border:1px solid #E5EAF2;
    border-radius:15px;
    padding:13px 15px;
    color:#667085;
    font-size:.8rem;
    line-height:1.55;
}


/* ---------- Executive Fleet Dashboard ---------- */
.executive-hero{
    position:relative;
    overflow:hidden;
    min-height:310px;
    padding:42px 46px;
    border-radius:28px;
    color:#FFFFFF;
    background:
        radial-gradient(circle at 84% 20%,rgba(56,189,248,.28),transparent 30%),
        radial-gradient(circle at 72% 120%,rgba(37,99,235,.52),transparent 42%),
        linear-gradient(135deg,#091225 0%,#102044 53%,#164A92 100%);
    box-shadow:0 24px 60px rgba(15,23,42,.18);
    margin-bottom:24px;
}
.executive-hero-kicker{
    color:#8ED8FB;
    text-transform:uppercase;
    letter-spacing:.13em;
    font-size:.76rem;
    font-weight:850;
    margin-bottom:13px;
}
.executive-hero-title{
    max-width:710px;
    font-size:3rem;
    line-height:1.04;
    font-weight:900;
    letter-spacing:-.04em;
    margin-bottom:16px;
}
.executive-hero-copy{
    max-width:660px;
    color:#CED9EC;
    font-size:1rem;
    line-height:1.7;
}
.executive-badges{
    display:flex;
    flex-wrap:wrap;
    gap:9px;
    margin-top:22px;
}
.executive-badge{
    background:rgba(255,255,255,.09);
    border:1px solid rgba(255,255,255,.14);
    border-radius:999px;
    padding:7px 11px;
    color:#EAF2FF;
    font-size:.74rem;
    font-weight:750;
}
.fleet-visual{
    position:absolute;
    right:48px;
    top:46px;
    width:330px;
    height:220px;
}
.fleet-road{
    position:absolute;
    right:-25px;
    bottom:24px;
    width:370px;
    height:3px;
    background:linear-gradient(90deg,transparent,#70B7FF,transparent);
    opacity:.55;
}
.fleet-truck-body{
    position:absolute;
    right:66px;
    top:63px;
    width:186px;
    height:82px;
    border-radius:15px 15px 9px 9px;
    background:linear-gradient(135deg,#F6FAFF,#C9E3FF);
    box-shadow:0 18px 28px rgba(0,0,0,.20);
}
.fleet-truck-cab{
    position:absolute;
    right:17px;
    top:88px;
    width:72px;
    height:57px;
    border-radius:13px 17px 8px 8px;
    background:linear-gradient(135deg,#DCEEFF,#9CCFFF);
    box-shadow:0 18px 28px rgba(0,0,0,.18);
}
.fleet-window{
    position:absolute;
    right:28px;
    top:98px;
    width:37px;
    height:23px;
    border-radius:7px;
    background:#17345F;
    opacity:.9;
}
.fleet-wheel{
    position:absolute;
    top:137px;
    width:31px;
    height:31px;
    border-radius:50%;
    background:#0B1224;
    border:6px solid #53647F;
}
.fleet-wheel.one{right:48px;}
.fleet-wheel.two{right:199px;}
.fleet-signal{
    position:absolute;
    right:267px;
    top:39px;
    width:42px;
    height:42px;
    border-radius:50%;
    background:rgba(34,197,94,.16);
    border:1px solid rgba(34,197,94,.40);
    box-shadow:0 0 0 12px rgba(34,197,94,.05);
}
.fleet-signal:after{
    content:'';
    position:absolute;
    left:14px;
    top:14px;
    width:12px;
    height:12px;
    border-radius:50%;
    background:#22C55E;
}
.health-card{
    background:#FFFFFF;
    border:1px solid #E5EAF2;
    border-radius:19px;
    padding:20px 21px;
    min-height:145px;
    box-shadow:0 8px 24px rgba(15,23,42,.045);
}
.health-dot{
    width:10px;
    height:10px;
    border-radius:50%;
    display:inline-block;
    margin-right:7px;
}
.health-label{
    color:#667085;
    font-size:.77rem;
    font-weight:800;
    text-transform:uppercase;
    letter-spacing:.07em;
}
.health-value{
    color:#101828;
    font-size:1.95rem;
    line-height:1.1;
    font-weight:900;
    margin-top:8px;
}
.health-sub{
    color:#98A2B3;
    font-size:.78rem;
    margin-top:7px;
}
.risk-kpi{
    background:#FFFFFF;
    border:1px solid #E5EAF2;
    border-radius:17px;
    padding:17px 18px;
    box-shadow:0 7px 21px rgba(15,23,42,.04);
}
.risk-kpi-icon{
    font-size:1.28rem;
    margin-bottom:8px;
}
.risk-kpi-label{
    color:#667085;
    font-size:.76rem;
    font-weight:750;
}
.risk-kpi-value{
    color:#101828;
    font-size:1.45rem;
    font-weight:900;
    margin-top:3px;
}
.insight-item{
    display:flex;
    gap:12px;
    align-items:flex-start;
    padding:14px 0;
    border-bottom:1px solid #EEF1F5;
}
.insight-item:last-child{
    border-bottom:none;
}
.insight-number{
    flex:0 0 auto;
    width:29px;
    height:29px;
    border-radius:9px;
    background:#EEF4FF;
    color:#2563EB;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:.74rem;
    font-weight:900;
}
.insight-text{
    color:#475467;
    font-size:.88rem;
    line-height:1.55;
}
.quick-action{
    position:relative;
    overflow:hidden;
    border-radius:22px;
    padding:25px 27px;
    background:linear-gradient(135deg,#EEF4FF,#F8FBFF);
    border:1px solid #D9E5F5;
}
@media (max-width: 950px){
    .fleet-visual{
        opacity:.20;
        right:-70px;
    }
    .executive-hero-title{
        font-size:2.35rem;
    }
}


/* ---------- Hero badge readability ---------- */
.executive-badge{
    background:rgba(37,99,235,.28) !important;
    border:1px solid rgba(147,197,253,.42) !important;
    color:#FFFFFF !important;
    font-weight:800 !important;
    box-shadow:inset 0 1px 0 rgba(255,255,255,.08);
}

.executive-badge,
.executive-badge *{
    color:#FFFFFF !important;
}


/* ---------- Download PDF button ---------- */
[data-testid="stMain"] [data-testid="stDownloadButton"] > button{
    background:#0F172A !important;
    color:#FFFFFF !important;
    border:1px solid #0F172A !important;
    border-radius:12px !important;
    min-height:46px !important;
    font-weight:800 !important;
    box-shadow:0 8px 18px rgba(15,23,42,.12) !important;
}

[data-testid="stMain"] [data-testid="stDownloadButton"] > button p,
[data-testid="stMain"] [data-testid="stDownloadButton"] > button span,
[data-testid="stMain"] [data-testid="stDownloadButton"] > button svg{
    color:#FFFFFF !important;
    fill:#FFFFFF !important;
}

[data-testid="stMain"] [data-testid="stDownloadButton"] > button:hover{
    background:#1D4ED8 !important;
    color:#FFFFFF !important;
    border-color:#1D4ED8 !important;
    transform:translateY(-1px);
}

[data-testid="stMain"] [data-testid="stDownloadButton"] > button:hover p,
[data-testid="stMain"] [data-testid="stDownloadButton"] > button:hover span,
[data-testid="stMain"] [data-testid="stDownloadButton"] > button:hover svg{
    color:#FFFFFF !important;
    fill:#FFFFFF !important;
}

</style>
""",unsafe_allow_html=True)


label_map=dict(
    ml.df[['Maintenance_Level_Code','Maintenance_Level']]
    .drop_duplicates()
    .sort_values('Maintenance_Level_Code')
    .values
)

reverse_label_map={v:k for k,v in label_map.items()}


def safe_number(value,default=0.0):
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except:
        return float(default)


def representative_example(level):
    subset=ml.df[ml.df['Maintenance_Level']==level].copy()

    categorical=[
        'Make_and_Model','Vehicle_Type','Route_Info','Maintenance_Type',
        'Brake_Condition','Weather_Conditions','Road_Conditions'
    ]

    example={}

    for col in ml.x.columns:
        if col=='Last_Maintenance_Date':
            dates=pd.to_datetime(subset[col],errors='coerce').dropna().sort_values()
            example[col]=dates.iloc[len(dates)//2].date()
        elif col in categorical:
            mode=subset[col].mode()
            example[col]=mode.iloc[0] if len(mode)>0 else subset[col].iloc[0]
        else:
            series=pd.to_numeric(subset[col],errors='coerce').dropna()
            value=float(series.median()) if len(series)>0 else 0.0
            if col=='Year_of_Manufacture':
                value=int(round(value))
            example[col]=value

    return example


examples={
    level:representative_example(level)
    for level in label_map.values()
}


input_keys=[
    'Make_and_Model','Year_of_Manufacture','Vehicle_Type','Usage_Hours','Route_Info',
    'Load_Capacity','Actual_Load','Last_Maintenance_Date','Maintenance_Type',
    'Maintenance_Cost','Engine_Temperature','Tire_Pressure','Fuel_Consumption',
    'Battery_Status','Vibration_Levels','Oil_Quality','Brake_Condition',
    'Failure_History','Anomalies_Detected','Predictive_Score',
    'Weather_Conditions','Road_Conditions','Delivery_Times',
    'Downtime_Maintenance','Impact_on_Efficiency','Severity_Score'
]


def build_default_values():
    categorical=[
        'Make_and_Model','Vehicle_Type','Route_Info','Maintenance_Type',
        'Brake_Condition','Weather_Conditions','Road_Conditions'
    ]

    defaults={}

    for col in input_keys:
        if col=='Last_Maintenance_Date':
            dates=pd.to_datetime(ml.df[col],errors='coerce').dropna().sort_values()
            defaults[col]=dates.iloc[len(dates)//2].date()

        elif col in categorical:
            mode=ml.df[col].mode()
            defaults[col]=mode.iloc[0] if len(mode)>0 else ml.df[col].dropna().iloc[0]

        else:
            series=pd.to_numeric(ml.df[col],errors='coerce').dropna()
            value=float(series.median()) if len(series)>0 else 0.0

            if col=='Year_of_Manufacture':
                value=int(round(value))

            defaults[col]=value

    return defaults


def actual_example(level):
    subset=ml.df[ml.df['Maintenance_Level']==level].copy()

    if len(subset)==0:
        return build_default_values()

    # Use a real raw row from the dataset, not invented values.
    row=subset.iloc[len(subset)//2]

    values={}

    for col in input_keys:
        if col=='Last_Maintenance_Date':
            value=pd.to_datetime(row[col],errors='coerce')
            if pd.isna(value):
                value=pd.Timestamp('2026-01-01')
            values[col]=value.date()
        elif col=='Year_of_Manufacture':
            values[col]=int(row[col])
        else:
            values[col]=row[col]

    return values


default_values=build_default_values()

examples={
    level:actual_example(level)
    for level in label_map.values()
}


if 'form_values' not in st.session_state:
    st.session_state.form_values=default_values.copy()

if 'form_version' not in st.session_state:
    st.session_state.form_version=0


def replace_form_values(new_values):
    st.session_state.form_values=new_values.copy()
    st.session_state.form_version+=1


def load_example(level):
    replace_form_values(examples[level])


def reset_form():
    replace_form_values(default_values)


def go_to_prediction():
    st.session_state.nav_page='Predict Maintenance'


def go_to_planner():
    st.session_state.nav_page='Maintenance Planning'


def get_historical_estimate(
    target_column,
    maintenance_level,
    vehicle_type,
    maintenance_type
):
    data=ml.df.copy()

    if target_column=='Downtime_Maintenance':
        data[target_column]=pd.to_numeric(
            data[target_column],
            errors='coerce'
        ).clip(lower=0)
    else:
        data[target_column]=pd.to_numeric(
            data[target_column],
            errors='coerce'
        )

    filters=[
        (
            (data['Maintenance_Level']==maintenance_level)&
            (data['Vehicle_Type']==vehicle_type)&
            (data['Maintenance_Type']==maintenance_type),
            f"{maintenance_level} · {vehicle_type} · {maintenance_type}"
        ),
        (
            (data['Maintenance_Level']==maintenance_level)&
            (data['Maintenance_Type']==maintenance_type),
            f"{maintenance_level} · {maintenance_type}"
        ),
        (
            (data['Maintenance_Level']==maintenance_level)&
            (data['Vehicle_Type']==vehicle_type),
            f"{maintenance_level} · {vehicle_type}"
        ),
        (
            data['Maintenance_Level']==maintenance_level,
            f"{maintenance_level} maintenance records"
        ),
        (
            pd.Series(True,index=data.index),
            "All historical records"
        )
    ]

    selected=None
    scope_label=None

    for mask,label in filters:
        candidate=data.loc[mask,target_column].dropna()

        if len(candidate)>=20:
            selected=candidate
            scope_label=label
            break

    if selected is None or len(selected)==0:
        selected=data[target_column].dropna()
        scope_label="All historical records"

    if len(selected)==0:
        return {
            'median':0.0,
            'q25':0.0,
            'q75':0.0,
            'count':0,
            'scope':scope_label
        }

    return {
        'median':float(selected.median()),
        'q25':float(selected.quantile(.25)),
        'q75':float(selected.quantile(.75)),
        'count':int(len(selected)),
        'scope':scope_label
    }


def get_cost_planning_estimate(
    maintenance_level,
    vehicle_type,
    maintenance_type
):
    data=ml.df.copy()

    data['Maintenance_Cost']=pd.to_numeric(
        data['Maintenance_Cost'],
        errors='coerce'
    )

    valid=data.dropna(subset=['Maintenance_Cost']).copy()

    def stats(series):
        series=series.dropna()

        if len(series)==0:
            return {
                'median':0.0,
                'q25':0.0,
                'q75':0.0,
                'q90':0.0,
                'count':0
            }

        return {
            'median':float(series.median()),
            'q25':float(series.quantile(.25)),
            'q75':float(series.quantile(.75)),
            'q90':float(series.quantile(.90)),
            'count':int(len(series))
        }

    exact=stats(
        valid.loc[
            (valid['Maintenance_Level']==maintenance_level)&
            (valid['Vehicle_Type']==vehicle_type)&
            (valid['Maintenance_Type']==maintenance_type),
            'Maintenance_Cost'
        ]
    )

    level_vehicle=stats(
        valid.loc[
            (valid['Maintenance_Level']==maintenance_level)&
            (valid['Vehicle_Type']==vehicle_type),
            'Maintenance_Cost'
        ]
    )

    level_only=stats(
        valid.loc[
            valid['Maintenance_Level']==maintenance_level,
            'Maintenance_Cost'
        ]
    )

    type_only=stats(
        valid.loc[
            valid['Maintenance_Type']==maintenance_type,
            'Maintenance_Cost'
        ]
    )

    global_stats=stats(
        valid['Maintenance_Cost']
    )

    # Conservative planning estimate:
    # do not rely on one small subgroup median only.
    estimated=max(
        exact['median'],
        level_vehicle['median'],
        level_only['median'],
        type_only['median']
    )

    lower=max(
        exact['q25'],
        level_vehicle['q25'],
        level_only['q25'],
        type_only['q25']
    )

    upper=max(
        exact['q75'],
        level_vehicle['q75'],
        level_only['q75'],
        type_only['q75'],
        estimated
    )

    # The source dataset contains relatively low Oil Change examples.
    # For planning, use a conservative historical benchmark rather than
    # allowing the estimate to fall below the broader dataset baseline.
    if str(maintenance_type).strip().lower()=='oil change':
        estimated=max(
            estimated,
            global_stats['median'],
            type_only['q75']
        )

        lower=max(
            lower,
            global_stats['q25']
        )

        upper=max(
            upper,
            global_stats['q75'],
            type_only['q90'],
            estimated
        )

    lower=min(lower,estimated)
    upper=max(upper,estimated)

    scope=(
        f"{maintenance_level} maintenance · "
        f"{vehicle_type} · {maintenance_type}"
    )

    return {
        'estimated':int(round(estimated)),
        'lower':int(round(lower)),
        'upper':int(round(upper)),
        'count':max(
            exact['count'],
            level_vehicle['count'],
            level_only['count'],
            type_only['count']
        ),
        'scope':scope
    }


def preprocess_for_prediction(values):
    row=pd.DataFrame([{
        'Vehicle_ID':0,
        'Make_and_Model':values['Make_and_Model'],
        'Year_of_Manufacture':values['Year_of_Manufacture'],
        'Vehicle_Type':values['Vehicle_Type'],
        'Usage_Hours':values['Usage_Hours'],
        'Route_Info':values['Route_Info'],
        'Load_Capacity':values['Load_Capacity'],
        'Actual_Load':values['Actual_Load'],
        'Last_Maintenance_Date':values['Last_Maintenance_Date'],
        'Maintenance_Type':values['Maintenance_Type'],
        'Maintenance_Cost':values['Maintenance_Cost'],
        'Engine_Temperature':values['Engine_Temperature'],
        'Tire_Pressure':values['Tire_Pressure'],
        'Fuel_Consumption':values['Fuel_Consumption'],
        'Battery_Status':values['Battery_Status'],
        'Vibration_Levels':values['Vibration_Levels'],
        'Oil_Quality':values['Oil_Quality'],
        'Brake_Condition':values['Brake_Condition'],
        'Failure_History':values['Failure_History'],
        'Anomalies_Detected':values['Anomalies_Detected'],
        'Predictive_Score':values['Predictive_Score'],
        'Weather_Conditions':values['Weather_Conditions'],
        'Road_Conditions':values['Road_Conditions'],
        'Delivery_Times':values['Delivery_Times'],
        'Downtime_Maintenance':values['Downtime_Maintenance'],
        'Impact_on_Efficiency':values['Impact_on_Efficiency'],
        'Severity_Score':values['Severity_Score']
    }])

    row['Vibration_Levels']=row['Vibration_Levels'].clip(lower=0)
    row['Vibration_Levels']=np.log1p(row['Vibration_Levels'])

    row['Oil_Quality']=np.where(row['Oil_Quality']>ml.upper_bound,ml.upper_bound,row['Oil_Quality'])
    row['Oil_Quality']=np.where(row['Oil_Quality']<ml.lower_bound,ml.lower_bound,row['Oil_Quality'])
    row['Oil_Quality']=row['Oil_Quality'].clip(upper=100.0)

    row['Brake_Condition']=row['Brake_Condition'].map({'Poor':0,'Fair':1,'Good':2})

    row['Failure_History']=(row['Failure_History']>=0.5).astype(int)
    row['Anomalies_Detected']=(row['Anomalies_Detected']>=0.5).astype(int)

    row['Predictive_Score']=row['Predictive_Score'].clip(lower=0.0,upper=1.0)
    row['Predictive_Score']=np.log1p(row['Predictive_Score'])

    row['Weather_Conditions']=row['Weather_Conditions'].map({
        'Clear':0,'Rainy':1,'Snowy':2,'Windy':3
    })

    row['Road_Conditions']=row['Road_Conditions'].map({
        'Urban':0,'Rural':1,'Highway':2
    })

    row['Delivery_Times']=np.log1p(row['Delivery_Times'])

    row['Downtime_Maintenance']=row['Downtime_Maintenance'].clip(lower=0.0)
    row['Downtime_Maintenance']=np.log1p(row['Downtime_Maintenance'])

    row['Impact_on_Efficiency']=np.log1p(row['Impact_on_Efficiency'])
    row['Severity_Score']=np.log1p(row['Severity_Score'])

    row=row.drop(columns=['Vehicle_ID'])

    row=ml.target_encoder.transform(row)
    row=row.rename(columns={'Make_and_Model':'Make_and_Model_Encoded'})

    row['Vehicle_Age']=ml.current_year-row['Year_of_Manufacture']
    row=row.drop(columns=['Year_of_Manufacture'])

    row['Vehicle_Type']=row['Vehicle_Type'].map({'Van':0,'Truck':1})

    row['Usage_Hours']=row['Usage_Hours'].clip(lower=0)
    row['Usage_Hours']=np.log1p(row['Usage_Hours'])

    row=row.drop(columns=['Route_Info'])

    row['capacity_ratio']=row['Actual_Load']/row['Load_Capacity']
    row=row.drop(columns=['Load_Capacity','Actual_Load'])

    row['Last_Maintenance_Date']=pd.to_datetime(row['Last_Maintenance_Date'])
    row['Days_Since_Maintenance']=(ml.reference_date-row['Last_Maintenance_Date']).dt.days
    row=row.drop(columns=['Last_Maintenance_Date'])

    encoded=ml.one_hot_encoder.transform(row[['Maintenance_Type']])
    encoded_df=pd.DataFrame(
        encoded,
        columns=ml.encoded_cols,
        index=row.index
    )

    row=pd.concat(
        [row.drop(columns=['Maintenance_Type']),encoded_df],
        axis=1
    )

    row['Maintenance_Cost']=np.log1p(row['Maintenance_Cost'])

    row=row.drop(columns=['Engine_Temperature'])

    row['Tire_Pressure']=np.log1p(row['Tire_Pressure'])
    row['Fuel_Consumption']=np.log1p(row['Fuel_Consumption'])

    row['Is_Battery_Elevated']=(row['Battery_Status']>47.5).astype(int)
    row=row.drop(columns=['Battery_Status'])

    row['Is_Overloaded']=(row['capacity_ratio']>1.0).astype(int)

    row=row.reindex(columns=ml.x_train.columns,fill_value=0)

    return row



def calculate_risk_score(probabilities):
    weights={
        'Normal':0,
        'Minor':50,
        'Major':100
    }

    score=0.0

    for class_code,probability in zip(ml.rf.classes_,probabilities):
        label=label_map[int(class_code)]
        score+=float(probability)*weights.get(label,50)

    return float(np.clip(score,0,100))


def risk_level_details(score):
    if score<35:
        return "Low","#22C55E","Vehicle risk is currently within the lower range."
    elif score<70:
        return "Moderate","#F59E0B","The vehicle shows a moderate maintenance-risk profile."
    else:
        return "High","#EF4444","The vehicle shows a high maintenance-risk profile."


def make_risk_gauge(score):
    from matplotlib.patches import Wedge

    fig,ax=plt.subplots(figsize=(6.2,3.2))

    segments=[
        (120,180,"#22C55E"),
        (60,120,"#F59E0B"),
        (0,60,"#EF4444")
    ]

    for theta1,theta2,color in segments:
        ax.add_patch(
            Wedge(
                (0,0),
                1,
                theta1,
                theta2,
                width=.22,
                facecolor=color,
                edgecolor="white",
                linewidth=2
            )
        )

    angle=np.deg2rad(180-(score/100)*180)
    needle_x=.72*np.cos(angle)
    needle_y=.72*np.sin(angle)

    ax.plot(
        [0,needle_x],
        [0,needle_y],
        linewidth=4,
        solid_capstyle="round",
        color="#0F172A"
    )

    ax.scatter(
        [0],
        [0],
        s=120,
        color="#0F172A",
        zorder=5
    )

    ax.text(
        0,
        .26,
        f"{score:.0f}",
        ha="center",
        va="center",
        fontsize=29,
        fontweight="bold",
        color="#0F172A"
    )

    ax.text(
        0,
        .08,
        "RISK SCORE",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#667085"
    )

    ax.text(-.94,-.08,"LOW",ha="left",fontsize=8,fontweight="bold",color="#667085")
    ax.text(0,-.08,"MODERATE",ha="center",fontsize=8,fontweight="bold",color="#667085")
    ax.text(.94,-.08,"HIGH",ha="right",fontsize=8,fontweight="bold",color="#667085")

    ax.set_xlim(-1.12,1.12)
    ax.set_ylim(-.14,1.08)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    plt.tight_layout(pad=.2)

    return fig


def get_local_sensitivity(processed,predicted_class):
    class_index=list(ml.rf.classes_).index(predicted_class)
    original_probability=float(
        ml.rf.predict_proba(processed)[0][class_index]
    )

    numeric_baseline=ml.x_train.median(numeric_only=True)

    rows=[]

    for feature in processed.columns:
        if feature not in numeric_baseline.index:
            continue

        baseline_value=numeric_baseline[feature]

        if pd.isna(baseline_value):
            continue

        changed=processed.astype(float).copy()
        changed.loc[:,feature]=float(baseline_value)

        changed_probability=float(
            ml.rf.predict_proba(changed)[0][class_index]
        )

        signed_change=original_probability-changed_probability

        rows.append({
            "Feature":feature.replace("_"," "),
            "Impact":abs(signed_change),
            "Direction":"Supports prediction" if signed_change>=0 else "Offsets prediction"
        })

    sensitivity=pd.DataFrame(rows)

    if len(sensitivity)==0:
        return sensitivity

    return sensitivity.sort_values(
        "Impact",
        ascending=False
    ).head(5)


def render_sensitivity_bars(sensitivity):
    if sensitivity.empty:
        st.info("Local sensitivity could not be calculated for this prediction.")
        return

    max_impact=max(float(sensitivity['Impact'].max()),1e-9)

    for _,row in sensitivity.iterrows():
        impact=float(row['Impact'])
        normalized=float(np.clip(impact/max_impact,0,1))
        direction=row['Direction']
        feature=row['Feature']

        c1,c2=st.columns([0.72,0.28])

        with c1:
            st.markdown(f"**{feature}**")

        with c2:
            if direction=="Supports prediction":
                st.caption("🔵 Supports prediction")
            else:
                st.caption("⚪ Offsets prediction")

        st.progress(
            normalized,
            text=f"{impact*100:.2f} percentage-point probability sensitivity"
        )

        st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)


def evaluate_vehicle(values):
    processed=preprocess_for_prediction(values)

    prediction=int(ml.rf.predict(processed)[0])
    probabilities=ml.rf.predict_proba(processed)[0]
    prediction_label=label_map[prediction]

    class_index=list(ml.rf.classes_).index(prediction)
    confidence=float(probabilities[class_index])

    risk_score=calculate_risk_score(probabilities)
    risk_level,risk_color,risk_message=risk_level_details(risk_score)

    return {
        "values":values,
        "processed":processed,
        "prediction":prediction,
        "prediction_label":prediction_label,
        "probabilities":probabilities,
        "confidence":confidence,
        "risk_score":risk_score,
        "risk_level":risk_level,
        "risk_color":risk_color,
        "risk_message":risk_message
    }



def row_to_raw_values(row):
    values={}

    for col in input_keys:
        if col=='Last_Maintenance_Date':
            value=pd.to_datetime(row[col],errors='coerce')
            if pd.isna(value):
                value=pd.Timestamp('2026-01-01')
            values[col]=value.date()

        elif col=='Year_of_Manufacture':
            values[col]=int(row[col])

        else:
            values[col]=row[col]

    return values


def build_midrange_examples():
    target_scores={
        'Normal':22.0,
        'Minor':60.0,
        'Major':82.0
    }

    processed_all=pd.concat(
        [ml.x_train,ml.x_test],
        axis=0
    ).sort_index()

    probabilities=ml.rf.predict_proba(processed_all)
    predicted_codes=ml.rf.predict(processed_all)

    risk_scores=[]

    for probs in probabilities:
        risk_scores.append(
            calculate_risk_score(probs)
        )

    score_df=pd.DataFrame(
        {
            'Predicted_Code':predicted_codes,
            'Risk_Score':risk_scores
        },
        index=processed_all.index
    )

    selected_examples={}

    for level,target in target_scores.items():
        class_code=reverse_label_map[level]

        true_level=ml.df.loc[
            score_df.index,
            'Maintenance_Level'
        ]

        candidates=score_df[
            (score_df['Predicted_Code']==class_code)&
            (true_level==level)
        ].copy()

        if len(candidates)==0:
            selected_examples[level]=actual_example(level)
            continue

        chosen_index=(
            candidates['Risk_Score']
            .sub(target)
            .abs()
            .idxmin()
        )

        selected_examples[level]=row_to_raw_values(
            ml.df.loc[chosen_index]
        )

    return selected_examples


def build_condition_snapshot(values):
    load_capacity=max(float(values['Load_Capacity']),.01)
    actual_load=float(values['Actual_Load'])
    load_ratio=actual_load/load_capacity

    brake=str(values['Brake_Condition'])
    anomalies=float(values['Anomalies_Detected'])>=.5
    failure=float(values['Failure_History'])>=.5
    battery_elevated=float(values['Battery_Status'])>47.5

    vibration=float(values['Vibration_Levels'])
    vibration_series=pd.to_numeric(
        ml.df['Vibration_Levels'],
        errors='coerce'
    ).dropna()

    vibration_q75=float(
        vibration_series.quantile(.75)
    ) if len(vibration_series)>0 else vibration

    if load_ratio>1:
        load_value="Overloaded"
        load_status="Attention"
        load_detail=f"{load_ratio:.2f}x of rated capacity"
    else:
        load_value="Within Capacity"
        load_status="Good"
        load_detail=f"{load_ratio:.2f}x of rated capacity"

    if brake=='Good':
        brake_status="Good"
    elif brake=='Fair':
        brake_status="Monitor"
    else:
        brake_status="Attention"

    if vibration>vibration_q75:
        vibration_value="Above Typical"
        vibration_status="Monitor"
    else:
        vibration_value="Typical Range"
        vibration_status="Good"

    snapshot=[
        {
            'label':'Load Status',
            'value':load_value,
            'status':load_status,
            'detail':load_detail
        },
        {
            'label':'Brake Condition',
            'value':brake,
            'status':brake_status,
            'detail':'Current condition'
        },
        {
            'label':'Anomalies',
            'value':'Detected' if anomalies else 'Not Detected',
            'status':'Attention' if anomalies else 'Good',
            'detail':'Anomaly flag detected' if anomalies else 'No anomaly flag detected'
        },
        {
            'label':'Vibration',
            'value':vibration_value,
            'status':vibration_status,
            'detail':'Higher than the typical fleet range' if vibration>vibration_q75 else 'Within the typical fleet range'
        },
        {
            'label':'Failure History',
            'value':'Previous Failure' if failure else 'No Failure Flag',
            'status':'Attention' if failure else 'Good',
            'detail':'Previous failure recorded' if failure else 'No previous failure recorded'
        },
        {
            'label':'Battery',
            'value':'Elevated' if battery_elevated else 'Normal',
            'status':'Monitor' if battery_elevated else 'Good',
            'detail':'Elevated battery reading' if battery_elevated else 'Battery reading within expected range'
        }
    ]

    return snapshot


def planning_priority_for_prediction(prediction_label):
    return {
        'Normal':'Routine',
        'Minor':'Medium',
        'Major':'High'
    }.get(prediction_label,'Medium')


def build_prediction_pdf(result,snapshot,priority):
    buffer=BytesIO()

    values=result['values']
    prediction=result['prediction_label']
    confidence=result['confidence']*100
    risk_score=result['risk_score']

    probability_rows=[
        (
            label_map[int(class_code)],
            float(probability)*100
        )
        for class_code,probability
        in zip(ml.rf.classes_,result['probabilities'])
    ]

    with PdfPages(buffer) as pdf:
        fig=plt.figure(figsize=(8.27,11.69))
        fig.patch.set_facecolor('white')

        fig.text(
            .08,.94,
            "Vehicle Maintenance Intelligence",
            fontsize=22,
            fontweight='bold',
            color='#0F172A'
        )

        fig.text(
            .08,.91,
            "Predictive Maintenance Report",
            fontsize=11,
            color='#64748B'
        )

        fig.text(
            .08,.865,
            "VEHICLE",
            fontsize=8,
            fontweight='bold',
            color='#2563EB'
        )
        fig.text(
            .08,.84,
            str(values['Make_and_Model']),
            fontsize=15,
            fontweight='bold',
            color='#101828'
        )
        fig.text(
            .08,.815,
            f"Vehicle type: {values['Vehicle_Type']}",
            fontsize=10,
            color='#475467'
        )

        fig.text(
            .08,.765,
            "PREDICTION SUMMARY",
            fontsize=8,
            fontweight='bold',
            color='#2563EB'
        )

        summary_lines=[
            f"Maintenance level: {prediction}",
            f"Risk score: {risk_score:.0f}/100",
            f"Prediction confidence: {confidence:.1f}%",
            f"Planning priority: {priority}"
        ]

        y=.735
        for line in summary_lines:
            fig.text(
                .09,y,line,
                fontsize=11,
                color='#101828'
            )
            y-=.031

        fig.text(
            .08,.585,
            "CLASS PROBABILITY BREAKDOWN",
            fontsize=8,
            fontweight='bold',
            color='#2563EB'
        )

        y=.555
        for label,probability in probability_rows:
            fig.text(
                .09,y,
                f"{label}: {probability:.1f}%",
                fontsize=10,
                color='#334155'
            )
            y-=.028

        fig.text(
            .08,.445,
            "VEHICLE CONDITION SNAPSHOT",
            fontsize=8,
            fontweight='bold',
            color='#2563EB'
        )

        y=.415
        for item in snapshot:
            fig.text(
                .09,y,
                f"{item['label']}: {item['value']}",
                fontsize=10,
                fontweight='bold',
                color='#101828'
            )
            fig.text(
                .47,y,
                f"{item['status']} - {item['detail']}",
                fontsize=8.5,
                color='#64748B'
            )
            y-=.043

        fig.text(
            .08,.11,
            "This report summarizes the trained model output and the vehicle inputs. "
            "Condition flags are descriptive and are not causal mechanical diagnoses.",
            fontsize=8.5,
            color='#64748B',
            wrap=True
        )

        fig.text(
            .08,.075,
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
            fontsize=8,
            color='#94A3B8'
        )

        plt.axis('off')
        pdf.savefig(fig,bbox_inches='tight')
        plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()


# Override the earlier examples with real dataset rows selected
# to produce less-extreme, mid-range risk scores where available.
examples=build_midrange_examples()


def page_header(kicker,title,copy):
    st.markdown(
        f"""
        <div class="section-label">{kicker}</div>
        <div class="section-title">{title}</div>
        <div class="section-copy">{copy}</div>
        """,
        unsafe_allow_html=True
    )



pages=['Dashboard','Predict Maintenance','Maintenance Planning','Model Performance','Data Insights']

if 'nav_page' not in st.session_state:
    st.session_state.nav_page='Dashboard'

st.markdown(
    """
    <div style="
        display:flex;
        align-items:center;
        justify-content:space-between;
        margin-bottom:4px;
    ">
        <div>
            <div style="font-size:1.12rem;font-weight:850;color:#0F172A;">
                🚘 VM Intelligence
            </div>
            <div style="font-size:.76rem;color:#667085;margin-top:2px;">
                Predictive maintenance platform
            </div>
        </div>
        <div style="
            font-size:.73rem;
            font-weight:750;
            color:#475467;
            background:#EEF4FF;
            border:1px solid #D6E4FF;
            padding:7px 12px;
            border-radius:999px;
        ">
            Random Forest · """ + f"{len(ml.x_train):,}" + """ training records
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

page=st.radio(
    "Main navigation",
    pages,
    horizontal=True,
    key='nav_page',
    label_visibility='collapsed'
)

st.markdown("<div style='height:10px'></div>",unsafe_allow_html=True)


if page=='Dashboard':
    total_vehicles=len(ml.df)

    maintenance_counts=(
        ml.df['Maintenance_Level']
        .value_counts()
        .reindex(['Normal','Minor','Major'])
        .fillna(0)
        .astype(int)
    )

    normal_count=int(maintenance_counts.get('Normal',0))
    minor_count=int(maintenance_counts.get('Minor',0))
    major_count=int(maintenance_counts.get('Major',0))

    normal_pct=(normal_count/total_vehicles*100) if total_vehicles else 0
    minor_pct=(minor_count/total_vehicles*100) if total_vehicles else 0
    major_pct=(major_count/total_vehicles*100) if total_vehicles else 0

    attention_count=minor_count+major_count
    attention_pct=(attention_count/total_vehicles*100) if total_vehicles else 0

    anomaly_binary=(
        pd.to_numeric(
            ml.df['Anomalies_Detected'],
            errors='coerce'
        ).fillna(0)>=0.5
    )
    anomaly_pct=float(anomaly_binary.mean()*100)

    poor_brake_pct=float(
        (
            ml.df['Brake_Condition']
            .astype(str)
            .str.strip()
            .str.lower()
            .eq('poor')
        ).mean()*100
    )

    load_capacity=pd.to_numeric(
        ml.df['Load_Capacity'],
        errors='coerce'
    )
    actual_load=pd.to_numeric(
        ml.df['Actual_Load'],
        errors='coerce'
    )
    valid_load=load_capacity.gt(0)&load_capacity.notna()&actual_load.notna()

    overloaded_pct=float(
        (
            actual_load[valid_load]>load_capacity[valid_load]
        ).mean()*100
    ) if valid_load.any() else 0.0

    maintenance_dates=pd.to_datetime(
        ml.df['Last_Maintenance_Date'],
        errors='coerce'
    )
    days_since_service=(
        ml.reference_date-maintenance_dates
    ).dt.days

    avg_days_since=float(
        days_since_service[
            days_since_service.notna()
        ].clip(lower=0).mean()
    )

    hero_html=f'''<div class="executive-hero">
<div class="executive-hero-kicker">Predictive Fleet Intelligence</div>
<div class="executive-hero-title">See fleet risk before it becomes downtime.</div>
<div class="executive-hero-copy">
Monitor maintenance exposure across the fleet, identify the strongest risk signals,
and move from reactive servicing to data-driven maintenance decisions.
</div>
<div class="executive-badges">
<span class="executive-badge">{total_vehicles:,} vehicles analyzed</span>
<span class="executive-badge">Random Forest engine</span>
<span class="executive-badge">3 maintenance levels</span>
</div>
<div class="fleet-visual">
<div class="fleet-signal"></div>
<div class="fleet-road"></div>
<div class="fleet-truck-body"></div>
<div class="fleet-truck-cab"></div>
<div class="fleet-window"></div>
<div class="fleet-wheel one"></div>
<div class="fleet-wheel two"></div>
</div>
</div>'''

    st.markdown(hero_html,unsafe_allow_html=True)

    page_header(
        "Fleet health",
        "Maintenance status at a glance",
        "A business view of how the fleet is distributed across maintenance levels."
    )

    k1,k2,k3,k4=st.columns(4)

    with k1:
        st.markdown(
            f'''<div class="health-card">
<div class="health-label">Total Fleet</div>
<div class="health-value">{total_vehicles:,}</div>
<div class="health-sub">Vehicle records monitored</div>
</div>''',
            unsafe_allow_html=True
        )

    with k2:
        st.markdown(
            f'''<div class="health-card">
<div class="health-label"><span class="health-dot" style="background:#22C55E;"></span>Normal</div>
<div class="health-value">{normal_pct:.1f}%</div>
<div class="health-sub">{normal_count:,} vehicles</div>
</div>''',
            unsafe_allow_html=True
        )

    with k3:
        st.markdown(
            f'''<div class="health-card">
<div class="health-label"><span class="health-dot" style="background:#F59E0B;"></span>Minor</div>
<div class="health-value">{minor_pct:.1f}%</div>
<div class="health-sub">{minor_count:,} vehicles</div>
</div>''',
            unsafe_allow_html=True
        )

    with k4:
        st.markdown(
            f'''<div class="health-card">
<div class="health-label"><span class="health-dot" style="background:#EF4444;"></span>Major</div>
<div class="health-value">{major_pct:.1f}%</div>
<div class="health-sub">{major_count:,} vehicles</div>
</div>''',
            unsafe_allow_html=True
        )

    st.markdown("<br>",unsafe_allow_html=True)

    left,right=st.columns([.86,1.14],gap="large")

    with left:
        page_header(
            "Maintenance overview",
            "Fleet status distribution",
            "Share of vehicles in Normal, Minor and Major maintenance states."
        )

        fig,ax=plt.subplots(figsize=(5.5,4.8))

        donut_values=[
            normal_count,
            minor_count,
            major_count
        ]

        donut_labels=[
            'Normal',
            'Minor',
            'Major'
        ]

        donut_colors=[
            '#22C55E',
            '#F59E0B',
            '#EF4444'
        ]

        wedges,_,autotexts=ax.pie(
            donut_values,
            labels=None,
            autopct='%1.1f%%',
            startangle=90,
            colors=donut_colors,
            pctdistance=.82,
            wedgeprops={
                'width':.34,
                'edgecolor':'white',
                'linewidth':3
            }
        )

        percent_text_colors=[
            '#FFFFFF',
            '#101828',
            '#FFFFFF'
        ]

        for autotext,text_color in zip(
            autotexts,
            percent_text_colors
        ):
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')
            autotext.set_color(text_color)

        ax.text(
            0,
            .05,
            f"{attention_pct:.1f}%",
            ha='center',
            va='center',
            fontsize=28,
            fontweight='bold',
            color='#101828'
        )

        ax.text(
            0,
            -.13,
            "NEED ATTENTION",
            ha='center',
            va='center',
            fontsize=8.5,
            fontweight='bold',
            color='#667085'
        )

        legend_labels=[
            f"Normal  {normal_pct:.1f}%",
            f"Minor  {minor_pct:.1f}%",
            f"Major  {major_pct:.1f}%"
        ]

        legend=ax.legend(
            wedges,
            legend_labels,
            loc='lower center',
            bbox_to_anchor=(.5,-.10),
            ncol=3,
            frameon=False,
            fontsize=9.5,
            handlelength=1,
            handletextpad=.5,
            columnspacing=1.4
        )

        for text in legend.get_texts():
            text.set_color('#344054')
            text.set_fontweight('semibold')

        ax.set_aspect('equal')
        fig.patch.set_alpha(0)
        plt.tight_layout()
        plt.subplots_adjust(bottom=.16)

        st.pyplot(fig,use_container_width=True)
        plt.close(fig)

    with right:
        page_header(
            "Risk indicators",
            "Operational warning signals",
            "Four fleet-level indicators derived directly from vehicle condition and maintenance data."
        )

        r1,r2=st.columns(2)

        with r1:
            st.markdown(
                f'''<div class="risk-kpi">
<div class="risk-kpi-icon">⚠️</div>
<div class="risk-kpi-label">Anomalies Detected</div>
<div class="risk-kpi-value">{anomaly_pct:.1f}%</div>
</div>''',
                unsafe_allow_html=True
            )

        with r2:
            st.markdown(
                f'''<div class="risk-kpi">
<div class="risk-kpi-icon">🛑</div>
<div class="risk-kpi-label">Poor Brake Condition</div>
<div class="risk-kpi-value">{poor_brake_pct:.1f}%</div>
</div>''',
                unsafe_allow_html=True
            )

        st.markdown("<div style='height:14px'></div>",unsafe_allow_html=True)

        r3,r4=st.columns(2)

        with r3:
            st.markdown(
                f'''<div class="risk-kpi">
<div class="risk-kpi-icon">📦</div>
<div class="risk-kpi-label">Overloaded Vehicles</div>
<div class="risk-kpi-value">{overloaded_pct:.1f}%</div>
</div>''',
                unsafe_allow_html=True
            )

        with r4:
            st.markdown(
                f'''<div class="risk-kpi">
<div class="risk-kpi-icon">🗓️</div>
<div class="risk-kpi-label">Avg. Days Since Maintenance</div>
<div class="risk-kpi-value">{avg_days_since:,.0f}</div>
</div>''',
                unsafe_allow_html=True
            )

        st.markdown("<br>",unsafe_allow_html=True)

        st.markdown(
            f'''<div class="intel-card">
<div class="intel-eyebrow">Fleet attention</div>
<div class="intel-title">{attention_count:,} vehicles require attention</div>
<div class="intel-copy">
Minor and Major classes together represent <b>{attention_pct:.1f}%</b> of the current dataset.
</div>
</div>''',
            unsafe_allow_html=True
        )

    st.markdown("<br><br>",unsafe_allow_html=True)

    page_header(
        "Fleet segmentation",
        "Maintenance risk by vehicle type",
        "Compare the maintenance-level mix across vehicle categories."
    )

    vehicle_mix=pd.crosstab(
        ml.df['Vehicle_Type'],
        ml.df['Maintenance_Level'],
        normalize='index'
    ).mul(100)

    vehicle_mix=vehicle_mix.reindex(
        columns=['Normal','Minor','Major'],
        fill_value=0
    )

    st.bar_chart(
        vehicle_mix,
        height=350
    )

    st.caption(
        "Values are percentages within each vehicle type, so each vehicle-type row represents its own maintenance distribution."
    )

    st.markdown("<br><br>",unsafe_allow_html=True)

    page_header(
        "Operational insights",
        "What stands out in the fleet",
        "Automatically generated observations from the current dataset."
    )

    major_by_type=(
        pd.crosstab(
            ml.df['Vehicle_Type'],
            ml.df['Maintenance_Level'],
            normalize='index'
        )
        .get('Major',pd.Series(dtype=float))
        .mul(100)
    )

    if len(major_by_type)>0:
        highest_major_type=str(major_by_type.idxmax())
        highest_major_pct=float(major_by_type.max())
    else:
        highest_major_type="N/A"
        highest_major_pct=0.0

    most_common_maintenance=(
        ml.df['Maintenance_Type']
        .mode()
        .iloc[0]
        if not ml.df['Maintenance_Type'].mode().empty
        else "N/A"
    )

    usage_by_level=(
        ml.df.groupby('Maintenance_Level')['Usage_Hours']
        .mean()
    )

    if len(usage_by_level)>0:
        highest_usage_level=str(usage_by_level.idxmax())
        highest_usage_value=float(usage_by_level.max())
    else:
        highest_usage_level="N/A"
        highest_usage_value=0.0

    insights=[
        f"<b>{highest_major_type}</b> has the highest Major-maintenance share at <b>{highest_major_pct:.1f}%</b> within its vehicle type.",
        f"<b>{anomaly_pct:.1f}%</b> of vehicle records contain a detected anomaly.",
        f"<b>{most_common_maintenance}</b> is the most common recorded maintenance type in the dataset.",
        f"<b>{highest_usage_level}</b> vehicles have the highest average usage at approximately <b>{highest_usage_value:,.0f} hours</b>."
    ]

    insight_html='<div class="intel-card">'

    for i,text in enumerate(insights,1):
        insight_html+=(
            '<div class="insight-item">'
            f'<div class="insight-number">{i}</div>'
            f'<div class="insight-text">{text}</div>'
            '</div>'
        )

    insight_html+='</div>'

    st.markdown(
        insight_html,
        unsafe_allow_html=True
    )

    st.markdown("<br><br>",unsafe_allow_html=True)

    action_left,action_right=st.columns([1.25,.75],gap="large")

    with action_left:
        st.markdown(
            '''<div class="quick-action">
<div class="section-label">Prediction lab</div>
<div style="font-size:1.55rem;font-weight:900;color:#101828;margin-top:5px;">
Analyze an individual vehicle
</div>
<div style="color:#667085;line-height:1.65;margin-top:7px;max-width:700px;">
Enter vehicle condition and operational data, load a ready-made Normal / Minor / Major example,
then review the risk score, probability breakdown and vehicle condition snapshot.
</div>
</div>''',
            unsafe_allow_html=True
        )

    with action_right:
        st.markdown("<div style='height:27px'></div>",unsafe_allow_html=True)

        st.button(
            "Analyze a Vehicle  →",
            use_container_width=True,
            on_click=go_to_prediction,
            key="dashboard_prediction_cta"
        )


elif page=='Predict Maintenance':
    page_header(
        "Prediction lab",
        "Predict a vehicle's maintenance level",
        "Enter vehicle information manually or load a realistic auto-fill example."
    )

    st.markdown(
        '''
        <div class="example-box">
            <div style="font-size:1rem;font-weight:800;color:#101828;">
                Try an auto-fill example
            </div>
            <div style="color:#667085;font-size:.86rem;margin-top:4px;">
                Each preset uses representative values from that maintenance class in your dataset.
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    ex1,ex2,ex3,ex4=st.columns(4)

    available_levels=list(label_map.values())

    with ex1:
        if len(available_levels)>0:
            st.button(
                "🟢 Normal Example",
                use_container_width=True,
                key="normal_example_button",
                on_click=load_example,
                args=(available_levels[0],)
            )

    with ex2:
        if len(available_levels)>1:
            st.button(
                "🟠 Minor Example",
                use_container_width=True,
                key="minor_example_button",
                on_click=load_example,
                args=(available_levels[1],)
            )

    with ex3:
        if len(available_levels)>2:
            st.button(
                "🔴 Major Example",
                use_container_width=True,
                key="major_example_button",
                on_click=load_example,
                args=(available_levels[2],)
            )

    with ex4:
        st.button(
            "↺ Reset Form",
            use_container_width=True,
            key="reset_form_button",
            on_click=reset_form
        )

    st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)

    categorical_values={
        col:sorted(ml.df[col].dropna().astype(str).unique().tolist())
        for col in [
            'Make_and_Model','Vehicle_Type','Route_Info','Maintenance_Type',
            'Brake_Condition','Weather_Conditions','Road_Conditions'
        ]
    }

    fv=st.session_state.form_values
    v=st.session_state.form_version

    def option_index(options,value):
        value=str(value)
        return options.index(value) if value in options else 0

    with st.form(f"prediction_form_{v}"):
        st.markdown("### Vehicle & Usage")
        a,b,c=st.columns(3)

        with a:
            make_model=st.selectbox(
                "Make & Model",
                categorical_values['Make_and_Model'],
                index=option_index(
                    categorical_values['Make_and_Model'],
                    fv['Make_and_Model']
                ),
                key=f'Make_and_Model_{v}'
            )

            year=int(st.number_input(
                "Year of Manufacture",
                min_value=1980,
                max_value=2026,
                value=int(fv['Year_of_Manufacture']),
                step=1,
                key=f'Year_of_Manufacture_{v}'
            ))

            vehicle_type=st.selectbox(
                "Vehicle Type",
                categorical_values['Vehicle_Type'],
                index=option_index(
                    categorical_values['Vehicle_Type'],
                    fv['Vehicle_Type']
                ),
                key=f'Vehicle_Type_{v}'
            )

        with b:
            usage=st.number_input(
                "Usage Hours",
                min_value=0.0,
                value=max(0.0,float(fv['Usage_Hours'])),
                step=100.0,
                key=f'Usage_Hours_{v}'
            )

            capacity=st.number_input(
                "Load Capacity",
                min_value=0.01,
                value=max(0.01,float(fv['Load_Capacity'])),
                step=1.0,
                key=f'Load_Capacity_{v}'
            )

            actual_load=st.number_input(
                "Actual Load",
                min_value=0.0,
                value=max(0.0,float(fv['Actual_Load'])),
                step=1.0,
                key=f'Actual_Load_{v}'
            )

        with c:
            route=st.selectbox(
                "Route Information",
                categorical_values['Route_Info'],
                index=option_index(
                    categorical_values['Route_Info'],
                    fv['Route_Info']
                ),
                key=f'Route_Info_{v}'
            )

            last_maintenance=st.date_input(
                "Last Maintenance Date",
                value=fv['Last_Maintenance_Date'],
                key=f'Last_Maintenance_Date_{v}'
            )

            maintenance_type=st.selectbox(
                "Maintenance Type",
                categorical_values['Maintenance_Type'],
                index=option_index(
                    categorical_values['Maintenance_Type'],
                    fv['Maintenance_Type']
                ),
                key=f'Maintenance_Type_{v}'
            )

        st.markdown("---")
        st.markdown("### Condition & Maintenance")
        a,b,c=st.columns(3)

        with a:
            maintenance_cost=st.number_input(
                "Maintenance Cost",
                min_value=0.0,
                value=max(0.0,float(fv['Maintenance_Cost'])),
                step=100.0,
                key=f'Maintenance_Cost_{v}'
            )

            engine_temp=st.number_input(
                "Engine Temperature",
                value=float(fv['Engine_Temperature']),
                step=1.0,
                key=f'Engine_Temperature_{v}'
            )

            tire_pressure=st.number_input(
                "Tire Pressure",
                min_value=0.01,
                value=max(0.01,float(fv['Tire_Pressure'])),
                step=1.0,
                key=f'Tire_Pressure_{v}'
            )

        with b:
            fuel=st.number_input(
                "Fuel Consumption",
                min_value=0.0,
                value=max(0.0,float(fv['Fuel_Consumption'])),
                step=1.0,
                key=f'Fuel_Consumption_{v}'
            )

            battery=st.number_input(
                "Battery Status",
                value=float(fv['Battery_Status']),
                step=1.0,
                key=f'Battery_Status_{v}'
            )

            vibration=st.number_input(
                "Vibration Levels",
                value=float(fv['Vibration_Levels']),
                step=.1,
                key=f'Vibration_Levels_{v}'
            )

        with c:
            oil=st.number_input(
                "Oil Quality",
                value=float(fv['Oil_Quality']),
                step=1.0,
                key=f'Oil_Quality_{v}'
            )

            brake=st.selectbox(
                "Brake Condition",
                categorical_values['Brake_Condition'],
                index=option_index(
                    categorical_values['Brake_Condition'],
                    fv['Brake_Condition']
                ),
                key=f'Brake_Condition_{v}'
            )

            failure=st.number_input(
                "Failure History",
                value=float(fv['Failure_History']),
                step=.1,
                key=f'Failure_History_{v}'
            )

        st.markdown("---")
        st.markdown("### Risk, Environment & Efficiency")
        a,b,c=st.columns(3)

        with a:
            anomalies=st.number_input(
                "Anomalies Detected",
                value=float(fv['Anomalies_Detected']),
                step=.1,
                key=f'Anomalies_Detected_{v}'
            )

            predictive=st.number_input(
                "Predictive Score",
                min_value=0.0,
                max_value=1.0,
                value=min(1.0,max(0.0,float(fv['Predictive_Score']))),
                step=.01,
                key=f'Predictive_Score_{v}'
            )

            weather=st.selectbox(
                "Weather Conditions",
                categorical_values['Weather_Conditions'],
                index=option_index(
                    categorical_values['Weather_Conditions'],
                    fv['Weather_Conditions']
                ),
                key=f'Weather_Conditions_{v}'
            )

        with b:
            road=st.selectbox(
                "Road Conditions",
                categorical_values['Road_Conditions'],
                index=option_index(
                    categorical_values['Road_Conditions'],
                    fv['Road_Conditions']
                ),
                key=f'Road_Conditions_{v}'
            )

            delivery=st.number_input(
                "Delivery Times",
                min_value=0.0,
                value=max(0.0,float(fv['Delivery_Times'])),
                step=1.0,
                key=f'Delivery_Times_{v}'
            )

            downtime=st.number_input(
                "Downtime Maintenance",
                value=float(fv['Downtime_Maintenance']),
                step=.01,
                key=f'Downtime_Maintenance_{v}'
            )

        with c:
            impact=st.number_input(
                "Impact on Efficiency",
                min_value=0.0,
                value=max(0.0,float(fv['Impact_on_Efficiency'])),
                step=.01,
                key=f'Impact_on_Efficiency_{v}'
            )

            severity=st.number_input(
                "Severity Score",
                min_value=0.0,
                value=max(0.0,float(fv['Severity_Score'])),
                step=.01,
                key=f'Severity_Score_{v}'
            )

            st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)
            st.info(
                "The prediction uses the trained Random Forest model and the same preprocessing logic used during training."
            )

        submitted=st.form_submit_button(
            "⚡ Predict Maintenance Level",
            use_container_width=True
        )

    if submitted:
        values={
            'Make_and_Model':make_model,
            'Year_of_Manufacture':year,
            'Vehicle_Type':vehicle_type,
            'Usage_Hours':usage,
            'Route_Info':route,
            'Load_Capacity':capacity,
            'Actual_Load':actual_load,
            'Last_Maintenance_Date':last_maintenance,
            'Maintenance_Type':maintenance_type,
            'Maintenance_Cost':maintenance_cost,
            'Engine_Temperature':engine_temp,
            'Tire_Pressure':tire_pressure,
            'Fuel_Consumption':fuel,
            'Battery_Status':battery,
            'Vibration_Levels':vibration,
            'Oil_Quality':oil,
            'Brake_Condition':brake,
            'Failure_History':failure,
            'Anomalies_Detected':anomalies,
            'Predictive_Score':predictive,
            'Weather_Conditions':weather,
            'Road_Conditions':road,
            'Delivery_Times':delivery,
            'Downtime_Maintenance':downtime,
            'Impact_on_Efficiency':impact,
            'Severity_Score':severity
        }

        st.session_state.last_prediction=evaluate_vehicle(values)

    if 'last_prediction' in st.session_state:
        result=st.session_state.last_prediction

        values=result['values']
        processed=result['processed']
        prediction=result['prediction']
        prediction_label=result['prediction_label']
        probabilities=result['probabilities']
        confidence=result['confidence']
        risk_score=result['risk_score']
        risk_level=result['risk_level']
        risk_color=result['risk_color']
        risk_message=result['risk_message']

        style_map={
            'Normal':(
                'result-normal',
                '🟢',
                'Vehicle condition is within the normal maintenance range.'
            ),
            'Minor':(
                'result-minor',
                '🟠',
                'The vehicle shows indicators that may require minor maintenance attention.'
            ),
            'Major':(
                'result-major',
                '🔴',
                'The vehicle shows stronger indicators associated with major maintenance.'
            )
        }

        css_class,icon,message=style_map.get(
            prediction_label,
            ('result-minor','🔧','Maintenance prediction completed.')
        )

        st.markdown("<br>",unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="{css_class}">
                <div class="result-label">Predicted maintenance level</div>
                <div class="result-title">{icon} {prediction_label}</div>
                <div class="result-sub">{message}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>",unsafe_allow_html=True)

        # --------------------------------------------------
        # Risk gauge + confidence overview
        # --------------------------------------------------
        gauge_col,summary_col=st.columns([.56,.44],gap="large")

        with gauge_col:
            st.markdown(
                """
                <div class="section-label">Risk intelligence</div>
                <div class="section-title">Vehicle risk gauge</div>
                <div class="section-copy">
                    A 0–100 score derived from the model's full Normal, Minor and Major probability distribution.
                </div>
                """,
                unsafe_allow_html=True
            )

            gauge_fig=make_risk_gauge(risk_score)
            st.pyplot(gauge_fig,use_container_width=True)
            plt.close(gauge_fig)

        with summary_col:
            st.markdown("<div style='height:28px'></div>",unsafe_allow_html=True)

            load_ratio_value=float(values['Actual_Load'])/max(float(values['Load_Capacity']),.01)
            days_since_service=(pd.Timestamp('2026-01-01')-pd.Timestamp(values['Last_Maintenance_Date'])).days

            risk_card_html=f'''<div class="intel-card">
<div class="intel-eyebrow">Current risk status</div>
<div class="intel-title">{risk_level} Maintenance Risk</div>
<div style="margin-top:14px;">
<span class="risk-pill" style="background:{risk_color}18;color:{risk_color};border:1px solid {risk_color}35;">● {risk_level} Risk</span>
</div>
<div class="intel-copy">{risk_message}</div>
<hr>
<div style="display:flex;justify-content:space-between;margin-bottom:10px;">
<span style="color:#667085;">Prediction confidence</span>
<b style="color:#101828;">{confidence*100:.1f}%</b>
</div>
<div style="display:flex;justify-content:space-between;margin-bottom:10px;">
<span style="color:#667085;">Risk score</span>
<b style="color:#101828;">{risk_score:.0f}/100</b>
</div>
<div style="display:flex;justify-content:space-between;margin-bottom:10px;">
<span style="color:#667085;">Load ratio</span>
<b style="color:#101828;">{load_ratio_value:.2f}x</b>
</div>
<div style="display:flex;justify-content:space-between;">
<span style="color:#667085;">Days since service</span>
<b style="color:#101828;">{days_since_service:,}</b>
</div>
</div>'''

            st.markdown(
                risk_card_html,
                unsafe_allow_html=True
            )

        st.markdown("<br>",unsafe_allow_html=True)

        # --------------------------------------------------
        # Probability distribution
        # --------------------------------------------------
        page_header(
            "Prediction profile",
            "Class probability breakdown",
            "The model's confidence distribution across all three maintenance levels."
        )

        probability_df=pd.DataFrame({
            'Maintenance Level':[
                label_map[int(c)]
                for c in ml.rf.classes_
            ],
            'Probability':[
                float(p)*100
                for p in probabilities
            ]
        }).set_index('Maintenance Level')

        st.bar_chart(probability_df,height=280)

        st.markdown("<br>",unsafe_allow_html=True)

        # --------------------------------------------------
        # Vehicle condition snapshot
        # --------------------------------------------------
        page_header(
            "Vehicle health",
            "Vehicle Condition Snapshot",
            "A descriptive summary of the current vehicle inputs and operational warning flags."
        )

        snapshot=build_condition_snapshot(values)
        priority=planning_priority_for_prediction(prediction_label)

        status_icons={
            'Good':'🟢',
            'Monitor':'🟠',
            'Attention':'🔴'
        }

        first_row=st.columns(3,gap="medium")
        second_row=st.columns(3,gap="medium")

        for i,item in enumerate(snapshot):
            target_col=first_row[i] if i<3 else second_row[i-3]

            with target_col:
                with st.container(border=True):
                    st.caption(item['label'].upper())
                    st.markdown(
                        f"### {status_icons.get(item['status'],'⚪')} {item['value']}"
                    )
                    st.caption(
                        f"{item['status']} · {item['detail']}"
                    )

        st.markdown("<br>",unsafe_allow_html=True)

        priority_col,download_col=st.columns([1.2,.8],gap="large")

        with priority_col:
            with st.container(border=True):
                st.caption("MAINTENANCE PRIORITY")
                st.markdown(f"## {priority}")
                st.caption(
                    f"Planning priority derived from the predicted maintenance level: {prediction_label}."
                )

        with download_col:
            pdf_bytes=build_prediction_pdf(
                result,
                snapshot,
                priority
            )

            safe_vehicle_name=''.join(
                c if c.isalnum() else '_'
                for c in str(values['Make_and_Model'])
            ).strip('_')

            st.markdown("<div style='height:9px'></div>",unsafe_allow_html=True)
            st.download_button(
                "⬇ Download PDF Report",
                data=pdf_bytes,
                file_name=f"{safe_vehicle_name}_maintenance_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_prediction_pdf"
            )

        st.markdown("<br><br>",unsafe_allow_html=True)

        with st.container(border=True):
            p1,p2=st.columns([1.35,.65],gap="large")

            with p1:
                st.markdown("### Next step: Maintenance Planning")
                st.caption(
                    "Schedule the maintenance activity and estimate its historical planning cost for this predicted maintenance level."
                )

            with p2:
                st.button(
                    "Open Maintenance Planning →",
                    use_container_width=True,
                    on_click=go_to_planner,
                    key="prediction_to_planner"
                )


elif page=='Maintenance Planning':
    page_header(
        "Post-prediction planning",
        "Maintenance Planning",
        "Turn the latest maintenance prediction into a practical plan with scheduling and historical cost estimation."
    )

    if 'last_prediction' not in st.session_state:
        st.warning(
            "No vehicle prediction is available yet. Run a prediction first, then return here to build the maintenance plan."
        )

        st.button(
            "Go to Predict Maintenance →",
            use_container_width=False,
            on_click=go_to_prediction,
            key="planner_no_prediction_cta"
        )

    else:
        result=st.session_state.last_prediction
        values=result['values']
        predicted_level=result['prediction_label']
        vehicle_type=str(values['Vehicle_Type'])
        vehicle_model=str(values['Make_and_Model'])
        prediction_confidence=float(result['confidence'])*100

        level_priority_map={
            'Normal':'Routine',
            'Minor':'Medium',
            'Major':'High'
        }

        default_priority=level_priority_map.get(
            predicted_level,
            'Medium'
        )

        maintenance_types=sorted(
            ml.df['Maintenance_Type']
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        current_maintenance_type=str(
            values.get(
                'Maintenance_Type',
                maintenance_types[0] if maintenance_types else ''
            )
        )

        if current_maintenance_type not in maintenance_types and maintenance_types:
            current_maintenance_type=maintenance_types[0]

        summary1,summary2,summary3=st.columns(3)

        with summary1:
            st.metric(
                "Vehicle",
                vehicle_model
            )

        with summary2:
            st.metric(
                "Predicted Level",
                predicted_level
            )

        with summary3:
            st.metric(
                "Prediction Confidence",
                f"{prediction_confidence:.1f}%"
            )

        st.markdown("<br>",unsafe_allow_html=True)

        # -------------------------------------------------
        # Maintenance Scheduling
        # -------------------------------------------------
        page_header(
            "1 · Maintenance scheduling",
            "Schedule the maintenance activity",
            "Create an in-session maintenance plan for the predicted vehicle."
        )

        with st.container(border=True):
            sc1,sc2=st.columns(2,gap="large")

            with sc1:
                planned_date=st.date_input(
                    "Planned Maintenance Date",
                    value=pd.Timestamp.today().date(),
                    key="planned_maintenance_date"
                )

                selected_maintenance_type=st.selectbox(
                    "Maintenance Type",
                    maintenance_types,
                    index=(
                        maintenance_types.index(current_maintenance_type)
                        if current_maintenance_type in maintenance_types
                        else 0
                    ),
                    key="planned_maintenance_type"
                )

            with sc2:
                priorities=['Routine','Medium','High','Critical']

                planned_priority=st.selectbox(
                    "Priority",
                    priorities,
                    index=priorities.index(default_priority),
                    key="planned_priority"
                )

                assigned_to=st.text_input(
                    "Workshop / Technician",
                    placeholder="Optional",
                    key="planned_assigned_to"
                )

            maintenance_notes=st.text_area(
                "Planning Notes",
                placeholder="Optional maintenance notes...",
                key="planned_notes"
            )

            save_plan=st.button(
                "Save Maintenance Plan",
                use_container_width=True,
                key="save_maintenance_plan"
            )

        if 'maintenance_plans' not in st.session_state:
            st.session_state.maintenance_plans=[]

        if save_plan:
            save_cost_base=get_cost_planning_estimate(
                predicted_level,
                vehicle_type,
                selected_maintenance_type
            )

            save_priority_multiplier={
                'Routine':1.00,
                'Medium':1.10,
                'High':1.25,
                'Critical':1.45
            }[planned_priority]

            saved_estimated_cost=int(
                round(
                    save_cost_base['estimated']*
                    save_priority_multiplier
                )
            )

            plan={
                'Vehicle':vehicle_model,
                'Vehicle Type':vehicle_type,
                'Predicted Level':predicted_level,
                'Maintenance Type':selected_maintenance_type,
                'Planned Date':str(planned_date),
                'Priority':planned_priority,
                'Estimated Cost':saved_estimated_cost,
                'Workshop / Technician':assigned_to if assigned_to else 'Not assigned',
                'Notes':maintenance_notes if maintenance_notes else ''
            }

            st.session_state.maintenance_plans.append(plan)

            st.success(
                f"Maintenance plan saved for {vehicle_model} on {planned_date}."
            )

        # Use current selector choice when available; before form submission,
        # Streamlit still provides the selected value from the form widgets.
        planning_maintenance_type=selected_maintenance_type

        st.markdown("<br><br>",unsafe_allow_html=True)

        # -------------------------------------------------
        # Cost Estimation
        # -------------------------------------------------
        page_header(
            "2 · Cost estimation",
            "Historical maintenance cost estimate",
            "Estimate based on similar records in the existing dataset, not a separate cost-prediction model."
        )

        cost_estimate=get_cost_planning_estimate(
            predicted_level,
            vehicle_type,
            planning_maintenance_type
        )

        priority_multiplier_map={
            'Routine':1.00,
            'Medium':1.10,
            'High':1.25,
            'Critical':1.45
        }

        priority_multiplier=priority_multiplier_map[
            planned_priority
        ]

        adjusted_cost={
            'estimated':int(
                round(
                    cost_estimate['estimated']*
                    priority_multiplier
                )
            ),
            'lower':int(
                round(
                    cost_estimate['lower']*
                    priority_multiplier
                )
            ),
            'upper':int(
                round(
                    cost_estimate['upper']*
                    priority_multiplier
                )
            )
        }

        cost1,cost2,cost3=st.columns(3)

        with cost1:
            st.metric(
                "Estimated Cost",
                f"{adjusted_cost['estimated']:,}"
            )

        with cost2:
            st.metric(
                "Typical Lower Range",
                f"{adjusted_cost['lower']:,}"
            )

        with cost3:
            st.metric(
                "Typical Upper Range",
                f"{adjusted_cost['upper']:,}"
            )

        st.caption(
            f"Priority adjustment: {planned_priority} × {priority_multiplier:.2f}. "
            "This multiplier is a planning assumption: Routine 1.00, Medium 1.10, "
            "High 1.25, Critical 1.45."
        )

        if str(planning_maintenance_type).strip().lower()=='oil change':
            st.caption(
                f"Planning basis: {cost_estimate['scope']}. "
                "Oil Change uses a conservative historical benchmark to avoid underestimating planning cost. "
                "Cost units follow the source dataset; no currency is assumed."
            )
        else:
            st.caption(
                f"Planning basis: {cost_estimate['scope']}. "
                "The estimate combines similar historical maintenance-level, vehicle-type and maintenance-type records. "
                "Cost units follow the source dataset; no currency is assumed."
            )

        st.markdown("<br><br>",unsafe_allow_html=True)

        if st.session_state.maintenance_plans:
            with st.expander(
                f"Saved Maintenance Plans ({len(st.session_state.maintenance_plans)})"
            ):
                plans_df=pd.DataFrame(
                    st.session_state.maintenance_plans
                )

                st.dataframe(
                    plans_df,
                    use_container_width=True,
                    hide_index=True
                )


elif page=='Model Performance':
    page_header(
        "Model evaluation",
        "How the algorithms compare",
        "Accuracy, precision, recall and F1-score across all trained classification models."
    )

    table=ml.results_df.copy().sort_values('Accuracy',ascending=False)

    display_table=table.copy()
    for col in ['Accuracy','Precision','Recall','F1-Score']:
        if col in display_table.columns:
            display_table[col]=(display_table[col]*100).map(lambda x:f"{x:.2f}%")

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("<br>",unsafe_allow_html=True)
    left,right=st.columns([1.05,.95],gap="large")

    with left:
        page_header(
            "Comparison",
            "Accuracy by model",
            "A direct comparison of test-set accuracy."
        )
        chart_df=table[['Model','Accuracy']].set_index('Model')*100
        st.bar_chart(chart_df,height=360)

    with right:
        best=table.iloc[0]
        st.markdown(
            f"""
            <div class="card">
                <div class="section-label">Top performer</div>
                <div style="font-size:1.65rem;font-weight:850;">{best['Model']}</div>
                <div style="font-size:2.25rem;font-weight:900;color:#2563EB;margin-top:14px;">
                    {best['Accuracy']*100:.2f}%
                </div>
                <div style="color:#667085;">Test accuracy</div>
                <hr>
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:#667085;">Precision</span>
                    <b>{best['Precision']*100:.2f}%</b>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:9px;">
                    <span style="color:#667085;">Recall</span>
                    <b>{best['Recall']*100:.2f}%</b>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:9px;">
                    <span style="color:#667085;">F1-score</span>
                    <b>{best['F1-Score']*100:.2f}%</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>",unsafe_allow_html=True)
    page_header(
        "Error analysis",
        "Random Forest confusion matrix",
        "Actual versus predicted maintenance levels on the held-out test set."
    )

    cm=confusion_matrix(ml.y_test,ml.y_pred_rf)

    fig,ax=plt.subplots(figsize=(7,4.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=list(label_map.values()),
        yticklabels=list(label_map.values()),
        ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Random Forest - Confusion Matrix')
    plt.tight_layout()
    st.pyplot(fig,use_container_width=False)
    plt.close(fig)


elif page=='Data Insights':
    page_header(
        "Data intelligence",
        "What drives maintenance predictions",
        "Explore the most influential model features and key patterns in the vehicle dataset."
    )

    importance=pd.DataFrame({
        'Feature':ml.x_train.columns,
        'Importance':ml.rf.feature_importances_
    }).sort_values('Importance',ascending=False).head(10)

    left,right=st.columns([1.15,.85],gap="large")

    with left:
        st.markdown("### Top 10 Random Forest features")
        imp_chart=importance.set_index('Feature')
        st.bar_chart(imp_chart,height=390)

    with right:
        st.markdown("### Most influential signal")
        top=importance.iloc[0]
        st.markdown(
            f"""
            <div class="card">
                <div class="section-label">#1 Feature</div>
                <div style="font-size:1.7rem;font-weight:850;">{top['Feature']}</div>
                <div style="font-size:2.1rem;color:#2563EB;font-weight:900;margin-top:12px;">
                    {top['Importance']*100:.1f}%
                </div>
                <div style="color:#667085;">Random Forest feature importance</div>
                <hr>
                <div style="color:#667085;line-height:1.7;">
                    Feature importance shows how strongly the trained Random Forest
                    uses each transformed input when making classification decisions.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>",unsafe_allow_html=True)
    c1,c2=st.columns(2,gap="large")

    with c1:
        st.markdown("### Vehicle type by maintenance level")
        vehicle_mix=pd.crosstab(
            ml.df['Maintenance_Level'],
            ml.df['Vehicle_Type']
        )
        st.bar_chart(vehicle_mix,height=330)

    with c2:
        st.markdown("### Average usage hours")
        usage_by_level=(
            ml.df.groupby('Maintenance_Level')['Usage_Hours']
            .mean()
            .reindex(list(label_map.values()))
            .rename('Average Usage Hours')
        )
        st.bar_chart(usage_by_level,height=330)

    st.markdown("<br>",unsafe_allow_html=True)

    with st.expander("View Raw Dataset",expanded=False):
        st.caption(
            "A compact preview of the source vehicle records used in this project."
        )

        preview_cols=[
            'Make_and_Model','Vehicle_Type','Usage_Hours','Maintenance_Type',
            'Oil_Quality','Brake_Condition','Severity_Score','Maintenance_Level'
        ]

        st.dataframe(
            ml.df[preview_cols].head(50),
            use_container_width=True,
            hide_index=True
        )


st.markdown(
    """
    <div class="footer">
        Vehicle Maintenance Intelligence · Machine Learning Classification Dashboard
    </div>
    """,
    unsafe_allow_html=True
)
