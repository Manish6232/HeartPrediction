import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import os


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Stroke Prediction · akarsh",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    m  = joblib.load(os.path.join(BASE_DIR, "knn_heart_model.pkl"))
    sc = joblib.load(os.path.join(BASE_DIR, "heart_scaler.pkl"))
    ec = joblib.load(os.path.join(BASE_DIR, "heart_columns.pkl"))

    return m, sc, ec

model, scaler, expected_columns = load_model()

# ═══════════════════════════════════════════════════════════════════════════════
#  CSS  — pure styles, zero <script> tags (Streamlit would render them as text)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Bebas+Neue&display=swap');

:root {
  --bg:       #080810;
  --card:     #0f0f1a;
  --card2:    #13131f;
  --border:   rgba(255,255,255,0.06);
  --border2:  rgba(255,255,255,0.12);
  --red:      #f03a3a;
  --red2:     #ff6060;
  --red-glow: rgba(240,58,58,0.18);
  --green:    #22d3a0;
  --green-dim:rgba(34,211,160,0.13);
  --muted:    #6b6a85;
  --sub:      #9896b4;
  --text:     #eeedf8;
}

/* ── Hide Streamlit shell ── */
#MainMenu,
header[data-testid="stHeader"],
footer,
[data-testid="stDecoration"],
[data-testid="stToolbar"],
[data-testid="stStatusWidget"] { display: none !important; }

/* ── Background ── */
html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main,
.main .block-container { background: var(--bg) !important; }

.block-container {
  max-width: 760px !important;
  padding: 0 1.4rem 5rem !important;
}

/* ── Fonts & text ── */
body, p, div, span, label, input, select, textarea, button {
  font-family: 'DM Sans', sans-serif !important;
  color: var(--text) !important;
}

/* ── Cursor ── */
* { cursor: none !important; }

/* ══════════════════════════════
   SLIDERS
══════════════════════════════ */
/* Thumb */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
  background:  var(--red) !important;
  border:      2.5px solid rgba(255,255,255,0.22) !important;
  box-shadow:  0 0 0 5px var(--red-glow), 0 0 16px rgba(240,58,58,0.5) !important;
  width:  20px !important;
  height: 20px !important;
  cursor: none !important;
  transition:  box-shadow .2s, transform .2s !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]:hover {
  box-shadow: 0 0 0 9px rgba(240,58,58,0.22), 0 0 26px rgba(240,58,58,0.6) !important;
  transform: scale(1.15) !important;
}
/* Filled track */
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stSliderTrackFill"] {
  background: linear-gradient(90deg, #c02020, var(--red)) !important;
  height: 4px !important;
  border-radius: 4px !important;
}
/* Empty track */
[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
  background: rgba(255,255,255,0.08) !important;
  height: 4px !important;
  border-radius: 4px !important;
}
/* Value label above thumb */
[data-testid="stSlider"] p {
  color: var(--red2) !important;
  font-size: 13px !important;
  font-weight: 700 !important;
}
[data-testid="stSlider"] { cursor: none !important; }

/* ══════════════════════════════
   SELECTBOX
══════════════════════════════ */
[data-testid="stSelectbox"] > div > div {
  background:  var(--card2) !important;
  border:      1px solid var(--border2) !important;
  border-radius: 10px !important;
  font-size:   14px !important;
  min-height:  46px !important;
  transition:  border-color .2s, box-shadow .2s !important;
  cursor:      none !important;
}
[data-testid="stSelectbox"] > div > div:hover,
[data-testid="stSelectbox"] > div > div:focus-within {
  border-color: var(--red) !important;
  box-shadow:   0 0 0 3px var(--red-glow) !important;
}
[data-testid="stSelectbox"] svg {
  color: var(--muted) !important;
  fill:  var(--muted) !important;
}
/* Dropdown list */
[data-testid="stSelectbox"] ul {
  background:    #18182a !important;
  border:        1px solid var(--border2) !important;
  border-radius: 10px !important;
  padding:       4px !important;
}
[data-testid="stSelectbox"] li {
  color:         var(--text) !important;
  border-radius: 7px !important;
  font-size:     14px !important;
  padding:       9px 12px !important;
  cursor:        none !important;
}
[data-testid="stSelectbox"] li:hover,
[data-testid="stSelectbox"] li[aria-selected="true"] {
  background: var(--red-glow) !important;
  color:      var(--red2) !important;
}

/* ══════════════════════════════
   NUMBER INPUT
══════════════════════════════ */
[data-testid="stNumberInput"] input {
  background:    var(--card2) !important;
  border:        1px solid var(--border2) !important;
  border-radius: 10px !important;
  font-size:     14px !important;
  height:        46px !important;
  cursor:        none !important;
  transition:    border-color .2s, box-shadow .2s !important;
}
[data-testid="stNumberInput"] input:focus {
  border-color: var(--red) !important;
  box-shadow:   0 0 0 3px var(--red-glow) !important;
  outline:      none !important;
}
[data-testid="stNumberInput"] button {
  background:   var(--card2) !important;
  border-color: var(--border2) !important;
  color:        var(--muted) !important;
  cursor:       none !important;
}
[data-testid="stNumberInput"] button:hover {
  background:   var(--red-glow) !important;
  color:        var(--red2) !important;
  border-color: var(--red) !important;
}

/* ══════════════════════════════
   WIDGET LABELS
══════════════════════════════ */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
  color:          var(--sub) !important;
  font-size:      11px !important;
  font-weight:    600 !important;
  letter-spacing: 0.07em !important;
  text-transform: uppercase !important;
}

/* ══════════════════════════════
   PREDICT BUTTON
══════════════════════════════ */
[data-testid="stButton"] > button {
  background:    linear-gradient(135deg, #f03a3a 0%, #a81a1a 100%) !important;
  color:         #fff !important;
  border:        none !important;
  border-radius: 12px !important;
  font-family:   'Bebas Neue', sans-serif !important;
  font-size:     22px !important;
  letter-spacing: 0.14em !important;
  padding:       15px 0 !important;
  width:         100% !important;
  box-shadow:    0 6px 28px rgba(240,58,58,0.4),
                 inset 0 1px 0 rgba(255,255,255,0.12) !important;
  cursor:        none !important;
  transition:    transform .18s, box-shadow .22s !important;
  position:      relative !important;
  overflow:      hidden !important;
}
[data-testid="stButton"] > button::before {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(135deg, rgba(255,255,255,0.09) 0%, transparent 55%);
  pointer-events: none;
}
[data-testid="stButton"] > button:hover {
  transform:  translateY(-3px) !important;
  box-shadow: 0 14px 36px rgba(240,58,58,0.55),
              inset 0 1px 0 rgba(255,255,255,0.16) !important;
}
[data-testid="stButton"] > button:active {
  transform: translateY(0px) !important;
}
[data-testid="stButton"] > button p {
  color:          #fff !important;
  font-family:    'Bebas Neue', sans-serif !important;
  font-size:      22px !important;
  letter-spacing: 0.14em !important;
}

/* ══════════════════════════════
   RESULT ALERTS
══════════════════════════════ */
[data-testid="stAlert"] {
  border-radius: 14px !important;
  padding:       22px 24px !important;
  font-size:     15px !important;
  font-weight:   500 !important;
  line-height:   1.6 !important;
  animation:     alertPop .45s cubic-bezier(.34,1.56,.64,1) both !important;
  border:        none !important;
}
@keyframes alertPop {
  from { opacity:0; transform:scale(0.93) translateY(10px); }
  to   { opacity:1; transform:scale(1) translateY(0); }
}
/* Error — high risk */
div[data-testid="stAlert"][data-baseweb="notification"] {
  background:  rgba(240,58,58,0.12) !important;
  border-left: 3px solid var(--red) !important;
}
div[data-testid="stAlert"][data-baseweb="notification"] p,
div[data-testid="stAlert"][data-baseweb="notification"] strong {
  color: #ffaaaa !important;
}
/* Success — low risk */
div[data-testid="stAlert"][kind="success"],
.stSuccess > div[data-testid="stAlert"] {
  background:  var(--green-dim) !important;
  border-left: 3px solid var(--green) !important;
}
div[data-testid="stAlert"][kind="success"] p,
div[data-testid="stAlert"][kind="success"] strong {
  color: #a0ffe0 !important;
}
[data-testid="stAlert"] svg { flex-shrink: 0 !important; }

/* ══════════════════════════════
   MISC
══════════════════════════════ */
[data-testid="stHorizontalBlock"] { gap: 14px !important; }
[data-testid="column"]            { padding: 0 !important; }
hr {
  border:     none !important;
  border-top: 1px solid var(--border) !important;
  margin:     18px 0 !important;
}
::-webkit-scrollbar       { width: 4px; }
::-webkit-scrollbar-thumb { background: #2a2a3f; border-radius: 4px; }

/* ══════════════════════════════
   ANIMATIONS
══════════════════════════════ */
@keyframes hxBadgePulse {
  0%,100% { opacity:1; transform:scale(1); }
  50%     { opacity:.35; transform:scale(.5); }
}
@keyframes hxFadeDown {
  from { opacity:0; transform:translateY(-20px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes hxFadeUp {
  from { opacity:0; transform:translateY(16px); }
  to   { opacity:1; transform:translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  CURSOR  — must live in components.html so <script> actually executes
# ═══════════════════════════════════════════════════════════════════════════════
components.html("""
<!DOCTYPE html><html><head>
<style>
  html,body{margin:0;padding:0;background:transparent;overflow:hidden;height:0}
</style>
</head><body>
<script>
(function(){
  var pd = window.parent.document;

  function el(id, css){
    var e = pd.getElementById(id);
    if(!e){ e = pd.createElement('div'); e.id = id; pd.body.appendChild(e); }
    e.setAttribute('style', css);
    return e;
  }

  var dot = el('__hx_dot',
    'position:fixed;z-index:2147483647;width:13px;height:13px;border-radius:50%;' +
    'background:#f03a3a;pointer-events:none;top:-40px;left:-40px;' +
    'transition:width .22s,height .22s,background .22s;will-change:transform;');

  var ring = el('__hx_ring',
    'position:fixed;z-index:2147483646;width:38px;height:38px;border-radius:50%;' +
    'border:1.5px solid rgba(240,58,58,.55);pointer-events:none;top:-60px;left:-60px;' +
    'transition:width .26s,height .26s,border-color .26s;will-change:transform;');

  /* Trail particles */
  var pts = [];
  for(var i=0;i<9;i++){
    var s  = Math.max(1, 4.5 - i*0.42);
    var al = Math.max(0.04, 0.42 - i*0.045);
    var t  = el('__hx_tr'+i,
      'position:fixed;z-index:'+(2147483640-i)+';width:'+s+'px;height:'+s+'px;' +
      'border-radius:50%;background:rgba(240,58,58,'+al+');pointer-events:none;' +
      'top:-20px;left:-20px;will-change:transform;');
    pts.push({el:t, x:-100, y:-100});
  }

  var mx=-100, my=-100, rx=-100, ry=-100;

  function lerp(a,b,t){ return a+(b-a)*t; }

  pd.addEventListener('mousemove', function(e){
    mx=e.clientX; my=e.clientY;
    dot.style.transform='translate('+(mx-6.5)+'px,'+(my-6.5)+'px)';
  },{passive:true});

  /* Ring follow */
  (function loopRing(){
    rx=lerp(rx,mx,.14); ry=lerp(ry,my,.14);
    ring.style.transform='translate('+(rx-19)+'px,'+(ry-19)+'px)';
    requestAnimationFrame(loopRing);
  })();

  /* Trail follow */
  (function loopTrail(){
    for(var i=pts.length-1;i>=0;i--){
      var src=i===0?{x:mx,y:my}:pts[i-1];
      pts[i].x=lerp(pts[i].x,src.x,.38);
      pts[i].y=lerp(pts[i].y,src.y,.38);
      pts[i].el.style.transform='translate('+(pts[i].x-2.5)+'px,'+(pts[i].y-2.5)+'px)';
    }
    requestAnimationFrame(loopTrail);
  })();

  /* Hover grow */
  pd.addEventListener('mouseover',function(e){
    var hit=e.target.closest(
      'button,[role="slider"],select,input,[data-baseweb="select"],li,[data-baseweb="menu-item"]'
    );
    if(hit){
      dot.style.width='22px';  dot.style.height='22px';
      dot.style.background='#ff6060';
      ring.style.width='54px'; ring.style.height='54px';
      ring.style.borderColor='rgba(240,58,58,.82)';
    } else {
      dot.style.width='13px';  dot.style.height='13px';
      dot.style.background='#f03a3a';
      ring.style.width='38px'; ring.style.height='38px';
      ring.style.borderColor='rgba(240,58,58,.55)';
    }
  });
})();
</script>
</body></html>
""", height=0, scrolling=False)

# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:48px 0 32px;animation:hxFadeDown .7s ease both">
  <!-- Badge -->
  <div style="display:inline-flex;align-items:center;gap:8px;
    background:rgba(240,58,58,.1);border:1px solid rgba(240,58,58,.26);
    border-radius:999px;padding:5px 16px 5px 10px;
    font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
    color:#ff7070;margin-bottom:22px">
    <span style="width:7px;height:7px;background:#f03a3a;border-radius:50%;
      display:inline-block;animation:hxBadgePulse 1.6s ease infinite"></span>
    Cardiac Risk Assessment
  </div>

  <!-- Title -->
  <div style="font-family:'Bebas Neue',sans-serif;
    font-size:clamp(52px,9vw,90px);line-height:.9;letter-spacing:.04em;
    color:#eeedf8;margin-bottom:16px">
    HEART <span style="color:#f03a3a">STROKE</span><br>PREDICTION
  </div>

  <!-- Subtitle -->
  <p style="color:#6b6a85;font-size:14px;line-height:1.65;
    max-width:400px;margin:0 auto 8px">
    Enter your clinical parameters to assess your cardiovascular disease risk using a trained KNN model.
  </p>
  <p style="color:#3d3c52;font-size:12px;margin:0">
    by <span style="color:#f03a3a;font-weight:600">akarsh</span>
  </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  STATS STRIP
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;
  margin-bottom:30px;animation:hxFadeUp .7s .15s ease both">
  <div style="background:#0f0f1a;border:1px solid rgba(255,255,255,.06);
    border-radius:12px;padding:16px 10px;text-align:center">
    <div style="font-family:'Bebas Neue',sans-serif;font-size:30px;
      color:#f03a3a;line-height:1">KNN</div>
    <div style="font-size:10px;color:#3d3c52;letter-spacing:.1em;
      text-transform:uppercase;margin-top:5px">Algorithm</div>
  </div>
  <div style="background:#0f0f1a;border:1px solid rgba(255,255,255,.06);
    border-radius:12px;padding:16px 10px;text-align:center">
    <div style="font-family:'Bebas Neue',sans-serif;font-size:30px;
      color:#f03a3a;line-height:1">11</div>
    <div style="font-size:10px;color:#3d3c52;letter-spacing:.1em;
      text-transform:uppercase;margin-top:5px">Features</div>
  </div>
  <div style="background:#0f0f1a;border:1px solid rgba(255,255,255,.06);
    border-radius:12px;padding:16px 10px;text-align:center">
    <div style="font-family:'Bebas Neue',sans-serif;font-size:30px;
      color:#f03a3a;line-height:1">LIVE</div>
    <div style="font-size:10px;color:#3d3c52;letter-spacing:.1em;
      text-transform:uppercase;margin-top:5px">Inference</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER — section divider
# ═══════════════════════════════════════════════════════════════════════════════
def section_header(icon: str, label: str):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin:26px 0 14px">
      <span style="font-size:16px">{icon}</span>
      <span style="font-size:10px;font-weight:700;letter-spacing:.15em;
        text-transform:uppercase;color:#3d3c52;white-space:nowrap">{label}</span>
      <div style="flex:1;height:1px;background:rgba(255,255,255,.05)"></div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  FORM  — all inputs collected BEFORE the predict button
# ═══════════════════════════════════════════════════════════════════════════════

# ── Section 1: Personal Info ─────────────────────────────────────────────────
section_header("🫀", "Personal Info")

col_a, col_b = st.columns(2)
with col_a:
    inp_age = st.slider("Age (years)", min_value=18, max_value=100, value=40, step=1)
with col_b:
    inp_sex = st.selectbox(
        "Sex",
        options=["M", "F"],
        format_func=lambda x: "Male" if x == "M" else "Female"
    )

inp_chest_pain = st.selectbox(
    "Chest Pain Type",
    options=["ATA", "NAP", "TA", "ASY"],
    format_func=lambda x: {
        "ATA": "ATA — Atypical Angina",
        "NAP": "NAP — Non-Anginal Pain",
        "TA":  "TA  — Typical Angina",
        "ASY": "ASY — Asymptomatic",
    }[x]
)

# ── Section 2: Vitals & Lab ───────────────────────────────────────────────────
section_header("🩺", "Vitals & Lab Results")

col_c, col_d = st.columns(2)
with col_c:
    inp_resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120, step=1
    )
    inp_fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        options=[0, 1],
        format_func=lambda x: "Yes (1)" if x else "No (0)"
    )
with col_d:
    inp_cholesterol = st.number_input(
        "Cholesterol (mg/dL)", min_value=100, max_value=600, value=200, step=1
    )
    inp_resting_ecg = st.selectbox(
        "Resting ECG",
        options=["Normal", "ST", "LVH"],
        format_func=lambda x: {
            "Normal": "Normal",
            "ST":     "ST — Wave Abnormality",
            "LVH":    "LVH — Left Ventricular Hypertrophy",
        }[x]
    )

inp_max_hr = st.slider(
    "Max Heart Rate Achieved (bpm)", min_value=60, max_value=220, value=150, step=1
)

# ── Section 3: Exercise & ECG ─────────────────────────────────────────────────
section_header("🏃", "Exercise & ECG Parameters")

col_e, col_f = st.columns(2)
with col_e:
    inp_exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        options=["N", "Y"],
        format_func=lambda x: "Yes" if x == "Y" else "No"
    )
    inp_slope = st.selectbox(
        "ST Slope",
        options=["Up", "Flat", "Down"]
    )
with col_f:
    inp_oldpeak = st.slider(
        "Oldpeak — ST Depression", min_value=0.0, max_value=6.0, value=1.0, step=0.1
    )

# ── Summary row ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:#0f0f1a;border:1px solid rgba(255,255,255,.06);
  border-radius:12px;padding:16px 20px;margin:20px 0 24px;
  display:grid;grid-template-columns:repeat(3,1fr);gap:12px;font-size:13px">
  <div><span style="color:#3d3c52;font-size:10px;text-transform:uppercase;
    letter-spacing:.08em;display:block;margin-bottom:3px">Age / Sex</span>
    <span style="color:#eeedf8;font-weight:600">{inp_age} yrs &nbsp;·&nbsp;
    {"Male" if inp_sex == "M" else "Female"}</span></div>
  <div><span style="color:#3d3c52;font-size:10px;text-transform:uppercase;
    letter-spacing:.08em;display:block;margin-bottom:3px">Resting BP / Cholesterol</span>
    <span style="color:#eeedf8;font-weight:600">{inp_resting_bp} mmHg &nbsp;·&nbsp; {inp_cholesterol} mg/dL</span></div>
  <div><span style="color:#3d3c52;font-size:10px;text-transform:uppercase;
    letter-spacing:.08em;display:block;margin-bottom:3px">Max HR / Oldpeak</span>
    <span style="color:#eeedf8;font-weight:600">{inp_max_hr} bpm &nbsp;·&nbsp; {inp_oldpeak}</span></div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PREDICT BUTTON
# ═══════════════════════════════════════════════════════════════════════════════
clicked = st.button("🫀  ANALYZE CARDIAC RISK", use_container_width=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
if clicked:
    # Build one-hot encoded input row
    raw = {
        "Age":         inp_age,
        "RestingBP":   inp_resting_bp,
        "Cholesterol": inp_cholesterol,
        "FastingBS":   inp_fasting_bs,
        "MaxHR":       inp_max_hr,
        "Oldpeak":     inp_oldpeak,
        # One-hot columns
        f"Sex_{inp_sex}":                        1,
        f"ChestPainType_{inp_chest_pain}":       1,
        f"RestingECG_{inp_resting_ecg}":         1,
        f"ExerciseAngina_{inp_exercise_angina}": 1,
        f"ST_Slope_{inp_slope}":                 1,
    }

    df = pd.DataFrame([raw])

    # Fill any missing expected columns with 0
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder to match training column order
    df = df[expected_columns]

    # Scale & predict
    scaled = scaler.transform(df)
    result = model.predict(scaled)[0]

    # ── Show single result ────────────────────────────────────────────────────
    if result == 1:
        st.error(
            "⚠️ **High Risk of Heart Disease**\n\n"
            "Your clinical parameters indicate elevated cardiovascular risk. "
            "Please consult a cardiologist for a comprehensive evaluation and personalised care plan."
        )
    else:
        st.success(
            "✅ **Low Risk of Heart Disease**\n\n"
            "Your parameters suggest a lower risk profile. "
            "Maintain a healthy lifestyle, regular check-ups, and monitor key vitals over time."
        )
