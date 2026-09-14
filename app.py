import os, json, re
from typing import TypedDict, List, Dict, Any
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

load_dotenv()
app = Flask(__name__)

LOGISTICS_KB = [
    {"topic": "route_planning", "keywords": ["route", "distance", "reroute", "long", "highway"], "guidance": "For routes over 40km, stage alternate arterial corridors to bypass metro bottlenecks and preload dynamic waypoint buffers."},
    {"topic": "traffic", "keywords": ["traffic", "congestion", "heavy", "severe", "gridlock"], "guidance": "In severe traffic conditions, stage deliveries via off-peak dispatch slots or switch to dedicated lane courier units."},
    {"topic": "weather", "keywords": ["weather", "rain", "storm", "snow", "fog"], "guidance": "Inclement weather requires a minimum 25-40% transit speed buffer and active waterproof cargo containment."},
    {"topic": "delivery_windows", "keywords": ["window", "tight", "express", "deadline", "slot"], "guidance": "Windows under 1.5 hours requiring urban transit above 45km/h must be flagged for depot priority staging."},
    {"topic": "driver_availability", "keywords": ["driver", "availability", "single", "limited", "backup"], "guidance": "Maintain a standby driver reserve to absorb route exhaustion and mechanical breakdowns without dispatch lapses."},
    {"topic": "safety_buffers", "keywords": ["safety", "buffer", "contingency", "oversized", "delay"], "guidance": "Oversized cargo or historical delays exceeding 25% require an explicit 30-minute staging slack and proactive ETA notifications."}
]

TRAFFIC_SCORES = {"low": 0.08, "moderate": 0.30, "heavy": 0.65, "severe": 0.95}
WEATHER_SCORES = {"clear": 0.05, "rain": 0.32, "fog": 0.45, "snow": 0.70, "storm": 0.90}
DRIVER_SCORES = {"high": 0.05, "normal": 0.20, "limited": 0.65, "single": 0.90}
PARCEL_SCORES = {"small": 0.05, "medium": 0.20, "large": 0.45, "oversized": 0.75}

class DeliveryState(TypedDict):
    delivery_data: Dict[str, Any]
    risk_factors: Dict[str, Any]
    retrieved_knowledge: List[str]
    analysis: Dict[str, Any]
    recommendation: Dict[str, Any]

@tool(description="Calculates deterministic risk scores, factor impacts, and risk classification.")
def risk_calculator(data: Dict[str, Any]) -> Dict[str, Any]:
    dist = float(data.get("distance", 15))
    win = max(0.2, float(data.get("time_window", 2)))
    hist = float(data.get("historical_delay_rate", 15))
    hist_rate = hist / 100.0 if hist > 1.0 else hist
    t_val = TRAFFIC_SCORES.get(str(data.get("traffic", "moderate")).lower(), 0.30)
    w_val = WEATHER_SCORES.get(str(data.get("weather", "clear")).lower(), 0.05)
    d_val = DRIVER_SCORES.get(str(data.get("driver_availability", "normal")).lower(), 0.20)
    p_val = PARCEL_SCORES.get(str(data.get("parcel_size", "medium")).lower(), 0.20)
    dist_factor = min(1.0, dist / 80.0)
    implied_speed = dist / win
    speed_pressure = min(1.0, max(0.05, (implied_speed - 15.0) / 45.0))
    weights = {
        "traffic": (t_val, 0.22, "Traffic Congestion"), "weather": (w_val, 0.18, "Weather Severity"),
        "window": (speed_pressure, 0.20, "Time Window Pressure"), "driver": (d_val, 0.15, "Driver Scarcity"),
        "history": (hist_rate, 0.15, "Historical Delay Frequency"), "distance": (dist_factor, 0.05, "Route Distance"),
        "parcel": (p_val, 0.05, "Cargo Dimensions")
    }
    raw_score = sum(val * wt for val, wt, _ in weights.values())
    score = round(min(1.0, max(0.02, raw_score)), 2)
    level = "LOW" if score < 0.35 else "MEDIUM" if score < 0.65 else "HIGH"
    breakdown = sorted([{"name": label, "impact": round(val * wt, 3), "severity": round(val, 2)} for val, wt, label in weights.values()], key=lambda x: x["impact"], reverse=True)
    return {"risk_score": score, "risk_level": level, "main_risk_factors": breakdown[:3], "factor_breakdown": breakdown, "estimation_label": "AI-powered heuristic delivery risk estimation", "implied_speed_kmh": round(implied_speed, 1)}

@tool(description="Retrieves logistics domain knowledge cards matching query keywords.")
def logistics_retriever(query: str) -> List[str]:
    tokens = set(re.findall(r"\w+", query.lower()))
    scored = []
    for item in LOGISTICS_KB:
        overlap = sum(1 for kw in item["keywords"] if kw in tokens)
        if overlap > 0:
            scored.append((overlap, item["guidance"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [g for _, g in scored[:3]] or [LOGISTICS_KB[0]["guidance"], LOGISTICS_KB[3]["guidance"]]

@tool(description="Evaluates what-if scenario adjustments against baseline delivery parameters.")
def scenario_calculator(base_data: Dict[str, Any], adjustments: Dict[str, Any]) -> Dict[str, Any]:
    orig_res = risk_calculator.invoke({"data": base_data})
    mod_data = dict(base_data)
    mod_data.update(adjustments)
    new_res = risk_calculator.invoke({"data": mod_data})
    delta = round(new_res["risk_score"] - orig_res["risk_score"], 2)
    return {"original_score": orig_res["risk_score"], "original_level": orig_res["risk_level"], "new_score": new_res["risk_score"], "new_level": new_res["risk_level"], "delta": delta, "adjusted_data": mod_data}

@tool(description="Generates actionable mitigation advice and justification using LLM or domain heuristics.")
def recommendation_generator(risk_data: Dict[str, Any], context: str) -> Dict[str, Any]:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    score = risk_data.get("risk_score", 0.5)
    level = risk_data.get("risk_level", "MEDIUM")
    factors = [f["name"] for f in risk_data.get("main_risk_factors", [])]
    if api_key:
        models = [os.environ.get("GEMINI_MODEL", ""), "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        for m in models:
            if not m: continue
            try:
                llm = ChatGoogleGenerativeAI(model=m, google_api_key=api_key, temperature=0.2)
                prompt = ChatPromptTemplate.from_template("Logistics AI. Risk score {score} ({level}), top factors: {factors}. Guidance: {context}. Respond strictly in JSON with keys 'actions' (list of 3 tactical advice strings) and 'explanation' (2-3 sentences).")
                res = (prompt | llm).invoke({"score": score, "level": level, "factors": ", ".join(factors), "context": context})
                txt = res.content.strip()
                if "```json" in txt: txt = txt.split("```json")[1].split("```")[0].strip()
                elif "```" in txt: txt = txt.split("```")[1].split("```")[0].strip()
                parsed = json.loads(txt)
                if "actions" in parsed and "explanation" in parsed:
                    return {"actions": parsed["actions"], "explanation": parsed["explanation"]}
            except Exception:
                continue
    fallback_map = {
        "Traffic": "Re-route through arterial ring roads or transition to an off-peak staging slot.",
        "Weather": "Enforce wet-weather braking margins and confirm sealed weatherproofing on cargo.",
        "Window": "Negotiate a 45-minute window extension or allocate express depot priority dispatch.",
        "Driver": "Mobilize a secondary on-call driver to share route density and prevent delivery failure.",
        "Delay": "Send proactive ETA tracking links to recipient to reduce first-attempt handoff delays.",
        "Distance": "Establish a mid-route checkpoint to verify fuel reserves and vehicle status.",
        "Cargo": "Equip vehicle with mechanical offloading gear to prevent delivery turnaround delays."
    }
    actions = []
    for f in factors:
        for k, v in fallback_map.items():
            if k in f and v not in actions:
                actions.append(v)
    if not actions:
        actions = ["Maintain current dispatch sequence with continuous GPS tracking active."]
    exp = f"Delivery assessed at {level} risk ({score:.2f}) under an AI-powered heuristic delivery risk estimation. Primary drivers include {', '.join(factors) if factors else 'baseline metrics'}. Operational mitigation is recommended prior to dispatch."
    return {"actions": actions[:3], "explanation": exp}

def receive_delivery_data(state: DeliveryState) -> Dict[str, Any]:
    raw = state.get("delivery_data", {})
    try: dist = max(1.0, min(float(raw.get("distance", 15)), 500.0))
    except (ValueError, TypeError): dist = 15.0
    try: win = max(0.2, min(float(raw.get("time_window", 2)), 24.0))
    except (ValueError, TypeError): win = 2.0
    try: hist = max(0.0, min(float(raw.get("historical_delay_rate", 15)), 100.0))
    except (ValueError, TypeError): hist = 15.0
    clean = {
        "distance": dist, "traffic": str(raw.get("traffic", "moderate")).lower(),
        "weather": str(raw.get("weather", "clear")).lower(), "time_window": win,
        "parcel_size": str(raw.get("parcel_size", "medium")).lower(), "driver_availability": str(raw.get("driver_availability", "normal")).lower(),
        "historical_delay_rate": hist
    }
    return {"delivery_data": clean}

def calculate_risk_factors(state: DeliveryState) -> Dict[str, Any]:
    return {"risk_factors": risk_calculator.invoke({"data": state["delivery_data"]})}

def retrieve_logistics_knowledge(state: DeliveryState) -> Dict[str, Any]:
    top_str = " ".join([f["name"] for f in state["risk_factors"].get("main_risk_factors", [])])
    q = f"{state['delivery_data']['weather']} {state['delivery_data']['traffic']} {top_str}"
    return {"retrieved_knowledge": logistics_retriever.invoke({"query": q})}

def analyze_risk(state: DeliveryState) -> Dict[str, Any]:
    rf = state["risk_factors"]
    return {"analysis": {"score": rf["risk_score"], "level": rf["risk_level"], "label": rf["estimation_label"]}}

def generate_recommendation(state: DeliveryState) -> Dict[str, Any]:
    ctx = "\n".join(state.get("retrieved_knowledge", []))
    rec = recommendation_generator.invoke({"risk_data": state["risk_factors"], "context": ctx})
    return {"recommendation": rec}

workflow = StateGraph(DeliveryState)
for node_name, fn in [("receive_delivery_data", receive_delivery_data), ("calculate_risk_factors", calculate_risk_factors), ("retrieve_logistics_knowledge", retrieve_logistics_knowledge), ("analyze_risk", analyze_risk), ("generate_recommendation", generate_recommendation)]:
    workflow.add_node(node_name, fn)
workflow.add_edge(START, "receive_delivery_data")
workflow.add_edge("receive_delivery_data", "calculate_risk_factors")
workflow.add_edge("calculate_risk_factors", "retrieve_logistics_knowledge")
workflow.add_edge("retrieve_logistics_knowledge", "analyze_risk")
workflow.add_edge("analyze_risk", "generate_recommendation")
workflow.add_edge("generate_recommendation", END)
graph_app = workflow.compile()

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Delivery Risk Advisor | AI Logistics Intelligence</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root {
  --bg: rgb(11, 15, 25); --card: rgba(22, 30, 49, 0.88); --card-border: rgba(255, 255, 255, 0.08);
  --primary: rgb(99, 102, 241); --primary-glow: rgba(99, 102, 241, 0.25); --accent: rgb(56, 189, 248);
  --low: rgb(16, 185, 129); --low-bg: rgba(16, 185, 129, 0.15); --med: rgb(245, 158, 11);
  --med-bg: rgba(245, 158, 11, 0.15); --high: rgb(239, 68, 68); --high-bg: rgba(239, 68, 68, 0.15);
  --text: rgb(241, 245, 249); --muted: rgb(148, 163, 184);
}
* { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
body { background: var(--bg); color: var(--text); min-height: 100vh; padding: 24px 16px; background-image: radial-gradient(circle at 10% 20%, rgba(99,102,241,0.08) 0%, transparent 40%), radial-gradient(circle at 90% 80%, rgba(56,189,248,0.08) 0%, transparent 40%); }
.container { max-width: 1240px; margin: 0 auto; }
header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--card-border); flex-wrap: wrap; gap: 12px; }
.logo-group h1 { font-size: 1.5rem; font-weight: 800; background: linear-gradient(135deg, rgb(255,255,255) 30%, rgb(148,163,184)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.logo-group p { font-size: 0.8rem; color: var(--muted); margin-top: 2px; }
.badge-heuristic { background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.3); color: rgb(199,210,254); padding: 5px 12px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.layout-grid { display: grid; grid-template-columns: 390px 1fr; gap: 24px; }
@media (max-width: 960px) { .layout-grid { grid-template-columns: 1fr; } }
.panel { background: var(--card); border: 1px solid var(--card-border); border-radius: 16px; padding: 22px; backdrop-filter: blur(12px); box-shadow: 0 8px 32px rgba(0,0,0,0.35); }
.panel-title { font-size: 1rem; font-weight: 700; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; color: rgb(255,255,255); }
.form-group { margin-bottom: 13px; }
.form-group label { display: block; font-size: 0.78rem; font-weight: 600; color: var(--muted); margin-bottom: 5px; }
.input-control, select { width: 100%; background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; color: var(--text); padding: 8px 12px; font-size: 0.84rem; outline: none; transition: border 0.2s, box-shadow 0.2s; }
.input-control:focus, select:focus { border-color: var(--primary); box-shadow: 0 0 0 3px var(--primary-glow); }
.range-val { float: right; color: var(--accent); font-weight: 700; font-size: 0.8rem; }
.btn-submit { width: 100%; background: linear-gradient(135deg, rgb(99, 102, 241), rgb(79, 70, 229)); color: white; border: none; border-radius: 10px; padding: 12px; font-size: 0.9rem; font-weight: 700; cursor: pointer; transition: transform 0.15s, box-shadow 0.2s; box-shadow: 0 4px 14px var(--primary-glow); margin-top: 8px; }
.btn-submit:hover { transform: translateY(-1px); box-shadow: 0 6px 20px var(--primary-glow); }
.score-banner { display: grid; grid-template-columns: 150px 1fr; gap: 20px; align-items: center; background: rgba(15,23,42,0.6); border: 1px solid var(--card-border); border-radius: 14px; padding: 18px; margin-bottom: 20px; }
.score-circle { width: 124px; height: 124px; border-radius: 50%; display: flex; flex-direction: column; align-items: center; justify-content: center; margin: 0 auto; }
.score-num { font-size: 2.2rem; font-weight: 800; line-height: 1; }
.score-sub { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; font-weight: 700; }
.level-pill { display: inline-block; padding: 4px 14px; border-radius: 999px; font-size: 0.85rem; font-weight: 800; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.05em; }
.label-disclaimer { font-size: 0.73rem; color: var(--muted); line-height: 1.4; }
.factors-list { display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px; }
.factor-row { background: rgba(15,23,42,0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 10px; padding: 10px 14px; }
.factor-meta { display: flex; justify-content: space-between; font-size: 0.8rem; font-weight: 600; margin-bottom: 6px; }
.meter-bg { background: rgba(255,255,255,0.06); height: 6px; border-radius: 999px; overflow: hidden; }
.meter-fill { height: 100%; border-radius: 999px; transition: width 0.4s ease; }
.rec-box { background: rgba(99,102,241,0.06); border: 1px solid rgba(99,102,241,0.2); border-radius: 12px; padding: 14px; margin-bottom: 20px; }
.rec-title { font-size: 0.82rem; font-weight: 700; color: rgb(199,210,254); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.04em; }
.rec-item { font-size: 0.82rem; margin-bottom: 6px; display: flex; align-items: flex-start; gap: 8px; line-height: 1.4; color: rgb(226,232,240); }
.rec-item:before { content: "•"; color: var(--accent); font-weight: bold; }
.explanation-text { font-size: 0.82rem; color: rgb(203,213,225); line-height: 1.5; background: rgba(15,23,42,0.5); border-radius: 10px; padding: 12px; border-left: 3px solid var(--primary); margin-bottom: 20px; }
.whatif-section { background: rgba(15,23,42,0.7); border: 1px solid rgba(56,189,248,0.2); border-radius: 14px; padding: 16px; }
.whatif-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-top: 10px; }
.whatif-btn { background: rgba(30,41,59,0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; color: var(--text); padding: 8px 10px; font-size: 0.75rem; font-weight: 600; cursor: pointer; transition: all 0.2s; text-align: left; }
.whatif-btn:hover { background: rgba(56,189,248,0.15); border-color: var(--accent); }
.whatif-result { margin-top: 12px; padding: 10px 14px; background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); border-radius: 10px; font-size: 0.8rem; display: none; }
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="logo-group">
      <h1>Delivery Risk Advisor</h1>
      <p>Logistics Pre-Dispatch Feasibility & Contingency Engine</p>
    </div>
    <span class="badge-heuristic">AI-powered heuristic delivery risk estimation</span>
  </header>
  <div class="layout-grid">
    <div class="panel">
      <div class="panel-title"><span>Dispatch Parameters</span></div>
      <form id="riskForm" onsubmit="event.preventDefault(); runAssessment();">
        <div class="form-group">
          <label>Delivery Distance <span class="range-val" id="distDisplay">28 km</span></label>
          <input type="range" class="input-control" id="distance" min="2" max="120" value="28" oninput="document.getElementById('distDisplay').innerText=this.value+' km'">
        </div>
        <div class="form-group">
          <label>Traffic Level</label>
          <select id="traffic" class="input-control"><option value="low">Low (Free Flow)</option><option value="moderate" selected>Moderate (Suburban Delays)</option><option value="heavy">Heavy (Congested Arterials)</option><option value="severe">Severe (Gridlock / Peak Rush)</option></select>
        </div>
        <div class="form-group">
          <label>Weather Condition</label>
          <select id="weather" class="input-control"><option value="clear" selected>Clear / Mild</option><option value="rain">Rain / Wet Pavement</option><option value="fog">Dense Fog / Low Visibility</option><option value="snow">Snow / Ice Hazard</option><option value="storm">Severe Storm / High Winds</option></select>
        </div>
        <div class="form-group">
          <label>Delivery Time Window <span class="range-val" id="winDisplay">1.5 hrs</span></label>
          <input type="range" class="input-control" id="time_window" min="0.5" max="8" step="0.5" value="1.5" oninput="document.getElementById('winDisplay').innerText=this.value+' hrs'">
        </div>
        <div class="form-group">
          <label>Parcel Size</label>
          <select id="parcel_size" class="input-control"><option value="small">Small (Envelope / Pouch)</option><option value="medium" selected>Medium (Standard Box)</option><option value="large">Large (Bulky / Heavy)</option><option value="oversized">Oversized (Palletized / 2-Person)</option></select>
        </div>
        <div class="form-group">
          <label>Driver Availability</label>
          <select id="driver_availability" class="input-control"><option value="high">High (Multiple Backups Active)</option><option value="normal" selected>Normal (Standard Fleet Pool)</option><option value="limited">Limited (Peak Route Load)</option><option value="single">Single Driver (Zero Backup)</option></select>
        </div>
        <div class="form-group">
          <label>Historical Delay Rate <span class="range-val" id="histDisplay">18%</span></label>
          <input type="range" class="input-control" id="historical_delay_rate" min="0" max="60" value="18" oninput="document.getElementById('histDisplay').innerText=this.value+'%'">
        </div>
        <button type="submit" class="btn-submit" id="submitBtn">Calculate Risk</button>
      </form>
    </div>
    <div class="panel">
      <div class="panel-title"><span>Risk Assessment & Actions</span><span id="speedMetric" style="font-size:0.75rem; color:var(--accent);">Velocity: -- km/h</span></div>
      <div class="score-banner">
        <div class="score-circle" id="scoreCircle" style="background: var(--med-bg); border: 2px solid var(--med); color: var(--med);">
          <div class="score-num" id="scoreNum">--</div>
          <div class="score-sub" id="scoreSub">INDEX</div>
        </div>
        <div>
          <div><span class="level-pill" id="levelPill" style="background:var(--med-bg); color:var(--med);">STANDBY</span></div>
          <p class="label-disclaimer">AI-powered heuristic delivery risk estimation based on deterministic multi-factor logistics scoring and domain guidance.</p>
        </div>
      </div>
      <div class="panel-title" style="font-size:0.85rem;"><span>Primary Risk Vectors</span></div>
      <div class="factors-list" id="factorsList"></div>
      <div class="rec-box">
        <div class="rec-title">Operational Mitigation Actions</div>
        <div id="recItems"></div>
      </div>
      <div class="explanation-text" id="explanationBox">Run assessment to evaluate pre-dispatch delivery feasibility.</div>
      <div class="whatif-section">
        <div class="panel-title" style="font-size:0.85rem; margin-bottom:6px;"><span>What-If Scenario Sandbox</span><span style="font-size:0.7rem; color:var(--muted);">Simulate mitigation impact</span></div>
        <div class="whatif-grid">
          <button type="button" class="whatif-btn" onclick="applyWhatIf({traffic: 'low'})">Traffic to Low</button>
          <button type="button" class="whatif-btn" onclick="applyWhatIf({time_window: parseFloat(document.getElementById('time_window').value) + 1.5})">+1.5h Delivery Window</button>
          <button type="button" class="whatif-btn" onclick="applyWhatIf({driver_availability: 'high'})">Add Backup Drivers</button>
          <button type="button" class="whatif-btn" onclick="applyWhatIf({weather: 'clear'})">Clear Weather</button>
          <button type="button" class="whatif-btn" onclick="applyWhatIf({distance: Math.max(5, parseFloat(document.getElementById('distance').value) * 0.6)})">Shorten Route 40%</button>
        </div>
        <div class="whatif-result" id="whatifResult"></div>
      </div>
    </div>
  </div>
</div>
<script>
function getFormData() {
  return {
    distance: parseFloat(document.getElementById('distance').value),
    traffic: document.getElementById('traffic').value,
    weather: document.getElementById('weather').value,
    time_window: parseFloat(document.getElementById('time_window').value),
    parcel_size: document.getElementById('parcel_size').value,
    driver_availability: document.getElementById('driver_availability').value,
    historical_delay_rate: parseFloat(document.getElementById('historical_delay_rate').value)
  };
}
function updateUI(res) {
  const score = res.risk_score; const level = res.risk_level;
  document.getElementById('scoreNum').innerText = score.toFixed(2);
  document.getElementById('speedMetric').innerText = 'Req. Speed: ' + res.implied_speed_kmh + ' km/h';
  const colors = level === 'LOW' ? {bg:'var(--low-bg)', border:'var(--low)', text:'var(--low)'} : (level === 'MEDIUM' ? {bg:'var(--med-bg)', border:'var(--med)', text:'var(--med)'} : {bg:'var(--high-bg)', border:'var(--high)', text:'var(--high)'});
  const circle = document.getElementById('scoreCircle');
  circle.style.background = colors.bg; circle.style.borderColor = colors.border; circle.style.color = colors.text;
  const pill = document.getElementById('levelPill');
  pill.style.background = colors.bg; pill.style.color = colors.text; pill.innerText = level + ' RISK';
  const factorsList = document.getElementById('factorsList');
  factorsList.innerHTML = '';
  (res.main_risk_factors || []).forEach(f => {
    const pct = Math.min(100, Math.round((f.impact / 0.25) * 100));
    factorsList.innerHTML += `<div class="factor-row"><div class="factor-meta"><span>${f.name}</span><span>Severity ${(f.severity * 100).toFixed(0)}%</span></div><div class="meter-bg"><div class="meter-fill" style="width:${pct}%; background:${colors.text}"></div></div></div>`;
  });
  const recItems = document.getElementById('recItems');
  recItems.innerHTML = '';
  (res.recommendations || []).forEach(act => { recItems.innerHTML += `<div class="rec-item">${act}</div>`; });
  document.getElementById('explanationBox').innerText = res.explanation;
}
async function runAssessment() {
  const btn = document.getElementById('submitBtn');
  btn.innerText = 'Analyzing Graph...'; btn.disabled = true;
  try {
    const resp = await fetch('/api/assess', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(getFormData()) });
    const data = await resp.json();
    updateUI(data);
  } catch(e) { alert('Assessment error: ' + e); }
  finally { btn.innerText = 'Calculate Risk'; btn.disabled = false; }
}
async function applyWhatIf(adjustments) {
  const base = getFormData();
  try {
    const resp = await fetch('/api/what-if', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({base, adjustments}) });
    const res = await resp.json();
    const box = document.getElementById('whatifResult');
    const color = res.delta <= 0 ? 'var(--low)' : 'var(--high)';
    const sign = res.delta <= 0 ? '' : '+';
    box.style.display = 'block';
    box.innerHTML = `<strong>What-If Simulation:</strong> Score shifted from <strong>${res.original_score.toFixed(2)}</strong> (${res.original_level}) to <strong style="color:${color}">${res.new_score.toFixed(2)}</strong> (${res.new_level}). Shift: <strong style="color:${color}">${sign}${res.delta.toFixed(2)}</strong>`;
  } catch(e) { alert('Scenario error: ' + e); }
}
window.onload = runAssessment;
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/assess", methods=["POST"])
def assess():
    data = request.get_json(force=True) or {}
    result = graph_app.invoke({"delivery_data": data})
    rf = result["risk_factors"]
    return jsonify({
        "risk_score": rf["risk_score"], "risk_level": rf["risk_level"],
        "estimation_label": rf["estimation_label"], "main_risk_factors": rf["main_risk_factors"],
        "factor_breakdown": rf["factor_breakdown"], "implied_speed_kmh": rf["implied_speed_kmh"],
        "knowledge": result.get("retrieved_knowledge", []),
        "recommendations": result["recommendation"].get("actions", []),
        "explanation": result["recommendation"].get("explanation", "")
    })

@app.route("/api/what-if", methods=["POST"])
def what_if():
    payload = request.get_json(force=True) or {}
    res = scenario_calculator.invoke({"base_data": payload.get("base", {}), "adjustments": payload.get("adjustments", {})})
    return jsonify(res)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
