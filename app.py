import os
import pickle
import sqlite3
import uuid
from datetime import datetime, timezone
import json
import base64
from urllib.request import urlopen
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import openai  # OpenAI API for ChatGPT integration
try:
    import PyPDF2 as _pypdf
except Exception:
    _pypdf = None
try:
    from streamlit_js_eval import get_browser_info
except Exception:  # package missing or older version without this helper
    def get_browser_info():
        return {}

# Set up OpenAI API Key (prefer session state, fallback to env)
api_key = None
if "openai_api_key" in st.session_state and st.session_state.openai_api_key:
    api_key = st.session_state.openai_api_key
elif os.environ.get("OPENAI_API_KEY"):
    api_key = os.environ.get("OPENAI_API_KEY")
    st.session_state.openai_api_key = api_key

if api_key:
    openai.api_key = api_key
    # Support OpenRouter keys by switching API base
    try:
        if isinstance(api_key, str) and api_key.startswith("sk-or-"):
            openai.api_base = "https://openrouter.ai/api/v1"
        else:
            # Reset to OpenAI default if previously set
            openai.api_base = "https://api.openai.com/v1"
    except Exception:
        pass
else:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        openai.api_key = api_key
        st.session_state.openai_api_key = api_key
        try:
            if api_key.startswith("sk-or-"):
                openai.api_base = "https://openrouter.ai/api/v1"
            else:
                if hasattr(openai, "api_base"):
                    openai.api_base = "https://api.openai.com/v1"
        except Exception:
            pass
    else:
        # Defer key requirement to Chat tab; allow other tabs to function
        openai.api_key = None

# Using legacy OpenAI SDK interface (openai==0.28)

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Get the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# -------------------- Simple Visits Tracking (SQLite) --------------------
# Create a session-scoped visitor ID
if "visitor_id" not in st.session_state:
    st.session_state["visitor_id"] = str(uuid.uuid4())

@st.cache_resource
def _get_visits_db(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS visits (
            visitor_id TEXT NOT NULL,
            page TEXT NOT NULL,
            first_ts TEXT NOT NULL,
            last_ts TEXT NOT NULL,
            count INTEGER NOT NULL,
            user_agent TEXT,
            ip TEXT,
            PRIMARY KEY (visitor_id, page)
        )
        """
    )
    # Backfill columns if upgrading from older schema
    try:
        cur.execute("PRAGMA table_info(visits)")
        cols = {r[1] for r in cur.fetchall()}
        if 'user_agent' not in cols:
            cur.execute("ALTER TABLE visits ADD COLUMN user_agent TEXT")
        if 'ip' not in cols:
            cur.execute("ALTER TABLE visits ADD COLUMN ip TEXT")
        conn.commit()
    except Exception:
        pass
    conn.commit()
    return conn

visits_conn = _get_visits_db(os.path.join(working_dir, "analytics.db"))

def record_visit(page: str):
    try:
        cur = visits_conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cur.execute(
            "SELECT count FROM visits WHERE visitor_id=? AND page=?",
            (st.session_state["visitor_id"], page),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO visits (visitor_id, page, first_ts, last_ts, count, user_agent, ip) VALUES (?,?,?,?,?,?,?)",
                (
                    st.session_state["visitor_id"],
                    page,
                    now,
                    now,
                    1,
                    st.session_state.get('ua'),
                    st.session_state.get('ip'),
                ),
            )
        else:
            cur.execute(
                "UPDATE visits SET last_ts=?, count=count+1 WHERE visitor_id=? AND page=?",
                (now, st.session_state["visitor_id"], page),
            )
        visits_conn.commit()
    except Exception:
        pass

def capture_client_meta():
    # Only attempt once per session
    if st.session_state.get('ua') and st.session_state.get('ip'):
        return
    try:
        info = get_browser_info()
        if isinstance(info, dict) and info.get('userAgent'):
            st.session_state['ua'] = info['userAgent'][:512]
    except Exception:
        pass
    # IP via ipify (best-effort)
    try:
        with urlopen('https://api.ipify.org?format=json', timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            ip = data.get('ip')
            if ip:
                st.session_state['ip'] = ip
    except Exception:
        pass

# Cached model loader to avoid reloads on rerun
@st.cache_resource
def _load_models(base_dir: str):
    with open(f"{base_dir}/saved_models/diabetes_model.sav", "rb") as f1:
        dm = pickle.load(f1)
    with open(f"{base_dir}/saved_models/heart_disease_model.sav", "rb") as f2:
        hm = pickle.load(f2)
    with open(f"{base_dir}/saved_models/parkinsons_model.sav", "rb") as f3:
        pm = pickle.load(f3)
    return dm, hm, pm

diabetes_model, heart_disease_model, parkinsons_model = _load_models(working_dir)

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('HEALTH AI',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Chat with HealthBot'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person', 'chat-left-dots'],
                           default_index=0)

# Record landing and page switches
if st.session_state.get("last_selected") != selected:
    capture_client_meta()
    record_visit(selected)
    st.session_state["last_selected"] = selected

if 'admin_authed' not in st.session_state:
    st.session_state['admin_authed'] = False
if 'admin_panel_open' not in st.session_state:
    st.session_state['admin_panel_open'] = False
if 'show_admin_login' not in st.session_state:
    st.session_state['show_admin_login'] = False

authed_flag = 'true' if st.session_state.get('admin_authed') else 'false'
if st.session_state.get('admin_authed'):
    components.html(
    ('<div id="stAdminAuthed" data-authed="' + authed_flag + '"></div>') +
    """
    <style>
      .stAdminGearBtn{position:fixed;top:10px;right:200px;z-index:1000;width:36px;height:36px;border:none;border-radius:12px;
        background:linear-gradient(180deg,#1f2937,#0b1220);color:#e5e7eb;box-shadow:0 8px 20px rgba(0,0,0,.35);
        display:flex;align-items:center;justify-content:center;cursor:pointer;opacity:.98;transition:transform .15s ease, box-shadow .15s ease, opacity .15s ease; outline:none}
      .stAdminGearBtn:hover{opacity:1; transform:translateY(-1px) scale(1.02); box-shadow:0 12px 26px rgba(0,0,0,.45)}
      .stAdminGearBtn:focus-visible{box-shadow:0 0 0 3px rgba(59,130,246,.6), 0 10px 24px rgba(0,0,0,.45)}
      .stAdminGearBtn.authed{background:linear-gradient(180deg,#059669,#065f46)}
      .stAdminGearBtn .icon{font-size:18px;}
      .stAdminGearTip{position:fixed;top:52px;right:186px;background:#0b0f19;color:#e5e7eb;padding:6px 10px;border-radius:10px;font-size:12px;
        border:1px solid rgba(255,255,255,.08);box-shadow:0 8px 18px rgba(0,0,0,.35);display:none;white-space:nowrap;pointer-events:none}
      .stAdminGearTip::after{content:""; position:absolute; top:-6px; right:12px; border-width:6px; border-style:solid; border-color:transparent transparent #0b0f19 transparent}
      @media (max-width: 768px){
        .stAdminGearBtn{top:8px;right:132px}
        .stAdminGearTip{top:48px;right:120px}
      }
    </style>
    <script>
    (function(){
      const P=window.parent; const D=P.document; if(D.getElementById('stAdminGearBtn')) return;
      const header=D.querySelector('header[data-testid="stHeader"]')||D.body;
      const btn=D.createElement('button'); btn.id='stAdminGearBtn'; btn.className='stAdminGearBtn'; btn.title='Admin'; btn.setAttribute('aria-label','Admin');
      const authedEl=D.getElementById('stAdminAuthed'); const isAuthed=authedEl && authedEl.getAttribute('data-authed')==='true';
      if(isAuthed) btn.classList.add('authed');
      btn.innerHTML='<span class="icon">\u2699\uFE0F</span>';
      const tip=D.createElement('div'); tip.className='stAdminGearTip'; tip.textContent='Admin';
      (header||D.body).appendChild(btn); D.body.appendChild(tip);

      function findDeployEl(){
        const candidates = Array.from((header||D).querySelectorAll('*'));
        for(const el of candidates){
          try{
            const t = (el.textContent||'').trim();
            if(t === 'Deploy') return el;
          }catch(e){}
        }
        return null;
      }
      function placeGear(){
        const target = findDeployEl();
        if(target){
          const r = target.getBoundingClientRect();
          const spacing = 12; // px to the left of the Deploy text
          btn.style.top = Math.max(8, r.top + (r.height-36)/2) + 'px';
          btn.style.left = (r.left - spacing - 36) + 'px';
          btn.style.right = 'auto';
          tip.style.top = (r.top + r.height + 4) + 'px';
          tip.style.left = Math.max(8, r.left - spacing - 36) + 'px';
        } else {
          btn.style.right = '200px';
          btn.style.left = 'auto';
        }
      }
      placeGear();
      window.addEventListener('resize', placeGear);
      const mo = new MutationObserver(placeGear);
      if(header) mo.observe(header, {childList:true, subtree:true, characterData:true});

      btn.addEventListener('mouseenter',()=>{tip.style.display='block'});
      btn.addEventListener('mouseleave',()=>{tip.style.display='none'});
      btn.addEventListener('click',()=>{
        try{ const u=new URL(P.location); u.searchParams.set('admin','1'); P.location.href=u.toString(); }catch(e){}
      });
      btn.addEventListener('keydown',(e)=>{ if(e.key==='Enter' || e.key===' '){ e.preventDefault(); btn.click(); } });
    })();
    </script>
    """,
    height=0,
)

try:
    qp = st.query_params
    if qp.get('admin', '0') == '1':
        if st.session_state.get('admin_authed'):
            st.session_state['admin_panel_open'] = True
        else:
            st.session_state['show_admin_login'] = True
        try:
            del st.query_params['admin']
        except Exception:
            pass
except Exception:
    pass

top_spacer, top_admin_col = st.columns([9,1])
with top_admin_col:
    if st.button('⚙️', key='admin_btn'):
        st.session_state['show_admin_login'] = not st.session_state.get('show_admin_login', False)
        st.rerun()

if st.session_state.get('show_admin_login') and not st.session_state.get('admin_authed'):
    admin_pw_env = os.environ.get('ADMIN_PASSWORD')
    entered = st.text_input('Admin password', type='password', key='admin_pw_input')
    c1, c2 = st.columns(2)
    with c1:
        if st.button('Unlock', key='unlock_admin'):
            if admin_pw_env and entered == admin_pw_env:
                st.session_state['admin_authed'] = True
                st.session_state['admin_panel_open'] = True
                st.session_state['show_admin_login'] = False
                st.rerun()
            else:
                st.error('Invalid password or ADMIN_PASSWORD not set.')
    with c2:
        if st.button('Cancel', key='cancel_admin'):
            st.session_state['show_admin_login'] = False
            st.rerun()

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, step=1, value=0)
    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0.0, max_value=300.0, step=1.0, value=120.0)
    with col3:
        BloodPressure = st.number_input('Blood Pressure value', min_value=0.0, max_value=200.0, step=1.0, value=70.0)
    with col1:
        SkinThickness = st.number_input('Skin Thickness value', min_value=0.0, max_value=99.0, step=1.0, value=20.0)
    with col2:
        Insulin = st.number_input('Insulin Level', min_value=0.0, max_value=900.0, step=1.0, value=80.0)
    with col3:
        BMI = st.number_input('BMI value', min_value=0.0, max_value=80.0, step=0.1, value=24.0)
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0, max_value=2.5, step=0.01, value=0.5)
    with col2:
        Age = st.number_input('Age of the Person', min_value=0, max_value=120, step=1, value=30)

    diab_diagnosis = ''
    precautions = ''
    if st.button('Diabetes Test Result'):
        try:
            # Convert inputs to float and validate ranges
            user_input = [
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ]
            if not (0 <= user_input[0] <= 20):
                st.warning("Number of Pregnancies must be between 0 and 20.")
            elif not (0 <= user_input[1] <= 300):
                st.warning("Glucose Level must be between 0 and 300.")
            elif not (0 <= user_input[2] <= 200):
                st.warning("Blood Pressure must be between 0 and 200.")
            elif not (0 <= user_input[3] <= 99):
                st.warning("Skin Thickness must be between 0 and 99.")
            elif not (0 <= user_input[4] <= 900):
                st.warning("Insulin Level must be between 0 and 900.")
            elif not (0 <= user_input[5] <= 80):
                st.warning("BMI must be between 0 and 80.")
            elif not (0 <= user_input[6] <= 2.5):
                st.warning("Diabetes Pedigree Function must be between 0 and 2.5.")
            elif not (0 <= user_input[7] <= 120):
                st.warning("Age must be between 0 and 120.")
            else:
                diab_prediction = diabetes_model.predict([user_input])
                diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'

                # Provide additional information based on prediction
                if diab_prediction[0] == 1:
                    precautions = '''
                    **Precautions:**
                     - Follow a healthy and balanced diet, focusing on whole grains, vegetables, lean proteins, and healthy fats.
                    - Engage in regular physical activity, such as walking, swimming, or strength training, for at least 30 minutes a day.
                    - Monitor blood sugar levels regularly to ensure they remain within the target range.
                    - Stay hydrated by drinking plenty of water throughout the day.
                    - Maintain a healthy weight to reduce the risk of complications.
                    - Get adequate sleep, as poor sleep can impact blood sugar regulation.
                    - Consult with a healthcare provider regularly to adjust your treatment plan as needed.
                    - Manage stress through relaxation techniques like meditation or yoga, as stress can elevate blood sugar levels.
                    - Limit alcohol intake and avoid smoking, as both can interfere with diabetes management.
                    - Be aware of the signs of high or low blood sugar (e.g., dizziness, fatigue, shaking) and know how to respond.
                    - Consider joining a support group to stay motivated and share experiences with others managing diabetes.
                 '''

                else:
                    precautions = '''
                    "Maintain this healthy life and stay active!"
                    '''
                st.success(diab_diagnosis)
                st.markdown(precautions, unsafe_allow_html=True)
        except ValueError:
            st.warning("Please enter valid numeric values.")

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, step=1, value=40)
    with col2:
        sex = st.selectbox('Sex', options=[('Female',0),('Male',1)], index=1, format_func=lambda x: x[0])[1]
    with col3:
        cp = st.selectbox('Chest Pain types', options=[('Typical angina',0),('Atypical angina',1),('Non-anginal pain',2),('Asymptomatic',3)], index=0, format_func=lambda x: x[0])[1]
    with col1:
        trestbps = st.number_input('Resting Blood Pressure', min_value=0.0, max_value=200.0, step=1.0, value=120.0)
    with col2:
        chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=0.0, max_value=600.0, step=1.0, value=200.0)
    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[('No',0),('Yes',1)], index=0, format_func=lambda x: x[0])[1]
    with col1:
        restecg = st.selectbox('Resting ECG results', options=[('Normal',0),('ST-T abnormality',1),('LV hypertrophy',2)], index=0, format_func=lambda x: x[0])[1]
    with col2:
        thalach = st.number_input('Maximum Heart Rate', min_value=0.0, max_value=220.0, step=1.0, value=150.0)
    with col3:
        exang = st.selectbox('Exercise Induced Angina', options=[('No',0),('Yes',1)], index=0, format_func=lambda x: x[0])[1]
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise', min_value=0.0, max_value=6.0, step=0.1, value=1.0)
    with col2:
        slope = st.selectbox('Slope of peak exercise ST', options=[('Upsloping',0),('Flat',1),('Downsloping',2)], index=1, format_func=lambda x: x[0])[1]
    with col3:
        ca = st.number_input('Major vessels colored by fluoroscopy', min_value=0, max_value=4, step=1, value=0)
    with col1:
        thal = st.selectbox('Thal', options=[('Normal',0),('Fixed defect',1),('Reversible defect',2)], index=0, format_func=lambda x: x[0])[1]

    heart_diagnosis = ''
    precautions = ''
    if st.button('Heart Disease Test Result'):
        try:
            # Convert inputs to float and validate ranges
            user_input = [
                float(age), float(sex), float(cp), float(trestbps),
                float(chol), float(fbs), float(restecg), float(thalach),
                float(exang), float(oldpeak), float(slope), float(ca), float(thal)
            ]
            if not (0 <= user_input[0] <= 120):
                st.warning("Age must be between 0 and 120.")
            elif not (0 <= user_input[1] <= 1):
                st.warning("Sex must be 0 or 1.")
            elif not (0 <= user_input[2] <= 3):
                st.warning("Chest Pain types must be between 0 and 3.")
            elif not (0 <= user_input[3] <= 200):
                st.warning("Resting Blood Pressure must be between 0 and 200.")
            elif not (0 <= user_input[4] <= 600):
                st.warning("Serum Cholesterol must be between 0 and 600.")
            elif not (0 <= user_input[5] <= 1):
                st.warning("Fasting Blood Sugar must be 0 or 1.")
            elif not (0 <= user_input[6] <= 2):
                st.warning("Resting ECG results must be between 0 and 2.")
            elif not (0 <= user_input[7] <= 220):
                st.warning("Maximum Heart Rate must be between 0 and 220.")
            elif not (0 <= user_input[8] <= 1):
                st.warning("Exercise Induced Angina must be 0 or 1.")
            elif not (0 <= user_input[9] <= 6.0):
                st.warning("ST Depression must be between 0 and 6.0.")
            elif not (0 <= user_input[10] <= 2):
                st.warning("Slope must be between 0 and 2.")
            elif not (0 <= user_input[11] <= 4):
                st.warning("Major Vessels must be between 0 and 4.")
            elif not (0 <= user_input[12] <= 2):
                st.warning("Thal must be between 0 and 2.")
            else:
                heart_prediction = heart_disease_model.predict([user_input])
                heart_diagnosis = 'The person is having heart disease' if heart_prediction[0] == 1 else 'The person does not have heart disease'

                # Provide additional information based on prediction
                if heart_prediction[0] == 1:
                   precautions = '''
                      **Precautions:**
                    - Monitor your cholesterol and blood pressure regularly to keep them within healthy ranges.
                     - Follow a heart-healthy diet, rich in fruits, vegetables, whole grains, lean proteins, and healthy fats (e.g., olive oil, avocados, nuts).
                    - Engage in regular physical activity, aiming for at least 150 minutes of moderate exercise or 75 minutes of vigorous exercise per week.
                    - Avoid smoking and excessive alcohol consumption, as both can increase the risk of heart disease.
                    - Maintain a healthy weight to reduce strain on the heart.
                    - Manage stress through relaxation techniques such as meditation, deep breathing, or yoga.
                    - Ensure adequate sleep each night, as poor sleep can impact heart health.
                    - Stay hydrated, and aim for 6-8 glasses of water a day.
                    - Reduce your intake of processed foods, trans fats, and added sugars.
                    - Regularly check your blood sugar levels if you have diabetes, as it can increase heart disease risk.
                    - Consult with a healthcare provider regularly for personalized treatment and prevention strategies.
                    - Limit your salt intake to help manage blood pressure and reduce fluid retention.
                    '''

                else:
                    precautions = '''
                    "Keep up the good work in maintaining heart health!"
                    '''
                st.success(heart_diagnosis)
                st.markdown(precautions, unsafe_allow_html=True)
        except ValueError:
            st.warning("Please enter valid numeric values.")

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    with col1:
        RAP = st.text_input('MDVP:RAP')
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
    with col3:
        DDP = st.text_input('MDVP:DDP')
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
    with col1:
        APQ3 = st.text_input('MDVP:APQ3')
    with col2:
        APQ5 = st.text_input('MDVP:APQ5')
    with col3:
        APQ = st.text_input('MDVP:APQ')
    with col4:
        DDA = st.text_input('Shimmer:DDA')
    with col5:
        NHR = st.text_input('NHR')
    with col1:
        HNR = st.text_input('HNR')
    with col2:
        RPDE = st.text_input('RPDE')
    with col3:
        DFA = st.text_input('DFA')
    with col4:
        spread1 = st.text_input('spread1')
    with col5:
        spread2 = st.text_input('spread2')
    with col1:
        D2 = st.text_input('D2')
    with col2:
        PPE = st.text_input('PPE')

    parkinsons_diagnosis = ''
    precautions = ''
    if st.button("Parkinson's Test Result"):
        try:
            # Convert inputs to float
            user_input = [
                float(fo), float(fhi), float(flo), float(Jitter_percent), float(Jitter_Abs),
                float(RAP), float(PPQ), float(DDP), float(Shimmer), float(Shimmer_dB),
                float(APQ3), float(APQ5), float(APQ), float(DDA), float(NHR), float(HNR),
                float(RPDE), float(DFA), float(spread1), float(spread2), float(D2), float(PPE)
            ]

            if any(x < 0 or x > 1000 for x in user_input):  # Example range for input values
                st.warning("Please enter values within a reasonable range (0-1000).")
            else:
                parkinsons_prediction = parkinsons_model.predict([user_input])
                parkinsons_diagnosis = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease"
                if parkinsons_prediction[0] == 1:
                    precautions = '''
                     **Precautions:**
                     - Follow your doctor’s advice for medication and therapy, including speech and physical therapy, as recommended.
                     - Stay physically and mentally active by engaging in exercises that promote flexibility, strength, and balance.
                     - Maintain a healthy diet, focusing on antioxidant-rich foods, fiber, and healthy fats to support brain health.
                     - Stay hydrated and avoid dehydration, as it can worsen symptoms.
                      - Get adequate rest and avoid sleep disturbances to support brain function.
                     - Establish a regular daily routine to reduce stress and improve motor control.
                    - Join support groups or connect with others who have Parkinson’s to share experiences and coping strategies.
                    - Avoid falls by ensuring your living space is clear of obstacles and using assistive devices if needed.
                     - Manage stress through relaxation techniques like meditation, yoga, or mindfulness.
                    - Stay up-to-date with regular check-ups and adjust treatments as needed to manage symptoms.
                     - Limit caffeine intake as it can sometimes interfere with Parkinson’s medication.
                    - Stay positive and engaged with hobbies, social activities, and interests to boost mental well-being.
                    '''

                else:
                   precautions = '''
                "Maintain your healthy lifestyle and keep active!"
                '''
                st.success(parkinsons_diagnosis)
                st.markdown(precautions, unsafe_allow_html=True)
        except ValueError:
            st.warning("Please enter valid numeric values.")
# Chatbot Page
if selected == 'Chat with HealthBot':
    st.title("Chat with HealthBot 🩺")
    st.info("This assistant provides general information and is not a medical professional. For medical advice, please consult a qualified clinician.")

    # If API key is missing, allow user to enter it securely at runtime
    if not openai.api_key:
        st.warning("OPENAI_API_KEY is not set. Enter a valid key below to use the chatbot.")
        key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", key="key_input")
        if st.button("Save API Key"):
            if key_input and key_input.strip():
                key_val = key_input.strip()
                st.session_state.openai_api_key = key_val
                openai.api_key = key_val
                try:
                    if key_val.startswith("sk-or-"):
                        openai.api_base = "https://openrouter.ai/api/v1"
                    else:
                        if hasattr(openai, "api_base"):
                            openai.api_base = "https://api.openai.com/v1"
                except Exception:
                    pass
                st.success("API key saved for this session.")
                st.rerun()
            else:
                st.error("Please enter a valid API key.")
        # If still no key, return early to avoid calling API
        if not openai.api_key:
            st.stop()
    else:
        # Key is set; proceed silently without banners or test button
        pass

    # Model settings (persist in session)
    base = getattr(openai, "api_base", "")
    st.session_state.chat_model = "openrouter/auto" if base.startswith("https://openrouter.ai") else "gpt-3.5-turbo"
    st.session_state.chat_temp = 0.2

    # Initialize messages in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm your health assistant. How can I help you today?"}]

    # Utility to cap history length (keep system/assistant greeting + last 6 exchanges)
    def capped_history(base_messages):
        # Preserve the first assistant greeting
        head = base_messages[:1]
        tail = base_messages[1:]
        # Keep last 12 items (6 user+assistant turns)
        tail = tail[-12:]
        return head + tail

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            st.markdown(f"**HealthBot:** {message['content']}")
        else:
            st.markdown(f"**You:** {message['content']}")

    # Clear the input box if flagged, BEFORE rendering the widget
    if st.session_state.get("clear_user_input", False):
        st.session_state["user_input"] = ""
        st.session_state["clear_user_input"] = False

    # If a transcript arrived via query params from the mic component, prefill the input
    try:
        qp = st.query_params
        _stt = qp.get('stt')
        if _stt:
            st.session_state["user_input"] = _stt
            # Flag auto submit if requested
            if qp.get('autosend', '0') == '1':
                st.session_state['auto_submit'] = True
            # remove params to avoid repeated fills
            try:
                del st.query_params['stt']
            except Exception:
                pass
            try:
                del st.query_params['autosend']
            except Exception:
                pass
        # Handle attach toggle from polished buttons
        _att = qp.get('attach')
        if _att is not None:
            st.session_state['show_attach'] = (_att == '1')
            try:
                del st.query_params['attach']
            except Exception:
                pass
    except Exception:
        pass

    if 'show_attach' not in st.session_state:
        st.session_state['show_attach'] = False

    # Chat input in a form to avoid reruns while typing
    with st.form("chat_form", clear_on_submit=False):
        components.html(
            """
            <style>
            .chat-shell{border:1px solid #e5e7eb;border-radius:18px;background:#ffffff;padding:12px 14px;box-shadow:0 4px 16px rgba(0,0,0,0.05);}
            .chat-shell .ctrl-row{margin-top:8px}
            .pill-btn{display:inline-flex;align-items:center;gap:6px;background:#f3f4f6;color:#111827;border:1px solid #e5e7eb;border-radius:9999px;padding:6px 10px;font:500 12px system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;cursor:pointer}
            .pill-btn:hover{background:#e5e7eb}
            .pill-btn.disabled{opacity:.6;cursor:default}
            .pill-btn.listening{background:#fee2e2;border-color:#fecaca}
            </style>
            <script>
            (function(){
              const P=window.parent, D=P.document;
              function decorate(){
                const input = D.querySelector('input[placeholder="Type your message here..."]');
                if(!input) return false;
                
                let form = input.closest('form');
                if(!form) return false;
                const host = form.parentElement; if(!host) return false;
                if(!host.classList.contains('chat-shell')) host.classList.add('chat-shell');
                return true;
              }
              if(!decorate()){
                const mo = new MutationObserver(()=>{ if(decorate()) mo.disconnect(); });
                mo.observe(D.body, {childList:true, subtree:true});
                window.addEventListener('load', ()=>{ decorate(); });
              }
            })();
            </script>
            """,
            height=0,
        )
        col_in, col_btn = st.columns([11, 1])
        with col_in:
            user_input = st.text_input("Your Message:", key="user_input", placeholder="Type your message here...", label_visibility="collapsed")
            st.caption("Press Enter to send • Shift+Enter for newline")
        with col_btn:
            submitted = st.form_submit_button("Send", use_container_width=True)

        # Controls row below the input: + Attach 
        ctrl_attach, _sp = st.columns([1,11])
        with ctrl_attach:
            if st.form_submit_button(" Attach"):
                st.session_state['show_attach'] = not st.session_state.get('show_attach', False)
                st.rerun()

    # Inline attachments panel (toggles with Attach)
    if st.session_state.get('show_attach'):
        files_col, photo_col = st.columns([2,1])
        with files_col:
            uploaded_files = st.file_uploader(
                "Upload documents (PDF/TXT/CSV/JSON or images)",
                accept_multiple_files=True,
                type=["pdf","txt","csv","json","png","jpg","jpeg","webp","heic","heif"],
            )
        with photo_col:
            if 'photo_mode' not in st.session_state:
                st.session_state['photo_mode'] = 'camera'
            pm_left, pm_right = st.columns(2)
            with pm_left:
                if st.button("Camera", type="secondary"):
                    st.session_state['photo_mode'] = 'camera'
                    st.rerun()
            with pm_right:
                if st.button("Library", type="secondary"):
                    st.session_state['photo_mode'] = 'library'
                    st.rerun()
            if st.session_state.get('photo_mode') == 'camera':
                captured_photo = st.camera_input("Take a photo")
            else:
                captured_photo = st.file_uploader(
                    "Upload from phone library",
                    accept_multiple_files=False,
                    type=["png","jpg","jpeg","webp","heic","heif"],
                )
        if uploaded_files is not None:
            st.session_state["uploaded_files_info"] = [getattr(f,'name','file') for f in uploaded_files]
            st.session_state["uploaded_files_objs"] = uploaded_files
        if captured_photo is not None:
            st.session_state["captured_photo"] = captured_photo

        # Attachment previews and management
        prev_left, prev_right = st.columns([3,1])
        with prev_left:
            st.caption("Attached:")
            files = list(st.session_state.get("uploaded_files_objs") or [])
            cam = st.session_state.get("captured_photo")
            if cam is not None:
                files = files + [cam]
            if files:
                for i, f in enumerate(files[:6]):
                    try:
                        name = getattr(f, 'name', f"photo_{i+1}")
                        mime = getattr(f, 'type', 'application/octet-stream')
                        data = f.getvalue() if hasattr(f, 'getvalue') else None
                        is_img = (mime.startswith('image/') or (name.lower().split('.')[-1] in ["png","jpg","jpeg","webp","heic","heif"]))
                        cols = st.columns([1,5,1])
                        with cols[0]:
                            if is_img and data is not None:
                                st.image(data, width=54, caption="")
                            else:
                                st.markdown("")
                        with cols[1]:
                            st.markdown(f"**{name}**\n\n<sub>{mime}</sub>", unsafe_allow_html=True)
                        with cols[2]:
                            if st.button("", key=f"att_remove_{i}"):
                                base_list = list(st.session_state.get("uploaded_files_objs") or [])
                                if cam is not None and i == len(files)-1 and f is cam:
                                    st.session_state.pop("captured_photo", None)
                                else:
                                    # remove by index if in base_list
                                    if i < len(base_list):
                                        base_list.pop(i)
                                        st.session_state["uploaded_files_objs"] = base_list
                                        st.session_state["uploaded_files_info"] = [getattr(x,'name','file') for x in base_list]
                                st.rerun()
                    except Exception:
                        pass
            else:
                st.info("No attachments yet.")
        with prev_right:
            if st.button("Clear all", type="secondary"):
                st.session_state.pop("uploaded_files_objs", None)
                st.session_state.pop("uploaded_files_info", None)
                st.session_state.pop("captured_photo", None)
                st.rerun()

    # Auto-submit if mic requested it
    if st.session_state.get('auto_submit') and st.session_state.get('user_input'):
        submitted = True
        st.session_state.pop('auto_submit', None)

    col_sp, col_clear = st.columns([11,1])
    clear_clicked = col_clear.button("Clear Chat", use_container_width=True)

    # JS handler for Enter-to-send (uses query params like the voice button)
    try:
        _soe = True
        _flag = 'true' if _soe else 'false'
        components.html(
            """
            <script>
            (function(){
              const P=window.parent, D=P.document; let wired=false;
              function wire(){
                if(wired) return; const input=D.querySelector('input[placeholder=\"Type your message here...\"]'); if(!input) return;
                input.addEventListener('keydown', function(e){
                  if(e.key==='Enter' && !e.shiftKey && """ + _flag + """ ){
                    e.preventDefault(); try{
                      const u=new URL(P.location); u.searchParams.set('stt', input.value||''); u.searchParams.set('autosend','1'); P.location.href=u.toString();
                    }catch(err){}
                  }
                });
                wired=true;
              }
              if(D.readyState==='complete') wire(); else window.addEventListener('load', wire);
            })();
            </script>
            """,
            height=0,
        )
    except Exception:
        pass

    if clear_clicked:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm your health assistant. How can I help you today?"}]
        st.session_state.pop("uploaded_files_info", None)
        st.session_state.pop("uploaded_files_objs", None)
        st.session_state.pop("captured_photo", None)
        st.session_state["clear_user_input"] = True
        st.rerun()

    if submitted:
        if user_input:
            content = user_input
            _att = []
            _names = st.session_state.get("uploaded_files_info") or []
            if _names:
                _att.append("files: " + ", ".join(_names[:3]) + ("..." if len(_names) > 3 else ""))
            if st.session_state.get("captured_photo") is not None:
                _att.append("photo: 1")
            if _att:
                content = content + "\n\n[" + " | ".join(_att) + "]"
            temp_user_msg = {"role": "user", "content": content}
            # Build attachment context to help the assistant use file contents
            attach_msgs = []
            try:
                def _is_text_like(mime, name):
                    ext = (name.rsplit('.',1)[-1].lower() if '.' in name else '')
                    return (mime.startswith('text/') or ext in {'txt','csv','json','md','py','js','html','css'})
                def _read_pdf_text(data):
                    if not _pypdf:
                        return None
                    try:
                        from io import BytesIO
                        reader = _pypdf.PdfReader(BytesIO(data))
                        pages = []
                        for i, p in enumerate(reader.pages[:5]):
                            pages.append(p.extract_text() or '')
                        return "\n".join(pages)
                    except Exception:
                        return None
                MAX_CHARS = 4000
                files = list(st.session_state.get("uploaded_files_objs") or [])
                cam = st.session_state.get("captured_photo")
                if cam is not None:
                    files.append(cam)
                if files:
                    att_text_parts = []
                    for f in files[:4]:
                        try:
                            name = getattr(f, 'name', 'attachment')
                            mime = getattr(f, 'type', 'application/octet-stream')
                            data = f.getvalue() if hasattr(f, 'getvalue') else None
                            snippet = None
                            code_lang = 'text'
                            if data is not None:
                                if _is_text_like(mime, name):
                                    try:
                                        text = data.decode('utf-8', errors='replace')
                                    except Exception:
                                        text = str(data[:4096])
                                    snippet = text[:MAX_CHARS]
                                    # Hint code block language
                                    if name.lower().endswith('.csv'): code_lang = 'csv'
                                    elif name.lower().endswith('.json'): code_lang = 'json'
                                elif name.lower().endswith('.pdf'):
                                    text = _read_pdf_text(data) or '[PDF text extraction unavailable]'
                                    snippet = (text or '')[:MAX_CHARS]
                                else:
                                    # Images or binaries: include a note only
                                    snippet = None
                            header = f"Attachment: {name} ({mime})"
                            if snippet:
                                att_text_parts.append(header + "\n\n" + f"```{code_lang}\n" + snippet + "\n```")
                            else:
                                att_text_parts.append(header + "\n(Content not inlined; image or binary)")
                        except Exception:
                            pass
                    if att_text_parts:
                        attach_msgs.append({"role":"user","content":"\n\n".join(att_text_parts)})
            except Exception:
                pass
            temp_messages = capped_history(st.session_state.messages + [temp_user_msg] + attach_msgs)

            try:
                # Debug log for request payload
                print("Debug: Sending the following payload to OpenAI API:")
                print(temp_messages)

                # Call ChatCompletion API (legacy SDK)
                model_name = "openrouter/auto" if getattr(openai, "api_base", "").startswith("https://openrouter.ai") else "gpt-3.5-turbo"
                # Try streaming for perceived speed, fallback to non-streaming
                assistant_reply = ""
                try:
                    import time as _t
                    _start = _t.time()
                    stream = openai.ChatCompletion.create(
                        model=model_name,
                        messages=temp_messages,
                        max_tokens=256,
                        temperature=0.2,
                        stream=True,
                        request_timeout=30,
                    )
                    reply_box = st.empty()
                    for chunk in stream:
                        delta = chunk["choices"][0]["delta"].get("content", "") if "choices" in chunk else ""
                        if delta:
                            assistant_reply += delta
                            reply_box.markdown(f"**HealthBot (typing):** {assistant_reply}")
                    elapsed = _t.time() - _start
                    # If streaming produced no content, fallback to non-streaming
                    if not assistant_reply.strip():
                        response = openai.ChatCompletion.create(
                            model=model_name,
                            messages=temp_messages,
                            max_tokens=512,
                            temperature=0.2,
                            request_timeout=30,
                        )
                        assistant_reply = response.choices[0].message["content"] or ""
                    if assistant_reply.strip():
                        reply_box.markdown(f"**HealthBot:** {assistant_reply}\n\n<sub>Responded in {elapsed:.1f}s</sub>")
                    else:
                        st.error("No response received from the chat provider. Please try again.")
                except Exception:
                    response = openai.ChatCompletion.create(
                        model=model_name,
                        messages=temp_messages,
                        max_tokens=256,
                        temperature=0.2,
                        request_timeout=30,
                    )
                    assistant_reply = response.choices[0].message["content"]

                # Limit response to 250 words
                assistant_reply_words = assistant_reply.split()
                if len(assistant_reply_words) > 250:
                    assistant_reply = " ".join(assistant_reply_words[:250]) + "..."

                # Commit both user and assistant messages only on success with non-empty reply
                if assistant_reply.strip():
                    st.session_state.messages = temp_messages + [{"role": "assistant", "content": assistant_reply}]
                    # chat messages not counted in visits by design
                else:
                    # Remove the temp user message since the request failed to produce an answer
                    st.error("The assistant did not return any content. Please try again.")
                st.session_state["clear_user_input"] = True

                # Debug log for API response
                print("Debug: Received response from OpenAI API:")
                try:
                    print(response)
                except NameError:
                    print({"message_preview": assistant_reply[:120]})

                # Rerun the app to refresh chat history only on success
                st.rerun()  # Updated to the new rerun function.

            except Exception as e:
                print("Error: Exception occurred while calling chat API.")
                print(e)
                msg = str(e)
                low = msg.lower()
                try:
                    fallback_resp = openai.ChatCompletion.create(
                        model=model_name,
                        messages=temp_messages,
                        max_tokens=128,
                        temperature=0.2,
                        request_timeout=20,
                    )
                    assistant_reply = fallback_resp.choices[0].message.get("content", "")
                    if assistant_reply.strip():
                        st.session_state.messages = temp_messages + [{"role": "assistant", "content": assistant_reply}]
                        st.session_state["clear_user_input"] = True
                        st.rerun()
                    else:
                        st.error("Unable to get a response right now. Please try again in a moment.")
                except Exception:
                    st.error("The chat service is temporarily unavailable. Please try again shortly.")

# Admin Panel
if st.session_state.get('admin_panel_open'):
    st.title('Admin Panel')
    top_l, top_r = st.columns([8,1])
    with top_r:
        if st.button('Logout', key='admin_logout'):
            st.session_state['admin_authed'] = False
            st.session_state['admin_panel_open'] = False
            st.session_state['show_admin_login'] = False
            st.rerun()
    # Require password from env
    admin_pw_env = os.environ.get('ADMIN_PASSWORD')
    if not st.session_state.get('admin_authed'):
        st.stop()

    # Filters
    st.subheader('Filters')
    scope = st.selectbox('Time range', ['All', 'Today', 'Last 7 days', 'Custom range'], index=1)
    where = ""
    params = []
    if scope == 'Today':
        where = "WHERE date(last_ts)=date('now')"
    elif scope == 'Last 7 days':
        where = "WHERE date(last_ts) >= date('now','-6 day')"
    elif scope == 'Custom range':
        colA, colB = st.columns(2)
        start = colA.date_input('Start (UTC)')
        end = colB.date_input('End (UTC)')
        where = "WHERE date(last_ts) BETWEEN ? AND ?"
        params = [str(start), str(end)]

    try:
        cur = visits_conn.cursor()
        # Unique visitors
        cur.execute(f"SELECT COUNT(DISTINCT visitor_id) FROM visits {where}", params)
        unique_visitors = cur.fetchone()[0] or 0
        # Total visits
        cur.execute(f"SELECT COALESCE(SUM(count),0) FROM visits {where}", params)
        total_visits = cur.fetchone()[0] or 0
        col1, col2 = st.columns(2)
        col1.metric('Unique visitors (session-based)', str(unique_visitors))
        col2.metric('Total visits', str(total_visits))

        # Visits by page
        st.subheader('Visits by page')
        cur.execute(f"SELECT page, SUM(count) FROM visits {where} GROUP BY page ORDER BY 2 DESC", params)
        rows = cur.fetchall()
        if rows:
            st.table([{ 'page': r[0], 'visits': r[1]} for r in rows])
        else:
            st.info('No visits recorded for this range.')

        # Detailed table
        st.subheader('Details')
        cur.execute(f"SELECT visitor_id, page, first_ts, last_ts, count, user_agent, ip FROM visits {where} ORDER BY last_ts DESC", params)
        details = cur.fetchall()
        if details:
            # Mask visitor IDs for privacy: show only first 4 and last 4 chars
            def _mask(v):
                return v if not isinstance(v, str) or len(v) <= 8 else f"{v[:4]}…{v[-4:]}"
            def _mask_ip(ip):
                if not isinstance(ip, str):
                    return ip
                parts = ip.split('.')
                if len(parts) == 4:
                    return '.'.join(parts[:3] + ['xxx'])
                # IPv6 or others: truncate
                return (ip[:12] + '…') if len(ip) > 12 else ip
            def _short(s):
                return s[:60] + '…' if isinstance(s, str) and len(s) > 60 else s
            data = [{'visitor_id': _mask(d[0]), 'page': d[1], 'first_ts': d[2], 'last_ts': d[3], 'count': d[4], 'user_agent': _short(d[5]), 'ip': _mask_ip(d[6])} for d in details]
            st.table(data)
            # CSV export
            import io, csv
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=['visitor_id','page','first_ts','last_ts','count','user_agent','ip'])
            writer.writeheader()
            writer.writerows(data)
            st.download_button('Download CSV', data=buf.getvalue(), file_name='visits.csv', mime='text/csv')
        else:
            st.info('No detailed rows for this range.')

        # Dangerous actions
        st.subheader('Danger zone')
        with st.form('reset_form'):
            st.warning('This will permanently delete all visit records.')
            confirm = st.checkbox('Yes, delete all data')
            do_reset = st.form_submit_button('Reset analytics data')
        if do_reset and confirm:
            try:
                cur.execute('DELETE FROM visits')
                visits_conn.commit()
                st.success('All visit data deleted.')
                st.rerun()
            except Exception as e:
                st.error(f'Failed to delete data: {e}')
    except Exception as e:
        st.error(f'Admin error: {e}')

# Persistent footer
st.markdown(
    """
    <style>
    .app-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 8px 0;
        color: rgba(240,240,240,0.95);
        font-weight: 600;
        background: rgba(20, 20, 20, 0.85);
        border-top: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 -2px 8px rgba(0,0,0,0.25);
        backdrop-filter: blur(6px);
        z-index: 1000;
    }
    </style>
    <div class="app-footer">COPYRIGHT: MANIK AND KESHAV</div>
    """,
    unsafe_allow_html=True,
)