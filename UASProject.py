import streamlit as st
import base64
import numpy as np
from joblib import load

# Fungsi untuk prediksi menggunakan model Random Forest
def predict_rf(tree_ls, X_test):
    pred_ls = []
    for x in X_test:
        ensemble_preds = [predict_tree(tree, x) for tree in tree_ls]
        final_pred = max(set(ensemble_preds), key=ensemble_preds.count)  # Kelas mayoritas
        pred_ls.append(final_pred)
    return np.array(pred_ls)

def predict_tree(tree, x):
    while not tree['is_leaf']:
        if x[tree['feature_idx']] <= tree['split_point']:
            tree = tree['left_child']
        else:
            tree = tree['right_child']
    return tree['prediction']

# GUI Streamlit
st.set_page_config(page_title="Cognify", page_icon="🧠", layout="centered")

def add_background(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    css = f"""
    <style>
    .block-container {{
        padding: 0 !important;
    }}
    header {{
        display: none;
    }}
    body {{
        margin: 0;
        padding: 0;
        overflow: hidden;
    }}
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 100vh;
        width: 100vw;
        margin: 0;
        padding: 0;
    }} 
    header, footer {{
        background-color: rgba(0, 0, 0, 0) !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = 1
if "positive" not in st.session_state:
    st.session_state.positive = False

def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1

# Page 1
if st.session_state.page == 1:
    add_background("Home Cognify.png")

    # Tambahkan spacing vertikal seperti sebelumnya
    for _ in range(27):  # Menggantikan multiple st.write("")
        st.write("")

    st.markdown("""
        <style>
        .stButton>button {
            background-color: grey;
            color: #0C2227;
            border: 1px solid grey;
            margin-left: 100px;  /* Tambahkan margin dari kiri */
            position: relative;  /* Tambahkan positioning */
            left: -200px;      /* Geser ke kiri */
        }
        .stButton>button:hover {
            background-color: #64D98A;
            color: #0C2227;
            border: #64D98A;
        }
        .stButton>button:active {
            background-color: #64D98A !important;
            color: #0C2227 !important;
            border: #64D98A;
        }
        .stButton>button:focus {
            background-color: #64D98A !important;
            color: #0C2227 !important;
            border: #64D98A !important;
            box-shadow: none !important;
        }
        .stButton>button:focus:not(:focus-visible) {
            outline: none !important;
        }
        
        .block-container {
        margin-left: 70px;   /* Lebih ke kiri */
        left: -150px;        /* Lebih ke kiri lagi */
        }
        
        div[data-testid="stVerticalBlock"] {
        margin-left: 70px;   /* Lebih ke kiri */
        left: -150px;        /* Lebih ke kiri lagi */
        }
        </style>
        """, unsafe_allow_html=True)
    
    if st.button("Start Your Detection Now"):
        next_page()

# Page 2
elif st.session_state.page == 2:
    add_background("Detection Cognify.png") 

    for _ in range(10): 
        st.write("")
    
    st.markdown("""
    <style>
    /* Style untuk number input container */
    .stNumberInput div[data-baseweb="input"] {
                # background-color: #1E1E1E !important;
        border: none !important;
    }
                
    /* Style untuk tombol plus dan minus saat hover */
    .stNumberInput button:hover {
        background-color: #64D98A !important;
        color: white !important;
    }
    
    /* Style untuk tombol plus dan minus saat active/diklik */
    .stNumberInput button:active {
        background-color: #64D98A !important;
        color: white !important;
    }
    
    /* Style untuk number input field */
    input[type="number"] {
                # background-color: #1E1E1E !important;
        color: white !important;
        border: none !important;
    }

    /* Menghilangkan border focus */
    .stNumberInput div[data-baseweb="input"]:focus-within {
        border: none !important;
        box-shadow: none !important;
    }
    
    .stButton>button:focus {
    background-color: #64D98A !important;
    color: #0C2227 !important;
    border: #64D98A !important;
    box-shadow: none !important;
    }
                
    /* Mencegah outline merah saat focus */
    .stButton>button:focus:not(:focus-visible) {
        outline: none !important;
    }
    </style>
""", unsafe_allow_html=True)
    
    # Gunakan kolom untuk tata letak text input
    col1, col2 = st.columns(2)

    with col1:
        memory_complaints = st.number_input(
            "Memory Complaints (0-10):",
            min_value=0.0, max_value=10.0, value=0.0, step=1.0,
            help="A measurement of how much a person feels they have problems with their memory, where 0 means no problems and 10 means very serious problems."
        )
        behavioral_problems = st.number_input(
            "Behavioral Problems (0-10):",
            min_value=0.0, max_value=10.0, value=0.0, step=1.0,
            help="An assessment of a person's behavioral changes or emotional problems, with 0 signifying no problem and 10 signifying very difficult behavior such as anger or confusion."
        )
        mmse = st.number_input(
            "MMSE Score (0-30):",
            min_value=0.0, max_value=30.0, value=0.0, step=1.0,
            help="A measurement of how much a person feels they have problems with their memory, where 0 means no problems and 10 means very serious problems."
        )

    with col2:
        adl = st.number_input(
            "ADL Score (0-10):",
            min_value=0.0, max_value=10.0, value=0.0, step=1.0,
            help="An assessment of the ability to perform daily activities such as eating, bathing, and dressing, with 0 indicating the ability to do everything by oneself and 10 indicating the inability to do anything by oneself."
        )
        functional_assessment = st.number_input(
            "Functional Assessment Score (0-10):",
            min_value=0.0, max_value=10.0, value=0.0, step=1.0,
            help="An evaluation of a person's ability to care for themselves, where 0 means fully independent and 10 means needing a lot of help."
        )

    # Load model dari file
    model_path = 'randomforest_model.joblib'
    forest = load(model_path)

    st.markdown("""
        <style>
        .stButton>button {
            background-color: grey;
            color: #0C2227;
            border: 1px solid grey;
        }
        .stButton>button:hover {
            background-color: #64D98A;
            color: #0C2227;
            border: #64D98A;
        }
        .stButton>button:active {
            background-color: #64D98A !important;
            color: #0C2227 !important;
            border: #64D98A;
        }
        </style>
        """, unsafe_allow_html=True)
        
    # Prediksi jika tombol ditekan
    if st.button("Predict"):
        # Data input dari user
        user_input = np.array([memory_complaints, behavioral_problems, mmse, adl, functional_assessment]).reshape(1, -1)

        # Prediksi menggunakan model
        user_prediction = predict_rf(forest, user_input)

        # Tentukan hasil berdasarkan ambang batas
        threshold = 0.5
        if np.mean(user_prediction) > threshold:
            result = "Terdeteksi Alzheimer"
            st.session_state.positive = 1
        else:
            result = "Tidak Terdeteksi Alzheimer"
            st.session_state.positive = 0

        # Set halaman ke Page 3
        st.session_state.page = 3

# Page 3
elif st.session_state.page == 3:
    # Positive prediction (detected Alzheimer's)
    if st.session_state.positive == 1:
        add_background("Result Positive.png")
    else:
        # Negative prediction (not detected)
        add_background("Result Negative.png")