import streamlit as st
import os
from keras.models import load_model as keras_load_model
from PIL import Image, ImageOps
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import base64
import time

# ================== THIẾT LẬP TRANG ==================
st.set_page_config(
    page_title="🌿 Trí Tuệ Nhân Tạo Nông Nghiệp",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== CSS TÙY CHỈNH ==================
def add_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    .main {
        padding: 0rem 1rem;
    }

    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }

    .main-header {
        background: linear-gradient(90deg, #00C851, #00695C);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        animation: slideDown 1s ease-out;
    }

    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .main-header p {
        color: #E8F5E8;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }

    .feature-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
        transition: all 0.3s ease;
        animation: fadeInUp 0.8s ease-out;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }

    .upload-section {
        background: linear-gradient(45deg, #E8F5E8, #F1F8E9);
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #4CAF50;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }

    .upload-section:hover {
        border-color: #2E7D32;
        background: linear-gradient(45deg, #C8E6C9, #DCEDC8);
    }

    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        height: 400px;
        overflow-y: auto;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }

    .user-message {
        background: linear-gradient(45deg, #2196F3, #21CBF3);
        color: white;
        padding: 1rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        display: block;
        animation: slideInRight 0.5s ease-out;
    }

    .bot-message {
        background: linear-gradient(45deg, #4CAF50, #8BC34A);
        color: white;
        padding: 1rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.5rem 0;
        max-width: 80%;
        animation: slideInLeft 0.5s ease-out;
    }

    .sidebar .stSelectbox > div > div {
        background: linear-gradient(45deg, #E8F5E8, #F1F8E9);
        border-radius: 10px;
    }

    .stButton > button {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
        background: linear-gradient(45deg, #45a049, #4CAF50);
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: scale(1.05);
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    .stFileUploader > div > div > div {
        text-align: center;
    }

    .success-animation {
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            transform: scale(1);
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ================== KHỞI TẠO MODEL VÀ CHATBOT ==================
@st.cache_resource
def load_model():
    """
    Load 2-stage Keras models:
    - Stage 1: Plant Type Classifier (from model/stage1.h5)
    - Stage 2: Disease Classifiers (from model/*.h5 files)
    """
    np.set_printoptions(suppress=True)

    try:
        # === STAGE 1: Load Plant Type Classifier ===
        stage1_model = keras_load_model("model/stage1.h5", compile=False)
        stage1_class_names = open("model/stage1.txt", "r").readlines()
        
        # Extract plant types from stage1 class names (remove "0 " prefix and newline)
        plant_types = [line.strip().split(' ', 1)[1] if ' ' in line else line.strip() for line in stage1_class_names]
        
        # === STAGE 2: Load Disease Classifiers ===
        stage2_models = {}
        stage2_class_names = {}
        
        # Available stage 2 models from kr.py
        available_models = ['apple', 'mango', 'tomato']
        
        for plant_type in available_models:
            model_path = f"model/{plant_type}.h5"
            txt_path = f"model/{plant_type}.txt"
            
            if os.path.exists(model_path) and os.path.exists(txt_path):
                stage2_models[plant_type] = keras_load_model(model_path, compile=False)
                stage2_class_names[plant_type] = open(txt_path, "r").readlines()
                print(f"✅ Loaded Stage 2 model for {plant_type}")
            else:
                st.warning(f"⚠️ Model cho {plant_type} không tồn tại tại {model_path}")
        
        return {
            'stage1_model': stage1_model,
            'stage1_class_names': stage1_class_names,
            'stage2_models': stage2_models,
            'stage2_class_names': stage2_class_names,
            'plant_types': plant_types
        }
    except Exception as e:
        st.error(f"❌ Không thể tải model: {e}")
        return None

@st.cache_resource
def load_chatbot():
    load_dotenv()

    try:
        loader = TextLoader("books/huong_dan_trong_cay.txt", encoding="utf-8")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        combined_text = "\n\n".join([chunk.page_content for chunk in chunks])

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        return llm, combined_text
    except Exception as e:
        st.error(f"❌ Không thể tải chatbot: {e}")
        return None, None

def predict_disease(image, models_dict):
    """
    Two-stage prediction using Keras models:
    1. Predict plant type
    2. Based on plant type, predict specific disease
    """
    if models_dict is None:
        return None, 0.0, None, 0.0

    try:
        stage1_model = models_dict['stage1_model']
        stage1_class_names = models_dict['stage1_class_names']
        stage2_models = models_dict['stage2_models']
        stage2_class_names = models_dict['stage2_class_names']
        plant_types = models_dict['plant_types']
        
        # Preprocess image for Keras model (224x224, normalized)
        img = image.convert("RGB")
        size = (224, 224)
        img_resized = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        image_array = np.asarray(img_resized)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        
        # Create batch with shape (1, 224, 224, 3)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # === STAGE 1: Predict Plant Type ===
        stage1_prediction = stage1_model.predict(data)
        plant_idx = np.argmax(stage1_prediction)
        plant_type = plant_types[plant_idx]
        plant_confidence = float(stage1_prediction[0][plant_idx])
        
        # === STAGE 2: Predict Disease ===
        if plant_type not in stage2_models:
            return plant_type, plant_confidence, "Model chưa được train", 0.0
        
        stage2_model = stage2_models[plant_type]
        disease_class_names = stage2_class_names[plant_type]
        
        stage2_prediction = stage2_model.predict(data)
        disease_idx = np.argmax(stage2_prediction)
        
        # Extract disease name (remove "0 " or "1 " prefix if exists)
        disease_raw = disease_class_names[disease_idx]
        disease = disease_raw.strip().split(' ', 1)[1] if ' ' in disease_raw else disease_raw.strip()
        disease_confidence = float(stage2_prediction[0][disease_idx])
        
        return plant_type, plant_confidence, disease, disease_confidence
        
    except Exception as e:
        st.error(f"❌ Lỗi dự đoán: {e}")
        return None, 0.0, None, 0.0

def chatbot_response(query, context_info, llm, combined_text):
    if llm is None:
        return "❌ Chatbot chưa được khởi tạo."

    try:
        messages = [
            SystemMessage(content="Bạn là trợ lý AI chuyên về nông nghiệp. Trả lời thân thiện, dễ hiểu cho nông dân Việt Nam."),
            HumanMessage(content=f"Tài liệu:\n{combined_text}\n\nThông tin bệnh cây: {context_info}\n\nCâu hỏi: {query}")
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"❌ Lỗi chatbot: {e}"

# ================== GIAO DIỆN CHÍNH ==================
def main():
    add_custom_css()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌿 Trí Tuệ Nhân Tạo Nông Nghiệp</h1>
        <p>🤖 Hệ thống nhận diện bệnh lá cây & Tư vấn nông nghiệp thông minh</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Chọn chức năng")

        page = st.selectbox(
            "📱 Điều hướng",
            ["🏠 Trang chủ", "🔬 Nhận diện bệnh", "💬 Chatbot tư vấn", "📊 Thống kê"]
        )

        st.markdown("---")
        st.markdown("### 🌟 Thông tin")
        st.info("🚀 Phiên bản: 2.0\n\n🎯 Tác giả: Nguyễn Quang Minh\n\n💡 Công nghệ: Keras + Streamlit")

    # Load models
    models_dict = load_model()
    llm, combined_text = load_chatbot()

    # Main content
    if page == "🏠 Trang chủ":
        show_home_page()
    elif page == "🔬 Nhận diện bệnh":
        show_disease_detection(models_dict)
    elif page == "💬 Chatbot tư vấn":
        show_chatbot(llm, combined_text)
    elif page == "📊 Thống kê":
        show_statistics()

def show_home_page():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🔬 Nhận diện bệnh lá cây</h3>
            <p>Sử dụng AI để phân tích hình ảnh lá cây và nhận diện các loại bệnh phổ biến với độ chính xác cao.</p>
            <ul>
                <li>✅ Hỗ trợ nhiều loại cây trồng</li>
                <li>✅ Độ chính xác > 90%</li>
                <li>✅ Kết quả tức thì</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h3>📊 Thống kê chi tiết</h3>
            <p>Theo dõi lịch sử phân tích và xu hướng bệnh cây để đưa ra các biện pháp phòng ngừa hiệu quả.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>💬 Chatbot tư vấn</h3>
            <p>Trò chuyện với AI chuyên gia nông nghiệp để nhận tư vấn về:</p>
            <ul>
                <li>🌱 Kỹ thuật trồng trọt</li>
                <li>🐛 Phòng chống sâu bệnh</li>
                <li>💊 Sử dụng thuốc bảo vệ thực vật</li>
                <li>🌿 Chăm sóc cây trồng</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        st.markdown("### 📈 Số liệu hệ thống")
        col1_metric, col2_metric, col3_metric = st.columns(3)

        with col1_metric:
            st.markdown("""
            <div class="metric-card">
                <h2>1,250+</h2>
                <p>Lượt phân tích</p>
            </div>
            """, unsafe_allow_html=True)

        with col2_metric:
            st.markdown("""
            <div class="metric-card">
                <h2>95.8%</h2>
                <p>Độ chính xác</p>
            </div>
            """, unsafe_allow_html=True)

        with col3_metric:
            st.markdown("""
            <div class="metric-card">
                <h2>24/7</h2>
                <p>Hỗ trợ</p>
            </div>
            """, unsafe_allow_html=True)

def show_disease_detection(models_dict):
    st.markdown("## 🔬 Nhận diện bệnh lá cây - 2 Giai đoạn")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div class="upload-section">
            <h3>📸 Tải lên hình ảnh lá cây</h3>
            <p>Hỗ trợ định dạng: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)

        # Image input options
        upload_option = st.radio(
            "Chọn cách tải ảnh:",
            ["📤 Tải file", "📷 Chụp ảnh"],
            key="disease_upload_option"
        )

        uploaded_file = None
        if upload_option == "📤 Tải file":
            uploaded_file = st.file_uploader(
                "Chọn ảnh lá cây",
                type=["jpg", "jpeg", "png"],
                help="Tải lên hình ảnh rõ ràng của lá cây cần phân tích"
            )
        else:
            # Camera input
            uploaded_file = st.camera_input("📷 Chụp ảnh lá cây")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="🖼️ Ảnh đã tải lên", use_container_width=True)

            if st.button("🔍 Phân tích ngay", use_container_width=True):
                with st.spinner("🤖 Đang phân tích..."):
                    time.sleep(1)  # Animation effect
                    plant_type, plant_conf, disease, disease_conf = predict_disease(image, models_dict)

                    if plant_type:
                        with col2:
                            # Stage 1 Result
                            st.markdown(f"""
                            <div class="feature-card success-animation">
                                <h3>� GIAI ĐOẠN 1: Loại cây</h3>
                                <h2 style="color: #2196F3;">🌿 {plant_type.upper()}</h2>
                                <h3>📊 Độ tin cậy: {plant_conf:.1%}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            st.progress(float(plant_conf))
                            
                            # Stage 2 Result
                            st.markdown(f"""
                            <div class="feature-card success-animation">
                                <h3>🔬 GIAI ĐOẠN 2: Bệnh cụ thể</h3>
                                <h2 style="color: #4CAF50;">🦠 {disease}</h2>
                                <h3>📊 Độ tin cậy: {disease_conf:.1%}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            st.progress(float(disease_conf))

                            # Overall assessment
                            if plant_conf > 0.8 and disease_conf > 0.8:
                                st.success("✅ Cả 2 giai đoạn đều có độ tin cậy cao!")
                            elif plant_conf > 0.6 and disease_conf > 0.6:
                                st.warning("⚠️ Kết quả có độ tin cậy trung bình")
                            else:
                                st.error("❌ Độ tin cậy thấp, vui lòng thử ảnh khác hoặc ảnh rõ hơn")

                            # Store result and image in session state for chatbot
                            st.session_state.disease_info = f"Loại cây: {plant_type} (tin cậy {plant_conf:.1%}). Bệnh: {disease} (tin cậy {disease_conf:.1%})"
                            st.session_state.analyzed_image = image

    with col2:
        if 'disease_info' not in st.session_state:
            st.markdown("""
            <div class="feature-card">
                <h3>💡 Hướng dẫn sử dụng</h3>
                <ol>
                    <li>📷 Chụp ảnh lá cây rõ ràng</li>
                    <li>📤 Tải ảnh lên hệ thống</li>
                    <li>🔍 Nhấn nút "Phân tích ngay"</li>
                    <li>🌱 Xem kết quả giai đoạn 1: Loại cây</li>
                    <li>🔬 Xem kết quả giai đoạn 2: Bệnh cụ thể</li>
                    <li>💬 Tư vấn thêm qua Chatbot</li>
                </ol>
                <br>
                <h3>🎯 Ưu điểm phương pháp 2 giai đoạn:</h3>
                <ul>
                    <li>✅ Nhận diện đúng loại cây trước</li>
                    <li>✅ Tránh nhầm lẫn giữa các loại cây</li>
                    <li>✅ Độ chính xác cao hơn cho bệnh cụ thể</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

def show_chatbot(llm, combined_text):
    st.markdown("## 💬 Chatbot tư vấn nông nghiệp")

    # Show image from disease detection if available
    if 'analyzed_image' in st.session_state and 'disease_info' in st.session_state:
        st.markdown("""
        <div class="feature-card">
            <h3>🖼️ Ảnh đã phân tích từ phần nhận diện bệnh</h3>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(st.session_state.analyzed_image, caption="📸 Ảnh lá cây đã phân tích", use_container_width=True)
            st.info(f"🔍 {st.session_state.disease_info}")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat messages display
    st.markdown("### 💬 Cuộc trò chuyện")

    # Create scrollable chat area with better styling
    if st.session_state.messages:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
                    <div class="user-message" style="max-width: 80%; margin-left: auto;">
                        👤 {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                    <div class="bot-message" style="max-width: 80%; margin-right: auto;">
                        🤖 {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="feature-card" style="text-align: center; margin: 20px 0;">
            <h4>🌟 Chào mừng đến với Chatbot tư vấn nông nghiệp!</h4>
            <p>Hãy đặt câu hỏi về nông nghiệp hoặc về kết quả phân tích bệnh lá cây.</p>
        </div>
        """, unsafe_allow_html=True)

    # Chat input area
    st.markdown("---")

    # Text input and send button
    col_input, col_send = st.columns([4, 1])

    with col_input:
        user_input = st.text_input(
            "💬 Nhập câu hỏi:",
            placeholder="Ví dụ: Cách phòng chống bệnh đốm lá ớt?",
            key="chat_input",
            label_visibility="collapsed"
        )

    with col_send:
        send_button = st.button("📤 Gửi", use_container_width=True)

    if send_button and user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get context from disease detection if available
        context_info = st.session_state.get('disease_info', "")

        # Get bot response
        with st.spinner("🤖 Đang suy nghĩ..."):
            response = chatbot_response(user_input, context_info, llm, combined_text)
            st.session_state.messages.append({"role": "assistant", "content": response})

        st.rerun()

    # Quick suggestions
    st.markdown("### 💡 Câu hỏi gợi ý")

    suggestions = [
        "🌱 Cách trồng và chăm sóc cây ớt?",
        "🐛 Phòng chống sâu bệnh trên cà chua",
        "💊 Thuốc nào tốt cho bệnh đốm lá?",
        "🌿 Kỹ thuật tưới nước cho rau xanh"
    ]

    # Display suggestions in 2 columns
    col1, col2 = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        if i % 2 == 0:
            with col1:
                if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": suggestion})
                    context_info = st.session_state.get('disease_info', "")
                    response = chatbot_response(suggestion, context_info, llm, combined_text)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
        else:
            with col2:
                if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": suggestion})
                    context_info = st.session_state.get('disease_info', "")
                    response = chatbot_response(suggestion, context_info, llm, combined_text)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()

def show_statistics():
    st.markdown("## 📊 Thống kê và Báo cáo")

    # Sample data for demonstration
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📈 Tổng phân tích",
            value="1,247",
            delta="23 hôm nay"
        )

    with col2:
        st.metric(
            label="🎯 Độ chính xác",
            value="95.8%",
            delta="2.1%"
        )

    with col3:
        st.metric(
            label="💬 Câu hỏi chatbot",
            value="856",
            delta="45 hôm nay"
        )

    with col4:
        st.metric(
            label="👥 Người dùng",
            value="234",
            delta="12 mới"
        )

    # Charts
    st.markdown("### 📈 Biểu đồ phân tích")

    col1, col2 = st.columns(2)

    with col1:
        # Sample chart data
        chart_data = {
            'Ngày': ['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'CN'],
            'Phân tích': [23, 45, 56, 78, 32, 67, 89]
        }
        st.bar_chart(chart_data, x='Ngày', y='Phân tích')

    with col2:
        # Disease distribution
        import pandas as pd
        disease_data = pd.DataFrame({
            'Loại bệnh': ['Bệnh đốm lá', 'Bệnh héo xanh', 'Bệnh thối rễ', 'Khỏe mạnh'],
            'Số lượng': [35, 28, 22, 15]
        })
        st.bar_chart(disease_data.set_index('Loại bệnh'))

if __name__ == "__main__":
    main()
