import streamlit as st
import cv2
import numpy as np
import google.generativeai as genai

from detector import EnhancedYOLODetector

def apply_ui_styles():
    with open('styles.css') as f:
        style = f.read()
    st.markdown(f'<style>{style}</style>', unsafe_allow_html=True)

st.set_page_config(
    page_title="Nhận dạng giống vật nuôi",
    layout="wide",
    initial_sidebar_state="collapsed"
)

apply_ui_styles()

# Header
st.markdown("<h1>Nhận dạng giống vật nuôi và tra cứu thông tin</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Hệ Thống Nhận Dạng Loài và Tra Cứu Thông Tin Dựa Trên AI</p>", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.markdown("### Công Nghệ Sử Dụng")
    st.markdown("""
    ####
    - Nhận Dạng Đối Tượng YOLO
    - Google Gemini AI
    - Tích Hợp Wikipedia API
    
    ---
    
    #### Hướng Dẫn Sử Dụng
    1. Tải lên hình ảnh động vật
    2. Đợi AI phân tích
    3. Xem kết quả chi tiết
    """)
    
    st.markdown("---")
    st.markdown("_Đây là bản demo và vẫn còn phát triển_")

# --- Main App ---
gemini_model_instance = None
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_api_key)
    gemini_model_instance = genai.GenerativeModel('gemini-2.5-flash')
except (KeyError, FileNotFoundError):
    st.error("Lỗi: Không tìm thấy 'GEMINI_API_KEY' trong cấu hình.")
    st.stop()
except Exception as e:
    st.error(f"Lỗi khi cấu hình Gemini API: {e}")
    st.stop()
detector = EnhancedYOLODetector(model_path='best.pt', gemini_model=gemini_model_instance)

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    uploaded_file = st.file_uploader(
        "Tải Lên Hình Ảnh Để Phân Tích",
        type=["jpg", "jpeg", "png"],
        help="Định dạng hỗ trợ: JPG, JPEG, PNG"
    )

    if uploaded_file is not None:
        st.markdown(f"**Tệp đã chọn:** {uploaded_file.name}")
        st.markdown(f"**Kích thước:** {uploaded_file.size / 1024:.2f} KB")

if uploaded_file is not None:
    image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(image_bytes, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.markdown("---")
    
    spinner_placeholder = st.empty()
    spinner_placeholder.markdown("""
        <div class="custom-spinner">
            <div class="custom-spinner-icon"></div>
            <span>Đang phân tích và tra cứu thông tin...</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Tạo một bản sao có thể ghi của frame để đảm bảo tương thích với PyTorch
    # Điều này khắc phục lỗi RuntimeError: "given numpy array is not writeable"
    frame_copy = frame.copy()
    results = detector.enhanced_detection(frame_copy)
    spinner_placeholder.empty()
        
    st.success('Phân tích hoàn tất!')
    st.markdown("---")
    st.markdown("### Hình Ảnh Đã Tải Lên")
    col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
    with col_img2:
        st.image(frame_rgb, caption="Hình ảnh được gửi để phân tích", use_container_width=True)
    if results['object_info']:
        st.markdown("---")
        st.markdown("## Kết Quả Nghiên Cứu")
        for class_name, info in results['object_info'].items():
            species_name = info['title'].replace('_', ' ').title()
            st.markdown(f"<div class='result-card'><h3 style='color: white; margin: 0;'>Loài được phát hiện: {species_name}</h3></div>", unsafe_allow_html=True)
            with st.expander("Thông Tin Chi Tiết Về Loài", expanded=True):
                st.markdown("#### Thông Tin Cơ Bản Và Cách Chăm Sóc")
                st.markdown(info['summary'])
                if info['wiki_url']:
                    st.markdown("---")
                    st.markdown("#### Tài Nguyên Bổ Sung")
                    st.markdown(f"[Đọc bài viết khoa học đầy đủ trên Wikipedia]({info['wiki_url']})")
    else:
        st.warning("Không phát hiện loài động vật nào trong hình ảnh được tải lên. Vui lòng thử hình ảnh khác có góc nhìn rõ hơn về động vật.")

else:

    st.info("Chào mừng! Vui lòng tải lên hình ảnh động vật để bắt đầu quá trình phân tích.")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Bước 1: Tải Lên")
        st.markdown("Chọn hình ảnh rõ nét về động vật từ thiết bị của bạn")
    with col2:
        st.markdown("#### Bước 2: Phân Tích")
        st.markdown("AI sẽ nhận dạng và phân tích")
    with col3:
        st.markdown("#### Bước 3: Học Hỏi")
        st.markdown("Xem thông tin chi tiết và tài liệu tham khảo")

# Footer
st.markdown("---")
st.markdown("<div class='footer'>Bản demo về SPCK - Computer Vision</div>", unsafe_allow_html=True)
