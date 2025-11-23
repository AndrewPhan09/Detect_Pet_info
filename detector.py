import streamlit as st
from ultralytics import YOLO
import requests
from urllib.parse import quote
import google.generativeai as genai

def get_gemini_animal_info(object_name, gemini_model):
    """Lấy và tóm tắt thông tin về một đối tượng bằng Gemini API."""
    if not gemini_model:
        return "Lỗi: Gemini model chưa được khởi tạo. Vui lòng cung cấp API Key."
    try:

        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = (
            f"Cung cấp thông tin về loài '{object_name}' bằng tiếng Việt theo định dạng sau:\n\n"
            f"**Thông tin sơ bộ:**\n"
            f"Một đoạn văn tóm tắt các đặc điểm chung. "
            f"Trong đoạn văn này, hãy **in đậm** những thông tin quan trọng nhất như nguồn gốc, kích thước, hoặc tuổi thọ.\n\n"
            f"**Cách chăm sóc:**\n"
            f"Liệt kê các gạch đầu dòng (bullet points) về lời khuyên chăm sóc, ví dụ:\n"
            f"- Chế độ ăn uống: ...\n"
            f"- Môi trường sống: ...\n"
            f"- Chăm sóc sức khỏe: ...\n\n"
            f"Chỉ cung cấp nội dung theo định dạng trên, không thêm bất kỳ văn bản giới thiệu hay kết luận nào khác."
        )
        response = gemini_model.generate_content(prompt)
        return response.text if response.parts else f"Không tìm thấy thông tin cho {object_name}."
    except Exception as e:
        return f"Lỗi khi truy xuất thông tin cho {object_name}: {e}"

class EnhancedYOLODetector:
    def __init__(self, model_path='best.pt', gemini_model=None):
        try:
            self.model = YOLO(model_path)
            self.class_names = self.model.names
        except Exception as e:
            st.error(f"Lỗi khi tải model YOLO tại '{model_path}': {e}")
            st.stop()
        
        self.gemini_model = gemini_model
        self.session = requests.Session()
        self.session.headers.update(
            {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )

    def detect_objects(self, image, conf_threshold=0.25):
        results = self.model(image, conf=conf_threshold, verbose=False)
        return results[0].plot()

    def get_object_info(self, object_name):
        wiki_info = None
        gemini_summary_text = None

        # 1. Lấy thông tin từ Wikipedia
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(object_name)}"
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            if 'title' in data and 'extract' in data:
                wiki_info = {
                    'title': data.get('title', object_name),
                    'summary': data.get('extract'),
                    'wiki_url': data.get('content_urls', {}).get('desktop', {}).get('page')
                }
        except requests.exceptions.RequestException:
            pass # Bỏ qua nếu không lấy được từ Wikipedia

        # 2. Lấy thông tin từ Gemini
        if self.gemini_model:
            gemini_summary_text = get_gemini_animal_info(object_name, self.gemini_model)

        # 3. Tổng hợp kết quả
        final_summary = gemini_summary_text if gemini_summary_text and "Lỗi" not in gemini_summary_text else (wiki_info.get('summary') if wiki_info else "Không tìm thấy thông tin chi tiết.")
        
        return {
            'title': object_name,
            'summary': final_summary,
            'wiki_url': wiki_info.get('wiki_url') if wiki_info else None
        }

    def enhanced_detection(self, image, conf_threshold=0.25):
        results = self.model(image, conf=conf_threshold, verbose=False)
        detections = results[0]
        annotated_image = detections.plot()
        
        object_info = {}
        processed_class_names = set()

        for box in detections.boxes:
            class_name = self.class_names[int(box.cls[0])]
            if class_name not in processed_class_names:
                info = self.get_object_info(class_name)
                if info:
                    object_info[class_name] = info
                processed_class_names.add(class_name)

        return {
            'annotated_image': annotated_image,
            'object_info': object_info
        }
