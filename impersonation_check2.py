import streamlit as st
import pandas as pd
import pytesseract
import cv2
import numpy as np
from PIL import Image
from deskew import determine_skew
import re

# ======================
# 📌 데이터 불러오기
# ======================
@st.cache_data
def load_data():
    file_path = "강원본부 직원 연락처.XLSX"
    df = pd.read_excel(file_path, dtype=str).fillna("")
    return df

# ======================
# 📌 이미지 전처리 및 OCR
# ======================
def extract_text_from_image(image):
    try:
        img_np = np.array(image.convert('RGB'))
        img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 기울기 보정
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        angle = determine_skew(grayscale)
        if angle is not None and abs(angle) > 1:
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 이진화
        thresh = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # 전체 텍스트 추출
        full_text = pytesseract.image_to_string(thresh, lang="kor+eng")

        return full_text, img
    except Exception as e:
        st.error(f"이미지 처리 중 오류: {e}")
        return "", None

# ======================
# 📌 OCR 결과 파싱
# ======================
def parse_business_card(full_text):
    lines = [line.strip() for line in full_text.split("\n") if line.strip()]
    name, dept, phone = "", "", ""
    
    # 1) 이름 추출
    for line in lines:
        match = re.fullmatch(r"^[가-힣\s]{2,5}$", line)
        if match:
            cleaned_name = match.group(0).replace(" ", "").strip()
            if 2 <= len(cleaned_name) <= 3 and not any(k in line for k in ["본부", "처", "부", "사업소", "역", "장", "팀장", "과장", "차장", "대리"]):
                name = cleaned_name
                break

    # 2) 부서 추출
    if "본부장" in full_text:
        dept = "강원본부"
    else:
        dept_patterns = [
            r"(동해기관차승무사업소|차량처|영업처|동해역|강릉역|평창역|태백역|평창전기사업소)",
            r"(강원본부)"
        ]
        for pattern in dept_patterns:
            match = re.search(pattern, full_text)
            if match:
                dept = match.group(1).strip()
                break
    
    # 3) 전화번호 추출
    phone_patterns = [
        r"T\.?\s*(033-520-[\d]{4}|520-[\d]{4})",
        r"(\d{3}-\d{4})",
        r"전화\s*:\s*(\d{3,4}-\d{4})",
        r"(033-520-\d{4})"
    ]
    for pattern in phone_patterns:
        match = re.search(pattern, full_text)
        if match:
            phone = match.group(1).strip()
            if len(phone) > 4 and ('520' in phone or '033' in phone):
                phone = phone.split('-')[-1]  # 뒷 4자리만 추출
            break
    
    return {"성명": name, "부서": dept, "전화번호": phone}

# ======================
# 📌 메인 앱
# ======================
def main():
    st.markdown(
        """
        <div style="text-align:center; margin-bottom:20px;">
            <span style="background-color:#EAF3FF; color:#003366; padding:6px 14px; border-radius:20px; font-size:14px; font-weight:600;">
                공공기관 사칭 피해 예방
            </span>
            <h1 style="color:#003366; margin-top:10px;">코레일 강원본부 직원 확인 시스템</h1>
            <h4 style="color:#666;">안전한 민원 응대를 위한 신원 확인</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_data()

    # ======================
    # 🔘 확인 방법 선택 (스타일 통일 및 중앙 정렬)
    # ======================
    st.markdown(
        """
        <style>
            .stButton > button {
                width: 100%;
                height: 4.5rem; /* 버튼 높이 증가 */
                font-size: 1.3rem; /* 폰트 크기 증가 */
                font-weight: 600;
                border-radius: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 세션 상태 초기화
    if "mode" not in st.session_state:
        st.session_state["mode"] = None

    # 버튼을 중앙에 배치
    with st.container():
        left_col, center_col, right_col = st.columns([1, 2, 1])
        with center_col:
            ocr_button = st.button("📷 사진으로 검색하기 (OCR)", key="ocr_btn")
            manual_button = st.button("✍ 직접 검색하기", key="manual_btn")
    
    if ocr_button:
        st.session_state["mode"] = "ocr"
    elif manual_button:
        st.session_state["mode"] = "manual"


    st.markdown(
        """
        <div style="margin-top:20px; color:#555; font-size:14px; line-height:1.6; text-align:center;">
        · 본 도구는 참고용이며, 최종 확인은 소속 부서 <b>공식 내선</b> 또는 <b>대표번호</b>를 통해 진행하세요.<br>
        · 입력 및 이미지 데이터는 외부로 전송되지 않습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ======================
    # 📌 기능 실행
    # ======================
    if st.session_state["mode"] == "ocr":
        st.subheader("② 사진 업로드하여 검색 (OCR)")
        uploaded_file = st.file_uploader("직원 정보가 담긴 사진 업로드", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="업로드한 이미지", use_container_width=True)

            full_text, preprocessed_img = extract_text_from_image(image)
            if preprocessed_img is not None:
                st.text_area("📜 추출된 텍스트", full_text, height=150)
                
                extracted_info = parse_business_card(full_text)
                st.write("🔍 추출된 정보:", extracted_info)

                filled_inputs = {k: v for k, v in extracted_info.items() if v}
                if len(filled_inputs) >= 2:
                    condition = pd.Series([True] * len(df))
                    for col, val in filled_inputs.items():
                        condition &= df[col].str.contains(val, na=False)
                    result = df[condition]

                    if not result.empty:
                        st.success("✅ 해당 번호로 전화하여 추가로 확인바랍니다.")
                        st.dataframe(result)
                    else:
                        st.error("❌ 일치하는 직원이 존재하지 않습니다.")
                else:
                    st.warning("⚠ 2개 이상의 항목을 찾을 수 없습니다.")

    elif st.session_state["mode"] == "manual":
        st.subheader("① 직접 입력하여 검색")
        dept_input = st.text_input("부서 입력")
        name_input = st.text_input("성명 입력")
        phone_input = st.text_input("전화번호 입력")

        if st.button("조회하기"):
            inputs = {"부서": dept_input.strip(), "성명": name_input.strip(), "전화번호": phone_input.strip()}
            filled_inputs = {k: v for k, v in inputs.items() if v}
            if len(filled_inputs) < 2:
                st.warning("⚠ 최소 2개 이상의 항목을 입력해야 합니다.")
            else:
                condition = pd.Series([True] * len(df))
                for col, val in filled_inputs.items():
                    condition &= df[col].str.contains(val, na=False)
                result = df[condition]
                if not result.empty:
                    st.success("✅ 일치\n해당 번호로 전화하여 추가로 확인바랍니다.")
                    st.dataframe(result)
                else:
                    st.error("❌ 존재하지 않음")

if __name__ == "__main__":
    main()
