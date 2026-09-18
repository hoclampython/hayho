# -*- coding: utf-8 -*-
"""
HỆ THỐNG DỰ BÁO VÀ PHÂN TÍCH GIAN LẬN BÁO CÁO TÀI CHÍNH (BCTC)
Sử dụng Mô hình Beneish M-Score, Hồi quy Logistic & XGBoost Classifier
Tác giả: Chuyên gia Phân tích Dữ liệu Tài chính & Web App
"""

import os
import io
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# CẤU HÌNH TRANG STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Phân Tích Gian Lận BCTC | Beneish M-Score & XGBoost AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tùy chỉnh CSS giao diện chuyên nghiệp
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .badge-safe {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
    }
    .badge-fraud {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# TỪ ĐIỂN Ý NGHĨA & HỆ SỐ M-SCORE GỐC
# ==============================================================================
FEATURE_COLS = ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'TATA', 'LVGI']

MEANING_MAP = {
    'Hệ số chặn (Intercept)': 'Mức rủi ro cơ sở khi các chỉ số ở mức bằng 0',
    'DSRI': 'Số ngày thu tiền bình quân (Tăng rủi ro ghi nhận doanh thu ảo)',
    'GMI': 'Tỷ lệ lãi gộp suy giảm (Tăng áp lực thao túng BCTC)',
    'AQI': 'Chất lượng tài sản (Tăng rủi ro vốn hóa chi phí)',
    'SGI': 'Tăng trưởng doanh thu (Tăng áp lực thổi phồng doanh thu)',
    'DEPI': 'Chỉ số khấu hao (Tác động giảm chi phí khấu hao)',
    'SGAI': 'Chi phí Bán hàng & QLDN (Tác động suy giảm hiệu quả quản lý)',
    'TATA': 'Biến động dồn tích (Chất lượng lợi nhuận thấp, CFO kém)',
    'LVGI': 'Đòn bẩy tài chính (Tăng rủi ro đòn bẩy nợ)'
}

BENEISH_COEFFS = {
    'Intercept': -4.84,
    'DSRI': 0.920,
    'GMI': 0.528,
    'AQI': 0.404,
    'SGI': 0.892,
    'DEPI': 0.115,
    'SGAI': -0.172,
    'TATA': 4.037,
    'LVGI': 0.0327
}
BENEISH_CUTOFF = -1.78

def calculate_beneish_mscore(row_data):
    """Tính điểm Beneish M-Score chuẩn gốc (1999) 8 biến."""
    score = BENEISH_COEFFS['Intercept']
    for col in FEATURE_COLS:
        score += BENEISH_COEFFS[col] * float(row_data[col])
    return score

# ==============================================================================
# HÀM LOAD & XỬ LÝ DỮ LIỆU CÓ CACHING
# ==============================================================================
@st.cache_data
def load_default_dataset():
    """Đọc dữ liệu mặc định từ file MScore_data.csv nếu có."""
    default_path = "MScore_data.csv"
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    return None

def compute_metrics(y_true, y_pred, y_proba):
    """Hàm tính toán các chỉ số đo lường hiệu năng."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'auc': auc, 'cm': cm, 'fpr': fpr, 'tpr': tpr
    }

@st.cache_data
def train_models(data_df, test_size=0.20, random_state=42, c_param=1.0, n_estimators=100, max_depth=3, learning_rate=0.1):
    """Huấn luyện cả Logistic Regression và XGBoost Classifier."""
    X = data_df[FEATURE_COLS]
    y = data_df['FRAUD_FLAG']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 1. Logistic Regression
    lr_model = LogisticRegression(C=c_param, random_state=random_state, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_proba = lr_model.predict_proba(X_test)[:, 1]
    lr_metrics = compute_metrics(y_test, lr_pred, lr_proba)

    # Bảng hệ số hồi quy Logistic
    intercept = lr_model.intercept_[0]
    coefficients = lr_model.coef_[0]
    coef_df = pd.DataFrame({
        'Chỉ số / Biến': ['Hệ số chặn (Intercept)'] + FEATURE_COLS,
        'Hệ số (Beta)': [intercept] + list(coefficients),
        'Tỷ số chênh (Odds Ratio)': [np.exp(intercept)] + list(np.exp(coefficients))
    })
    coef_df['Ý nghĩa kinh tế trong phát hiện gian lận'] = coef_df['Chỉ số / Biến'].map(MEANING_MAP)

    # 2. XGBoost Classifier
    xgb_model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = compute_metrics(y_test, xgb_pred, xgb_proba)

    # Mức độ quan trọng của biến (Feature Importance) XGBoost
    importance_df = pd.DataFrame({
        'Chỉ số / Biến': FEATURE_COLS,
        'Feature Importance': xgb_model.feature_importances_
    }).sort_values(by='Feature Importance', ascending=False)
    importance_df['Ý nghĩa kinh tế'] = importance_df['Chỉ số / Biến'].map(MEANING_MAP)

    data_splits = (X_train, X_test, y_train, y_test)
    
    return {
        'lr': (lr_model, coef_df, lr_metrics),
        'xgb': (xgb_model, importance_df, xgb_metrics),
        'splits': data_splits
    }

# ==============================================================================
# SIDEBAR QUẢN LÝ DỮ LIỆU & THIẾT LẬP
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
    st.markdown("## **Cấu hình & Dữ liệu**")
    
    data_source = st.radio(
        "Nguồn dữ liệu huấn luyện:",
        ["Dữ liệu mặc định (MScore_data.csv)", "Tải lên file CSV mới"],
        index=0
    )

    df_active = None
    if data_source == "Dữ liệu mặc định (MScore_data.csv)":
        df_active = load_default_dataset()
        if df_active is None:
            st.error("⚠️ Không tìm thấy file `MScore_data.csv` trong thư mục. Vui lòng tải file lên!")
        else:
            st.success(f"✅ Đã tải dữ liệu mặc định: **{len(df_active)}** mẫu")
    else:
        uploaded_file = st.file_uploader("Chọn file CSV chứa dữ liệu BCTC", type=["csv"])
        if uploaded_file is not None:
            try:
                df_active = pd.read_csv(uploaded_file)
                st.success(f"✅ Tải lên thành công: **{len(df_active)}** mẫu")
            except Exception as e:
                st.error(f"Lỗi khi đọc file: {e}")

    st.markdown("---")
    st.markdown("### **Tham số Huấn luyện Chung**")
    test_pct = st.slider("Tỷ lệ tập kiểm tra (Test Size)", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
    random_seed = st.number_input("Random State", min_value=1, max_value=999, value=42)
    
    st.markdown("### **Tham số Logistic Regression**")
    c_regularization = st.selectbox("Điều chuẩn C (Inverse Regularization)", [0.01, 0.1, 1.0, 10.0, 100.0], index=2)

    st.markdown("### **Tham số XGBoost**")
    xgb_n_estimators = st.slider("Số lượng cây (n_estimators)", min_value=10, max_value=300, value=100, step=10)
    xgb_max_depth = st.slider("Độ sâu tối đa cây (max_depth)", min_value=1, max_value=10, value=3, step=1)
    xgb_lr = st.selectbox("Tốc độ học (learning_rate)", [0.01, 0.05, 0.1, 0.2, 0.3], index=2)

    st.markdown("---")
    st.markdown("💡 **Thông tin mô hình:**")
    st.caption("Ứng dụng kết hợp giữa **Beneish M-Score cổ điển**, **Logistic Regression** và thuật toán Boosting tiên tiến **XGBoost** để phát hiện gian lận báo cáo tài chính.")

# ==============================================================================
# KIỂM TRA ĐIỀU KIỆN DỮ LIỆU ĐỂ HUẤN LUYỆN
# ==============================================================================
if df_active is None:
    st.warning("⚠️ Vui lòng cung cấp file dữ liệu CSV để ứng dụng bắt đầu hoạt động.")
    st.stop()

missing_cols = [col for col in FEATURE_COLS + ['FRAUD_FLAG'] if col not in df_active.columns]
if missing_cols:
    st.error(f"❌ Dữ liệu đang thiếu các cột bắt buộc: {missing_cols}. Vui lòng kiểm tra lại cấu trúc file!")
    st.stop()

# Huấn luyện các mô hình
model_results = train_models(
    df_active, test_size=test_pct, random_state=random_seed,
    c_param=c_regularization, n_estimators=xgb_n_estimators,
    max_depth=xgb_max_depth, learning_rate=xgb_lr
)

lr_model, coef_df, lr_metrics = model_results['lr']
xgb_model, importance_df, xgb_metrics = model_results['xgb']
X_train, X_test, y_train, y_test = model_results['splits']

# ==============================================================================
# GIAO DIỆN CHÍNH - HEADER & TABS
# ==============================================================================
st.markdown('<div class="main-title">HỆ THỐNG DỰ BÁO RỦI RO GIAN LẬN BÁO CÁO TÀI CHÍNH</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ứng dụng Khoa học Dữ liệu (Logistic Regression & XGBoost) và Mô hình Beneish M-Score trong Kiểm toán, Ngân hàng</div>', unsafe_allow_html=True)

# Thanh tóm tắt nhanh hiệu năng (So sánh LR vs XGBoost)
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
with col_m1:
    st.metric("Tổng số doanh nghiệp", f"{len(df_active):,} DN")
with col_m2:
    fraud_count = df_active['FRAUD_FLAG'].sum()
    st.metric("DN gian lận (Thực tế)", f"{fraud_count} DN", f"{fraud_count/len(df_active)*100:.1f}%", delta_color="inverse")
with col_m3:
    st.metric("Accuracy (LR vs XGB)", f"{lr_metrics['accuracy']*100:.1f}% | {xgb_metrics['accuracy']*100:.1f}%")
with col_m4:
    st.metric("Recall (LR vs XGB)", f"{lr_metrics['recall']*100:.1f}% | {xgb_metrics['recall']*100:.1f}%")
with col_m5:
    st.metric("AUC - ROC (LR vs XGB)", f"{lr_metrics['auc']:.3f} | {xgb_metrics['auc']:.3f}")

# Các Tab chức năng
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 1. Khám Phá Dữ Liệu (EDA)",
    "🧠 2. Huấn Luyện & Đánh Giá Mô Hình",
    "🔍 3. Chẩn Đoán Doanh Nghiệp Đơn Lẻ",
    "📂 4. Dự Báo Hàng Loạt (Batch Prediction)",
    "📚 5. Từ Điển & Phương Pháp Luận"
])

# ==============================================================================
# TAB 1: KHÁM PHÁ DỮ LIỆU (EDA)
# ==============================================================================
with tab1:
    st.subheader("1. Tổng quan Tập Dữ liệu M-Score")
    
    col_t1_left, col_t1_right = st.columns([3, 2])
    with col_t1_left:
        st.markdown("**5 dòng dữ liệu đầu tiên:**")
        st.dataframe(df_active.head(10), use_container_width=True)
    
    with col_t1_right:
        st.markdown("**Tỷ lệ nhãn Gian Lận (FRAUD_FLAG):**")
        pie_data = df_active['FRAUD_FLAG'].value_counts().reset_index()
        pie_data.columns = ['Trạng thái', 'Số lượng']
        pie_data['Trạng thái'] = pie_data['Trạng thái'].map({0: '0: An toàn (Non-Fraud)', 1: '1: Gian lận (Fraud)'})
        
        fig_pie = px.pie(
            pie_data, names='Trạng thái', values='Số lượng',
            color='Trạng thái',
            color_discrete_map={'0: An toàn (Non-Fraud)': '#10B981', '1: Gian lận (Fraud)': '#EF4444'},
            hole=0.45
        )
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=260)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("2. Phân Phối & So Sánh 8 Chỉ Số Theo Nhóm An Toàn / Gian Lận")
    
    selected_metric = st.selectbox(
        "Chọn chỉ số tài chính muốn xem phân phối chi tiết:",
        FEATURE_COLS,
        format_func=lambda x: f"{x} - {MEANING_MAP[x]}"
    )
    
    col_box, col_hist = st.columns(2)
    with col_box:
        fig_box = px.box(
            df_active, x='FRAUD_FLAG', y=selected_metric, color='FRAUD_FLAG',
            labels={'FRAUD_FLAG': 'Trạng thái', selected_metric: f'Giá trị {selected_metric}'},
            color_discrete_map={0: '#10B981', 1: '#EF4444'},
            title=f"Biểu đồ Hộp (Boxplot) của {selected_metric}"
        )
        fig_box.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig_box, use_container_width=True)

    with col_hist:
        fig_hist = px.histogram(
            df_active, x=selected_metric, color='FRAUD_FLAG', barmode='overlay',
            labels={'FRAUD_FLAG': 'Trạng thái', selected_metric: f'Giá trị {selected_metric}'},
            color_discrete_map={0: '#10B981', 1: '#EF4444'},
            title=f"Phân phối tần suất (Histogram) của {selected_metric}",
            opacity=0.7
        )
        fig_hist.update_layout(height=360)
        st.plotly_chart(fig_hist, use_container_width=True)

    with st.expander("📊 Xem Bảng Thống Kê Mô Tả (Descriptive Statistics)"):
        st.dataframe(df_active.describe().T.round(3), use_container_width=True)

# ==============================================================================
# TAB 2: HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH (SO SÁNH LOGISTIC REGRESSION & XGBOOST)
# ==============================================================================
with tab2:
    st.subheader("1. Đặc Tính & Hệ Số/Mức Đội Quan Trọng Của Biến Trong Mô Hình")
    
    tab_lr_exp, tab_xgb_exp = st.tabs(["📌 Logistic Regression (Hệ số Beta)", "🌲 XGBoost (Feature Importance)"])
    
    with tab_lr_exp:
        def highlight_beta(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return 'color: #DC2626; font-weight: bold;'
                elif val < 0:
                    return 'color: #16A34A; font-weight: bold;'
            return ''

        styler = coef_df.style
        if hasattr(styler, 'map'):
            styler = styler.map(highlight_beta, subset=['Hệ số (Beta)'])
        elif hasattr(styler, 'applymap'):
            styler = styler.applymap(highlight_beta, subset=['Hệ số (Beta)'])

        st.dataframe(
            styler.format({'Hệ số (Beta)': '{:.4f}', 'Tỷ số chênh (Odds Ratio)': '{:.4f}'}),
            use_container_width=True
        )

    with tab_xgb_exp:
        col_xgb_fi, col_xgb_chart = st.columns([1, 1])
        with col_xgb_fi:
            st.dataframe(
                importance_df.style.format({'Feature Importance': '{:.4f}'}),
                use_container_width=True
            )
        with col_xgb_chart:
            fig_fi = px.bar(
                importance_df, x='Feature Importance', y='Chỉ số / Biến', orientation='h',
                title="Mức Độ Quan Trọng Của Biến Theo XGBoost",
                color='Feature Importance', color_continuous_scale='Viridis'
            )
            fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=300)
            st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")
    st.subheader("2. Đánh Giá Hiệu Năng Chi Tiết Trên Tập Kiểm Thử (Test Set)")
    
    col_cm_lr, col_cm_xgb = st.columns(2)
    
    with col_cm_lr:
        st.markdown("**Ma trận Nhầm lẫn (Logistic Regression):**")
        fig_cm_lr = px.imshow(
            lr_metrics['cm'],
            x=['Dự báo An toàn (0)', 'Dự báo Gian lận (1)'],
            y=['Thực tế An toàn (0)', 'Thực tế Gian lận (1)'],
            color_continuous_scale='Blues',
            text_auto=True, title="Confusion Matrix - Logistic Regression"
        )
        fig_cm_lr.update_layout(height=340, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_cm_lr, use_container_width=True)

    with col_cm_xgb:
        st.markdown("**Ma trận Nhầm lẫn (XGBoost Classifier):**")
        fig_cm_xgb = px.imshow(
            xgb_metrics['cm'],
            x=['Dự báo An toàn (0)', 'Dự báo Gian lận (1)'],
            y=['Thực tế An toàn (0)', 'Thực tế Gian lận (1)'],
            color_continuous_scale='Oranges',
            text_auto=True, title="Confusion Matrix - XGBoost Classifier"
        )
        fig_cm_xgb.update_layout(height=340, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_cm_xgb, use_container_width=True)

    st.markdown("---")
    st.markdown("**So Sánh Đường Cong ROC (ROC Curve Comparison):**")
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=lr_metrics['fpr'], y=lr_metrics['tpr'],
        mode='lines', name=f'Logistic Regression (AUC = {lr_metrics["auc"]:.4f})',
        line=dict(color='#2563EB', width=3)
    ))
    fig_roc.add_trace(go.Scatter(
        x=xgb_metrics['fpr'], y=xgb_metrics['tpr'],
        mode='lines', name=f'XGBoost Classifier (AUC = {xgb_metrics["auc"]:.4f})',
        line=dict(color='#D97706', width=3)
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines', name='Random Guess',
        line=dict(color='#9CA3AF', dash='dash', width=2)
    ))
    fig_roc.update_layout(
        xaxis_title='Tỷ lệ Báo động giả (False Positive Rate)',
        yaxis_title='Tỷ lệ Phát hiện gian lận (True Positive Rate)',
        title="Đường cong ROC Curve So Sánh Giữa Logistic Regression & XGBoost",
        height=400, margin=dict(t=40, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")
    st.subheader("3. Bảng Bảng So Sánh Chỉ Số Hiệu Năng Mô Hình")
    summary_df = pd.DataFrame({
        'Chỉ số Đánh Giá': ['Accuracy', 'Precision', 'Recall / Sensitivity', 'F1-Score', 'AUC - ROC'],
        'Logistic Regression': [
            f"{lr_metrics['accuracy']:.4f} ({lr_metrics['accuracy']*100:.2f}%)",
            f"{lr_metrics['precision']:.4f}",
            f"{lr_metrics['recall']:.4f}",
            f"{lr_metrics['f1']:.4f}",
            f"{lr_metrics['auc']:.4f}"
        ],
        'XGBoost Classifier': [
            f"{xgb_metrics['accuracy']:.4f} ({xgb_metrics['accuracy']*100:.2f}%)",
            f"{xgb_metrics['precision']:.4f}",
            f"{xgb_metrics['recall']:.4f}",
            f"{xgb_metrics['f1']:.4f}",
            f"{xgb_metrics['auc']:.4f}"
        ],
        'Ý nghĩa Kiểm toán & Quản trị Rủi ro': [
            'Tỷ lệ dự báo đúng tổng thể trên toàn bộ mẫu kiểm tra',
            'Xác suất gian lận thực sự khi mô hình phát cờ cảnh báo',
            'Khả năng bắt trúng các vụ gian lận thực tế (Tránh lọt tội)',
            'Điểm trung bình điều hòa giữa Precision và Recall',
            'Khả năng phân biệt doanh nghiệp An toàn vs Gian lận'
        ]
    })
    st.table(summary_df)

# ==============================================================================
# TAB 3: CHẨN ĐOÁN DOANH NGHIỆP ĐƠN LẺ (SINGLE COMPANY DIAGNOSIS)
# ==============================================================================
with tab3:
    st.subheader("Chẩn Đoán Rủi Ro Báo Cáo Tài Chính Của Một Doanh Nghiệp")
    st.markdown("Nhập 8 chỉ số tài chính tính toán từ BCTC kỳ gần nhất để kiểm tra nguy cơ thao túng số liệu:")

    # Nút chọn kịch bản mẫu
    col_demo1, col_demo2, col_demo3 = st.columns(3)
    if col_demo1.button("🟢 Tải mẫu: DN Tài Chính Lành Mạnh"):
        st.session_state.custom_inputs = {
            'DSRI': 0.75, 'GMI': 0.95, 'AQI': 0.70, 'SGI': 1.05,
            'DEPI': 0.90, 'SGAI': 0.95, 'TATA': 0.02, 'LVGI': 0.90
        }
    if col_demo2.button("🔴 Tải mẫu: DN Nguy Cơ Gian Lận Rất Cao"):
        st.session_state.custom_inputs = {
            'DSRI': 1.85, 'GMI': 1.70, 'AQI': 1.45, 'SGI': 1.65,
            'DEPI': 1.30, 'SGAI': 1.35, 'TATA': 0.18, 'LVGI': 1.40
        }
    if col_demo3.button("🟡 Tải mẫu: DN Vùng Ranh Giới (Borderline)"):
        st.session_state.custom_inputs = {
            'DSRI': 1.25, 'GMI': 1.15, 'AQI': 1.05, 'SGI': 1.20,
            'DEPI': 1.05, 'SGAI': 1.10, 'TATA': 0.08, 'LVGI': 1.10
        }

    if 'custom_inputs' not in st.session_state:
        st.session_state.custom_inputs = {
            'DSRI': 1.00, 'GMI': 1.00, 'AQI': 1.00, 'SGI': 1.10,
            'DEPI': 1.00, 'SGAI': 1.00, 'TATA': 0.05, 'LVGI': 1.00
        }

    # Form nhập liệu
    with st.form("single_predict_form"):
        col_in1, col_in2, col_in3, col_in4 = st.columns(4)
        
        with col_in1:
            val_dsri = st.number_input("1. DSRI (Số ngày thu tiền)", value=float(st.session_state.custom_inputs['DSRI']), step=0.05, format="%.3f")
            val_depi = st.number_input("5. DEPI (Khấu hao tài sản)", value=float(st.session_state.custom_inputs['DEPI']), step=0.05, format="%.3f")
        with col_in2:
            val_gmi = st.number_input("2. GMI (Suy giảm lãi gộp)", value=float(st.session_state.custom_inputs['GMI']), step=0.05, format="%.3f")
            val_sgai = st.number_input("6. SGAI (Chi phí BH & QLDN)", value=float(st.session_state.custom_inputs['SGAI']), step=0.05, format="%.3f")
        with col_in3:
            val_aqi = st.number_input("3. AQI (Chất lượng tài sản)", value=float(st.session_state.custom_inputs['AQI']), step=0.05, format="%.3f")
            val_tata = st.number_input("7. TATA (Biến động dồn tích)", value=float(st.session_state.custom_inputs['TATA']), step=0.02, format="%.3f")
        with col_in4:
            val_sgi = st.number_input("4. SGI (Tăng trưởng doanh thu)", value=float(st.session_state.custom_inputs['SGI']), step=0.05, format="%.3f")
            val_lvgi = st.number_input("8. LVGI (Đòn bẩy tài chính)", value=float(st.session_state.custom_inputs['LVGI']), step=0.05, format="%.3f")

        submit_btn = st.form_submit_button("🚀 Tiến Hành Phân Tích & Chẩn Đoán", use_container_width=True)

    input_data = {
        'DSRI': val_dsri, 'GMI': val_gmi, 'AQI': val_aqi, 'SGI': val_sgi,
        'DEPI': val_depi, 'SGAI': val_sgai, 'TATA': val_tata, 'LVGI': val_lvgi
    }

    # Tính toán dự báo cho 3 phương pháp
    input_df = pd.DataFrame([input_data])
    lr_prob = lr_model.predict_proba(input_df)[0, 1]
    xgb_prob = xgb_model.predict_proba(input_df)[0, 1]
    m_score_val = calculate_beneish_mscore(input_data)

    st.markdown("---")
    st.subheader("Kết Quả Phân Tích Đa Mô Hình")

    col_res1, col_res2, col_res3 = st.columns(3)
    
    with col_res1:
        st.markdown("**1. Logistic Regression:**")
        if lr_prob >= 0.5:
            st.markdown(f"<div class='metric-card' style='border-left-color: #EF4444;'><h4>Xác suất Gian lận</h4><h2 style='color:#EF4444;'>{lr_prob*100:.1f}%</h2><span class='badge-fraud'>⚠️ CẢNH BÁO NGUY CƠ CAO</span></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='metric-card' style='border-left-color: #10B981;'><h4>Xác suất Gian lận</h4><h2 style='color:#10B981;'>{lr_prob*100:.1f}%</h2><span class='badge-safe'>✅ BCTC AN TOÀN</span></div>", unsafe_allow_html=True)

    with col_res2:
        st.markdown("**2. XGBoost Classifier:**")
        if xgb_prob >= 0.5:
            st.markdown(f"<div class='metric-card' style='border-left-color: #EF4444;'><h4>Xác suất Gian lận</h4><h2 style='color:#EF4444;'>{xgb_prob*100:.1f}%</h2><span class='badge-fraud'>⚠️ CẢNH BÁO NGUY CƠ CAO</span></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='metric-card' style='border-left-color: #10B981;'><h4>Xác suất Gian lận</h4><h2 style='color:#10B981;'>{xgb_prob*100:.1f}%</h2><span class='badge-safe'>✅ BCTC AN TOÀN</span></div>", unsafe_allow_html=True)

    with col_res3:
        st.markdown("**3. Beneish M-Score (1999):**")
        if m_score_val > BENEISH_CUTOFF:
            st.markdown(f"<div class='metric-card' style='border-left-color: #EF4444;'><h4>Điểm M-Score</h4><h2 style='color:#EF4444;'>{m_score_val:.3f}</h2><span class='badge-fraud'>CÓ DẤU HIỆU THAO TÚNG</span><br><small>(M-Score > -1.78)</small></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='metric-card' style='border-left-color: #10B981;'><h4>Điểm M-Score</h4><h2 style='color:#10B981;'>{m_score_val:.3f}</h2><span class='badge-safe'>AN TOÀN / KHÔNG THAO TÚNG</span><br><small>(M-Score ≤ -1.78)</small></div>", unsafe_allow_html=True)

    # Đóng góp của từng chỉ số vào Log-Odds (Logistic Regression)
    st.markdown("### 📊 Phân Tích Đóng Góp Của Từng Chỉ Số (Logistic Regression)")
    coef_dict = dict(zip(FEATURE_COLS, lr_model.coef_[0]))
    contributions = {col: input_data[col] * coef_dict[col] for col in FEATURE_COLS}
    contrib_df = pd.DataFrame(list(contributions.items()), columns=['Chỉ số', 'Tác động Log-Odds'])
    contrib_df['Màu sắc'] = contrib_df['Tác động Log-Odds'].apply(lambda x: '#EF4444' if x > 0 else '#10B981')
    contrib_df = contrib_df.sort_values(by='Tác động Log-Odds', ascending=True)

    fig_contrib = px.bar(
        contrib_df, x='Tác động Log-Odds', y='Chỉ số', orientation='h',
        color='Màu sắc', color_discrete_map="identity",
        title="Mức Độ Đóng Góp Vào Rủi Ro (Tăng (+) / Giảm (-))"
    )
    fig_contrib.update_layout(height=320, showlegend=False)
    st.plotly_chart(fig_contrib, use_container_width=True)

# ==============================================================================
# TAB 4: DỰ BÁO HÀNG LOẠT (BATCH PREDICTION)
# ==============================================================================
with tab4:
    st.subheader("Dự Báo & Sàng Lọc Rủi Ro Cho Danh Sách Hàng Loạt Doanh Nghiệp")
    st.markdown("Tải lên file CSV chứa danh sách các doanh nghiệp cùng 8 chỉ số BCTC để hệ thống tự động chấm điểm và phân loại rủi ro:")

    # Tải file mẫu
    template_df = pd.DataFrame({
        'Ma_DN': ['DN_A', 'DN_B', 'DN_C', 'DN_D'],
        'DSRI': [1.32, 0.54, 1.54, 1.02],
        'GMI': [0.70, 1.06, 1.78, 0.26],
        'AQI': [0.78, 0.86, 1.19, 0.75],
        'SGI': [1.31, 0.87, 1.49, 1.35],
        'DEPI': [1.03, 0.92, 1.21, 0.76],
        'SGAI': [1.18, 0.98, 1.16, 1.00],
        'TATA': [-0.08, 0.19, 0.15, 0.09],
        'LVGI': [1.07, 0.85, 1.36, 1.43]
    })
    
    csv_template = template_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Tải file CSV mẫu kiểm tra (Template Batch)",
        data=csv_template,
        file_name="mau_kiem_tra_bctc.csv",
        mime="text/csv"
    )

    uploaded_batch = st.file_uploader("Chọn file CSV dữ liệu doanh nghiệp cần thẩm định", type=["csv"], key="batch_upload")

    if uploaded_batch is not None:
        try:
            batch_data = pd.read_csv(uploaded_batch)
            missing_batch_cols = [c for c in FEATURE_COLS if c not in batch_data.columns]
            
            if missing_batch_cols:
                st.error(f"❌ File thiếu các cột chỉ số: {missing_batch_cols}")
            else:
                st.success(f"✅ Đã tải danh sách gồm **{len(batch_data)}** doanh nghiệp!")
                
                # Tính toán dự báo
                lr_probs = lr_model.predict_proba(batch_data[FEATURE_COLS])[:, 1]
                xgb_probs = xgb_model.predict_proba(batch_data[FEATURE_COLS])[:, 1]
                batch_mscores = [calculate_beneish_mscore(row) for _, row in batch_data.iterrows()]
                
                result_batch = batch_data.copy()
                result_batch['Xác suất (Logistic)'] = np.round(lr_probs, 4)
                result_batch['Xác suất (XGBoost)'] = np.round(xgb_probs, 4)
                result_batch['Phân loại (XGBoost)'] = np.where(xgb_probs >= 0.5, '⚠️ Nguy cơ Gian lận', '✅ An toàn')
                result_batch['Điểm Beneish M-Score'] = np.round(batch_mscores, 3)
                result_batch['Kết luận Beneish'] = np.where(np.array(batch_mscores) > BENEISH_CUTOFF, 'Thao túng', 'Bình thường')
                
                def get_risk_level(prob):
                    if prob >= 0.70:
                        return 'Đỏ - Rủi ro Rất cao'
                    elif prob >= 0.40:
                        return 'Vàng - Rủi ro Đáng chú ý'
                    else:
                        return 'Xanh - Rủi ro Thấp'

                result_batch['Cấp độ Rủi ro (XGBoost)'] = result_batch['Xác suất (XGBoost)'].apply(get_risk_level)

                # Hiển thị kết quả
                st.markdown("### Kết Quả Phân Tích & Chấm Điểm Danh Sách:")
                st.dataframe(result_batch, use_container_width=True)

                # Biểu đồ phân bổ mức độ rủi ro & Tương quan
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    fig_risk = px.histogram(
                        result_batch, x='Cấp độ Rủi ro (XGBoost)', color='Cấp độ Rủi ro (XGBoost)',
                        color_discrete_map={
                            'Đỏ - Rủi ro Rất cao': '#EF4444',
                            'Vàng - Rủi ro Đáng chú ý': '#F59E0B',
                            'Xanh - Rủi ro Thấp': '#10B981'
                        },
                        title="Phân Bổ Cấp Độ Rủi Ro Theo XGBoost"
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)

                with col_b2:
                    fig_scatter = px.scatter(
                        result_batch, x='Xác suất (Logistic)', y='Xác suất (XGBoost)',
                        color='Cấp độ Rủi ro (XGBoost)',
                        color_discrete_map={
                            'Đỏ - Rủi ro Rất cao': '#EF4444',
                            'Vàng - Rủi ro Đáng chú ý': '#F59E0B',
                            'Xanh - Rủi ro Thấp': '#10B981'
                        },
                        title="Tương Quan Xác Suất Giữa Logistic Regression & XGBoost",
                        hover_data=list(batch_data.columns)
                    )
                    fig_scatter.add_vline(x=0.5, line_dash="dash", line_color="blue", annotation_text="Ngưỡng LR (0.5)")
                    fig_scatter.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Ngưỡng XGB (0.5)")
                    st.plotly_chart(fig_scatter, use_container_width=True)

                # Xuất file kết quả
                csv_out = result_batch.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📤 Xuất Báo Cáo Thẩm Định (Download CSV Kết Quả)",
                    data=csv_out,
                    file_name="ket_qua_tham_dinh_bctc.csv",
                    mime="text/csv"
                )
        except Exception as err:
            st.error(f"Lỗi trong quá trình xử lý hàng loạt: {err}")

# ==============================================================================
# TAB 5: TỪ ĐIỂN CHỈ SỐ & PHƯƠNG PHÁP LUẬN BCTC
# ==============================================================================
with tab5:
    st.subheader("Phương Pháp Luận Phát Hiện Thao Túng Báo Cáo Tài Chính")
    st.markdown("""
    Giáo sư **Messod Beneish** (Đại học Indiana, 1999) đã phát triển mô hình toán học định lượng để phát hiện khả năng một công ty đang **thổi phồng lợi nhuận kế toán hoặc che giấu chi phí**. 
    Mô hình bao gồm **8 chỉ số cốt lõi** đo lường các méo mó tài chính:
    """)

    tabs_guide = st.tabs([
        "1. DSRI", "2. GMI", "3. AQI", "4. SGI", 
        "5. DEPI", "6. SGAI", "7. TATA", "8. LVGI"
    ])

    with tabs_guide[0]:
        st.markdown("### DSRI - Days Sales in Receivables Index (Chỉ số Thu Tiền)")
        st.latex(r"DSRI = \frac{\text{Khoản phải thu}_t / \text{Doanh thu}_t}{\text{Khoản phải thu}_{t-1} / \text{Doanh thu}_{t-1}}")
        st.info("💡 **Ý nghĩa:** DSRI > 1 cho thấy số ngày thu tiền bình quân kéo dài bất thường. Đây là dấu hiệu cổ điển của việc **ghi nhận doanh thu non**, nới lỏng tín dụng thương mại quá mức hoặc tạo lập **doanh thu ảo**.")

    with tabs_guide[1]:
        st.markdown("### GMI - Gross Margin Index (Chỉ số Lãi Gộp)")
        st.latex(r"GMI = \frac{\text{Biên lãi gộp}_{t-1}}{\text{Biên lãi gộp}_t}")
        st.info("💡 **Ý nghĩa:** GMI > 1 báo hiệu biên lợi nhuận gộp đang bị suy giảm so với năm trước. Khi triển vọng kinh doanh đi xuống, ban lãnh đạo đối mặt với áp lực lớn phải **xào nấu số liệu** để làm đẹp lợi nhuận báo cáo.")

    with tabs_guide[2]:
        st.markdown("### AQI - Asset Quality Index (Chỉ số Chất Lượng Tài Sản)")
        st.latex(r"AQI = \frac{1 - (\text{Tài sản ngắn hạn}_t + \text{TSCĐ thuần}_t + \text{Chứng khoán}_t)/\text{Tổng tài sản}_t}{1 - (\text{Tài sản ngắn hạn}_{t-1} + \text{TSCĐ thuần}_{t-1} + \text{Chứng khoán}_{t-1})/\text{Tổng tài sản}_{t-1}}")
        st.info("💡 **Ý nghĩa:** AQI > 1 phản ánh tỷ lệ tài sản vô hình, chi phí trả trước dài hạn hoặc các khoản đầu tư khó định giá đang tăng lên. Doanh nghiệp có thể đang **vốn hóa chi phí** thay vì hạch toán vào chi phí trong kỳ.")

    with tabs_guide[3]:
        st.markdown("### SGI - Sales Growth Index (Chỉ số Tăng Trưởng Doanh Thu)")
        st.latex(r"SGI = \frac{\text{Doanh thu thuần}_t}{\text{Doanh thu thuần}_{t-1}}")
        st.info("💡 **Ý nghĩa:** Tăng trưởng doanh thu cao không hẳn là tiêu cực, nhưng các doanh nghiệp tăng trưởng nóng thường chịu áp lực nặng nề từ kỳ vọng thị trường và dễ thực hiện hành vi gian lận khi đà tăng trưởng chững lại.")

    with tabs_guide[4]:
        st.markdown("### DEPI - Depreciation Index (Chỉ số Tỷ Lệ Khấu Hao)")
        st.latex(r"DEPI = \frac{\text{Tỷ lệ khấu hao}_{t-1}}{\text{Tỷ lệ khấu hao}_t}")
        st.info("💡 **Ý nghĩa:** DEPI > 1 biểu thị tốc độ khấu hao tài sản cố định đang bị kéo dài hoặc giảm xuống. Doanh nghiệp có thể đã thay đổi ước tính kế toán (tăng thời gian sử dụng hữu ích) để **giảm chi phí khấu hao**, từ đó nâng cao lợi nhuận.")

    with tabs_guide[5]:
        st.markdown("### SGAI - Sales, General & Administrative Index (Chỉ số Chi Phí Bán Hàng & QLDN)")
        st.latex(r"SGAI = \frac{\text{Chi phí SG&A}_t / \text{Doanh thu}_t}{\text{Chi phí SG&A}_{t-1} / \text{Doanh thu}_{t-1}}")
        st.info("💡 **Ý nghĩa:** SGAI > 1 cho thấy chi phí vận hành tăng nhanh hơn doanh thu, phản ánh sự suy giảm hiệu quả quản trị và tạo động cơ bù đắp lợi nhuận thông qua các thủ thuật kế toán.")

    with tabs_guide[6]:
        st.markdown("### TATA - Total Accruals to Total Assets (Tổng Biến Động Dồn Tích)")
        st.latex(r"TATA = \frac{\text{Lợi nhuận thuần HĐKD} - \text{Dòng tiền thuần từ HĐKD (CFO)}}{\text{Tổng tài sản}}")
        st.info("💡 **Ý nghĩa:** TATA đo lường sự chênh lệch giữa lợi nhuận kế toán và dòng tiền thực tế thu về. TATA càng cao đồng nghĩa lợi nhuận phần lớn là lợi nhuận trên giấy tờ (chất lượng lợi nhuận thấp, rủi ro BCTC rất cao).")

    with tabs_guide[7]:
        st.markdown("### LVGI - Leverage Index (Chỉ số Đòn Bẩy Tài Chính)")
        st.latex(r"LVGI = \frac{\text{Tổng nợ}_t / \text{Tổng tài sản}_t}{\text{Tổng nợ}_{t-1} / \text{Tổng tài sản}_{t-1}}")
        st.info("💡 **Ý nghĩa:** LVGI > 1 cho thấy đòn bẩy nợ tăng lên. Doanh nghiệp chịu rủi ro vỡ nợ hoặc vi phạm các cam kết vay nợ (loan covenants), thôi thúc việc bóp méo số liệu để đạt chuẩn cấp tín dụng của ngân hàng.")

    st.markdown("---")
    st.markdown("""
    ### ⚖️ Công Thức M-Score Gốc (Beneish, 1999):
    $$M = -4.84 + 0.920 \times DSRI + 0.528 \times GMI + 0.404 \times AQI + 0.892 \times SGI + 0.115 \times DEPI - 0.172 \times SGAI + 4.037 \times TATA + 0.0327 \times LVGI$$
    
    - **Điểm M > -1.78:** Khả năng cao doanh nghiệp đang thao túng BCTC (Likely Manipulator).
    - **Điểm M ≤ -1.78:** Doanh nghiệp ít có khả năng thao túng BCTC (Non-manipulator).
    """)

# Footer bản quyền
st.markdown("---")
st.markdown(
    "<center><small>Hệ Thống Phân Tích BCTC & Phát Hiện Gian Lận | Kết hợp Beneish, Logistic Regression & XGBoost Classifier | Streamlit App</small></center>",
    unsafe_allow_html=True
)
