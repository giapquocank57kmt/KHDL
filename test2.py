import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Title
st.title("🌦️ Ứng dụng dự báo thời tiết thông minh")


# Load dữ liệu
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("weatherHistory.csv")
        df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True).dt.tz_localize(None)

        # Chuẩn hóa tên cột
        df.columns = df.columns.str.strip().str.replace(' ', '_')

        # Kiểm tra các cột có sẵn
        cols = ['Formatted_Date', 'Temperature_(C)']
        if 'Precip_Type' in df.columns or 'Precipitation_Sum_(mm)' in df.columns:
            cols.extend(['Humidity', 'Pressure_(millibars)', 'Precip_Type'])
            if 'Precipitation_Sum_(mm)' in df.columns:
                cols.append('Precipitation_Sum_(mm)')

        df = df[cols].rename(columns={
            'Formatted_Date': 'ds',
            'Temperature_(C)': 'temperature'
        })

        return df

    except Exception as e:
        st.error(f"Lỗi khi đọc dữ liệu: {str(e)}")
        return pd.DataFrame()


df = load_data()


# Ước lượng lượng mưa nếu không có sẵn
def estimate_precipitation(df):
    if 'Precipitation_Sum_(mm)' in df.columns:
        df['precipitation'] = df['Precipitation_Sum_(mm)']
        return df

    st.warning("Không tìm thấy dữ liệu lượng mưa, đang ước lượng từ độ ẩm và áp suất...")

    # Tạo cột is_rain dựa trên Precip_Type nếu có
    if 'Precip_Type' in df.columns:
        df['is_rain'] = df['Precip_Type'].apply(lambda x: 1 if x == 'rain' else 0)
    else:
        # Ước lượng is_rain dựa trên độ ẩm
        df['is_rain'] = np.where(df['Humidity'] > 80, 1, 0)

    # Ước lượng lượng mưa bằng heuristic
    df['precipitation'] = np.where(
        df['is_rain'] == 1,
        (df['Humidity'] - 75) * (1010 - df['Pressure_(millibars)']) / 20,
        0
    )
    df['precipitation'] = df['precipitation'].clip(lower=0)

    return df


if not df.empty:
    df = estimate_precipitation(df)
    st.subheader("📊 Dữ liệu thời tiết")
    st.dataframe(df.head(100))

    # Tùy chọn dự báo
    options = ['Nhiệt độ', 'Lượng mưa']
    option = st.selectbox("🔍 Chọn yếu tố cần dự báo:", options)

    # Cài đặt dự báo
    st.sidebar.header("⚙️ Cài đặt dự báo")
    periods = st.sidebar.slider("Số ngày dự báo:", 1, 14, 7)
    confidence = st.sidebar.slider("Mức độ tin cậy:", 80, 95, 90)


    # Hàm dự báo cải tiến
    def enhanced_forecast(df, target_col, label):
        try:
            df_model = df[['ds', target_col]].rename(columns={target_col: 'y'}).dropna()

            if len(df_model) < 2:
                st.error("Không đủ dữ liệu để dự báo")
                return

            model = Prophet(interval_width=confidence / 100)
            model.fit(df_model)

            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)

            # Biểu đồ
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_model['ds'], df_model['y'], label='Dữ liệu lịch sử', color='blue')
            ax.plot(forecast['ds'], forecast['yhat'], label='Dự báo', color='red', linestyle='--')

            # Vùng tin cậy
            ax.fill_between(
                forecast['ds'],
                forecast['yhat_lower'],
                forecast['yhat_upper'],
                color='gray', alpha=0.2, label=f'Khoảng tin cậy {confidence}%'
            )

            ax.set_title(f"Dự báo {label} {periods} ngày tới", fontsize=14)
            ax.set_xlabel("Ngày", fontsize=12)
            ax.set_ylabel(label, fontsize=12)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Hiển thị dữ liệu dự báo
            st.subheader(f"📈 Kết quả dự báo {label}")
            forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
            forecast_df.columns = ['Ngày', 'Giá trị dự báo', 'Giá trị thấp nhất', 'Giá trị cao nhất']
            st.dataframe(forecast_df.set_index('Ngày'))

        except Exception as e:
            st.error(f"Lỗi khi dự báo: {str(e)}")


    # Thực hiện dự báo
    if option == 'Nhiệt độ':
        enhanced_forecast(df, 'temperature', 'Nhiệt độ (°C)')
    elif option == 'Lượng mưa':
        enhanced_forecast(df, 'precipitation', 'Lượng mưa (mm)')
else:
    st.warning("Không có dữ liệu để hiển thị. Vui lòng kiểm tra file dữ liệu.")