import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================
# WEB PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Smart Order Trcking System", page_icon="🚚", layout="wide")

st.markdown("""
    <div style='background-color:#1565C0; padding:15px; border-radius:10px; text-align:center;'>
        <h1 style='color:white; margin:0;'>🚚 SMART ORDER TRACKING SYSTEM USING AI</h1>
        <p style='color:white; margin:0;'>Logistics Order Tracking & AI Prediction Platform </p>
    </div>
    <br>
""", unsafe_allow_html=True)

# ==========================================
# DATA PROCESSING & AI TRAINING (Logistic Regression)
# ==========================================
@st.ccahe_data
def load_and_train_model(file):
    df_primary = pd.read_excel(file, sheet_name="Primary data")
    df_ai = pd.read_excel(file, sheet_name="Refined")

    # Clean data
    df_primary = df_primary.drop_duplicates().dropna(subset=["BookingID"])

    # Encode condition text
    le_condition = LabelEncoder()
    df_ai["condition_encoded"] = le_condition.fit_transform(df_ai["condition_text"].astype(str))

    # Feature Selection (Exact match with untitled5.py)
    features = [
        "Fixed Costs",
        "Maintenance",
        "condition_encoded"
    ]
    target = "On time Delivery"

    data = df_ai[features + [target]].copy()
    data.dropna(inplace=True)

    X = data[features]
    y = data[target]

    # Train Logistic Regression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))

    return df_primary, df_ai, model, le_condition, acc

# ==========================================
# SIDEBAR / FILE UPLOAD
# ==========================================
st.sidebar.header("📂 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (Excel format)", type=["xlsx"])

if uploaded_file is None:
    st.info("👈 Please upload the Dataset file on the left sidebar to begin.")
else:
    with st.spinner('Reading data and training AI model...'):
        df_primary, df_ai, model, le_condition, accuracy = load_and_train_model(uploaded_file)
    
    st.sidebar.success(f"✅ AI Model Ready! Accuracy: {accuracy*100:.2f}%")

    # Create 3 Tabs
    tab1, tab2, tab3 = st.tabs([" Dashboard & Statistics", " GPS Tracking Map", " AI Prediction & Search"])

    # ------------------------------------------
    # TAB 1: DASHBOARD
    # ------------------------------------------
    with tab1:
        st.subheader("📦 Order Overview")
        col1, col2, col3, col4 = st.columns(4)
        total_orders = len(df_primary)
        on_time = df_primary["ontime"].notna().sum()
        delay = df_primary["delay"].notna().sum()
        total_distance = df_primary["TRANSPORTATION_DISTANCE_IN_KM"].sum()
        
        col1.metric("Total Orders", total_orders)
        col2.metric("On Time", f"{on_time}")
        col3.metric("Delayed", f"{delay}")
        col4.metric("Total Distance", f"{total_distance:.0f} km")

        st.markdown("---")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("**Delivery Statistics**")
            fig1, ax1 = plt.subplots(figsize=(5,3))
            ax1.bar(["On Time", "Delay"], [on_time, delay], color=['#2ca02c', '#d62728'])
            st.pyplot(fig1)

        with col_chart2:
            st.markdown("**Top 10 Vehicle Types**")
            top_vehicle = df_primary["vehicleType"].value_counts().head(10)
            fig2, ax2 = plt.subplots(figsize=(5,3))
            top_vehicle.plot(kind="bar", color='#1f77b4', ax=ax2)
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig2)

    # ------------------------------------------
    # TAB 2: GOOGLE MAP
    # ------------------------------------------
    with tab2:
        st.subheader(" GPS Tracking Map")
        map_df = df_primary.dropna(subset=["Curr_lat", "Curr_lon"])
        
        if not map_df.empty:
            shipment_map = folium.Map(location=[map_df["Curr_lat"].mean(), map_df["Curr_lon"].mean()], zoom_start=6)
            for _, row in map_df.head(100).iterrows(): 
                popup_info = f"<b>ID:</b> {row['BookingID']}<br><b>Vehicle:</b> {row['vehicleType']}<br><b>To:</b> {row['Destination_Location']}"
                folium.Marker(
                    location=[row["Curr_lat"], row["Curr_lon"]],
                    popup=popup_info,
                    tooltip=row["BookingID"]
                ).add_to(shipment_map)
            
            st_folium(shipment_map, width=1000, height=500)
        else:
            st.warning("No valid GPS coordinates found.")

    # ------------------------------------------
    # TAB 3: AI PREDICTION
    # ------------------------------------------
    with tab3:
        st.subheader(" AI Prediction & Search")
        
        booking_id = st.text_input("Enter Booking ID:")
        
        if st.button("Search Order"):
            if booking_id == "":
                st.warning("Please enter a Booking ID.")
            else:
                primary_result = df_primary[df_primary["BookingID"].astype(str) == booking_id]
                
                if primary_result.empty:
                    st.error("❌ Booking ID not found .")
                else:
                    primary = primary_result.iloc[0]
                    delivery_id = primary["BookingID"]
                    refined_result = df_ai[df_ai["Delivery Id"].astype(str) == str(delivery_id)]
                    
                    if refined_result.empty:
                        st.error("❌ No AI features found for this Booking ID.")
                    else:
                        refined = refined_result.iloc[0]
                        
                        # AI Prediction
                        condition_code = le_condition.transform([refined["condition_text"]])[0]
                        X_new = pd.DataFrame([{
                            "Fixed Costs": refined["Fixed Costs"],
                            "Maintenance": refined["Maintenance"],
                            "condition_encoded": condition_code
                        }])
                        
                        prediction = model.predict(X_new)[0]
                        prob = model.predict_proba(X_new)[0]
                        confidence = max(prob) * 100
                        
                        st.markdown("---")
                        res_col1, res_col2 = st.columns(2)
                        
                        with res_col1:
                            st.info("📦 **ORDER & SHIPMENT INFORMATION**")
                            st.write(f"**Booking ID:** {primary['BookingID']}")
                            st.write(f"**Customer:** {primary['customerNameCode']}")
                            st.write(f"**Material:** {primary['Material Shipped']}")
                            st.write(f"**Current Location:** {primary['Current_Location']}")
                            st.write(f"**Distance:** {primary['TRANSPORTATION_DISTANCE_IN_KM']} km")
                            st.write(f"**Planned ETA:** {primary['Planned_ETA']}")
                            st.write(f"**Driver:** {primary['Driver_Name']}")

                        with res_col2:
                            if prediction == 1:
                                st.success(f"### 🟢 AI PREDICTION: ON TIME")
                                st.write(f"**Confidence:** {confidence:.2f}%")
                                st.write("**Recommendation:** Proceed with planned delivery.")
                            else:
                                st.error(f"### 🔴 AI PREDICTION: DELAYED")
                                st.write(f"**Confidence:** {confidence:.2f}%")
                                st.write("**Recommendation:** Monitor shipment and notify customer.")
