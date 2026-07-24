import io, re

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================
# Page setup
# =========================
st.set_page_config(
    page_title="Smart Order Tracking System",
    page_icon="🚚",
    layout="wide"
)

st.markdown("""
<div style="background:#1565C0;padding:18px;border-radius:10px;text-align:center;">
    <h1 style="color:white;margin:0;">🚚 SMART ORDER TRACKING SYSTEM USING AI</h1>
    <p style="color:white;margin:0;">On-Time Delivery KPI Monitoring and Shipment Risk Analysis</p>
</div><br>
""", unsafe_allow_html=True)

NULL_TOKENS = {"", "NULL", "NA", "N/A", "NONE", "NAN"}

# =========================
# Data loading + model
# =========================
def clean_text(series, default="Unknown"):
    s = series.astype(str).str.strip()
    return s.mask(s.str.upper().isin(NULL_TOKENS), default).astype(str)

def get_value(df, label):
    row = df[df.iloc[:, 0].astype(str).str.strip().eq(label)]
    if row.empty:
        return None
    value = pd.to_numeric(row.iloc[0, 1], errors="coerce")
    return None if pd.isna(value) else int(value)

def get_target_rate(df, default=90.0):
    for cell in df.to_numpy().flatten():
        match = re.search(r"target\s*:\s*([\d.]+)\s*%", str(cell), re.I)
        if match:
            return float(match.group(1))
    return default

@st.cache_data(show_spinner=False)
def load_data_and_train(file_bytes):
    excel = pd.ExcelFile(io.BytesIO(file_bytes))
    sheet_names = {s.replace(" ", "").lower(): s for s in excel.sheet_names}

    dashboard_sheet = sheet_names.get("sheet2")
    primary_sheet = sheet_names.get("primarydata")

    if dashboard_sheet is None or primary_sheet is None:
        raise ValueError("The Excel file must contain 'Sheet2' and 'Primary data'.")

    df_dash = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=dashboard_sheet,
        header=None,
        keep_default_na=False
    )

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=primary_sheet,
        keep_default_na=False
    )

    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()
    df["BookingID"] = clean_text(df["BookingID"], default="")
    df = df[~df["BookingID"].str.upper().isin(NULL_TOKENS)].drop_duplicates().copy()

    status = (
        df["Status"].astype(str)
        .str.strip().str.lower()
        .str.replace(r"[\s_-]+", " ", regex=True)
    )
    df["target"] = status.map({
        "on time": 1,
        "ontime": 1,
        "delay": 0,
        "delayed": 0
    })
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)
    df["Status"] = df["target"].map({1: "On Time", 0: "Delay"})

    dashboard_on_time = get_value(df_dash, "On Time")
    dashboard_delay = get_value(df_dash, "Delay")
    dashboard_total = get_value(df_dash, "Grand Total")

    primary_total = len(df)
    primary_on_time = int(df["target"].sum())
    primary_delay = primary_total - primary_on_time

    dashboard_on_time = primary_on_time if dashboard_on_time is None else dashboard_on_time
    dashboard_delay = primary_delay if dashboard_delay is None else dashboard_delay
    dashboard_total = dashboard_on_time + dashboard_delay if dashboard_total is None else dashboard_total

    target_rate = get_target_rate(df_dash)
    current_rate = dashboard_on_time / dashboard_total * 100
    gap = max(target_rate - current_rate, 0)
    required_on_time = int(np.ceil(dashboard_total * target_rate / 100))
    additional_needed = max(required_on_time - dashboard_on_time, 0)

    cat_features = ["GpsProvider", "Market/Regular ", "vehicleType", "Month"]
    num_features = ["TRANSPORTATION_DISTANCE_IN_KM"]
    features = cat_features + num_features

    for col in cat_features:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = clean_text(df[col])

    df["TRANSPORTATION_DISTANCE_IN_KM"] = pd.to_numeric(
        clean_text(df["TRANSPORTATION_DISTANCE_IN_KM"], default=np.nan),
        errors="coerce"
    )
    total_distance = df["TRANSPORTATION_DISTANCE_IN_KM"].sum(skipna=True)
    distance_median = df["TRANSPORTATION_DISTANCE_IN_KM"].median()
    distance_median = 0 if pd.isna(distance_median) else distance_median
    df["TRANSPORTATION_DISTANCE_IN_KM"] = df["TRANSPORTATION_DISTANCE_IN_KM"].fillna(distance_median)

    for col in ["Curr_lat", "Curr_lon"]:
        df[col] = pd.to_numeric(clean_text(df[col], default=np.nan), errors="coerce")

    X = df[features].copy()
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("preprocess", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", StandardScaler(), num_features)
        ])),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42))
    ])

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    def group_performance(column):
        result = df.groupby(column).agg(
            Total_Shipments=("BookingID", "count"),
            On_Time_Shipments=("target", "sum")
        ).reset_index()
        result["Delayed_Shipments"] = result["Total_Shipments"] - result["On_Time_Shipments"]
        result["On_Time_Rate"] = result["On_Time_Shipments"] / result["Total_Shipments"] * 100
        result["Delay_Rate"] = 100 - result["On_Time_Rate"]
        result = result[result["Total_Shipments"] >= 20]
        return result.sort_values(["Delayed_Shipments", "On_Time_Rate"], ascending=[False, True])

    kpi = {
        "total": dashboard_total,
        "on_time": dashboard_on_time,
        "delay": dashboard_delay,
        "target_rate": target_rate,
        "current_rate": current_rate,
        "gap": gap,
        "required_on_time": required_on_time,
        "additional_needed": additional_needed,
        "total_distance": total_distance,
        "overall_delay_rate": primary_delay / primary_total * 100,
        "distance_threshold": df["TRANSPORTATION_DISTANCE_IN_KM"].quantile(0.75)
    }

    long_distance_rows = df[
        df["TRANSPORTATION_DISTANCE_IN_KM"] >= kpi["distance_threshold"]
    ]

    if long_distance_rows.empty:
        kpi["long_distance_delay_rate"] = 0.0
        kpi["long_distance_shipments"] = 0
    else:
        kpi["long_distance_delay_rate"] = (
            1 - long_distance_rows["target"].mean()
        ) * 100
        kpi["long_distance_shipments"] = len(long_distance_rows)

    groups = {
        "vehicle": group_performance("vehicleType"),
        "shipment": group_performance("Market/Regular "),
        "month": group_performance("Month")
    }

    return df, model, accuracy, features, cat_features, num_features, distance_median, kpi, groups

def get_group_result(table, column, value):
    """Return the historical performance row matching one shipment factor."""
    matched = table[table[column].astype(str).eq(str(value))]
    return None if matched.empty else matched.iloc[0]


def highest_risk_row(table):
    """Return the group with the highest historical delay rate."""
    if table.empty:
        return None
    return table.sort_values(
        ["Delay_Rate", "Total_Shipments"],
        ascending=[False, False]
    ).iloc[0]


def dashboard_risk_items(groups, kpi):
    """Create the four system-level delay risks displayed on the Dashboard."""
    items = []

    vehicle = highest_risk_row(groups["vehicle"])
    if vehicle is not None:
        items.append({
            "title": "Vehicle Risk",
            "evidence": (
                f"vehicleType '{vehicle['vehicleType']}' is associated with a "
                f"{vehicle['Delay_Rate']:.2f}% historical delay rate across "
                f"{int(vehicle['Total_Shipments']):,} shipments."
            ),
            "mitigation": (
                "Inspect vehicle readiness, review maintenance records, and "
                "consider assigning a more suitable vehicle before dispatch."
            )
        })

    shipment_type = highest_risk_row(groups["shipment"])
    if shipment_type is not None:
        items.append({
            "title": "Shipment Type Risk",
            "evidence": (
                f"Market/Regular '{shipment_type['Market/Regular ']}' is associated "
                f"with a {shipment_type['Delay_Rate']:.2f}% historical delay rate "
                f"across {int(shipment_type['Total_Shipments']):,} shipments."
            ),
            "mitigation": (
                "Review handling requirements and prioritize operational "
                "follow-up for this shipment type."
            )
        })

    month = highest_risk_row(groups["month"])
    if month is not None:
        items.append({
            "title": "Operating-Period Risk",
            "evidence": (
                f"Month '{month['Month']}' is associated with a "
                f"{month['Delay_Rate']:.2f}% historical delay rate across "
                f"{int(month['Total_Shipments']):,} shipments."
            ),
            "mitigation": (
                "Allocate additional vehicles or drivers and prepare a backup "
                "operating plan for this period."
            )
        })

    items.append({
        "title": "Long-Distance Risk",
        "evidence": (
            f"Shipments of at least {kpi['distance_threshold']:,.0f} km record a "
            f"{kpi['long_distance_delay_rate']:.2f}% historical delay rate."
        ),
        "mitigation": (
            "Review the route, departure time, travel allowance, and backup "
            "delivery plan before dispatch."
        )
    })

    return items


def shipment_delay_risks(shipment, groups, kpi):
    """
    Identify shipment-specific risks from Vehicle Type, Shipment Type,
    Month, and Transportation Distance. GPS Provider is excluded.
    """
    risks = []

    checks = [
        (
            "Vehicle-related risk",
            groups["vehicle"],
            "vehicleType",
            shipment.get("vehicleType", "Unknown"),
            "Inspect vehicle readiness and consider assigning a more suitable "
            "vehicle before dispatch."
        ),
        (
            "Shipment-type risk",
            groups["shipment"],
            "Market/Regular ",
            shipment.get("Market/Regular ", "Unknown"),
            "Review handling requirements and prioritize operational follow-up."
        ),
        (
            "Operating-period risk",
            groups["month"],
            "Month",
            shipment.get("Month", "Unknown"),
            "Allocate additional operating resources and prepare a backup "
            "delivery plan."
        )
    ]

    for title, table, column, value, mitigation in checks:
        row = get_group_result(table, column, value)
        if row is None:
            continue

        if row["Delay_Rate"] >= kpi["overall_delay_rate"]:
            risks.append({
                "title": title,
                "evidence": (
                    f"{column.strip()} '{value}' is associated with a "
                    f"{row['Delay_Rate']:.2f}% historical delay rate across "
                    f"{int(row['Total_Shipments']):,} shipments."
                ),
                "mitigation": mitigation,
                "score": float(row["Delay_Rate"])
            })

    distance = float(
        pd.to_numeric(
            pd.Series([
                shipment.get(
                    "TRANSPORTATION_DISTANCE_IN_KM",
                    kpi["distance_threshold"]
                )
            ]),
            errors="coerce"
        ).fillna(kpi["distance_threshold"]).iloc[0]
    )

    if distance >= kpi["distance_threshold"]:
        risks.append({
            "title": "Long-distance risk",
            "evidence": (
                f"The transportation distance is {distance:,.0f} km, which is "
                f"at or above the historical 75th-percentile threshold of "
                f"{kpi['distance_threshold']:,.0f} km."
            ),
            "mitigation": (
                "Review the route, departure schedule, travel-time allowance, "
                "and backup delivery plan before dispatch."
            ),
            "score": 100.0
        })

    return sorted(
        risks,
        key=lambda item: item["score"],
        reverse=True
    )[:4]


def risk_card(title, evidence, mitigation):
    """Render one risk card without Markdown indentation errors."""
    card_html = (
        '<div style="'
        'border:1px solid #dddddd;'
        'border-radius:10px;'
        'padding:18px;'
        'margin-bottom:14px;'
        'background:#ffffff;'
        'min-height:230px;'
        'box-sizing:border-box;'
        '">'
        '<div style="'
        'color:#C62828;'
        'font-size:18px;'
        'font-weight:700;'
        'margin-bottom:16px;'
        '">'
        f'⚠️ {title}'
        '</div>'
        '<div style="margin-bottom:12px;">'
        '<b>Risk evidence:</b>'
        '</div>'
        '<div style="margin-bottom:16px;">'
        f'{evidence}'
        '</div>'
        '<div style="margin-bottom:10px;">'
        '<b>Mitigation action:</b>'
        '</div>'
        '<div>'
        f'{mitigation}'
        '</div>'
        '</div>'
    )

    st.markdown(
        card_html,
        unsafe_allow_html=True
    )


# =========================
# App
# =========================
st.sidebar.header("📂 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (Excel format)", type=["xlsx"])

if uploaded_file is None:
    st.info("👈 Please upload the dataset file on the left sidebar to begin.")
    st.stop()

with st.spinner("Reading data and training AI model..."):
    df, model, accuracy, features, cat_features, num_features, distance_median, kpi, groups = load_data_and_train(uploaded_file.getvalue())

st.sidebar.success(f"✅ AI Model Ready! Accuracy: {accuracy * 100:.2f}%")

tab1, tab2, tab3 = st.tabs(["Dashboard", "Route Map", "Shipment Analysis"])

# =========================
# Dashboard
# =========================
with tab1:
    st.subheader("Historical Shipment Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Shipments", f"{kpi['total']:,}")
    c2.metric("On-Time Shipments", f"{kpi['on_time']:,}")
    c3.metric("Delayed Shipments", f"{kpi['delay']:,}")
    c4.metric("Total Distance", f"{kpi['total_distance']:,.0f} km")

    st.markdown("---")
    st.subheader("90% On-Time Delivery Target Assessment")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current On-Time Rate", f"{kpi['current_rate']:.2f}%")
    c2.metric("Target On-Time Rate", f"{kpi['target_rate']:.2f}%")
    c3.metric("Gap to 90% Target", f"{kpi['gap']:.2f}%")
    if kpi["current_rate"] < kpi["target_rate"]:
        c4.error("BELOW TARGET")
    else:
        c4.success("TARGET ACHIEVED")

    st.warning(
        f"To reach the {kpi['target_rate']:.0f}% target at the current shipment volume, "
        f"at least {kpi['required_on_time']:,} shipments must be delivered on time. "
        f"Improvement equivalent to {kpi['additional_needed']:,} additional on-time shipments is required."
    )

    st.markdown("---")
    st.subheader("AI Prediction Overview")
    a1, a2, a3 = st.columns(3)
    a1.info("**AI Prediction Output**\n\nPredicted Delivery Status\n\n**On Time / Delayed**")
    a2.info("**Prediction Scope**\n\nIndividual shipments retrieved by **Booking ID**")
    a3.info("**Input Factors**\n\nGPS Provider, Shipment Type, Vehicle Type, Month, and Distance")
    st.success(
        f"**Business Use:** The AI model identifies shipments with a high risk of delay and supports improvement toward the {kpi['target_rate']:.0f}% on-time delivery target."
    )

    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    bars = ax.bar(["Current On-Time Rate", "Target"], [kpi["current_rate"], kpi["target_rate"]])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Current On-Time Performance vs. 90% Target")
    for bar, value in zip(bars, [kpi["current_rate"], kpi["target_rate"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.2f}%", ha="center")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)

    st.markdown("---")
    st.subheader("Key Delay Risks and Mitigation Actions")

    dashboard_risks = dashboard_risk_items(groups, kpi)
    risk_columns = st.columns(2)

    for i, risk in enumerate(dashboard_risks):
        with risk_columns[i % 2]:
            risk_card(
                risk["title"],
                risk["evidence"],
                risk["mitigation"]
            )

# =========================
# Route Map
# =========================
with tab2:
    st.subheader("Shipment Route Visualization")
    st.info("This module visualizes historical shipment locations and supports transportation review.")

    map_df = df.dropna(subset=["Curr_lat", "Curr_lon"])
    if map_df.empty:
        st.warning("No valid GPS coordinates found.")
    else:
        m = folium.Map(
            location=[map_df["Curr_lat"].mean(), map_df["Curr_lon"].mean()],
            zoom_start=6
        )

        for _, row in map_df.head(100).iterrows():
            popup = (
                f"<b>Booking ID:</b> {row.get('BookingID', 'N/A')}<br>"
                f"<b>Vehicle:</b> {row.get('vehicleType', 'N/A')}<br>"
                f"<b>Status:</b> {row.get('Status', 'N/A')}<br>"
                f"<b>Destination:</b> {row.get('Destination_Location', 'N/A')}"
            )
            folium.Marker(
                [row["Curr_lat"], row["Curr_lon"]],
                popup=popup,
                tooltip=str(row.get("BookingID", "N/A"))
            ).add_to(m)

        st_folium(m, width=1200, height=550)

# =========================
# Shipment Analysis
# =========================
with tab3:
    st.subheader("Shipment Analysis")
    st.info(
        "**Prediction logic:** Booking ID retrieves the shipment record. "
        "The AI model then uses GPS Provider, Shipment Type, Vehicle Type, Month, and Distance to estimate delivery status."
    )

    booking_id = st.text_input("Enter Booking ID:")

    if st.button("Search Order"):
        result = df[df["BookingID"].astype(str).str.strip().eq(booking_id.strip())]

        if booking_id.strip() == "":
            st.warning("Please enter a Booking ID.")
        elif result.empty:
            st.error("❌ Booking ID not found.")
        else:
            shipment = result.iloc[0]
            X_new = pd.DataFrame([{feature: shipment.get(feature, "Unknown") for feature in features}])

            for col in cat_features:
                X_new[col] = clean_text(X_new[col])
            for col in num_features:
                X_new[col] = pd.to_numeric(X_new[col], errors="coerce").fillna(distance_median)

            prediction = int(model.predict(X_new)[0])
            prob = model.predict_proba(X_new)[0]
            on_time_prob = prob[list(model.named_steps["classifier"].classes_).index(1)] * 100
            delay_prob = 100 - on_time_prob
            identified_risks = shipment_delay_risks(shipment, groups, kpi)

            left, right = st.columns(2)

            with left:
                st.info("### ORDER & SHIPMENT INFORMATION")
                st.write(f"**Booking ID:** {shipment.get('BookingID', 'N/A')}")
                st.write(f"**Customer:** {shipment.get('customerNameCode', 'N/A')}")
                st.write(f"**Material:** {shipment.get('Material Shipped', 'N/A')}")
                st.markdown("---")
                st.write(f"**Origin:** {shipment.get('Origin_Location', 'N/A')}")
                st.write(f"**Destination:** {shipment.get('Destination_Location', 'N/A')}")
                st.write(f"**Current Location:** {shipment.get('Current_Location', 'N/A')}")
                st.write(f"**Distance:** {shipment.get('TRANSPORTATION_DISTANCE_IN_KM', 'N/A')} km")
                st.write(f"**Planned ETA:** {shipment.get('Planned_ETA', 'N/A')}")

            with right:
                if prediction == 1:
                    st.success("### 🟢 AI DELIVERY STATUS ASSESSMENT")
                    st.write("## ON TIME")
                    recommendation = "Maintain the planned schedule and continue routine shipment monitoring."
                else:
                    st.error("### 🔴 AI DELIVERY STATUS ASSESSMENT")
                    st.write("## DELAYED")
                    recommendation = (
                        "Apply the mitigation actions linked to the identified delay risks."
                    )

                st.write(f"**Estimated On-Time Probability:** {on_time_prob:.2f}%")
                st.write(f"**Estimated Delay Probability:** {delay_prob:.2f}%")
                st.markdown("---")
                st.write(f"**Operational Recommendation:** {recommendation}")
                st.write(f"**Vehicle:** {shipment.get('vehicleType', 'N/A')}")
                st.write(f"**Driver:** {shipment.get('Driver_Name', 'N/A')}")

            if prediction == 1:
                st.success(
                    "**Delay Risk Review:** No major delay risk requires immediate "
                    "mitigation. Continue routine shipment monitoring."
                )
            else:
                if identified_risks:
                    top_risks = identified_risks[:2]

                    risk_names = " and ".join(
                        risk["title"] for risk in top_risks
                    )

                    mitigation_actions = "; ".join(
                        risk["mitigation"] for risk in top_risks
                    )

                    st.warning(
                        f"**Delay Risk Review:** {risk_names} identified.\n\n "
                        f"**Mitigation:** {mitigation_actions}"
                    )
                else:
                    st.warning(
                        "**Delay Risk Review:** No single major risk factor was "
                        "identified. Review vehicle readiness, route planning, and "
                        "the departure schedule before dispatch."
                    )        