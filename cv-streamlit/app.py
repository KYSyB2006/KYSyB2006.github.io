from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Belva YONTA · CV", page_icon="🎓", layout="wide")

ASSETS = Path(__file__).resolve().parent.parent / "assets"
DATA_URL = "https://raw.githubusercontent.com/uber-web/kepler.gl-data/master/nyctrips/data.csv"


# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.image(str(ASSETS / "logo-efrei.png"), width=140)
    st.markdown("### Belva YONTA")
    st.markdown("[yontabelva@gmail.com](mailto:yontabelva@gmail.com)")
    st.markdown("EFREI Paris · **2026 – 2028**")
    st.markdown("[GitHub](https://github.com/KYSyB2006) · [LinkedIn](https://www.linkedin.com/in/samyra-yonta)")
    st.divider()
    page = st.radio("Courses", [
        "About me",
        "Data Visualization & BI",
        "Mathematics for Data Science",
        "Machine Learning I",
        "API & Web Services",
    ])


# ---------------------------------------------------------------- About me
def about_me():
    st.title("Belva YONTA")
    st.subheader("Data & AI engineering student · Big Data and Machine Learning")
    st.success("Available now for a work-study contract (alternance) until summer 2028 · 3 days in company / 2 days at school")

    st.markdown("""
I turn data into decisions: reliable pipelines, dashboards that are useful to the business,
and Machine Learning models evaluated with rigour.
""")

    c1, c2, c3 = st.columns(3)
    c1.metric("TOEIC", "955 / 990")
    c2.metric("Preparatory classes", "Top of the class")
    c3.metric("EFREI cycle", "2026 – 2028")

    st.header("Education")
    st.markdown("""
- **2026 – 2028 · EFREI Paris** – engineering cycle (work-study), Big Data and Machine Learning
- **2025 – 2026 · UTBM, Belfort** – first year of the engineering cycle, artificial intelligence and information systems
- **2023 – 2025 · IUC, Douala** – engineering preparatory classes, top of the class
""")

    st.header("Projects")
    st.markdown("""
- **Wine quality classification** (Machine Learning, Streamlit, Docker) – [app](https://winequalityclassificator-efrei.streamlit.app/) · [code](https://github.com/KYSyB2006/WineQualityClassificator)
- **Semantic search for work-study offers** (pgvector, FastAPI) – [code](https://github.com/KYSyB2006/alternance-finder)
- **MediConnect**, from database to Power BI dashboard – [case study](https://kysyb2006.github.io/mediconnect-portfolio/)
- **Teeko game agent** – [case study](https://kysyb2006.github.io/teeko-portfolio/)
- **PLAY & LEARN**, user interface design – [case study](https://kysyb2006.github.io/play-and-learn-portfolio/)
""")

    st.header("Skills")
    st.markdown("""
- **Data & ML:** Python, pandas, NumPy, scikit-learn, statistics
- **Visualisation:** Power BI, DAX, Matplotlib, seaborn, Streamlit
- **Databases:** PostgreSQL, MySQL, MongoDB
- **Tools:** Git, GitHub Actions, Docker, Linux, C / C++, Java
- **Languages:** French (native), English C1
""")
    st.markdown("Full portfolio: [kysyb2006.github.io](https://kysyb2006.github.io/)")


# ---------------------------------------------------------------- Data Visualization course
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"])
    df["trip_duration"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    return df


def zone(lat, lon):
    conditions = [
        lat.between(40.62, 40.67) & lon.between(-73.83, -73.75),     # JFK airport
        lat.between(40.76, 40.78) & lon.between(-73.89, -73.86),     # LaGuardia airport
        lat.between(40.70, 40.73) & lon.between(-74.02, -73.97),     # south Manhattan
        lat.between(40.73, 40.77) & lon.between(-74.01, -73.96),     # central Manhattan
        lat.between(40.77, 40.88) & lon.between(-74.00, -73.93),     # north Manhattan
    ]
    names = ["JFK", "LaGuardia", "South Manhattan", "Central Manhattan", "North Manhattan"]
    return np.select(conditions, names, default="Other")


@st.cache_data
def prepare(raw):
    # Remove impossible trips: zero distance, zero or negative fare, zero or negative duration
    df = raw[(raw["trip_distance"] > 0) & (raw["fare_amount"] > 0) & (raw["trip_duration"] > 0)].copy()
    df["distance_class"] = pd.cut(df["trip_distance"], [0, 1, 2, 3, 5, 10, 20, 100]).astype(str)
    df["pickup_zone"] = zone(df["pickup_latitude"], df["pickup_longitude"])
    df["dropoff_zone"] = zone(df["dropoff_latitude"], df["dropoff_longitude"])
    df["price_per_mile"] = df["fare_amount"] / df["trip_distance"]
    df["speed_mph"] = df["trip_distance"] / (df["trip_duration"] / 60)
    df["band"] = (df["trip_distance"] * 2).round() / 2                 # 0.5-mile bands: 1.59 -> 1.5
    return df


def gap_by_zone(df):
    # Gap = fare - average fare of trips in the same 0.5-mile band
    gap = df["fare_amount"] - df.groupby("band")["fare_amount"].transform("mean")
    return pd.DataFrame({
        "pick-up": gap.groupby(df["pickup_zone"]).mean(),
        "drop-off": gap.groupby(df["dropoff_zone"]).mean(),
    }).round(2)


def data_viz_course():
    st.caption("Course")
    st.title("Data Visualization & Business Intelligence")
    st.markdown("""
**Professor:** Dr Mano Joseph MATHEW, EFREI Paris · [mano.mathew@efrei.fr](mailto:mano.mathew@efrei.fr)

**Assigned dataset:** [New York yellow taxi trips, 15 January 2015](https://github.com/uber-web/kepler.gl-data/tree/master/nyctrips)
(kepler.gl sample data, [raw CSV file](https://raw.githubusercontent.com/uber-web/kepler.gl-data/master/nyctrips/data.csv))
""")

    raw = load_data()
    df = prepare(raw)

    tab1, tab2, tab3 = st.tabs(["Dataset", "Analysis", "Presentation"])

    # ---------- Dataset
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Trips", f"{len(raw):,}")
        c2.metric("Columns", raw.shape[1] - 2)          # the 2 computed columns are not counted
        c3.metric("Missing values", int(raw.isna().sum().sum()))
        st.dataframe(raw.head(100))
        st.subheader("Summary statistics")
        st.dataframe(raw[["passenger_count", "trip_distance", "fare_amount", "tip_amount", "total_amount", "trip_duration"]].describe().round(2))
        st.subheader("Cleaning")
        st.markdown(f"""
- {(raw['trip_distance'] == 0).sum()} trips with a distance of 0 miles
- {(raw['fare_amount'] <= 0).sum()} trips with a fare of $0 or less
- {(raw['trip_duration'] <= 0).sum()} trip with a duration of 0 minutes or less

**{len(raw) - len(df)} trips removed, {len(df):,} kept.**
""")

    # ---------- Analysis
    with tab2:
        st.subheader("1. Where do the trips happen?")
        st.map(df.sample(5000, random_state=0), latitude="pickup_latitude", longitude="pickup_longitude", size=20)
        st.dataframe(pd.DataFrame({
            "pick-ups": df["pickup_zone"].value_counts(),
            "drop-offs": df["dropoff_zone"].value_counts(),
        }))

        st.subheader("2. Trips and price by distance class (miles)")
        classes = df.groupby("distance_class").agg(
            trips=("fare_amount", "size"),
            median_fare=("fare_amount", "median"),
            median_price_per_mile=("price_per_mile", "median"),
            median_speed_mph=("speed_mph", "median"),
        )
        order = ["(0, 1]", "(1, 2]", "(2, 3]", "(3, 5]", "(5, 10]", "(10, 20]", "(20, 100]"]
        classes = classes.loc[order]
        c1, c2 = st.columns(2)
        c1.bar_chart(classes["trips"], sort=False, x_label="Distance class (miles)", y_label="Trips")
        c2.bar_chart(classes["median_price_per_mile"], sort=False, x_label="Distance class (miles)", y_label="Median price per mile ($)")
        st.dataframe(classes.round(2))

        st.subheader("3. At equal distance, median fare ($) by zone")
        side = st.radio("Zone of", ["pick-up", "drop-off"], horizontal=True)
        side = "pickup_zone" if side == "pick-up" else "dropoff_zone"
        table = df.pivot_table(index="distance_class", columns=side, values="fare_amount", aggfunc="median")
        count = df.pivot_table(index="distance_class", columns=side, values="fare_amount", aggfunc="size")
        st.dataframe(table[count >= 30].loc[order].style.background_gradient(cmap="Blues", axis=None).format("{:.2f}", na_rep=""))

        st.subheader("4. By how much? Average gap with trips of similar distance ($)")
        fixed = df["fare_amount"] == 52
        c1, c2 = st.columns(2)
        c1.markdown("**All trips**")
        c1.dataframe(gap_by_zone(df))
        c2.markdown(f"**Without the {fixed.sum():,} trips at exactly $52**")
        c2.dataframe(gap_by_zone(df[~fixed]))

        st.subheader("5. Speed and price per mile by pick-up hour")
        by_hour = df.groupby("pickup_hour")[["speed_mph", "price_per_mile"]].median()
        c1, c2 = st.columns(2)
        c1.bar_chart(by_hour["speed_mph"], x_label="Pick-up hour", y_label="Median speed (mph)", color="#7B3FC0")
        c2.bar_chart(by_hour["price_per_mile"], x_label="Pick-up hour", y_label="Median price per mile ($)", color="#C0392B")
        st.write("Correlation between speed and price per mile, hour by hour:",
                 round(by_hour["speed_mph"].corr(by_hour["price_per_mile"]), 2))

    # ---------- Presentation
    with tab3:
        st.subheader("At equal distance, does the place of pick-up or drop-off change the price?")
        c1, c2, c3 = st.columns(3)
        c1.metric("Pick-ups in Manhattan zones", "91 %")
        c2.metric("Gap inside Manhattan", "< $0.50")
        c3.metric("Speed vs price per mile", "−0.97")
        st.markdown("""
**One day, mostly in Manhattan.** The 97,500 trips kept were all picked up on 15 January 2015.
91% of pick-ups and 87% of drop-offs fall in the three Manhattan zones.

**The price is not proportional to distance.** The median price per mile falls from $7.50 for trips under
1 mile to $2.55 above 20 miles. Trips over 5 miles are only 14% of trips but bring 35% of the fares.
This is why places are compared at equal distance.

**Inside Manhattan, the place barely matters.** Compared with trips of similar distance, the average gap stays
under $0.50: central Manhattan is slightly more expensive, north and south Manhattan slightly cheaper.

**The JFK zone stands out.** 1,771 trips cost exactly $52, and 92% of them connect JFK and Manhattan,
whatever their distance: this looks like a fixed price. Without these trips, metered trips from JFK cost
$1.93 less than trips of similar distance, and trips to JFK $2.19 less.

**Speed drives the price per mile.** At 4 am taxis move at 19.2 mph and a mile costs $3.73; around 11 am they move
at about 8 mph and a mile costs $6.25. Hour by hour, the correlation is −0.97: the slower the taxis, the more
expensive each mile.

**Limits:** one single day of data, approximate rectangular zones, and a few trips lasting several hours
(up to almost 24 hours) are kept.
""")
        st.markdown("Analysis notebook: `sample.ipynb` (UberDataAnalysis project).")


# ---------------------------------------------------------------- Other courses
def other_course(name):
    st.caption("Course")
    st.title(name)
    st.info("Content coming soon.")


if page == "About me":
    about_me()
elif page == "Data Visualization & BI":
    data_viz_course()
else:
    other_course(page)
