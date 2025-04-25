# Streamlit-Time-Series-Forecaster
⏰ Streamlit Time-Series Forecaster
A one-file Streamlit app that lets anyone:

Upload any CSV that has

one date column (daily, weekly, monthly, hourly, …)

one or more numeric columns

Pick the series and forecast horizon.

Click “Run forecast” – the app trains a model in seconds:

Prophet (falls back to auto-ARIMA if Prophet isn’t installed)

See & download an interactive Plotly chart and a CSV of the forecast.

This project is a proof-of-concept for portfolios and demos—not an enterprise system.
Need production-grade forecasting, CI/CD, or MLOps? → drtomharty.com/bio

✨ Features

Feature	Details
Zero-setup	One Python file (app.py) + requirements.txt.
Model auto-selection	Prophet → auto-ARIMA fallback if Prophet wheel fails to build.
Any time frequency	Choose D, W, M, or H.
Interactive chart	Plotly line plot with confidence bands.
Cached I/O	@st.cache_data & @st.cache_resource keep it snappy.
Download buttons	One-click CSV for the forecast.


🚀 Quick Start (local)
bash
Copy
Edit
# clone the repo
git clone https://github.com/THartyMBA/streamlit-timeseries-forecaster.git
cd streamlit-timeseries-forecaster

# create & activate a virtualenv (optional)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# install deps
pip install -r requirements.txt

# run the app
streamlit run app.py
Open http://localhost:8501 in your browser.

☁️ Deploy on Streamlit Cloud (free)
Push this folder to GitHub (public or private repo).

Log in to streamlit.io/cloud → New app.

Select repo / branch, click Deploy.

🗂️ Repo structure
bash
Copy
Edit
/app.py              ← the entire app
/requirements.txt    ← minimal dependencies
/README.md           ← you’re reading it
🛠️ Requirements

Package	Notes
streamlit	UI framework
pandas	CSV I/O
plotly	interactive charts
prophet	primary forecaster (optional)
pmdarima	auto-ARIMA fallback
📝 License
MIT License
Creative Commons Zero (CC0) — do anything you want, but credit is appreciated.

🙏 Acknowledgements
Facebook Prophet

pmdarima

Streamlit

Enjoy forecasting! 🎉
