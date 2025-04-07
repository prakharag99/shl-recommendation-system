import streamlit as st
import requests
import pandas as pd

# For local testing, use: "http://localhost:8000/recommend"
#API_URL = "https://your-backend-url/recommend"  # update with your deployed backend URL
API_URL = "http://127.0.0.1:8000/recommend"

st.set_page_config(page_title="SHL Assessment Recommendation System", layout="wide")

st.title("SHL Assessment Recommendation System")
st.markdown("Enter a job description or natural language query to get relevant SHL assessments.")

# Input form
query_input = st.text_area("Enter your query or job description URL", height=150)

if st.button("Get Recommendations"):
    if not query_input.strip():
        st.error("Please enter a valid query.")
    else:
        # Call the backend API
        with st.spinner("Fetching recommendations..."):
            try:
                response = requests.post(API_URL, json={"query": query_input})
                if response.status_code == 200:
                    data = response.json()
                    recommendations = data.get("recommendations", [])
                    if recommendations:
                        st.success("Recommendations retrieved!")
                        # Create a DataFrame for display
                        df = pd.DataFrame(recommendations)
                        # Make assessment name clickable
                        def make_clickable(url, name):
                            return f'<a href="{url}" target="_blank">{name}</a>'
                        df["name"] = df.apply(lambda row: make_clickable(row["url"], row["name"]), axis=1)
                        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
                    else:
                        st.warning("No recommendations found for your query.")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
