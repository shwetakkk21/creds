import streamlit as st
import pandas as pd
import numpy as np
import json
import pdfplumber
import re
import plotly.graph_objects as go
import requests # Necessary for API communication

# --- CONFIGURATION ---
# CRITICAL: Use the public URL of your deployed FastAPI Space.
API_ENDPOINT_URL = "https://yashkumfux-credit-scoring.hf.space"

# --- Page Configuration ---
st.set_page_config(
    page_title="Beneficiary Credit Scoring Dashboard",
    layout="wide"
)

# --- NOTE: All local model loading and prediction logic has been removed. ---

# --- Helper & Charting Functions ---

def analyze_bank_statement(uploaded_file):
    """Extracts and calculates average salary from a PDF bank statement."""
    # (Local function to handle file upload and text extraction)
    if uploaded_file is None:
        return None, "No file uploaded."
    salaries = []
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    # Regex to find common salary/credit transaction strings
                    matches = re.finditer(r'(?i)(?:SALARY|SAL|SAL-TRANSFER|SALARY CREDIT)\s.*?([\d,]+\.\d{2})', text)
                    for match in matches:
                        salaries.append(float(match.group(1).replace(',', '')))
    except Exception as e:
        return None, f"Could not process PDF. Error: {e}"
    if not salaries:
        return 0, "Analyzed: No salary credits found."
    avg_salary = np.mean(salaries)
    return avg_salary, f"Verified: Average monthly income is ₹{avg_salary:,.2f}"

def create_gauge_chart(score, title):
    """Creates a Plotly gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': 'red'},
                {'range': [40, 70], 'color': 'orange'},
                {'range': [70, 100], 'color': 'green'}],
        }))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_quadrant_chart(repayment_score, income_score):
    """Creates a risk quadrant chart, using 0.65 as the low-risk willingness threshold."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[repayment_score], y=[income_score],
        mode='markers',
        marker=dict(color='blue', size=20, symbol='star'),
        name='Applicant Profile'
    ))
    # Low Risk Willingness Threshold (W1)
    fig.add_shape(type="line", x0=0.65, y0=0, x1=0.65, y1=1, line=dict(color="grey", width=2, dash="dash"))
    # High Need Ability Threshold (W2)
    fig.add_shape(type="line", x0=0, y0=0.5, x1=1, y1=0.5, line=dict(color="grey", width=2, dash="dash"))
    
    # Quadrant Labels (based on Repayment W1=0.65 and Income W2=0.5)
    fig.add_annotation(x=0.325, y=0.75, text="High Risk,<br>Higher Income", showarrow=False)
    fig.add_annotation(x=0.825, y=0.75, text="Low Risk,<br>Lower Need", showarrow=False)
    fig.add_annotation(x=0.325, y=0.25, text="High Risk,<br>High Need", showarrow=False)
    fig.add_annotation(x=0.825, y=0.25, text="Low Risk,<br>High Need", showarrow=False)
    
    fig.update_xaxes(range=[0, 1], title_text="Repayment Score (Willingness to Pay)")
    fig.update_yaxes(range=[0, 1], title_text="Income Score (Ability to Pay)")
    fig.update_layout(title_text="Risk Quadrant Analysis", height=450)
    return fig


# --- UI Layout ---
st.title("Beneficiary Credit Scoring & Digital Lending")

# Initialize session state for results if it doesn't exist
if 'results' not in st.session_state:
    st.session_state.results = None

# --- Sidebar for User Input (The 31 Feature Form) ---
with st.sidebar.form(key='applicant_form'):
    st.header("Applicant Information Form")
    
    st.subheader("Personal & Loan Details")
    user_inputs = {
        'name': st.text_input("Full Name", "John Doe"),
        # Model B Demographic Inputs (Numeric)
        'Age': st.number_input("Age", 18, 100, 35),
        'head_of_household_age': st.number_input("HH Head Age", 18, 100, 35),
        'household_size_calculated': st.number_input("Household Size", 1, 20, 4),
        'avg_education_years_adults': st.number_input("Avg. Education Years (Adults)", 0.0, 25.0, 12.0, step=0.1),
        'num_internet_users': st.number_input("Internet Users in HH", 0, 10, 0),

        # Model B Demographic Inputs (Categorical)
        'Social_Group_of_HH_Head': st.selectbox("Social Group", ["ST", "SC", "OBC", "General"]),
        'Sector': st.radio("Sector", ["Rural", "Urban"]),
        'Religion_of_HH_Head': st.selectbox("Religion", ["Hindu", "Muslim", "Christian", "Other"]),
        'Max_Income_Activity': st.selectbox("Primary Income Source", ["Salaried", "Self-employed", "Business", "Farm Owner"]),
        'Ration_Card_Type': st.selectbox("Ration Card Type", ["APL", "BPL", "Antyodaya"]),
        'Type_of_Dwelling_Unit': st.selectbox("Type of Dwelling", ["Owned", "Hired/Rented", "Ancestral"]),
        'Land_Ownership': st.selectbox("Owns Agricultural Land", ["Yes", "No"]),

        # Loan/Credit Details (Model A Inputs)
        'LoanAmount': st.number_input("Loan Amount Requested (INR)", 10000, 1000000, 50000, 5000),
        'LoanPurpose': st.selectbox("Loan Purpose", ["Business", "Personal", "Education", "Home Improvement"]),
        'LoanTerm': st.number_input("Loan Term (Months)", 6, 120, 60),
        'Income': st.number_input("Claimed Net Monthly Income", 5000, 500000, 25000, 1000),
        'MonthsEmployed': st.number_input("Time Employed (Months)", 0, 600, 60),
        'EmploymentType': st.selectbox("Employment Type", ["Full-time", "Self-employed", "Contract", "Unemployed"]),
        'MaritalStatus': st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed"]),
        'HasDependents': st.selectbox("Has Dependents", ["Yes", "No"]),
    }
    
    st.subheader("Financial & Risk Inputs")
    user_inputs.update({
        'CreditScore': st.number_input("Traditional Credit Score (0 for Invisible)", 0, 900, 0),
        'DTIRatio': st.number_input("Debt-to-Income Ratio (Input/Verified)", 0.0, 1.0, 0.20, 0.01),
        'NumCreditLines': st.number_input("Number of Credit Lines", 0, 20, 0),
        'InterestRate': st.number_input("Interest Rate (%)", 0.0, 30.0, 11.5),
        'HasMortgage': st.selectbox("Has Mortgage", ["Yes", "No"]),
        'HasCoSigner': st.selectbox("Has CoSigner", ["Yes", "No"]),
        'Education': st.selectbox("Education Level", ["High School", "Bachelor's", "Primary", "None"]),
    })

    st.subheader("Consumption Proxies")
    user_inputs.update({
        'fuel_expenditure': st.number_input("Avg Monthly Fuel/Electric Spend (INR)", 0, 5000, 1200, 100),
        'comm_expenditure': st.number_input("Avg Monthly Comm/Mobile Spend (INR)", 0, 3000, 500, 50),
        'existing_emi': st.number_input("Total Existing Monthly EMIs", 0, 200000, 5000, 500), # Used for local DTI calculation
    })
    
    st.subheader("Assets & Subsidies (for X1, X2 calculation)")
    user_inputs.update({
        'Asset_Car': st.checkbox("Owns a Car (5 pts)"),
        'Asset_Refrigerator': st.checkbox("Owns a Refrigerator (2 pts)"),
        'Asset_WM': st.checkbox("Owns a Washing Machine (3 pts)"),
        'Scheme_Proxy': st.selectbox("Receives Major Subsidy Benefits?", ["Yes", "No"]),
    })
    
    uploaded_statement = st.file_uploader("Upload Bank Statement PDF", type="pdf")
    submit_button = st.form_submit_button(label='Assess Creditworthiness (LIVE API)')

# --- Calculation Logic (Sending Data to FastAPI) ---
if submit_button:
    if API_ENDPOINT_URL == "https://yashkumfuv-credit-scoring.hf.space":
        # Check if the user needs to upload a bank statement to verify the income
        verified_income, verification_message = analyze_bank_statement(uploaded_statement)
        net_monthly_income = verified_income if verified_income is not None and verified_income > 0 else user_inputs['Income']
        
        with st.spinner('Analyzing application with LIVE API...'):
            # --- 1. Calculate Engineered Scores (Local Logic for Proxies) ---
            asset_score_value = 0
            if user_inputs['Asset_Car']: asset_score_value += 5
            if user_inputs['Asset_Refrigerator']: asset_score_value += 2
            if user_inputs['Asset_WM']: asset_score_value += 3
            
            scheme_index_value = 4.0 if user_inputs['Scheme_Proxy'] == "Yes" else 1.0 # BPL=4.0, APL=1.0

            # --- 2. Assemble the FINAL API Payload (All 31 Features) ---
            api_payload = {
                "Age": user_inputs['Age'], "Income": net_monthly_income, "LoanAmount": user_inputs['LoanAmount'], 
                "CreditScore": user_inputs['CreditScore'], "MonthsEmployed": user_inputs['MonthsEmployed'], 
                "NumCreditLines": user_inputs['NumCreditLines'], "InterestRate": user_inputs['InterestRate'], 
                "LoanTerm": user_inputs['LoanTerm'], "DTIRatio": user_inputs['DTIRatio'],
                "Education": user_inputs['Education'], "EmploymentType": user_inputs['EmploymentType'], 
                "MaritalStatus": user_inputs['MaritalStatus'], "HasMortgage": user_inputs['HasMortgage'], 
                "HasDependents": user_inputs['HasDependents'], "LoanPurpose": user_inputs['LoanPurpose'], 
                "HasCoSigner": user_inputs['HasCoSigner'], 
                
                "Sector": user_inputs['Sector'], "Social_Group_of_HH_Head": user_inputs['Social_Group_of_HH_Head'], 
                "Max_Income_Activity": user_inputs['Max_Income_Activity'], "Type_of_Dwelling_Unit": user_inputs['Type_of_Dwelling_Unit'],
                "Land_Ownership": user_inputs['Land_Ownership'], "Ration_Card_Type": user_inputs['Ration_Card_Type'], 
                "Religion_of_HH_Head": user_inputs['Religion_of_HH_Head'], "head_of_household_age": user_inputs['head_of_household_age'], 
                "household_size_calculated": user_inputs['household_size_calculated'], "avg_education_years_adults": user_inputs['avg_education_years_adults'],
                "num_internet_users": user_inputs['num_internet_users'], "fuel_expenditure": user_inputs['fuel_expenditure'], 
                "comm_expenditure": user_inputs['comm_expenditure'], 
                "Asset_Score_X1": asset_score_value,
                "Scheme_Index_X2": scheme_index_value
            }

            # --- 3. Send Request to FastAPI ---
            try:
                response = requests.post(f"{API_ENDPOINT_URL}/score", json=api_payload)
                response.raise_for_status() 
                api_results = response.json()
                
                if "error" in api_results:
                    st.error(f"API Error: {api_results['error']}")
                    st.session_state.results = None
                    st.stop()

                # --- 4. Process API Results for Dashboard ---
                dti_local = user_inputs['existing_emi'] / net_monthly_income if net_monthly_income > 0 else 1
                
                st.session_state.results = {
                    'user_inputs': user_inputs, 
                    'net_monthly_income': net_monthly_income,
                    'verification_message': verification_message, 
                    'repayment_score': api_results['repayment_score'],
                    'income_score': api_results['income_proxy_score'],
                    'pci': np.random.randint(650, 850), # Placeholder for PCI
                    'wds': np.random.uniform(0.6, 0.95), # Placeholder for WDS
                    'existing_emi': user_inputs['existing_emi'], 
                    'dti': dti_local,
                    'existing_debt': user_inputs['existing_emi'] / 0.02, 
                    'composite_score': api_results['composite_score'],
                    'risk_band_full': api_results['risk_band']
                }

            except requests.exceptions.RequestException as e:
                st.error(f"🔴 Connection Error: Could not reach the API. Check your network or URL. Details: {e}")
                st.session_state.results = None
            except Exception as e:
                st.error(f"🔴 Unknown Processing Error in Frontend: {e}")
                st.session_state.results = None
    else:
        st.error("🚨 API Endpoint not set correctly! Please update API_ENDPOINT_URL.")


# --- Main Area for Results (Display Logic) ---
if st.session_state.results is None:
    st.info("Please fill out the form on the left and click 'Assess Creditworthiness' to view the analysis.")
else:
    results = st.session_state.results
    user_inputs = results['user_inputs']
    
    tab_list = [
        "User Profile", "Decision & Explainability", "Composite Score", 
        "Repayment Behavior", "Income Estimation", "Financial Health", 
        "Download/Share", "Loan Simulator"
    ]
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(tab_list)

    with tab1:
        st.subheader(f"Profile for: {user_inputs['name']}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Loan Details**")
            st.markdown(f"- **Amount Requested:** ₹{user_inputs['LoanAmount']:,.0f}")
            st.markdown(f"- **Purpose:** {user_inputs['LoanPurpose']}")
        with col2:
            st.markdown("**Demographics**")
            st.markdown(f"- **Age:** {user_inputs['Age']}")
            st.markdown(f"- **Sector:** {user_inputs['Sector']}")
            st.markdown(f"- **Household Size:** {user_inputs['household_size_calculated']}")
        st.markdown("---")
        st.markdown("**Socio-Economic & Lifestyle**")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown(f"- **Primary Income:** {user_inputs['Max_Income_Activity']}")
            st.markdown(f"- **Dwelling Type:** {user_inputs['Type_of_Dwelling_Unit']}")
            st.markdown(f"- **Owns a Car:** {'Yes' if user_inputs['Asset_Car'] else 'No'}")
        with col4:
            st.markdown(f"- **Social Group:** {user_inputs['Social_Group_of_HH_Head']}")
            st.markdown(f"- **Ration Card Type:** {user_inputs['Ration_Card_Type']}")
            st.markdown(f"- **Has Subsidy/Ration:** {user_inputs['Scheme_Proxy']}")

    with tab2:
        st.subheader("Decision & Explainability")
        
        if "Low Risk" in results['risk_band_full']:
            decision_text = "Auto-Approve (Direct Digital Lending)"
            st.success(f"## **{decision_text}**")
        else:
            decision_text = "Manual Review / Reject"
            st.error(f"## **{decision_text}**")
            
        st.subheader("Key Decision Factors")
        st.markdown(f"- **Final Risk Band:** {results['risk_band_full']}")
        st.markdown(f"- **Composite Score:** {results['composite_score']:.2%}")
        st.markdown(f"- **Willingness to Pay (Repayment Score):** {results['repayment_score']:.2%}.")
        st.markdown(f"- **Ability to Pay (Income Proxy Score):** {results['income_score']:.2%}.")
        st.markdown(f"- **Debt-to-Income:** Healthy DTI of {results['dti']:.0%}.")


    with tab3:
        st.subheader("Composite Score & Risk Band")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Composite Score", f"{results['composite_score']:.2%}")
            st.markdown(f"**Risk Band:** {results['risk_band_full']}")
        with col2:
            st.plotly_chart(create_quadrant_chart(results['repayment_score'], results['income_score']), use_container_width=True)

    with tab4:
        st.subheader("Repayment Behavior Analysis (Model A)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Repayment Score", f"{results['repayment_score']:.2%}")
        col2.metric("Periodic Credit Info (PCI)", f"{results['pci']}")
        col3.metric("Wallet Debit Score (WDS)", f"{results['wds']:.2%}")
        st.plotly_chart(create_gauge_chart(results['repayment_score'], "Repayment Score"), use_container_width=True)

    with tab5:
        st.subheader("Income Estimation (Model B)")
        st.info(results['verification_message'])
        st.bar_chart({"Category": ["Claimed", "Verified/Used"], "Income": [user_inputs['Income'], results['net_monthly_income']]})

    with tab6:
        st.subheader("Financial Health")
        col1, col2, col3 = st.columns(3)
        col1.metric("Existing Debt (Est.)", f"₹{results['existing_debt']:,.0f}")
        col2.metric("Existing Monthly EMI", f"₹{results['existing_emi']:,.0f}")
        col3.metric("Debt-to-Income (DTI) Ratio", f"{results['dti']:.0%}", "🔴 High" if results['dti'] > 0.5 else "🟢 Low")
    
    with tab7:
        st.subheader("Generate & Share Report")
        if st.button("Download Loan Report (PDF)"):
            st.info("Feature coming soon! This would generate a PDF summary.")
        
        st.markdown(f'<a href="mailto:?subject=Loan Application for {user_inputs["name"]}&body=Final Decision: {decision_text}. Credit Score: {results["composite_score"]:.2%}">Share via Email</a>', unsafe_allow_html=True)
        
    with tab8:
        st.subheader("Loan Simulator")
        st.info("The Loan Simulator demonstrates how adjusting variables impacts the score. This requires a separate API call.")
        
        # --- Loan Simulator Logic ---
        # This sends a separate payload to the LIVE API with adjusted values
        with st.form(key='simulator_form'):
            sim_col1, sim_col2 = st.columns(2)
            
            with sim_col1:
                st.markdown("#### Adjust Your Financial Profile")
                sim_income = st.slider("Net Monthly Income", 5000, 500000, int(results['net_monthly_income']), 1000)
                sim_emi = st.slider("Total Monthly EMIs", 0, 200000, user_inputs['existing_emi'], 500)
                sim_missed_pmnt = st.slider("Simulated Credit Score Impact", 0.0, 1.0, results['repayment_score'], 0.01)
                
                simulate_button = st.form_submit_button(label="Run Simulation")

            if simulate_button:
                # Create a new payload based on the original but with simulated changes
                sim_payload = api_payload.copy() # Use the last successful payload as base
                
                # Apply simulated changes
                sim_payload['Income'] = sim_income # Direct income adjustment
                sim_payload['DTIRatio'] = sim_emi / sim_income if sim_income > 0 else 1 # Recalculated DTI
                # Note: Repayment Score must be simulated directly for this demo's logic simplification
                
                try:
                    sim_response = requests.post(f"{API_ENDPOINT_URL}/score", json=sim_payload)
                    sim_response.raise_for_status()
                    sim_api_results = sim_response.json()

                    sim_dti = sim_emi / sim_income if sim_income > 0 else 1
                    sim_composite_score = sim_api_results['composite_score']
                    
                    # Calculate change metrics
                    score_change = sim_composite_score - results['composite_score']
                    dti_change = sim_dti - results['dti']
                    
                    if "Low Risk" in sim_api_results['risk_band']:
                        sim_decision = "Auto-Approve"
                    else:
                        sim_decision = "Manual Review"

                    with sim_col2:
                        st.markdown("#### Simulated Outcome")
                        st.metric("Simulated Credit Score", f"{sim_composite_score:.2%}", f"{score_change:+.2%}")
                        st.metric("Simulated DTI Ratio", f"{sim_dti:.0%}", f"{dti_change:+.0%}")
                        
                        if "Approve" in sim_decision:
                            st.success(f"**Simulated Decision: {sim_decision}**")
                        else:
                            st.error(f"**Simulated Decision: {sim_decision}**")
                
                except requests.exceptions.RequestException as e:
                    st.error(f"Simulation Error: Could not connect to API. Details: {e}")
                    
            else:
                 # Default state for simulated outcome column
                with sim_col2:
                    st.markdown("#### Simulated Outcome")
                    st.metric("Simulated Credit Score", f"{results['composite_score']:.2%}")
                    st.metric("Simulated DTI Ratio", f"{results['dti']:.0%}")
                    if "Low Risk" in results['risk_band_full']:
                        st.success(f"**Simulated Decision: Auto-Approve**")
                    else:
                        st.error(f"**Simulated Decision: Manual Review**")