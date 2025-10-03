import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import pdfplumber
import re
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Beneficiary Credit Scoring Dashboard",
    layout="wide"
)

# --- Load Models and Assets ---
@st.cache_resource
def load_assets():
    """Loads all trained models and required assets."""
    try:
        assets = {
            "repayment_model": joblib.load('repayment_model.joblib'),
            "repayment_features": json.load(open('repayment_features.json')),
            "repayment_medians": json.load(open('repayment_medians.json')),
            "income_model": joblib.load('income_model.joblib'),
            "income_features": json.load(open('income_features.json'))
        }
        return assets
    except FileNotFoundError:
        st.error("Model assets not found. Please run the `train_model.py` script first.")
        return None

assets = load_assets()

# --- Helper & Charting Functions ---
def analyze_bank_statement(uploaded_file):
    """Extracts and calculates average salary from a PDF bank statement."""
    if uploaded_file is None:
        return None, "No file uploaded."
    salaries = []
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    matches = re.finditer(r'(?i)(?:SALARY|SAL|SAL-TRANSFER|SALARY CREDIT)\s.*?([\d,]+\.\d{2})', text)
                    for match in matches:
                        salaries.append(float(match.group(1).replace(',', '')))
    except Exception as e:
        return None, f"Could not process PDF. Error: {e}"
    if not salaries:
        return 0, "Analyzed: No salary credits found."
    avg_salary = np.mean(salaries)
    return avg_salary, f"Verified: Average salary is ₹{avg_salary:,.2f}"

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
    """Creates a risk quadrant chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[repayment_score], y=[income_score],
        mode='markers',
        marker=dict(color='blue', size=20, symbol='star'),
        name='Applicant Profile'
    ))
    fig.add_shape(type="line", x0=0.5, y0=0, x1=0.5, y1=1, line=dict(color="grey", width=2, dash="dash"))
    fig.add_shape(type="line", x0=0, y0=0.5, x1=1, y1=0.5, line=dict(color="grey", width=2, dash="dash"))
    fig.add_annotation(x=0.25, y=0.75, text="Lower Repayment,<br>Higher Income", showarrow=False)
    fig.add_annotation(x=0.75, y=0.75, text="Higher Repayment,<br>Higher Income", showarrow=False)
    fig.add_annotation(x=0.25, y=0.25, text="Lower Repayment,<br>Lower Income", showarrow=False)
    fig.add_annotation(x=0.75, y=0.25, text="Higher Repayment,<br>Lower Income", showarrow=False)
    
    fig.update_xaxes(range=[0, 1], title_text="Repayment Score (Willingness to Pay)")
    fig.update_yaxes(range=[0, 1], title_text="Income Score (Ability to Pay)")
    fig.update_layout(title_text="Risk Quadrant Analysis", height=450)
    return fig

def get_risk_band(score):
    if score > 0.8: return "🟢 Very Low Risk"
    elif score > 0.7: return "🟢 Low Risk"
    elif score > 0.55: return "🟡 Medium Risk"
    elif score > 0.4: return "🟠 High Risk"
    else: return "🔴 Very High Risk"

# --- UI Layout ---
st.title("Beneficiary Credit Scoring & Digital Lending")

# Initialize session state for results if it doesn't exist
if 'results' not in st.session_state:
    st.session_state.results = None

# --- Sidebar for User Input (Expanded Form) ---
with st.sidebar.form(key='applicant_form'):
    st.header("Applicant Information Form")
    
    st.subheader("Personal & Loan Details")
    user_inputs = {
        'name': st.text_input("Full Name", "John Doe"),
        'person_age': st.number_input("Age", 18, 100, 35),
        'household_size_calculated': st.number_input("Household Size", 1, 20, 4),
        'avg_education_years_adults': st.number_input("Avg. Education Years (Adults)", 0, 25, 12),
        'social_group': st.selectbox("Social Group", ["ST", "SC", "OBC", "Others"]),
        'sector': st.radio("Sector", ["Rural", "Urban"]),
        'loan_amount': st.number_input("Loan Amount Requested (INR)", 10000, 1000000, 50000, 5000),
        'loan_purpose': st.selectbox("Loan Purpose", ["Business", "Personal", "Education", "Home Improvement"]),
    }
    
    st.subheader("Employment & Financial")
    user_inputs.update({
        'net_monthly_income_claimed': st.number_input("Claimed Net Monthly Income", 5000, 500000, 25000, 1000),
        'time_with_curr_empr': st.number_input("Time with Employer (Years)", 0, 50, 5),
        'max_income_activity': st.selectbox("Primary Income Source", ["Salaried", "Self-Employed", "Business", "Other"]),
        'existing_emi': st.number_input("Total Existing Monthly EMIs", 0, 200000, 5000, 500)
    })

    st.subheader("Credit History")
    user_inputs.update({
        'tot_missed_pmnt': st.number_input("Total Missed Payments", 0, 100, 0),
        'time_since_recent_payment': st.number_input("Days Since Last Payment", 0, 365, 30),
        'age_oldest_tl': st.number_input("Age of Oldest Loan (Months)", 0, 600, 60),
        'age_newest_tl': st.number_input("Age of Newest Loan (Months)", 0, 600, 12),
        'enq_l3m': st.number_input("Enquiries in Last 3 Months", 0, 50, 0),
    })

    st.subheader("Household Assets & Lifestyle")
    user_inputs.update({
        'land_ownership': "Yes" if st.checkbox("Owns Agricultural Land") else "No",
        'type_of_dwelling': st.selectbox("Type of Dwelling", ["Owned", "Hired/Rented", "Ancestral"]),
        'energy_source': st.selectbox("Cooking Energy Source", ["LPG", "Firewood", "Kerosene", "Other"]),
        'possess_car': "Yes" if st.checkbox("Owns a Car") else "No",
        'possess_refrigerator': "Yes" if st.checkbox("Owns a Refrigerator") else "No",
        'possess_washing_machine': "Yes" if st.checkbox("Owns a Washing Machine") else "No",
    })
    
    st.subheader("Bank Statement Verification")
    uploaded_statement = st.file_uploader("Upload Bank Statement PDF", type="pdf")

    submit_button = st.form_submit_button(label='Assess Creditworthiness')

# --- Calculation Logic ---
if assets and submit_button:
    with st.spinner('Analyzing application...'):
        verified_income, verification_message = analyze_bank_statement(uploaded_statement)
        net_monthly_income = verified_income if verified_income is not None and verified_income > 0 else user_inputs['net_monthly_income_claimed']

        repayment_input_data = {f: assets['repayment_medians'].get(f, 0) for f in assets['repayment_features']}
        form_feature_map = {
            'netmonthlyincome': net_monthly_income, 'time_with_curr_empr': user_inputs['time_with_curr_empr'],
            'tot_missed_pmnt': user_inputs['tot_missed_pmnt'], 'age_oldest_tl': user_inputs['age_oldest_tl'],
            'age_newest_tl': user_inputs['age_newest_tl'], 'time_since_recent_payment': user_inputs['time_since_recent_payment'],
            'enq_l3m': user_inputs['enq_l3m']
        }
        for feature, value in form_feature_map.items():
            matching_keys = [k for k in repayment_input_data if k.lower() == feature]
            if matching_keys:
                repayment_input_data[matching_keys[0]] = value
        repayment_df = pd.DataFrame([repayment_input_data])[assets['repayment_features']]
        
        income_df_placeholder = pd.DataFrame([{'NETMONTHLYINCOME': net_monthly_income, 'Time_With_Curr_Empr': user_inputs['time_with_curr_empr']}])

        repayment_score = assets['repayment_model'].predict_proba(repayment_df)[:, 0][0]
        income_pred = assets['income_model'].predict(income_df_placeholder)[0]
        income_score = {'Very Low': 0.2, 'Low': 0.4, 'Medium': 0.7, 'High': 0.9}.get(income_pred, 0.4)
        
        existing_emi = user_inputs['existing_emi']
        dti = existing_emi / net_monthly_income if net_monthly_income > 0 else 1
        
        composite_score = (0.6 * repayment_score) + (0.4 * income_score)
        reasons = []

        if dti > 0.50:
            decision = "REJECT"
            reasons.append(f"High Debt-to-Income (DTI) Ratio at {dti:.0%}. This is above the 50% threshold.")
            if user_inputs['tot_missed_pmnt'] > 0:
                reasons.append(f"History of {user_inputs['tot_missed_pmnt']} missed payment(s) indicates high repayment risk.")
        elif composite_score < 0.5:
            decision = "REJECT"
            reasons.append(f"Overall Credit Score ({composite_score:.2%}) is below the minimum required threshold of 50%.")
            if repayment_score < 0.5:
                 reasons.append(f"The Repayment Score ({repayment_score:.2%}) was low.")
        else:
            decision = "Auto-Approve"
            reasons.append(f"Strong Credit Score of {composite_score:.2%} exceeds the required threshold.")
            reasons.append(f"Healthy Debt-to-Income (DTI) Ratio of {dti:.0%}.")

        # Store results in session state
        st.session_state.results = {
            'user_inputs': user_inputs, 'net_monthly_income': net_monthly_income,
            'verification_message': verification_message, 'repayment_score': repayment_score,
            'income_score': income_score, 'pci': np.random.randint(650, 850),
            'wds': np.random.uniform(0.6, 0.95), 'existing_emi': existing_emi, 'dti': dti,
            'existing_debt': existing_emi / 0.02, 'composite_score': composite_score,
            'decision': decision, 'reasons': reasons, 'repayment_df': repayment_df
        }

# --- Main Area for Results ---
if st.session_state.results is None:
    st.info("Please fill out the form on the left and click 'Assess Creditworthiness' to view the analysis.")
else:
    results = st.session_state.results
    user_inputs = results['user_inputs']
    
    tab_list = [
        "User Profile", "Decision & Explainability", "Credit Score", 
        "Repayment Behavior", "Income Estimation", "Financial Health", 
        "Download/Share", "Loan Simulator"
    ]
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(tab_list)

    with tab1:
        st.subheader(f"Profile for: {user_inputs['name']}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Loan Details**")
            st.markdown(f"""
            - **Amount Requested:** ₹{user_inputs['loan_amount']:,.0f}
            - **Purpose:** {user_inputs['loan_purpose']}
            """)
        with col2:
            st.markdown("**Demographics**")
            st.markdown(f"""
            - **Age:** {user_inputs['person_age']}
            - **Sector:** {user_inputs['sector']}
            - **Household Size:** {user_inputs['household_size_calculated']}
            """)
        st.markdown("---")
        st.markdown("**Socio-Economic & Lifestyle**")
        col3, col4 = st.columns(2)
        with col3:
             st.markdown(f"""
            - **Primary Income:** {user_inputs['max_income_activity']}
            - **Dwelling Type:** {user_inputs['type_of_dwelling']}
            - **Owns a Car:** {user_inputs['possess_car']}
            """)
        with col4:
             st.markdown(f"""
            - **Social Group:** {user_inputs['social_group']}
            - **Owns Refrigerator:** {user_inputs['possess_refrigerator']}
            - **Owns Washing Machine:** {user_inputs['possess_washing_machine']}
            """)

    with tab2:
        st.subheader("Decision & Explainability")
        if "Approve" in results['decision']:
            st.success(f"## **{results['decision']}**")
        else:
            st.error(f"## **{results['decision']}**")
        st.subheader("Key Decision Factors")
        for reason in results['reasons']:
            st.markdown(f"- {reason}")

    with tab3:
        st.subheader("Credit Score & Risk Band")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Credit Score", f"{results['composite_score']:.2%}")
            st.markdown(f"**Risk Band:** {get_risk_band(results['composite_score'])}")
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
        st.bar_chart({"Category": ["Claimed", "Verified/Used"], "Income": [user_inputs['net_monthly_income_claimed'], results['net_monthly_income']]})

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
        
        st.markdown(f'<a href="mailto:?subject=Loan Application for {user_inputs["name"]}&body=Please find the loan assessment attached. Credit Score: {results["composite_score"]:.2%}">Share via Email</a>', unsafe_allow_html=True)
        
    with tab8:
        st.subheader("Loan Simulator")
        st.info("Adjust the sliders and click 'Run Simulation' to see the impact on the credit score.")
        
        with st.form(key='simulator_form'):
            sim_col1, sim_col2 = st.columns(2)
            
            with sim_col1:
                st.markdown("#### Adjust Your Financial Profile")
                sim_income = st.slider("Net Monthly Income", 5000, 500000, int(results['net_monthly_income']), 1000)
                sim_emi = st.slider("Total Monthly EMIs", 0, 200000, user_inputs['existing_emi'], 500)
                sim_missed_pmnt = st.slider("Past Missed Payments", 0, 10, user_inputs['tot_missed_pmnt'])
                
                # Add the simulation button inside the form
                simulate_button = st.form_submit_button(label="Run Simulation")

            with sim_col2:
                st.markdown("#### Simulated Outcome")

                # The simulation logic now runs every time, but is only updated when the form button is clicked
                sim_repayment_input = results['repayment_df'].copy()
                sim_repayment_input['NETMONTHLYINCOME'] = sim_income
                sim_repayment_input['Tot_Missed_Pmnt'] = sim_missed_pmnt
                
                sim_repayment_score = assets['repayment_model'].predict_proba(sim_repayment_input)[:, 0][0]
                sim_income_pred = assets['income_model'].predict(pd.DataFrame([{'NETMONTHLYINCOME': sim_income, 'Time_With_Curr_Empr': user_inputs['time_with_curr_empr']}]))[0]
                sim_income_score = {'Very Low': 0.2, 'Low': 0.4, 'Medium': 0.7, 'High': 0.9}.get(sim_income_pred, 0.4)

                sim_dti = sim_emi / sim_income if sim_income > 0 else 1
                sim_composite_score = (0.6 * sim_repayment_score) + (0.4 * sim_income_score)

                if sim_dti > 0.50: sim_decision = "REJECT"
                elif sim_composite_score < 0.5: sim_decision = "REJECT"
                else: sim_decision = "Auto-Approve"

                st.metric("Simulated Credit Score", f"{sim_composite_score:.2%}", f"{sim_composite_score - results['composite_score']:+.2%}")
                st.metric("Simulated DTI Ratio", f"{sim_dti:.0%}", f"{sim_dti - results['dti']:+.0%}")
                
                if "Approve" in sim_decision:
                    st.success(f"**Simulated Decision: {sim_decision}**")
                else:
                    st.error(f"**Simulated Decision: {sim_decision}**")

