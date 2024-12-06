import streamlit as st
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter as KMFitter

# Override KaplanMeiterFitter
class KaplanMeierFitter(KMFitter):
    def __init__(self, *args, **kwargs):
        super(KaplanMeierFitter, self).__init__(*args, **kwargs)

    def cumulative_hazard_at_times(self, times, label=None):
        surv_prob = self.survival_function_at_times(times)
        surv_prob = np.clip(surv_prob, a_min=1e-6, a_max=1)
        return -np.log(surv_prob)

    def hazard_at_times(self, times, label=None):
        ref = np.array(self.event_table.index)
        nearest = [len(ref)-1 if ref[-1] < t else np.where(ref <= t)[0].max() for t in times]
        return pd.Series(
            self.event_table.iloc[nearest]['observed'].values / self.event_table.iloc[nearest]['at_risk'].values,
            index=times)

class BasicNetwork(nn.Module):
    def __init__(self, layers, activation='ReLU', dropout=0.25, batchnorm=False, bias=True):
        super(BasicNetwork, self).__init__()
        activation = getattr(nn, activation) if activation else None

        _layers = []
        for d, (units_in, units_out) in enumerate(zip(layers, layers[1:])):
            _layers.append(nn.Linear(units_in, units_out, bias=bias))
            if d < len(layers) - 2:
                if batchnorm:
                    _layers.append(nn.BatchNorm1d(units_out))
                if activation:
                    _layers.append(activation())
                if dropout:
                    _layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*_layers)

    def forward(self, x):
        return self.layers(x)

layers = [29, 16, 16, 1]
with torch.inference_mode():
    net = BasicNetwork(layers, batchnorm=True, dropout=0.)
    net.load_state_dict(torch.load('fair_deep_cox.pth'))
    net.eval()

baseline_data = pd.read_csv('train_time_event.csv', index_col=0)
baseline_hazard = KaplanMeierFitter()
baseline_hazard.fit(baseline_data['time'], baseline_data['event'].astype(bool))

st.title("Prostate Cancer Risk Calculator")

st.markdown("---")
st.markdown("### Basic Inputs")
#year_of_dx = st.number_input("Year of Diagnosis", value=None, min_value=1950, max_value=2024, step=1)
year_of_dx = st.selectbox("Year of Diagnosis", range(1950, 2026))
#age = st.number_input("Age", value=0, min_value=0, max_value=100, step=1)
age = st.selectbox("Age", range(0, 101))
race_eth = st.radio("Race/Ethnicity", ("Asian", "Hispnaic", "Non-hispanic Black", "Non-hispanic White"))
insurance = st.radio("Insurance Status", ("Not Insured", "Private Insurance/Managed Care", "Medicaid", "Medicare", "Other Government Insurance", "Unknown"))
income = st.radio("Household Income", ("below $30,000", " \$30,000 - \$34,999", " \$35,000 - \$45,999", "over $46,000", "N/A"))

st.markdown("---")
st.markdown("### Clinical Conditions")
facility = st.radio("Facility", ("Community Cancer Program", "Comprehensive Community Cancer Program", "Academic/Research Program (including NCI-designated Comprehensive Cancer Centers)", "Integrated Network Cancer Program", "N/A"))
clinical_t_stage = st.radio("Clinical T-Stage", ("T1", "T2", "T3", "T4"))
psa_combo = st.number_input("PSA Combo Score (needs to be updated)", value=0.0, min_value=0.0, step=0.1)
grade_clin_combo = st.radio("Grade Clinical Combo Score (needs to be updated)", ("1", "2", "3", "4", "5"))

st.markdown("---")
st.markdown("### Comorbidities")

comorbid_scores = {
    'Myocardial Infraction': 1,
    'Congestive Heart Failure': 1,
    'Peripheral Vascular Disease': 1,
    'Cerebrovascular Disease': 1,
    'Dementia': 1,
    'Chronic Pulmonary Disease': 1,
    'Rheumatologic Disease': 1,
    'Peptic Ulcer Disease': 1,
    'Mild Liver Disease': 1,
    'Diabetes': 1,
    'Disabetes with Chronic Complications': 2,
    'Hemiplegia or Paraplegia': 2,
    'Renal Disease': 2,
    'Moderate or Severe Liver Disease': 3,
    'AIDS': 6
}


comorbid = dict()
for c in comorbid_scores:
    comorbid[c] = st.checkbox(c, value=False)


race_eth_one_hot = {
    'Asian': 0,
    'Non-hispanic Black': 0,
    'Hispanic': 0,
    'Non-hispanic White': 0
}


insurance_one_hot = {
    'Not Insured': 0,
    'Private Insurance/Managed Care': 0,
    'Medicaid': 0,
    'Medicare': 0,
    'Other Government Insurance': 0,
    'Unknown': 0
}

facility_one_hot = {
    'Community Cancer Program': 0,
    'Comprehensive Community Cancer Program': 0,
    'Academic/Research Program (including NCI-designated Comprehensive Cancer Centers)': 0,
    'Integrated Network Cancer Program': 0,
    'N/A': 0
}

Tstage_one_hot = {
    'T1': 0,
    'T2': 0,
    'T3': 0,
    'T4': 0
}

income_one_hot = {
    'below $30,000': 0,
    '$30,000 - $34,999': 0,
    '$35,000 - $45,999': 0,
    'over $46,000': 0,
    'N/A': 0
}

grade_clinical_combo_one_hot = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
}

st.markdown("---")

#col1, col2, col3 = st.columns([1, 1, 1])  # Adjust ratios for spacing
#with col2:
if st.button("Predict", use_container_width=True):

    race_eth_one_hot[race_eth] = 1.
    insurance_one_hot[insurance] = 1.
    facility_one_hot[facility] = 1.
    Tstage_one_hot[clinical_t_stage] = 1.
    income_one_hot[income] = 1.
    cdcc_score = np.clip(sum([comorbid_scores[c] for c in comorbid_scores if comorbid[c]]), 0, 3)
    cdcc_one_hot = {
        0: 0,
        1: 0,
        2: 0,
        3: 0
    }

    cdcc_one_hot[cdcc_score] = 1.
    grade_clinical_combo_one_hot[grade_clin_combo] = 1.

    input_x = np.concatenate([
        [year_of_dx],
        pd.DataFrame(race_eth_one_hot, index=[0]).values.flatten()[1:],
        [age],
        pd.DataFrame(insurance_one_hot, index=[0]).values.flatten()[1:],
        pd.DataFrame(facility_one_hot, index=[0]).values.flatten()[1:-1],
        pd.DataFrame(income_one_hot, index=[0]).values.flatten()[1:],
        pd.DataFrame(cdcc_one_hot, index=[0]).values.flatten()[1:],
        [psa_combo],
        pd.DataFrame(Tstage_one_hot, index=[0]).values.flatten()[1:],
        pd.DataFrame(grade_clinical_combo_one_hot, index=[0]).values.flatten()[1:]
    ], axis=0).reshape(1, -1)

    input_x = torch.from_numpy(input_x).float()
    with torch.inference_mode():
        prop_hazard = torch.exp(net(input_x))

    times = np.arange(5, 185, 5)
    cumul_hazards = prop_hazard.numpy() * baseline_hazard.cumulative_hazard_at_times(times).values.reshape(-1, len(times))
    surv = np.exp(-cumul_hazards).flatten()

    res = pd.DataFrame([times, surv], index=['time', 'surv_prob']).T.set_index('time')

    fig, ax = plt.subplots()
    ax.plot(times, surv)
    ax.set_title('Survival Probabilities')
    ax.set_xlabel('Months')
    ax.set_ylabel('Probability')

    x_5y = 60
    y_5y = res.loc[60]["surv_prob"]
    x_10y = 120
    y_10y = res.loc[120]["surv_prob"]

    ax.scatter([x_5y], [y_5y], color="green", label="5-Year")
    ax.plot([x_5y, x_5y], [0, y_5y], color="green", linestyle="--")
    ax.plot([0, x_5y], [y_5y, y_5y], color="green", linestyle="--")

    ax.scatter([x_10y], [y_10y], color="red", label="10-Year")
    ax.plot([x_10y, x_10y], [0, y_10y], color="red", linestyle="--")
    ax.plot([0, x_10y], [y_10y, y_10y], color="red", linestyle="--")


    st.pyplot(fig)
    st.markdown(f'#### 5-Year Survival Probability: {res.loc[60]["surv_prob"]:.4f}')
    st.markdown(f'#### 10-Year Survival Probability: {res.loc[120]["surv_prob"]:.4f}')
    st.markdown("---")

