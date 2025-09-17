import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Advance - Fraud Detection",
    page_icon="<---->",
    layout="centered"
)

@st.cache_resource
def load_model():
    try:
        model = joblib.load('XGBoost.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_features(type_, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest):
    balance_diff_org = oldbalanceOrg - newbalanceOrig
    balance_diff_dest = newbalanceDest - oldbalanceDest
    
    type_CASH_IN = 1 if type_ == "CASH_IN" else 0
    type_CASH_OUT = 1 if type_ == "CASH_OUT" else 0
    type_DEBIT = 1 if type_ == "DEBIT" else 0
    type_PAYMENT = 1 if type_ == "PAYMENT" else 0
    type_TRANSFER = 1 if type_ == "TRANSFER" else 0
    
    features = pd.DataFrame({
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest],
        'balancedifforg': [balance_diff_org],
        'balancediffdest': [balance_diff_dest],
        'type_CASH_IN': [type_CASH_IN],
        'type_CASH_OUT': [type_CASH_OUT],
        'type_DEBIT': [type_DEBIT],
        'type_PAYMENT': [type_PAYMENT],
        'type_TRANSFER': [type_TRANSFER]
    })
    
    return features

def calculate_fraud_indicators(type_, amount_val, oldbalanceOrg_val, newbalanceOrig_val, oldbalanceDest_val, newbalanceDest_val):
    balance_diff_org = round(oldbalanceOrg_val - newbalanceOrig_val, 2)
    balance_diff_dest = round(newbalanceDest_val - oldbalanceDest_val, 2)
    
    EPSILON = 0.01
    
    legitimate_transfer = (type_ == "TRANSFER" and 
                         abs(balance_diff_org - amount_val) <= EPSILON and
                         abs(balance_diff_dest - amount_val) <= EPSILON)
    
    legitimate_cashout = (type_ == "CASH_OUT" and 
                         abs(balance_diff_org - amount_val) <= EPSILON and
                         abs(oldbalanceDest_val) <= EPSILON and
                         abs(newbalanceDest_val) <= EPSILON and
                         oldbalanceOrg_val >= amount_val * 1.2 and
                         amount_val <= 10000)
    
    legitimate_large_cashout = (type_ == "CASH_OUT" and
                              abs(balance_diff_org - amount_val) <= EPSILON and
                              abs(oldbalanceDest_val) <= EPSILON and
                              abs(newbalanceDest_val) <= EPSILON and
                              oldbalanceOrg_val >= amount_val * 1.2 and
                              newbalanceOrig_val >= amount_val * 0.2 and
                              amount_val >= 50000)
    
    transfer_fraud_static_receiver = (type_ == "TRANSFER" and 
                                    abs(balance_diff_org - amount_val) <= EPSILON and
                                    abs(balance_diff_dest) <= EPSILON and
                                    oldbalanceDest_val > EPSILON)
    
    transfer_fraud_full_balance = (type_ == "TRANSFER" and 
                                 abs(oldbalanceOrg_val - amount_val) <= EPSILON and
                                 abs(newbalanceOrig_val) <= EPSILON and
                                 abs(oldbalanceDest_val - newbalanceDest_val) <= EPSILON)
    
    cashout_fraud_static_receiver = (type_ == "CASH_OUT" and 
                                   abs(balance_diff_org - amount_val) <= EPSILON and
                                   oldbalanceDest_val > EPSILON and
                                   abs(oldbalanceDest_val - newbalanceDest_val) <= EPSILON)
    
    cashout_fraud_suspicious = (type_ == "CASH_OUT" and
                              amount_val >= 10000 and
                              oldbalanceOrg_val > 0 and
                              ((amount_val / oldbalanceOrg_val) >= 0.9 or
                               abs(newbalanceOrig_val) <= EPSILON) and
                              not legitimate_large_cashout)
    
    cashout_fraud_full_withdrawal = (type_ == "CASH_OUT" and
                                   abs(oldbalanceOrg_val - amount_val) <= EPSILON and
                                   abs(newbalanceOrig_val) <= EPSILON and
                                   amount_val >= 10000 and
                                   not legitimate_large_cashout)
    
    large_amount = amount_val >= 50000
    very_large_amount = amount_val >= 80000
    
    full_balance_depletion = (abs(oldbalanceOrg_val - amount_val) <= EPSILON and
                             abs(newbalanceOrig_val) <= EPSILON and
                             not legitimate_large_cashout)
    
    negative_balance = (newbalanceOrig_val < -EPSILON or newbalanceDest_val < -EPSILON)
    
    amount_mismatch = (abs(balance_diff_org - amount_val) > EPSILON or 
                      (type_ == "TRANSFER" and abs(balance_diff_dest - amount_val) > EPSILON))
    
    return {
        'legitimate_transfer': legitimate_transfer,
        'legitimate_cashout': legitimate_cashout,
        'legitimate_large_cashout': legitimate_large_cashout,
        'transfer_fraud_static_receiver': transfer_fraud_static_receiver,
        'transfer_fraud_full_balance': transfer_fraud_full_balance,
        'cashout_fraud_static_receiver': cashout_fraud_static_receiver,
        'cashout_fraud_suspicious': cashout_fraud_suspicious,
        'cashout_fraud_full_withdrawal': cashout_fraud_full_withdrawal,
        'large_amount': large_amount,
        'very_large_amount': very_large_amount,
        'full_balance_depletion': full_balance_depletion,
        'negative_balance': negative_balance,
        'amount_mismatch': amount_mismatch,
        'balance_diff_org': balance_diff_org,
        'balance_diff_dest': balance_diff_dest
    }

def is_fraud(indicators, type_, amount_val, oldbalanceOrg_val, newbalanceOrig_val, oldbalanceDest_val, newbalanceDest_val):
    if (indicators['legitimate_transfer'] or 
        indicators['legitimate_cashout'] or
        indicators['legitimate_large_cashout']):
        return False
    
    fraud_score = 0
    
    if type_ == "TRANSFER":
        if indicators['transfer_fraud_static_receiver']:
            fraud_score += 5
        if indicators['transfer_fraud_full_balance']:
            fraud_score += 4
    
    if type_ == "CASH_OUT":
        if indicators['cashout_fraud_static_receiver']:
            fraud_score += 5
        if indicators['cashout_fraud_full_withdrawal']:
            fraud_score += 5
        if indicators['cashout_fraud_suspicious']:
            fraud_score += 4
    
    if indicators['amount_mismatch']:
        fraud_score += 3
    
    if indicators['negative_balance']:
        fraud_score += 3
    
    if not indicators['legitimate_large_cashout']:
        if indicators['very_large_amount']:
            fraud_score += 2
        elif indicators['large_amount']:
            fraud_score += 1
    
        if indicators['full_balance_depletion'] and amount_val >= 10000:
            fraud_score += 3
    
    return fraud_score >= 3

model = load_model()

st.title("Advance fraud detector")
st.subheader("Transaction Fraud Detection System")
st.markdown("---")

col_left, col_right = st.columns([1.2, 1], gap="large")

with col_left:
    st.markdown("### Transaction Details")
    
    type_ = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"])
    
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        amount = st.text_input("Amount (Rs)", "1,250.00")
    with col1_2:
        oldbalanceOrg = st.text_input("Sender Old Balance", "8,500.00")
    
    col2_1, col2_2 = st.columns(2)
    with col2_1:
        newbalanceOrig = st.text_input("Sender New Balance", "7,250.00")
    with col2_2:
        oldbalanceDest = st.text_input("Receiver Old Balance", "3,200.00")
    
    newbalanceDest = st.text_input("Receiver New Balance", "4,450.00")
    
    predict_btn = st.button("🔍 Analyze Transaction", use_container_width=True, key="predict")

with col_right:
    st.markdown("### Information")
    st.info("Enter transaction details in the form and click 'Analyze Transaction' to detect fraud.")

st.markdown("### Prediction Result")

if predict_btn:
    try:
        amount_val = float(amount.replace(',', ''))
        oldbalanceOrg_val = float(oldbalanceOrg.replace(',', ''))
        newbalanceOrig_val = float(newbalanceOrig.replace(',', ''))
        oldbalanceDest_val = float(oldbalanceDest.replace(',', ''))
        newbalanceDest_val = float(newbalanceDest.replace(',', ''))
        
        if amount_val < 0:
            st.error("Amount cannot be negative")
            st.stop()
        
        features = preprocess_features(
            type_, amount_val, oldbalanceOrg_val, newbalanceOrig_val, 
            oldbalanceDest_val, newbalanceDest_val
        )
        
        indicators = calculate_fraud_indicators(
            type_, amount_val, oldbalanceOrg_val, newbalanceOrig_val, 
            oldbalanceDest_val, newbalanceDest_val
        )
        
        if model is not None:
            if hasattr(model, 'predict_proba'):
                try:
                    model_prob = model.predict_proba(features)[0][1]
                    
                    is_fraud_result = is_fraud(indicators, type_, amount_val, 
                                             oldbalanceOrg_val, newbalanceOrig_val, 
                                             oldbalanceDest_val, newbalanceDest_val)
                    
                    if indicators['legitimate_transfer'] or indicators['legitimate_cashout'] or indicators['legitimate_large_cashout']:
                        final_result = False
                    else:
                        if model_prob > 0.5 and is_fraud_result:
                            final_result = True
                        elif model_prob < 0.3 and not is_fraud_result:
                            final_result = False
                        else:
                            final_result = is_fraud_result
                        
                except Exception as pred_error:
                    final_result = is_fraud(indicators, type_, amount_val, 
                                          oldbalanceOrg_val, newbalanceOrig_val, 
                                          oldbalanceDest_val, newbalanceDest_val)
            else:
                model_pred = model.predict(features)[0]
                
                is_fraud_result = is_fraud(indicators, type_, amount_val, 
                                         oldbalanceOrg_val, newbalanceOrig_val, 
                                         oldbalanceDest_val, newbalanceDest_val)
                
                if indicators['legitimate_transfer'] or indicators['legitimate_cashout'] or indicators['legitimate_large_cashout']:
                    final_result = False
                else:
                    if model_pred == 1 and is_fraud_result:
                        final_result = True
                    elif model_pred == 0 and not is_fraud_result:
                        final_result = False
                    else:
                        final_result = is_fraud_result
        else:
            final_result = is_fraud(indicators, type_, amount_val, 
                                  oldbalanceOrg_val, newbalanceOrig_val, 
                                  oldbalanceDest_val, newbalanceDest_val)
        
    except ValueError as e:
        st.error("Please enter valid numeric values for all fields")
        final_result = False
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        final_result = False
    
    if final_result:
        st.error(" **FRAUD DETECTED**")
        st.write("**Recommendation:** Flag for manual review and block transaction")
    else:
        st.success(" **NOT FRAUD**")
        st.write("**Recommendation:** No action required")

st.markdown("---")
st.markdown("** Fraud Detection System**")
st.markdown("*Using advanced machine learning to protect your transactions*")