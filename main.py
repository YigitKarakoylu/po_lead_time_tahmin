import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import joblib

st.set_page_config(layout = "wide", page_title="PO Cycle Time Prediction", page_icon="")
st.title(":red[PO Cycle Time] Tahmin Uygulaması")


@st.cache_data
def get_pipeline():
    pipeline = joblib.load('/Users/yigitcankarakoylu/PycharmProjects/po_lead_time_tahmin/final_model_v4.joblib')
    return pipeline

main_tab, recommendation_tab = st.tabs(["Ana Sayfa", "Tahmin Sistemi"])

# Ana Sayfa
left_col, right_col = main_tab.columns(2)

left_col.header("Projenin Amacı")
left_col.write("""Projenin amacı, bir ürünün üretim hattında üretimine başlanması ile üretimin tamamlanması arasındaki süreyi tahmin eden bir arayüz oluşturmaktır.
                \n Bu sistemin yürürlüğe girmesi ile şirket daha iyi kapasite planlamayı ve gelecek siparişleri daha iyi yönetmeyi hedeflemektedir.
                \n Tahmin için kullanılacak veriler kabaca aşağıda tarif edilmiştir:
                \n * Malzeme Grubu                     
                \n * Malzeme Türü
                \n * Üretim Yeri
                \n * İşlem Sayısı
                \n * Miktar
                \n * Manuel İşçilik Süresi
                \n * Makina İşçilik Süresi  ...vb """)

right_col.image("graph3.gif")

# Öneri Sistemi

pipeline = get_pipeline()

col_features1, col_features2, col_recommendation = recommendation_tab.columns(3)

order_plan = col_features1.selectbox("Sipariş Planı", options=[1, 2, 3, 4, 5, 6, 7, 8, 9])

nat_ind = col_features1.selectbox("Millileştirilebilir mi?", options=['Evet', 'Hayır'])
nat_ind = 1 if nat_ind == 'Evet' else 0

ord_priority = col_features1.selectbox("Sipariş önceliği var mı?", options=['Evet', 'Hayır'])
ord_priority = 1 if ord_priority == 'Evet' else 0

is_proj = col_features1.selectbox("Proje altında takip ediliyor mu?", options=['Evet', 'Hayır'])
is_proj = 1 if is_proj == 'Evet' else 0

comp_availab = col_features1.selectbox("Tüm alt bileşenler sağlandı mı?", options=['Evet', 'Hayır'])
comp_availab = 1 if comp_availab == 'Evet' else 0

# Transform numerical input values
quantity = col_features2.number_input("Miktar", min_value=1, max_value=10000, value="min", step=10)
process_num = col_features2.number_input("İşlem Sayısı", min_value=1, max_value=65, value="min", step=1)
manual_labour = col_features2.number_input("Manuel İşçilik Saati", min_value=0.0, max_value=9000.0, value="min", step=0.5)
machine_labour = col_features2.number_input("Makina İşçilik Saati", min_value=0.0, max_value=6000.0, value="min", step=0.5)
z4_num = col_features2.number_input("Z4 Bildirim Sayısı", min_value=0, max_value=60, value="min", step=1)
z6_num = col_features2.number_input("Z6 Bildirim Sayısı", min_value=0, max_value=600, value="min", step=1)

# Create the DataFrame with transformed values
features = np.array([quantity, order_plan, process_num, nat_ind, ord_priority, is_proj, comp_availab,
                     manual_labour, machine_labour, z4_num, z6_num]).reshape(1, -1)

df = pd.DataFrame(features, columns=['QUANTITY', 'ORDER_PLAN', 'PROCESS_NUMBER', 'NATIONAL_INDICATOR',
                                     'ORDER_PRIORITY', 'IS_PROJECT', 'IS_ALL_COMPONENTS_AVAILABLE',
                                     'MANUAL_LABOUR_HOUR', 'MACHINE_LABOUR_HOUR', 'Z4_NOTIF_NUM', 'Z6_NOTIF_NUM'])

if col_recommendation.button("Öneri Getir!"):

    predictions = pipeline.predict(df)

    col_recommendation.write(f"**{predictions[0]}**")





