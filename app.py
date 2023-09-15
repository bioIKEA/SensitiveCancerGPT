import streamlit as st
import requests
import json


def call_api(input_prompt, tissue):
    headers = {
    "Authorization": f'Bearer ' + st.secrets('OPENAI_API_KEY'),
}
    data = {
  "prompt": input_prompt,
  "max_tokens": 60
}
    ##THIS IS ADA LUAD DRUG, CELL, MUT
    if tissue == 'LUAD':
        r = requests.post("https://api.openai.com/v1/engines/ada:ft-personal-2023-09-12-15-57-23/completions", headers=headers, json=data)
    elif tissue == 'THCA':
        ##THIS IS ADA THCA DRUG, CELL, MUT
        r = requests.post("https://api.openai.com/v1/engines/ada:ft-personal-2023-09-13-19-48-22/completions", headers=headers, json=data)
    elif tissue == 'COREAD':
        ##THIS IS ADA COREAD DRUG, CELL, MUT
        r = requests.post("https://api.openai.com/v1/engines/ada:ft-personal-2023-09-13-20-17-19/completions", headers=headers, json=data)
    elif tissue == 'BRCA':
        ##THIS IS ADA BRCA DRUG, CELL, MUT
        r = requests.post("https://api.openai.com/v1/engines/ada:ft-personal-2023-09-14-15-37-41/completions", headers=headers, json=data)
    elif tissue == 'LGG':
        ##THIS IS ADA LGG DRUG, CELL, MUT
        r = requests.post("https://api.openai.com/v1/engines/ada:ft-personal-2023-09-14-16-07-32/completions", headers=headers, json=data)
    
    out = json.loads(r.text)
    #print('out', out)
    #sys.exit()
    completion = out['choices'][0]['text']
    completion = completion.split('\n')
    completion = completion[0]
    #print('new completion', completion)
    return completion

#st.title("Drug Sensitivity Prediction")
st.title("SensitiveCancerGPT")
st.write('SensitiveCancerGPT is a web app for anti-cancer drug sensitivity prediction using GPT-3.')

default_drug_name = "ML323"
user_input_drug = st.text_input("Please enter drug name: ", default_drug_name)

default_cell_name = "USP1"
user_input_cell = st.text_input("Please enter cell line name: ", default_cell_name)

default_tissue_name = "LUAD"
option_tissue = st.selectbox(
    'Please select the tissue type',
    #('LUAD', 'THCA', 'COREAD', 'BRCA', 'LGG'))
    ('Lung adenocarcinoma (LUAD)', 'Breast invasive carcinoma (BRCA)', 'Colon and rectum adenocarcinoma (COREAD)', 'Thyroid carcinoma (THCA)', 'Brain Lower Grade Glioma (LGG)'))

if option_tissue == 'Lung adenocarcinoma (LUAD)':
    option_tissue = 'LUAD'
elif option_tissue == 'Breast invasive carcinoma (BRCA)':
    option_tissue = 'BRCA'
elif option_tissue == 'Colon and rectum adenocarcinoma (COREAD)':
    option_tissue = 'COREAD'
elif option_tissue == 'Thyroid carcinoma (THCA)':
    option_tissue = 'THCA'
elif option_tissue == 'Brain Lower Grade Glioma (LGG)':
    option_tissue = 'LGG'

st.write('You selected:', option_tissue)


if not user_input_drug or not user_input_cell:
  st.warning("Please fill out the required fields")

if user_input_drug and user_input_cell:
    input_prompt = "Decide in a single word if the drug's IC50 response to the target is sensitive or resistant. Drug compound and cell line: The drug is " + user_input_drug + ". " + "The cell line is " + user_input_cell +  "\n\n###\n\n '{}'"
else:
    input_prompt = "Decide in a single word if the drug's IC50 response to the target is sensitive or resistant. Drug compound and cell line: The drug is " + default_drug_name + ". " + "The cell line is " + default_cell_name +  "\n\n###\n\n '{}'"

if option_tissue:
    pred = call_api(input_prompt, option_tissue)
else:
    pred = call_api(input_prompt, default_tissue_name)
print('pred', pred)

st.button("Predict", type="primary")
st.write(pred)



