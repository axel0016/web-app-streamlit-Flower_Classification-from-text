import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write('''
# evento predection app
''')

st.sidebar.header('User Input Parameters')

def userinput():
    sepalego=st.sidebar.slider('sepalgo',0,10,5)
    sepaletii=st.sidebar.slider('sepaltii',0,10,5)
    anirogo=st.sidebar.slider('anirogo',0,20,10)
    anirotii=st.sidebar.slider('anirotii',0,20,10)
    data={'sepalgo':sepalego,
            'sepaltii':sepaletii,
            'anirogo':anirogo,
            'anirotii':anirotii}
    features=pd.DataFrame(data,index=[0])
    return features
st.subheader('User Input Parameters')
df=userinput()
st.write(df)

iris=datasets.load_iris()
x=iris.data
y=iris.target
clf=RandomForestClassifier()
clf.fit(x,y)

prediction=clf.predict(df)

st.subheader('your final decision is')
st.write(iris.target_names[prediction])


