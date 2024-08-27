import joblib
mnb = joblib.load('spam_model4.pkl')
pp = joblib.load('price_pred4.pkl')
scale = joblib.load('scaler4.pkl')
association = joblib.load('association_rules.pkl')

def get_spam(user_inp):
    v = joblib.load('count_vectorizer4.pkl')
    data_count = v.transform([user_inp])
    op = mnb.predict(data_count)
    return op

def format_number(number):
    number_str = str(number)
    number_str = number_str[::-1] 
    number_str = ','.join([number_str[i:i+2] for i in range(0, len(number_str), 2)])
    number_str = number_str[::-1] 
    number_str = ','.join([number_str[i:i+3] for i in range(0, len(number_str), 3)])
    return number_str

def get_price(user_entry):
    user_entry=[user_entry]
    user_entry = scale.transform(user_entry)
    model_pred = pp.predict(user_entry)
    return int(model_pred[0])


def associates(user_item):
    return association[ association['antecedents']=={user_item}]['consequents']

from streamlit_extras.let_it_rain import rain
def it_rains():
    rain(
        emoji="👷",
        font_size=54,
        falling_speed=5,
        animation_length="infinite",
    )

import streamlit as st
def app():   
    st.header("Welcome to Atharva's ML Models")
    with st.sidebar:
        st.header("Stuff")
        application = st.selectbox("ML Model:",["Spam Classification","House Price Prediction",'Association Rules','Text_Autocomplete'])

    if application == "Spam Classification":
        st.title("Spam-Ham Detection")
        text_input = st.text_input("Enter Your text to check:")

        if st.button("Classfy Message"):
            prediction = get_spam(text_input)
            if prediction == 1:
                st.success("The text is classified as HAM")
            else:
                st.error(f"The text is classified as SPAM")
    elif application == "House Price Prediction":
        st.title("House Price Prediction")
        
        area = st.number_input("Enter area of House:")
        bedrooms = st.slider("Bedroom:", 0, 10, 2)
        bathrooms = st.slider("Bathroom:", 0, 10, 2)
        stories = st.slider("Stories:", 0, 10, 2)
        mainroad = st.radio(
            "MainRoad",
            ["Yes","No"]
        )
        mainroad =  1 if mainroad=="Yes" else 0

        guestroom = st.radio(
            "GuestRoom",
            ["Yes","No"]
        )
        guestroom =  1 if guestroom=="Yes" else 0
        basement = st.radio(
            "Basement",
            ["Yes","No"]
        )
        basement =  1 if basement=="Yes" else 0
        hotwater = st.radio(
            "Hot Water",
            ["Yes","No"]
        )
        hotwater =  1 if hotwater=="Yes" else 0
        aircond = st.radio(
            "Air Conditioning",
            ["Yes","No"]
        )
        aircond =  1 if aircond=="Yes" else 0
        preface = st.radio(
            "Preface",
            ["Yes","No"]
        )
        parking = st.slider("Parking:", 0, 10, 2)

        preface =  1 if preface=="Yes" else 0
        
        furnishing = st.radio(
            "Furnishing",
            ["Furnished","Semi-furnished","Unfurnished"]
        )
        if furnishing=="Furnished":
            furnishing =0
        elif furnishing=='Semi-furnished':
            furnishing =1
        else:
            furnishing = 2
        
        
        user_inp = [area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwater,aircond,parking,preface,furnishing]
        if st.button("Predict Price"):
            prediction = get_price(user_inp)
            formatted_prediction = format_number(prediction)
            st.success(f"₹{formatted_prediction}/-")
            
    elif application=="Association Rules":
        st.title("Finding Association Rules")
        ui =st.radio(
            "Select Items",
            ["burgers","cake",'chocolate','eggs','french fries','milk','pancakes','spaghetti']
        )
        if st.button('Find Association'):
            st.info(f"Your {ui} could be best sold with:")
            asso = associates(ui)
            for key, value in asso.items():
                consequent_list = list(value) 
                consequent_string = ", ".join(consequent_list)
                st.write(f"{consequent_string}")
    elif application=="Text_Autocomplete":
        st.title('Text_Autocompletion')
        st.info('Application in development')
        it_rains()
        with st.spinner('Required Time->soon'):
            import time
            time.sleep(10)
            st.info('Still Work in Progress')
if __name__ == "__main__":
    app()
