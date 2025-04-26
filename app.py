
import streamlit as st
import joblib



# Streamlit App
st.title("📩 Spam vs Ham Classifier")
st.write("Enter a message and find out if it's spam or not.")
model = joblib.load("spam_classifier.pkl")  
vectorizer = joblib.load("count_vectorizer.pkl") 
transformer = joblib.load("tfidf_transformer.pkl") 

user_input = st.text_area("Your Message")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        message_vec = vectorizer.transform([user_input])
        
        # Step 2: Apply the same TF-IDF transformation
        message_tfidf = transformer.transform(message_vec)
        
        # Step 3: Predict with the model
        prediction = model.predict(message_tfidf)[0]
        if prediction == "spam":
            st.error("🚨 This message is classified as **SPAM**.")
        else:
            st.success("✅ This message is classified as **HAM** (not spam).")



