import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer

# Define the text summarization function
def summarize_text(text):
    model_name = "facebook/bart-large-cnn"  # Change to T5 if you prefer
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Tokenize input text and generate summary
    inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Streamlit UI
st.title("Text Summarization Tool")

text = st.text_area("Enter text to summarize:")

if st.button("Summarize"):
    if text:
        summary = summarize_text(text)
        st.write("Summary:")
        st.write(summary)
    else:
        st.write("Please enter text to summarize.")
