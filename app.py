import time
import streamlit as st
import SessionState
import numpy as np
from predictor import Predictor
from dataset.add_noise import SynthesizeData

state = SessionState.get(text_correct="", input="", noise="")
import nltk

def main():
    model, synther = load_model()
    st.title("Chương trình sửa lỗi chính tả tiếng Việt")
    # Load model
    state.input = ""
    state.noise = ""
    text_input = st.text_area("Nhập đầu vào:", value=state.input)
    text_input = text_input.strip()
    if st.button("Correct"):
        state.noise = text_input
        state.text_correct = model.spelling_correct(state.noise)
        st.text("Câu nhiễu: ")
        st.success(state.noise)
        st.text("Kết quả:")
        st.success(state.text_correct)


    if st.button("Add noise and Correct"):
        state.noise = synther.add_noise(text_input, percent_err=0.3)
        # state.output = noise_text
        state.text_correct = model.spelling_correct(state.noise)
        st.text("Câu nhiễu: ")
        st.success(state.noise)
        st.text("Kết quả:")
        st.success(state.text_correct)



    # state.sync()


@st.cache(allow_output_mutation=True)  # hash_func
def load_model():
    print("Loading model ...")
    nltk.download('punkt')
    model = Predictor(weight_path='weights/seq2seq.pth', have_att=True)
    synther = SynthesizeData()
    return model, synther


if __name__ == "__main__":
    main()