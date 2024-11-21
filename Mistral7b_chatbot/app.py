import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

# Load the fine-tuned model and tokenizer from Hugging Face Hub


@st.cache_resource
def load_model_and_tokenizer():
    # Use your model name from Hugging Face Hub
    model_path = "ItzShahzaib/mistral-finetuned-alpaca"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # Adjust device map if necessary (e.g., to 'cpu' if you don't have a GPU)
        device_map="cpu"
    )
    return model, tokenizer


# Initialize model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# Generate response


def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(
        "cpu")  # Use 'cpu' if you're not using a GPU
    generation_config = GenerationConfig(
        do_sample=True,
        top_k=1,
        temperature=0.1,
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id
    )
    outputs = model.generate(**inputs, generation_config=generation_config)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Streamlit UI
st.title("Instruction-Finetuned QA Chatbot")
st.markdown(
    """
    This is a chatbot fine-tuned on instruction-based datasets for high-quality responses.
    """
)

# User Input
user_input = st.text_area(
    "### Human: Ask your question here:", placeholder="Enter your question...")

if st.button("Generate Response"):
    if user_input.strip():
        with st.spinner("Generating response..."):
            assistant_prompt = f"###Human: {user_input.strip()} ###Assistant: "
            response = generate_response(assistant_prompt)
        st.markdown(f"**### Assistant:** {response}")
    else:
        st.warning("Please enter a question to get a response.")
