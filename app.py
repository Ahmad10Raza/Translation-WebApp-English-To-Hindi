from transformers import pipeline, MarianMTModel, MarianTokenizer

# Load the pre-trained MarianMT model and tokenizer for English to Hindi translation
model_name = 'Helsinki-NLP/opus-mt-en-hi'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Define a translation function
def translate_transformers(from_text):
    # Tokenize input text
    inputs = tokenizer(from_text, return_tensors='pt', truncation=True)

    # Generate translation
    translation_ids = model.generate(**inputs)
    translation_text = tokenizer.batch_decode(translation_ids, skip_special_tokens=True)[0]

    return translation_text

# Example usage
translated_text = translate_transformers('My name is Nick')
print(translated_text)

# Interface setup
import gradio as gr

interface = gr.Interface(
    fn=translate_transformers,
    inputs=gr.Textbox(lines=2, placeholder='Text to translate'),
    outputs='text'
)

# Launch the Gradio interface
interface.launch()
