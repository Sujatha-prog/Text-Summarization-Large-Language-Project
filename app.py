from flask import Flask, render_template, request
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

app = Flask(__name__)

# Load Pegasus model and tokenizer
model_name = 'google/pegasus-large'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        text = request.form['text']
        
        # Tokenize and generate summary
        inputs = tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = model.generate(**inputs)

        # Decode the generated summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return render_template('index.html', text=text, summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
