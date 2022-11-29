from flask import Flask, render_template, request
from newspaper import Article
from transformers import BartForConditionalGeneration, BartTokenizer



app = Flask(__name__)
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/summarize",methods=['POST','GET'])
def getsummary():
    body=request.form["data"]
    inputs = tokenizer([body], max_length=1024, return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=4000, early_stopping=True)
    summary = " ".join([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

    return render_template('index.html', summary=summary, text=body, show_summary=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug = True)
