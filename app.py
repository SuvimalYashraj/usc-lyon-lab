from flask import Flask, request, jsonify

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification


def preprocessing(questions):
    # Load the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Tokenize your dataset
    questions_tokenized = tokenizer(
        questions, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )

    return questions_tokenized


# Create a Flask application
app = Flask(__name__)


# Define a route for your API
@app.route("/predict", methods=["GET", "POST"])
# @cross_origin()
def predict():
    if request.method == "GET":
        input_data = request.args.get("input", "")
        return input_data

    # Get the request data as a JSON object
    data = request.get_json()

    # Extract the list of strings from the request data
    question_list = data.get("string_list")

    # Preprocess the input data
    val_loader = preprocessing(question_list)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    val_loader = {k: v.to(device) for k, v in val_loader.items()}

    # Load the trained model
    model = RobertaForSequenceClassification.from_pretrained('roberta-base',num_labels=3).to(device)
    model.load_state_dict(torch.load("roberta_combined_state_dict.pth"))
    model.to(device)
    model.eval()

    # Make predictions with the model
    outputs = model(**val_loader)
    predictions = torch.argmax(outputs.logits, dim=1).tolist()

    result = jsonify({"predictions": predictions})

    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
