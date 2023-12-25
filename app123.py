from flask import Flask, request, jsonify
import pickle

app123 = Flask(__name__)

# Load the pickled SVM model
with open('model1.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)
    
with open('vectorizer.pickle', 'rb') as f:
    # Load the pickle object
    vectorizer = pickle.load(f)


# Endpoint to receive POST requests with ingredients and return predictions
@app123.route('/', methods=['POST'])
def predict1():
    try:
        data = request.get_json()
        
        new_ingredients_vector = vectorizer.transform(data)
        

        # Make predictions
        predictions = svm_model.predict(new_ingredients_vector)

        # Prepare and return the predictions as JSON
        return jsonify( predictions.tolist())
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app123.run(debug=False,host='0.0.0.0')
