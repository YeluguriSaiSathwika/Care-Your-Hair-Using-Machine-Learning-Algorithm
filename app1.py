import numpy as np
import pickle
import pandas
import os
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open(r'model1.pkl', 'rb'))
d = pickle.load(open(r'dandruff.pkl', 'rb'))
hw = pickle.load(open(r'hair_washing.pkl', 'rb'))
p = pickle.load(open(r'pressure_level.pkl', 'rb'))
sa = pickle.load(open(r'school_assesssment.pkl', 'rb'))
sl = pickle.load(open(r'stress_level.pkl', 'rb'))
sb = pickle.load(open(r'shampoo_brand.pkl', 'rb'))
sw = pickle.load(open(r'swimming.pkl', 'rb'))


@app.route('/', methods=["POST", "GET"])
def home():
    return render_template("home.html")
@app.route('/predict', methods=["POST", "GET"])
def predict():
    return render_template("input.html")
@app.route('/submit', methods=["POST", "GET"])
def submit():
    stay_up_late = int(request.form['stay_up_late'])
    pressure_level = request.form['pressure_level']
    pressure_level = p.transform([pressure_level])
    coffee_consumed = int(request.form['coffee_consumed'])
    brain_working_duration = int(request.form['brain_working_duration'])
    school_assesssment = request.form['school_assesssment']
    school_assesssment = sa.transform([school_assesssment])
    stress_level = request.form['stress_level']
    stress_level = sl.transform([stress_level])
    shampoo_brand = request.form['shampoo_brand']
    shampoo_brand = sb.transform([shampoo_brand])
    swimming = request.form['swimming']
    swimming = sw.transform([swimming])
    hair_washing = request.form['hair_washing']
    hair_washing = hw.transform([hair_washing])
    hair_grease = int(request.form['hair_grease'])
    dandruff = request.form['dandruff']
    dandruff = d.transform([dandruff])

    names = [stay_up_late, pressure_level,
             coffee_consumed, brain_working_duration, school_assesssment,
             stress_level, shampoo_brand, swimming, hair_washing,
             hair_grease, dandruff]
    print(names)
    final_features = [np.array(names)]
    print(final_features)
    prediction = model.predict(final_features)
    print(prediction)
    #prediction = int(prediction)

    if prediction == 'A lot':
        return render_template("output.html", result="Alert! You have Excessive Hair loss: Address the urgency, take immediate action - nourish, strengthen, and consult a dermatologist, opt for protein-rich treatments, and consider supplements; Turn this period of excessive hair loss into a transformative journey towards renewal and recovery.")
    elif prediction == 'Few':
        return render_template("output.html", result="Enjoy the ease of Minimal Hair loss; sustain a regular care routine, hydrate, and indulge in occasional scalp massages and manage stress for luscious locks that shine with health and vitality!")
    elif prediction == 'Many':
        return render_template("output.html", result="No need to panic, you have slightly High Hair loss: address the root - assess nutrition, biotin-rich foods, and deep conditioning, minimize stress, and seek professional guidance to conquer the challenge of hair loss.")
    elif prediction == 'Medium':
        return render_template("output.html", result="Attention! You have Medium Hair loss : seek professional guidance, fortify your hair with a gentle touch, essential nutrients, and positive vibes to counteract hair loss and boost your confidence.")

if __name__ == "__main__":
    app.run(debug=False, port=1111)