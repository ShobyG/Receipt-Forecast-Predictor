from flask import Flask, render_template, request, redirect, flash
from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField
from wtforms.validators import DataRequired

from inference import inference


app = Flask(__name__)
# app.config['SECRET_KEY'] = 'your secret key'
app.secret_key = "a secret"

CHOICES = [(1, 'January'),
           (2, 'February'),
           (3, 'March'),
           (4,'April'),
           (5,'May'),
           (6,'June'),
           (7,'July'),
           (8,'August'),
           (9,'September'),
           (10,'October'),
           (11,'November'),
           (12,'December')]
class monthInputForm(FlaskForm):
    month = SelectField('Select a month',
                        choices=CHOICES,
                        validators=[DataRequired()])
    submit = SubmitField(label="Predict Total Receipt Count")


@app.route("/", methods=['GET', 'POST'])
def home():
    form = monthInputForm()
    if request.method == "POST":
        month_selected = form.month.data
        total_receipts = int(inference(month_selected))
        # print(total_receipts)
        month = None
        # print(month_selected)
        for i in CHOICES:
            if i[0] == int(month_selected):
                month = i[1]
                break
        return render_template("home.html", prediction_data=(month, total_receipts), form=form)
    return render_template("home.html", form=form)

# min_value = 7095414
# max_value = 10738865
# m = load_model('receipt_count.h5')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
