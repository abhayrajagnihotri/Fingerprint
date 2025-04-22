from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
import order_items

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'  # SQLite database file
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
login_user = None

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)
    repeat_psw = db.Column(db.String(80), nullable=False)

# Move db.create_all() inside the application context
#with app.app_context():
#    db.create_all()

@app.route('/')
def index():
    #users = User.query.all()
    #return render_template('index.html', users=users)
    return render_template('index.html')


@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/submit_form', methods=['POST'])
def submit_form():
    if request.method == 'POST':
        email = request.form['email']
        psw = request.form['psw']
        psw_repeat = request.form['psw-repeat']

        try:
            new_user = User(email=email, password=psw, repeat_psw=psw_repeat)
            db.session.add(new_user)
            db.session.commit()

            return redirect(url_for('index'))
        except:
            return "Invalid entries or User already exist"


@app.route('/order', methods=['GET', 'POST'])
def order():
    total = 0
    if request.method == 'POST':
        # Get values from the dropdowns
        l = []
        for i in range(1,23):
            #if request.form.get(str(i)) is not '0':
            if request.form.get(str(i)) != '0':
                l.append({'Item':order_items.orderList[i-1]['name'],
                    'Price':order_items.orderList[i-1]['price'],
                    'Quantity':request.form.get(str(i))})

                total = total + order_items.orderList[i-1]['price']*int(request.form.get(str(i)))
        total = total*1.1
    return render_template('order.html', order_data=l, total=total)


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/validate_user', methods=['POST'])
def validate_user():
    global login_user
    if request.method == 'POST':
        email = request.form['uname']
        psw = request.form['psw']

        login_user = User.query.filter_by(email=email).first()

        if login_user and login_user.password==psw:
            # Log the user in
            #return 'Login successful'
            #return redirect(url_for('index'), user=user)
            return render_template('index.html', user=login_user)
        else:
            return 'Login failed'


@app.route('/registered_users')
def registered_user():
    users = User.query.all()
    user_list = []

    for user in users:
        user_data = {
            'id': user.id,
            'email': user.email,
            'password': user.password
            # Add other fields as needed
        }
        user_list.append(user_data)

    return jsonify(users=user_list)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
