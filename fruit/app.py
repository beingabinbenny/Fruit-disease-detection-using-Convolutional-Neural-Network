from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import test


import os

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)  

    

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():

    return render_template('index.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    file = request.files['fileInput']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)



        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print(f"Uploaded file: {filename}")  # Print the filename

        image_path = f"/Users/abinbenny/Documents/adithya project/fruit/uploads/{filename}"
        predicted_class = test.classify_image(image_path)
        return render_template('index.html', predicted_class=predicted_class,filename=filename)
    
    return redirect(url_for('index'))






def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

if __name__ == '__main__':
    app.run(debug=True)
