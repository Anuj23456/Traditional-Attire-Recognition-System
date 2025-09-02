from flask import Flask, request, render_template, url_for
import os
import pickle
from PIL import Image
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = pickle.load(open('NoteBook/tunnedModel.pkl', 'rb'))

# Categories for prediction
categories = ['Adi', 'Apatani', 'Galo', 'Mishmi', 'Monpa', 'Nocte', 'Nyishi', 'Singpho', 'Tai Khampti']

# About information for each category
about_categories = {
    'Adi': (
    "<strong>About:</strong> The Adi tribe is one of the major tribal groups of Arunachal Pradesh, known for their vibrant culture, "
    "festivals like Solung, and expertise in agriculture and weaving.<br>"
    "<strong>Region:</strong> Siang districts, Upper Dibang Valley.<br>"
    "<strong>Traditional Dress:</strong> Men wear a sleeveless woolen coat called 'Gale.' Women wear a skirt called 'Gale' paired with "
    "blouses, along with silver ornaments.<br>"
    "<strong>Special Features:</strong> Skilled in bamboo and cane crafts, the Adis follow animistic beliefs but also incorporate elements "
    "of Hinduism. They are known for their community-based agricultural practice called Jhum (shifting cultivation)."
    ),

    'Apatani': (
    "<strong>About:</strong> The Apatani tribe is renowned for their sustainable agricultural practices and unique cultural identity. "
    "Apatani women are recognized for their facial tattoos and nose plugs, which are a part of their heritage.<br>"
    "<strong>Region:</strong> Ziro Valley, Lower Subansiri district.<br>"
    "<strong>Traditional Dress:</strong> Women wear a patterned cloth wrapped around the body with a sash, and men wear sleeveless shirts with a cane hat.<br>"
    "<strong>Special Features:</strong> Masters of wet rice cultivation and fish farming, they are known for their eco-friendly practices. "
    "The Apatani Valley is also a UNESCO World Heritage Site nominee."
    ),

    'Galo': (
    "<strong>About:</strong> The Galos are one of the subgroups of the Adi tribe, known for their agricultural economy and vibrant community festivals like Mopin.<br>"
    "<strong>Region:</strong> West Siang district.<br>"
    "<strong>Traditional Dress:</strong> Men wear a 'Galo Gale,' a long wraparound cloth, and women wear skirts and blouses made of woven fabric.<br>"
    "<strong>Special Features:</strong> The Galo tribe follows a unique system of family lineage called Aji, which helps maintain a structured genealogy. "
    "They are also skilled in traditional herbal medicine."
    ),

    'Mishmi': (
    "<strong>About:</strong> The Mishmi tribe is divided into three main groups—Idus, Digarus, and Mijus. They are known for their unique cultural traditions, "
    "animistic religious beliefs, and skilled craftsmanship in weaving and basketry.<br>"
    "<strong>Region:</strong> Dibang Valley, Lohit, and Anjaw districts.<br>"
    "<strong>Traditional Dress:</strong> Women wear finely woven skirts and blouses with elaborate silver jewelry, while men wear simple woven coats and headgear.<br>"
    "<strong>Special Features:</strong> The Mishmis are adept in herbal medicine and healing practices. Their festivals, like Reh, reflect deep spiritual connections with nature."
    ),

    'Monpa': (
    "<strong>About:</strong> The Monpa tribe has strong Tibetan cultural influences and is primarily Buddhist, following the Gelugpa sect of Tibetan Buddhism. "
    "They are known for their monasteries, festivals, and woodcraft.<br>"
    "<strong>Region:</strong> Tawang and West Kameng districts.<br>"
    "<strong>Traditional Dress:</strong> Men wear woolen gowns called 'Chuba,' while women wear brightly colored striped aprons with woolen dresses and jewelry.<br>"
    "<strong>Special Features:</strong> Skilled in carpet weaving, wood carving, and thangka painting. The Monpas celebrate Losar (New Year) and Torgya festivals."
    ),

    'Nocte': (
    "<strong>About:</strong> The Nocte tribe is known for its vibrant festivals, music, and dance, with strong traditions linked to agriculture and warrior heritage. "
    "They practice both animism and Christianity.<br>"
    "<strong>Region:</strong> Tirap district.<br>"
    "<strong>Traditional Dress:</strong> Men wear sleeveless vests with headgear decorated with hornbill feathers, while women wear woven skirts with beaded ornaments.<br>"
    "<strong>Special Features:</strong> They celebrate the Chalo-Loku festival, marking the harvest season. The Noctes are also known for their iron-smelting traditions."
    ),

    'Nyishi': (
    "<strong>About:</strong> The Nyishi tribe is the largest ethnic group in Arunachal Pradesh, recognized for their colorful traditions, community life, and "
    "animistic religious practices alongside Hindu and Christian influences.<br>"
    "<strong>Region:</strong> Papum Pare, Kurung Kumey, Kra Daadi, and East Kameng districts.<br>"
    "<strong>Traditional Dress:</strong> Men traditionally wear cane helmets with hornbill beaks, while women wear woven skirts with beads and silver ornaments.<br>"
    "<strong>Special Features:</strong> They celebrate the Nyokum Yullo festival, invoking deities for prosperity and harmony. Skilled in cane and bamboo crafts."
    ),

    'Singpho': (
    "<strong>About:</strong> The Singpho tribe shares close cultural ties with the Kachin people of Myanmar. They are known for introducing tea cultivation "
    "in Assam and Arunachal Pradesh.<br>"
    "<strong>Region:</strong> Changlang district.<br>"
    "<strong>Traditional Dress:</strong> Men wear sleeveless shirts with long lungis, while women wear brightly colored wraparound skirts and blouses with shawls.<br>"
    "<strong>Special Features:</strong> They practice Theravada Buddhism, and their lifestyle revolves around tea cultivation, agriculture, and community feasts."
    ),
    
    'Tai Khampti': (
    "<strong>About:</strong> The Tai Khampti tribe follows Theravada Buddhism and has a rich cultural heritage of literature, music, and dance. "
    "They use the Tai script and are known for their distinctive customs.<br>"
    "<strong>Region:</strong> Namsai district.<br>"
    "<strong>Traditional Dress:</strong> Men wear a simple shirt with lungi, while women wear elegant long skirts called 'Sinh' with blouses and shoulder cloths.<br>"
    "<strong>Special Features:</strong> Famous for their Poi-Pee-Mau festival (New Year), traditional cuisine, and goldsmithing. The Khamptis are also skilled artisans and farmers."
    ),
}

@app.route('/', methods=['GET', 'POST'])
def home():
    message = ""
    image_url = None
    prediction = None
    predicted_category = None
    about_text = None

    if request.method == 'POST':
        if 'image' not in request.files:
            message = "Choose a File"
        else:
            file = request.files['image']
            if file.filename == '':
                message = "No selected file"
            elif file:
                # Save the uploaded file
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                image_url = url_for('static', filename=f'uploads/{file.filename}')
                message = "Image uploaded successfully!"

                # Preprocess the image
                image = Image.open(filepath)
                image = image.resize((150, 150))  # Resize to 150x150
                image = np.array(image)  # Convert image to NumPy array

                # If the image is RGB, convert to grayscale or keep it as RGB based on model requirement
                if image.shape[-1] == 3:  # If RGB
                    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale (optional)

                # Reshape image for prediction
                image = image.reshape(1, -1)  # Flatten the image for SVC input (150*150,)

                # Predict using the model
                prediction = model.predict(image)
                predicted_category = categories[prediction[0]]  # Get the category from the prediction index
                about_text = about_categories.get(predicted_category, "Details about this category are not available.")

    return render_template('index.html', 
                           message=message, 
                           image_url=image_url, 
                           prediction=prediction, 
                           predicted_category=predicted_category,
                           about_text=about_text)

if __name__ == '__main__':
    app.run(debug=True)
