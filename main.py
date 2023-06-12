import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# Importing Data
crop = pd.read_csv('Crop_recommendation.csv')

# Preprocessing and Encoding
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9,
    'watermelon': 10, 'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15, 'blackgram': 16,
    'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}
crop['label_num'] = crop['label'].map(crop_dict)
crop.drop('label', axis=1, inplace=True)

# Train Test Split
X = crop.iloc[:, :-1]
y = crop.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training Model
rdf = RandomForestClassifier()
rdf.fit(X_train, y_train)

# Evaluation
y_pred = rdf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# GUI
def predict_crop():
    try:
        N = float(nitrogen_entry.get())
        P = float(phosphorous_entry.get())
        K = float(potassium_entry.get())
        temperature = float(temperature_entry.get())
        humidity = float(humidity_entry.get())
        pH = float(ph_entry.get())
        rainfall = float(rainfall_entry.get())

        input_values = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
        prediction = rdf.predict(input_values)
        crop_dict_reverse = {v: k for k, v in crop_dict.items()}

        if prediction[0] in crop_dict_reverse:
            result_label.config(text=f"The recommended crop is {crop_dict_reverse[prediction[0]]}")
        else:
            messagebox.showerror("Error",
                                 "Sorry, we could not determine the best crop to be cultivated with the provided data.")
    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter numeric values for all fields.")


# Create GUI window
window = tk.Tk()
window.title("Crop Recommendation System")
window.geometry("500x400")
window.configure(bg='#f0f0f0')

# Create input labels and entry fields
nitrogen_label = tk.Label(window, text="Nitrogen content:", bg='#f0f0f0')
nitrogen_label.pack()
nitrogen_entry = tk.Entry(window)
nitrogen_entry.pack()

phosphorous_label = tk.Label(window, text="Phosphorous content:", bg='#f0f0f0')
phosphorous_label.pack()
phosphorous_entry = tk.Entry(window)
phosphorous_entry.pack()

potassium_label = tk.Label(window, text="Potassium content:", bg='#f0f0f0')
potassium_label.pack()
potassium_entry = tk.Entry(window)
potassium_entry.pack()

temperature_label = tk.Label(window, text="Temperature:", bg='#f0f0f0')
temperature_label.pack()
temperature_entry = tk.Entry(window)
temperature_entry.pack()

humidity_label = tk.Label(window, text="Humidity:", bg='#f0f0f0')
humidity_label.pack()
humidity_entry = tk.Entry(window)
humidity_entry.pack()

ph_label = tk.Label(window, text="pH level:", bg='#f0f0f0')
ph_label.pack()
ph_entry = tk.Entry(window)
ph_entry.pack()

rainfall_label = tk.Label(window, text="Rainfall:", bg='#f0f0f0')
rainfall_label.pack()
rainfall_entry = tk.Entry(window)
rainfall_entry.pack()

# Create predict button
predict_button = tk.Button(window, text="Predict", command=predict_crop, bg='#007bff', fg='#ffffff')
predict_button.pack()

# Create result label
result_label = tk.Label(window, text="", bg='#f0f0f0', font=("Arial", 12, "bold"))
result_label.pack()

# Run the GUI
window.mainloop()
