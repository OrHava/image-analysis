import os
import cv2
import json
import torch
import time
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
from torchvision import models, transforms
import face_recognition
from deepface import DeepFace
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from datetime import datetime

# Load a pre-trained VGG16 model from PyTorch
model = models.vgg16(pretrained=True) 
model.eval()

# Define a transformation to preprocess the image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append((filename, img))
    return images

def detect_objects(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    image_tensor = preprocess(pil_image).unsqueeze(0)
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
    _, indices = torch.sort(predictions, descending=True)
    percentage = torch.nn.functional.softmax(predictions, dim=1)[0] * 100
    labels = [(idx.item(), percentage[idx].item()) for idx in indices[0][:5]]
    
    return labels

def analyze_images(images):
    results = []
    total_time = 0
    for i, (filename, img) in enumerate(images):
        print(f"Analyzing {filename}...")
        start_time = time.time()
        try:
            face_locations = face_recognition.face_locations(img)
            face_encodings = face_recognition.face_encodings(img, face_locations)
            face_analysis = DeepFace.analyze(img_path=img, actions=['age', 'gender', 'race', 'emotion'])
            object_detections = detect_objects(img)
            
            results.append({
                'filename': filename,
                'face_locations': face_locations,
                'face_analysis': face_analysis,
                'object_detections': object_detections
            })
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")
        
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        print(f"Time taken for {filename}: {elapsed_time:.2f} seconds")
        
        if i > 0:
            avg_time_per_image = total_time / (i + 1)
            remaining_images = len(images) - (i + 1)
            estimated_time_remaining = remaining_images * avg_time_per_image
            print(f"Estimated time remaining: {estimated_time_remaining / 60:.2f} minutes")

    return results

def save_results_to_file(results, base_name):
    count = 1
    output_file = f"{base_name}.json"
    while os.path.exists(output_file):
        output_file = f"{base_name}_{count}.json"
        count += 1
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")
    return output_file

def summarize_face_analysis(face_analysis):
    summary = []
    for face in face_analysis:
        age = face.get('age', 'N/A')
        gender = face.get('dominant_gender', 'N/A')
        race = face.get('dominant_race', 'N/A')
        emotion = face.get('dominant_emotion', 'N/A')
        summary.append(f"Age: {age}, Gender: {gender}, Race: {race}, Emotion: {emotion}")
    return summary

def summarize_object_detections(object_detections):
    return [f"Object ID: {int(obj[0])}, Confidence: {obj[1]:.2f}%" for obj in object_detections]

def display_summary(data, base_name):
    if data is None:
        print("No valid data to display.")
        return

    count = 1
    output_file = f"{base_name}.txt"
    while os.path.exists(output_file):
        output_file = f"{base_name}_{count}.txt"
        count += 1

    with open(output_file, 'w') as f:
        for entry in data:
            filename = entry.get('filename', 'Unknown')
            face_analysis = entry.get('face_analysis', [])
            object_detections = entry.get('object_detections', [])
            
            f.write(f"Summary for {filename}:\n")
            f.write("Face Analysis:\n")
            for face_summary in summarize_face_analysis(face_analysis):
                f.write(f" - {face_summary}\n")
            
            f.write("Object Detections:\n")
            for object_summary in summarize_object_detections(object_detections):
                f.write(f" - {object_summary}\n")
            
            f.write("\n" + "="*50 + "\n")
    
    print(f"Summary saved to {output_file}")
    return output_file

def analyze_json(file_path):
    data = load_json(file_path)
    if data:
        output_file = display_summary(data, "summary_output")
        show_statistics(data)
    else:
        print("Failed to load JSON data.")

def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
def show_statistics(data):
    ages = []
    gender_percentages = []
    race_percentages = []
    emotion_percentages = []
    dates = []
    confidences = []

    for entry in data:
        for face in entry.get('face_analysis', []):
            ages.append(face.get('age', 'N/A'))
            gender_percentages.append(face.get('gender', {}))
            race_percentages.append(face.get('race', {}))
            emotion_percentages.append(face.get('emotion', {}))
            confidences.append(face.get('face_confidence', 0))
            # Extract date from filename
            date_str = entry.get('filename', '').split('_')[0]  # Example: Extracting date from filename
            if date_str:
                try:
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    dates.append(date)
                except ValueError:
                    continue

    # Calculate average age
    avg_age = sum(filter(lambda x: isinstance(x, (int, float)), ages)) / len(ages) if ages else 0

    # Aggregate percentages
    def aggregate_percentages(percentages_list):
        aggregated = Counter()
        for percentages in percentages_list:
            for key, value in percentages.items():
                aggregated[key] += value
        total = sum(aggregated.values())
        return {k: v / total * 100 for k, v in aggregated.items()}

    aggregated_genders = aggregate_percentages(gender_percentages)
    aggregated_races = aggregate_percentages(race_percentages)
    aggregated_emotions = aggregate_percentages(emotion_percentages)

    # Plotting
    fig, axs = plt.subplots(3, 2, figsize=(12, 15))

    # Age Distribution
    axs[0, 0].hist(ages, bins=range(0, 100, 5), color='skyblue', edgecolor='black')
    axs[0, 0].set_title(f'Age Distribution (Avg: {avg_age:.2f})')

    # Gender Distribution
    axs[0, 1].pie(aggregated_genders.values(), labels=aggregated_genders.keys(), autopct='%1.1f%%', colors=plt.cm.Paired.colors)
    axs[0, 1].set_title('Gender Distribution')

    # Race Distribution
    axs[1, 0].pie(aggregated_races.values(), labels=aggregated_races.keys(), autopct='%1.1f%%', colors=plt.cm.Paired.colors)
    axs[1, 0].set_title('Race Distribution')

    # Emotion Distribution
    axs[1, 1].pie(aggregated_emotions.values(), labels=aggregated_emotions.keys(), autopct='%1.1f%%', colors=plt.cm.Paired.colors)
    axs[1, 1].set_title('Emotion Distribution')

    # Time-based emotion analysis
    if dates and emotion_percentages:
        emotion_over_time = pd.DataFrame({'Date': dates, 'Emotion': [max(emotion, key=emotion.get) for emotion in emotion_percentages]})
        emotion_over_time['Count'] = 1
        emotion_trend = emotion_over_time.groupby(['Date', 'Emotion']).count().unstack(fill_value=0)

        emotion_trend.plot(kind='line', ax=axs[2, 0], marker='o')
        axs[2, 0].set_title('Emotion Trends Over Time')
        axs[2, 0].set_xlabel('Date')
        axs[2, 0].set_ylabel('Count')
        axs[2, 0].legend(title='Emotion')
        axs[2, 0].tick_params(axis='x', rotation=45)

    # Confidence over time
    if dates and confidences:
        confidence_over_time = pd.DataFrame({'Date': dates, 'Confidence': confidences})
        confidence_over_time.groupby('Date').mean().plot(kind='line', ax=axs[2, 1], marker='o', color='purple')
        axs[2, 1].set_title('Average Confidence Over Time')
        axs[2, 1].set_xlabel('Date')
        axs[2, 1].set_ylabel('Average Confidence')
        axs[2, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
    
def start_image_analysis():
    folder_path = filedialog.askdirectory(title="Select Folder with Images")
    if not folder_path:
        messagebox.showwarning("No Folder Selected", "Please select a folder containing images.")
        return
    
    images = load_images_from_folder(folder_path)
    if not images:
        messagebox.showwarning("No Images Found", "No JPEG images found in the selected folder.")
        return
    
    results = analyze_images(images)
    output_file = save_results_to_file(results, "image_analysis_results")
    messagebox.showinfo("Analysis Complete", f"Image analysis complete. Results saved to {output_file}")

    analyze_json(output_file)

def start_json_analysis():
    file_path = filedialog.askopenfilename(title="Select JSON File", filetypes=[("JSON Files", "*.json")])
    if not file_path:
        messagebox.showwarning("No File Selected", "Please select a JSON file.")
        return
    
    analyze_json(file_path)

def start_summary_analysis():
    file_path = filedialog.askopenfilename(title="Select Summary File", filetypes=[("Text Files", "*.txt")])
    if not file_path:
        messagebox.showwarning("No File Selected", "Please select a summary text file.")
        return
    
    with open(file_path, 'r') as file:
        content = file.read()
        show_summary_statistics(content)

def show_summary_statistics(content):
    # Example: Display the content in a message box
    messagebox.showinfo("Summary Analysis", content)

def create_gui():
    root = tk.Tk()
    root.title("Image and JSON Analysis Tool")

    analyze_images_button = tk.Button(root, text="Analyze Images", command=start_image_analysis)
    analyze_images_button.pack(pady=10)

    analyze_json_button = tk.Button(root, text="Analyze JSON", command=start_json_analysis)
    analyze_json_button.pack(pady=10)

    analyze_summary_button = tk.Button(root, text="Analyze Summary", command=start_summary_analysis)
    analyze_summary_button.pack(pady=10)

    root.geometry("300x200")
    root.mainloop()

if __name__ == "__main__":
    create_gui()