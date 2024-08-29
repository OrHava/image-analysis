# Image and JSON Analysis Tool

## Overview

This project is a comprehensive tool for analyzing images and JSON files. It leverages several powerful libraries to detect objects, analyze facial features, and visualize data. The tool provides functionality for both real-time image analysis and post-analysis of saved results, with a user-friendly GUI for ease of use.

## Features

- **Image Analysis**:
  - Detect objects using a pre-trained VGG16 model.
  - Analyze facial features (age, gender, race, emotion) using DeepFace.
  - Save and summarize results in JSON and text formats.
  - Visualize data through charts and graphs.

- **JSON Analysis**:
  - Load and analyze results from JSON files.
  - Display statistical summaries and trends.

- **GUI Interface**:
  - Simple GUI to start image analysis, JSON analysis, and summary analysis.

## Requirements

To run this project, you need to install the following dependencies:

- **Python 3.x**: Ensure Python 3.x is installed on your system.
- **Libraries**: Install the required Python libraries using pip:
  ```bash
  pip install opencv-python-headless pillow torch torchvision face_recognition deepface matplotlib pandas


## Installation

1. Clone the Repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, you can manually install the required libraries as mentioned in the requirements.

## Usage

### Running the Tool

1. Start the GUI: Run the following command to launch the GUI:
   ```bash
   python <script-name>.py
   ```

2. Analyze Images:
   - Click the "Analyze Images" button in the GUI.
   - Select a folder containing JPEG images.
   - The tool will analyze the images, detect objects, and perform facial analysis.
   - Results will be saved to a JSON file and summarized in a text file.

3. Analyze JSON Files:
   - Click the "Analyze JSON" button in the GUI.
   - Select a JSON file containing analysis results.
   - The tool will display a summary of the data and statistical insights.

4. Analyze Summary Files:
   - Click the "Analyze Summary" button in the GUI.
   - Select a text file containing summary data.
   - The tool will display the content in a message box.

## Script Overview

- `load_images_from_folder(folder_path)`: Loads images from the specified folder.
- `detect_objects(image)`: Detects objects in an image using the VGG16 model.
- `analyze_images(images)`: Analyzes each image for faces and objects, and calculates processing time.
- `save_results_to_file(results, base_name)`: Saves analysis results to a JSON file.
- `summarize_face_analysis(face_analysis)`: Summarizes facial analysis results.
- `summarize_object_detections(object_detections)`: Summarizes object detection results.
- `display_summary(data, base_name)`: Saves a summary of the analysis to a text file.
- `analyze_json(file_path)`: Analyzes and displays summary from a JSON file.
- `show_statistics(data)`: Displays statistical charts and trends based on analyzed data.
- `create_gui()`: Creates and displays the GUI for user interactions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The DeepFace and face_recognition libraries for facial analysis.
- The torchvision library for object detection with VGG16.
- The matplotlib and pandas libraries for data visualization and analysis.

Feel free to modify and extend this tool to suit your needs. If you encounter any issues or have suggestions, please open an issue or submit a pull request.
## Pictures Example
<img width="883" alt="צילום מסך 2024-08-29 232944" src="https://github.com/user-attachments/assets/1ebddd7f-b0a0-40e5-be07-0a6d4aaed8fe">



