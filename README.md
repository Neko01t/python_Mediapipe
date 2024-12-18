# Hand Drawing Application

This project is a Hand Drawing Application that uses OpenCV and MediaPipe to enable drawing on the screen by tracking hand landmarks. Users can draw lines using their index finger, clear the canvas, and save their drawing as a JSON file.

## Features

- Real-time hand tracking using MediaPipe.
- Draw lines on the screen by pinching (bringing the thumb and index finger close).
- Clear the drawing canvas by pressing a key.
- Save drawn points as a JSON file for later use.

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/hand-drawing-app.git
   cd hand-drawing-app
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:

   ```bash
   python hand_drawing.py
   ```

2. Use your webcam to interact with the application:

   - Bring your thumb and index finger close to draw on the screen.
   - Press `c` to clear the canvas.
   - Press `q` to quit the application.

3. After exiting, the drawing data will be saved to a file named `drawing_data.json`.

## Key Bindings

- `q`: Quit the application.
- `c`: Clear the canvas.

## Output

- The application generates a JSON file named `drawing_data.json`, which contains the coordinates of the drawn points.

## How It Works

1. **Hand Detection**:

   - The application uses MediaPipe to detect hand landmarks in real time from the webcam feed.

2. **Drawing Logic**:

   - The distance between the thumb tip and index finger tip is calculated.
   - If the distance is below a threshold, drawing mode is activated, and the index finger's coordinates are recorded.

3. **Canvas Operations**:

   - The recorded coordinates are used to draw lines on the video frame.
   - Pressing `c` clears the list of recorded points, effectively clearing the canvas.

4. **Data Saving**:

   - The coordinates of the drawn points are saved in a JSON file when the application exits.

## Dependencies

- OpenCV: For capturing webcam feed and drawing.
- MediaPipe: For hand landmark detection.
- NumPy: For mathematical operations.

## Acknowledgements

- [MediaPipe](https://google.github.io/mediapipe/) for the excellent hand tracking module.
- [OpenCV](https://opencv.org/) for real-time computer vision capabilities.

## License

This project is licensed under the MIT License. Feel free to use and modify it as needed.

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to submit a pull request.

