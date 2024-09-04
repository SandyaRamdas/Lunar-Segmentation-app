## Project Setup Instructions
### Clone or Download the Repository:

Open the project directory in your editor (e.g., VS Code).
### Create a Virtual Environment:

Run the command: python -m venv <name_of_the_venv>.
### Activate the Virtual Environment:

On Windows: .\<name_of_the_venv>\Scripts\activate.
On macOS/Linux: source <name_of_the_venv>/bin/activate.
### Install the Required Packages:

Run: python -m pip install -r requirements.txt.
### Train Your Model:

Use the image_segmentation-ipynb.ipynb notebook to train your model via Kaggle.
### Add the Trained Model:
Add the trained model to your Project directory.

### Run the FastAPI App:

In the command prompt, run: uvicorn backend:app --reload.
### Run the Streamlit App:

Open a new command prompt and run: streamlit run frontend.py.
