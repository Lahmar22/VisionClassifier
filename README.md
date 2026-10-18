# Vision Classifier

An image classifier for six Intel scene categories: buildings, forest, glacier,
mountain, sea, and street.

## Run with Docker Compose

Build the image and start the Streamlit dashboard from the repository root:

```sh
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501). Stop the dashboard with
`Ctrl+C`, or run `docker compose down` in another terminal.

The image includes the model in `models/` and its class names file. The training
dataset and notebook are not included in the image.

## Run locally

Create and activate a virtual environment, then install the dependencies:

```sh
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Start the Streamlit app from the repository root:

```sh
python -m streamlit run dashboard/app.py
```

The dashboard loads Keras models from `models/` and class names from
`models/scene_classes.json`. Choose a model in the sidebar, upload a JPG, PNG,
or WebP image, and view the predicted scene with confidence scores for all
classes. The saved classifier expects RGB images resized to 150 by 150 pixels.
