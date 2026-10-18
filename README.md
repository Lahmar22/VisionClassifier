# Vision Classifier

An image classifier for six Intel scene categories: buildings, forest, glacier,
mountain, sea, and street.

## Run the dashboard

From the repository root, start the Streamlit app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

The dashboard loads Keras models from `models/` and class names from
`models/scene_classes.json`. Choose a model in the sidebar, upload a JPG, PNG,
or WebP image, and view the predicted scene with confidence scores for all
classes. The saved classifier expects RGB images resized to 150 by 150 pixels.