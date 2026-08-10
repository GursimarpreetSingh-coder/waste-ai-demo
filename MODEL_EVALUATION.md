# Model evaluation

To calculate real accuracy, precision, recall, F1 score, and a confusion matrix, use **unseen labelled images**. Do not use the same photos used to train the model.

Create this structure next to `evaluate_model.py`:

```text
test_data/
  dry/
    bottle.jpg
    paper.jpg
  wet/
    banana_peel.jpg
    food_scraps.jpg
```

Run:

```powershell
..\venv\Scripts\python.exe evaluate_model.py --data-dir test_data
```

The command prints the metrics and writes `evaluation_report.json`.

For a trustworthy result, aim for at least 100 unseen images per class. Include the lighting, backgrounds, camera angle, and waste items that will be used in the real app. Keep the two classes balanced.

`Accuracy` is the overall percentage of correct predictions. `Precision` measures how often a predicted class is correct. `Recall` measures how many real examples of that class were found. `F1 score` balances precision and recall; use macro F1 to judge whether both Dry and Wet are performing well.
