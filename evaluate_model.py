import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "waste_model_quant.tflite"
TEST_DIR = "test_data"

IMG_SIZE = 224

# Number of images to test from EACH class
IMAGES_PER_CLASS = 500

# ============================================================
# LOAD TFLITE MODEL
# ============================================================

print("Loading TFLite model...")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded!")

print("Input shape:", input_details[0]["shape"])
print("Input dtype:", input_details[0]["dtype"])
print("Output shape:", output_details[0]["shape"])
print("Output dtype:", output_details[0]["dtype"])

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    img = Image.open(image_path).convert("RGB")

    img = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img, dtype=np.float32)

    # Normalize to [-1, 1]
    img_array = (img_array / 127.5) - 1.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ============================================================
# PREDICTION
# ============================================================

def predict(image_path):

    input_data = preprocess_image(image_path)

    interpreter.set_tensor(
        input_details[0]["index"],
        input_data
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    # 0 = Dry Waste
    # 1 = Wet Waste

    pred_idx = np.argmax(output)

    return pred_idx


# ============================================================
# COLLECT FIRST 500 IMAGES FROM EACH CLASS
# ============================================================

true_labels = []
pred_labels = []

folders = ["dry", "wet"]

print("\n" + "=" * 60)
print("STARTING MODEL EVALUATION")
print("=" * 60)

print(f"Images per class : {IMAGES_PER_CLASS}")
print(f"Total images     : {IMAGES_PER_CLASS * 2}")

for idx, folder in enumerate(folders):

    folder_path = os.path.join(TEST_DIR, folder)

    if not os.path.exists(folder_path):

        print(f"ERROR: {folder_path} not found")
        continue

    # Get image files
    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(
            (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        )
    ]

    # Sort so the same first 500 are selected every time
    files.sort()

    # Take ONLY first 500
    files = files[:IMAGES_PER_CLASS]

    print(
        f"\n{folder.upper()} → "
        f"Testing {len(files)} images"
    )

    # ========================================================
    # RUN PREDICTIONS
    # ========================================================

    for image_number, img_file in enumerate(files, start=1):

        img_path = os.path.join(
            folder_path,
            img_file
        )

        try:

            pred = predict(img_path)

            # Ground truth
            true_labels.append(idx)

            # Model prediction
            pred_labels.append(pred)

            # Progress every 50 images
            if (
                image_number % 50 == 0
                or image_number == len(files)
            ):

                print(
                    f"  {image_number}/{len(files)} images processed"
                )

        except Exception as error:

            print(
                f"  SKIPPED: {img_file}"
                f" → {error}"
            )


# ============================================================
# CHECK RESULTS
# ============================================================

if len(true_labels) == 0:

    print("\nERROR: No images were successfully evaluated.")
    exit()


print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print(f"Images evaluated: {len(true_labels)}")


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    pred_labels
)

target_names = [
    "Dry Waste",
    "Wet Waste"
]

report = classification_report(
    true_labels,
    pred_labels,
    target_names=target_names,
    output_dict=True,
    zero_division=0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    f"\nOVERALL ACCURACY: "
    f"{accuracy * 100:.2f}%"
)

print("\nPer-Class Metrics:\n")

for class_name in target_names:

    precision = report[class_name]["precision"]
    recall = report[class_name]["recall"]
    f1 = report[class_name]["f1-score"]
    support = report[class_name]["support"]

    print(f"{class_name}")
    print(
        f"  Precision : {precision * 100:.2f}%"
    )
    print(
        f"  Recall    : {recall * 100:.2f}%"
    )
    print(
        f"  F1-Score  : {f1 * 100:.2f}%"
    )
    print(
        f"  Support   : {support}"
    )
    print()


# ============================================================
# MACRO F1
# ============================================================

macro_f1 = report["macro avg"]["f1-score"]

print(
    f"Macro F1-Score: "
    f"{macro_f1 * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    pred_labels
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names
)

plt.title(
    "Confusion Matrix - Waste Classification"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.close()

print(
    "\nSaved: confusion_matrix.png"
)


# ============================================================
# PERFORMANCE SUMMARY IMAGE
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 4)
)

ax.axis("tight")
ax.axis("off")

metrics_data = [

    [
        "Metric",
        "Dry Waste",
        "Wet Waste",
        "Overall"
    ],

    [
        "Accuracy",
        f"{accuracy * 100:.2f}%",
        f"{accuracy * 100:.2f}%",
        f"{accuracy * 100:.2f}%"
    ],

    [
        "Precision",
        f"{report['Dry Waste']['precision'] * 100:.2f}%",
        f"{report['Wet Waste']['precision'] * 100:.2f}%",
        "-"
    ],

    [
        "Recall",
        f"{report['Dry Waste']['recall'] * 100:.2f}%",
        f"{report['Wet Waste']['recall'] * 100:.2f}%",
        "-"
    ],

    [
        "F1-Score",
        f"{report['Dry Waste']['f1-score'] * 100:.2f}%",
        f"{report['Wet Waste']['f1-score'] * 100:.2f}%",
        "-"
    ],

    [
        "Macro F1",
        "-",
        "-",
        f"{macro_f1 * 100:.2f}%"
    ]
]


table = ax.table(
    cellText=metrics_data,
    loc="center",
    cellLoc="center",
    colWidths=[
        0.20,
        0.20,
        0.20,
        0.20
    ]
)

table.auto_set_font_size(False)

table.set_fontsize(12)

table.scale(
    1.2,
    1.5
)

plt.title(
    "WasteVision Model Performance"
)

plt.tight_layout()

plt.savefig(
    "metrics_summary.png",
    dpi=300
)

plt.close()

print(
    "Saved: metrics_summary.png"
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("🎉 EVALUATION FINISHED")
print("=" * 60)

print(
    f"Dry images tested : "
    f"{sum(label == 0 for label in true_labels)}"
)

print(
    f"Wet images tested : "
    f"{sum(label == 1 for label in true_labels)}"
)

print(
    f"Total tested      : "
    f"{len(true_labels)}"
)

print(
    f"Accuracy          : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Macro F1          : "
    f"{macro_f1 * 100:.2f}%"
)

print("\nGenerated files:")

print("  ✓ confusion_matrix.png")
print("  ✓ metrics_summary.png")

print("\nDone!")