#transfer learning with huggingface transformers
#here, i am going to use a pretrained computer vision model and do trasnfer learning to apply its pretraining onto my dataset by freezing the prior layers then adding some new ones to the end of the model which will be trained via backprop to customize it to my use case

from datasets import load_dataset
from transformers import AutoModelForImageClassification, AutoModel, AutoImageProcessor
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

dataset_name = "mmenendezg/pneumonia_x_ray"  # which HF dataset are we using?
ds = load_dataset(dataset_name)

# TODO: subsample train/test if the dataset is large (we discussed why)
train_ds = ds["train"]  # .shuffle(seed=42).select(range(___)) if needed
test_ds = ds["test"]    # same here

# --- 2. Load pretrained model + processor (frozen backbone) ---
model_name = "microsoft/resnet-18"  # which pretrained vision model?
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name) #AutoModel (what we used) — loads only the backbone, and explicitly does not attach any task-specific head
model.eval()  # TODO: why do we call .eval() here instead of .train()?
print(model)

#we want backbone, aka non head (head is one final layer for classifcation which we are going to replace with our own classifier)
# --- 3. Freeze all backbone parameters ---
for param in model.parameters():
    param.requires_grad = False  # False since its wasteful for pytroch to track the gradients on frozen backbone, we arent even computing gradeints for them becausd ethey wont be updated

# --- 4. Function to extract embeddings ---
#you can print this function to see the entire model structure aka what layers and what types
def get_embedding(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # TODO: which output field gives you a single embedding vector per image?
    # hint: look at outputs.last_hidden_state shape vs outputs.pooler_output
    embedding = outputs.pooler_output
    return embedding.squeeze().numpy()

# --- 5. Extract embeddings for train/test sets ---
X_train = [get_embedding(ex["image"]) for ex in train_ds]
y_train = [ex["label"] for ex in train_ds]

X_test = [get_embedding(ex["image"]) for ex in test_ds]
y_test = [ex["label"] for ex in test_ds]

# --- 6. Train a simple classifier head on frozen embeddings ---
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
##i chose a logisitc rgeression classifier because it is a simple linear classifier for binar classifcation
#also, the original resent model had a final ilnear layer, logisitc regression is also a linear classifier
#alsi i wanted to see if the logisitc regression performs well: if it does, that means our data is linearly separable
# --- 7. Evaluate ---
preds = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
