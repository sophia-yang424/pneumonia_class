This is a small app made by applying linear probing transfer learning evaluation on Resnet-18 Image classification CNN. In the Gradio app, you can flag noteworthy cases and save as csv to further train the model's head. New model head is a logistic regression classifier built for this use case.

I decided to start with linear probing, to decide on whether transferring learning or more expensive fine tuning was needed. My probed model received the following scores:
Accuracy: 0.8814102564102564
              precision    recall  f1-score   support

           0       0.99      0.69      0.81       234
           1       0.84      0.99      0.91       390
It appeared that the simpler linear probing approach had lopsided results in which only 69% of true negative labels were recalled, indicating ResNet 18 has mixed results on producing linearly separable embeddings for our images, which allows a simpler linear classifier head like logisti regression to perform well. The linear separability of the ResNet-18's backbone outputs is significant because it indicates whether of the complexity in the image data has been cleanly distinguished and separated such that embeddings of the same class largely sit in the same region across a linear boundary (though my LR head isnt learning any new things to characterize the data with, it is learning the boundary line itself to seperate those prefound directions in the data which are emboided in the embeddings, my LR head attempts to separate those embeddings in the multi dimensional space. The embeddings themslves are not human interpretable let alone cleanly packaged into binary categories, so my LR head separates them into binary categories. If it can successfully do so then that means the backbone/ResNet 18 performs sufficiently on our data.). In other words, this helps us determine if a more coputationalyl expensive approach like fine tuning the backbone is needed, since this can tell us how well ResNet as is performs without modifications in its backbone.


I preprocess the image data in the same way that the Resnet model does to ensure compatibility (under the hood the hugging face transformer library will resize the images and apply other needed image transformations exactly how resnet's own pretraining data did). I then allow the preprocessed image data to complete a forward pass (just call torch's model() function for each image iteratively) in the Resnet18 backbone to get the embeddings that would have been passed to the final linear classifier head in Resnet. Then I feed those into my logistic regression head which is trained on these embedding inputs. 
I do the same for the test data, then make my trained logistic regression classify those test image data imbeddings.

I then attempted to use the class_weight="balanced" parameter when defining my logistic regression head -- it produced the following results:
Accuracy: 0.8958333333333334
              precision    recall  f1-score   support

           0       0.98      0.74      0.84       234
           1       0.86      0.99      0.92       390

It seems that the recall of the negative class improved, but is still lacking compared to the positive class. This mens the pretraining for resNet is not sufficient for our classification task as it cannot recognize which cases are negative cases sufficiently. Thus my next step will be to examine if fine tuning produces better results.

the grad settings are turned off, since I froze the backbone, although I do not call backward() and step() to do backprop anywhere so no training/gradients/updates are made. This disabling of grad is just to make sure no memory is waste don pytorch making computation graphs which it does by default. I also loaded the backbone specifically, so excluding the head with HuggingFace AutoModel function which does that for whatever hugging face model you specify


why fine tune? because it improves your model performance from just using what it was pretrained with. Fine tuning because of the pretrained backbone not being able to recall normal class as well (predicts penuonia too much. The pretarined backbone finds the embeddings then we train a LR head to classify those as pneumonia/not pneumonia, the embeddings they did didnt separate the classes well enough to fit a good line to separate since LR is linear classifier). So i resolved this by fine tuning with a 50-50 split (800 pneumonia then using pytorch's random augmenter I augmented from only 200 normal examples to be 800) that way the model's backbone will be able to learn evenly how to make good linearly separable embeddings for both classes, rather than just being biased to say everything is pneumonia (class 1)

random seed based ensembling bc originally it was non deterministic


Fine tuning, used optuna to vary:
-learning rate (log sampling, to account for how linear sampling will give disproportionality certain regions with more widht. We do log sampling such that we sample float values for x from -6 to -4.398. derivation: log10(1e-6) = -6
log10(4e-5) = log10(4 * 10^-5) = log10(4) + log10(10^-5) ≈ 0.602 - 5 = -4.398 So, in the base-10 log-space, the range is roughly [-6, -4.398]. ) on range 1e^-6 to 4e^-5
-epochs from 2 to 3
-weight decay (for regularization to prevent big weights/overfitting): 0.075 to 0.2
-batch size: 8,16,32,64 (mini batch size which is how many examples used per update, 1600 training examples total)
-dropout: 0.1, 0.5
--started from bigger range to smaller based on the best performing trails of hyperparameters selected

fine tuned, using model 7:
hyperparamters: 'learning_rate': 3.713408826464528e-05, 'num_train_epochs': 3, 'weight_decay': 0.09730862815310555, 'per_device_train_batch_size': 16, seed = 42

Epoch	Train Loss	Val Loss	Accuracy	F1
1	0.155	0.457	0.815	0.803
2	0.083	0.236	0.925	0.924
3	0.033	0.345	0.905	0.903

--- Detailed Classification Report for Trial 7 Model ---
              precision    recall  f1-score   support

      normal       0.99      0.79      0.87        84
   pneumonia       0.86      0.99      0.92       116

    accuracy                           0.91       200
   macro avg       0.92      0.89      0.90       200
weighted avg       0.92      0.91      0.90       200

decide not to prune since my epoch number is low anyways (2-3) since we are fine tuning

load_best_model_at_end=True, each trial we keep the best epoch for accuracy not last epoch
I used optuna for hyperparamters, then did seed ensembling with following seeds: [42, 101, 202, 303, 404]

Loading model from /content/drive/MyDrive/my_ml_models/fine_tuned_pneumonia_model_best_hp_ensemble/model_seed_42/checkpoint-150...
seed 42 for optuna trial 7's hyperparamters

--- Classification Report for: checkpoint-150 ---
              precision    recall  f1-score   support

      normal       0.99      0.85      0.91        84
   pneumonia       0.90      0.99      0.94       116

    accuracy                           0.93       200
   macro avg       0.94      0.92      0.93       200
weighted avg       0.94      0.93      0.93       200



To run app:
run the mounting cell st notebook can access google drive to retrieve the model
run the gradio app code cell

Final model in : /content/drive/MyDrive/my_ml_models/fine_tuned_pneumonia_model_best_hp_ensemble/model_seed_42

Didnt do LoRa because im fine tuning a CNN which is small enough to where fine tuning isnt too costly, but when I do an NLP project LoRa is probably better because more parameters
