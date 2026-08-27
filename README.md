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


why fine tune? because it improves your model performance from just using what it was pretrained with
