# Image_deionising_auto_encoder
This projects tries to replicate a popular technique known as Image deionising .i.e noise removal from images.It uses a convolutional autoencoder which is able to remove noise from the image.
# Autoencoders
Autoencoders is an unsupervised learning technique which learns pixel to pixel mapping. It consists of an encoder-decoder network, out of which encoder tries to compress the image to extract usefull features and the decoder reconstructs the image.
The problem with autoencoders is that, the quality of output image is not very detailed. The reasons for the image being not very detailed is that the we are forcing the encoder(the last layers) to learn features of the entire image, the decoder uses this encoding to reconstruct the image. As the features provided were less so is the quality. 
A very good tutorial for <a href="https://www.youtube.com/watch?v=9zKuYvjFFS8&t=728s">autoencoders</a>
# How Does the approach work
I tool my stone paper scissor game dataset(Although I have provided the dataset in this repo but you can have a look at my "game of stone paper scissors vs man" if you want to know how the data was generated) for training the model. We take those image resize them and add noise to them, the image containg noise is fed as input and the same images without noise are used for y_labels.<br>
We can add noise to the images using opencv<br>
For loading, resizing, saving of images I used opencv<br>
I have used keras for training the autoencoder model.
# Have a look at the input and ouput yourself
![scissor](https://user-images.githubusercontent.com/24778913/45904332-5b914e80-be0a-11e8-9b47-c53eda638f92.png)
![paper](https://user-images.githubusercontent.com/24778913/45904370-6fd54b80-be0a-11e8-8596-f517879b74c4.png)
![stone](https://user-images.githubusercontent.com/24778913/45904389-84b1df00-be0a-11e8-8923-91d1934b17ae.png)

## Requirements for the project
0. Python 3.x
1. <a href="https://tensorflow.org">Tensorflow 1.5</a>
2. <a href="https://keras.io">Keras</a>
3. OpenCV 3.4(for loading,resizing images)
4. h5py(for saving trained model)
5. pyttsx3
6. A good grasp over convolutional neural networks. For online resources refer to standford cs231n, deeplearning.ai on coursera or cs231n by standford university
7. A good CPU (preferably with a GPU).
8. Time
9. datetime
10. Patience.... A lot of it.

## Installing the requirements
1. Start your terminal of cmd depending on your os.
  2. If you have a NVidia GPU then make sure you have the prerequisites for Tensorflow GPU installation (Refer to official site). Then use this commmand

    pip install -r requirements_gpu.txt

  3. In case you do not have a GPU then use this command

    pip install -r requirements_cpu.txt
## steps to run the repo 
1)Clone the repo<br>
2)Extract the data folders<br>
3)Install the requirements<br>
4)You can change the name of test image in runner.py<br>
5)Run "runner.py"

## Liked it
If you liked it you will surely like my other repos as well. You can also have a look at my youtube channel <a href="https://www.youtube.com/c/reactorscience">"reactor science"</a>. If you have any doubts you can contact me on my facebook page <a href="https://www.facebook.com/pg/reactorscience/about/">"reactor science"</a>

## References
1)Deep learning with python by Francois Chollet<br>
2)keras.io<br>
3)Deeplearning.ai by coursera(prof Andrew Ng)<br>
4)CS231n by stanford<br>
5)Pyimagesearch.com(Adrian Rosenberg)

