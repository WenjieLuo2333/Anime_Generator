A WGAN-GP modified from https://github.com/eriklindernoren/Keras-GAN is used to generate images.<br>
I modified the layers of models and change the size to fit the data set.<br>
It works fine on Mnist but not that ok on Anime dataset.<br>
Maybe it's because of the sequential structure.<br>

WGAN-gp can generate unclear images but fail to generate images with good resolution.<br>
![image](https://github.com/WenjieLuo2333/Anime_Generator/blob/master/WGAN-gp/WGAN-gp.png)<br>
