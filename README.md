# Anime_Generator<br>
A GAN used to genrate anime images.<br>

## Large GAN with resnet module<br>
performs worse than smaller sized model in similar structure.<br>

## WGAN-GP<br>
Conditional WGAN-GP works for mnist but not anime images.<br>
Normal WGAN-GP generates unclear anime images.<br>
![image](https://github.com/WenjieLuo2333/Anime_Generator/blob/master/WGAN-gp/WGAN-gp.png)<br>
Method to conditional :<br> G_input = multiply(noise,embedding_label)<br> D_input = multiply(flatten image,embedding_label)<br>
Way too easy,not work for complex images.<br>

## ACGAN with Resnet<br>
The result shows that the image at the same location generally has a same color of hair. It shows that ACGAN works to control the output.<br>
![image](https://github.com/WenjieLuo2333/Anime_Generator/blob/master/Res_ACGAN_Large/20400.png)
![image](https://github.com/WenjieLuo2333/Anime_Generator/blob/master/Res_ACGAN_Large/20600.png)
![image](https://github.com/WenjieLuo2333/Anime_Generator/blob/master/Res_ACGAN_Large/20800.png)<br>
But the images is not clear. More works need to be done.

## DRAGAN with out label<br>
Genrate Model trained. Controllable Version to be done.<br>

For Res_DRAGAN the result is generally good.<br>
![image](https://github.com/WenjieLuo2333/Anime_Generator/blob/master/Res_DRAGAN/Predict_2.png)<br>
And for Interpolation it seems that the entries of noise is realated to the feature of pictures.<br>
![image](https://github.com/WenjieLuo2333/Anime_Generator/blob/master/Res_DRAGAN/inter_2.png)<br>


## DRAGAN with out label<br>
run ```get_by_name``` function in ```/acgan/api_example.py``` to run the conditional generating demo.
