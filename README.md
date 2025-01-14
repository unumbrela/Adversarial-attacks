# 概述

真实图像经过攻击（增加扰动，如增加噪声等）以后，准确率会略微下降，而虚假图片（GAN等模型生成的）经过攻击以后，准确率下降较为明显。例如，本实验所用数据集为data文件夹，内含fake和real两个文件夹，其中各含有20种同类物品，共计8000张，经过训练以后，模型识别真实和假图片的准确率都很高（90%以上），而经过攻击以后，模型对于真实图片的准确率略微下降（80%左右

），而对于假图片的准确率显著降低（50%以下），本实验用进行六组小实验，涵盖PGD等常见攻击方法，均取得类似结果，以此证明：真图片的抗扰性强于虚假图片。





# 实验思路

总共分为六组实验，涵盖各种常见的攻击方法，包括PGD、FGSM、BIM、CW、DeepFool

以BIM攻击为例

首先，准备好data数据集，然后将其中的fake、real以及相应的生成对应的对抗攻击样本一起送入模型进行训练，这样就能提高模型的抗击打能力，如果只把fake、real送入训练，那么一经攻击，两者的准确率就都趋于0了，所以要把对应的对抗样本一起送入训练

然后，等到训练结束，便会显示攻击前模型对于真实和虚假图片的准确率（通常较高），随后调用攻击方法进行攻击，显示攻击后模型对于真假图片的准确率





# 实验结果

![image](https://github.com/user-attachments/assets/a0b8234f-0451-4094-b039-8c62640d93fb)



