# Rationalizing the Translation Elongation by Reinforcement Learning

# Environment
python 2.7      
pytorch 0.3.1       
numpy 1.15.2      
scipy 1.1.0     


# Note
The trained RiboRL model is in model/ folder. To reproduce the result in our paper, run the following code.

```bash
python runRiboRL.py  --cuda  --mode 1 --parallel --load model/best-model.pkl
```
and you will get the following results:
```bash
| reward -0.694 | mse 0.64586| corr 0.53031|
``` 

If you have any questions, please feel free to contact me :)    
Email: liuxg16@mails.tsinghua.edu.cn

