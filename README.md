# How to run this code?
1. Download Dataset & copy your dataset to your Project Directory
2. Open this project as root in Python IDE
3. Install requirements.txt```pip install -r requirements.txt```
4. Uncomment this conntent

![img.png](Documentation/img.png)

5. execute ```main.py``` to run the augmented images



# Link Dataset
[click to Download](https://drive.google.com/file/d/1HAeCIBUdCby4vkyT4E5L2yqxG_tiJx50/view?usp=sharing)

if you have downloaded it, you will see the directory arrangement like this :

![img.png](Documentation/img1.png)

### Augmented Images Test
#### Found 1016 validated image filenames.

Test: accuracy = 0.833333  ;  loss = 0.429308
- Precision:  0.945823927765237
- Recall:  0.945823927765237
- array ([[549,  24],[ 24, 419]])

![lossAugmented](parameter/lossAugmented.png)
![accAugmented](parameter/accAugmented.png)
![CM-augmented](parameter/CM-augmented.png)
![ROC-augmented.png](parameter/ROC-augmented.png)


### Normal Image Test
Test: accuracy = 0.812500  ;  loss = 0.488956 
- array([[169,  16],[ 17,  98]])
- Precision:  0.8596491228070176 
- Recall:  0.8521739130434782

![lossNormal](parameter/lossNormal.png)
![accNormal](parameter/accNormal.png)
![CM-normal.png](parameter/CM-normal.png)
![ROC-normal.png](parameter/ROC-normal.png)