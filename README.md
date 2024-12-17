## Jaffa-Sparrow
A tiny language model trained in native tamil. The model was trained on multilingual c4 (mc4), news articles, tiny books, wikipedia datasets. Training was done on 2 x 4060ti gpu (Single host multi device strategy) with data parallelism. The model was later finetuned on tamil converstions datasets to be able to carry out multi-turn conversations.

### Multi-GPU training 
Uses distributed training (single host, multi-device training). Trained on 4060ti x 2 gpus with 16gb vram each. Kindly modify those line as per your requirement

### Generated text after 4 epochs
![Screenshot from 2024-12-18 00-08-55](https://github.com/user-attachments/assets/5435cd9e-4ea3-4fd3-9fc9-446678e2727d)

