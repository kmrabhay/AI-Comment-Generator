#!/usr/bin/env python
# coding: utf-8

# This notebook is an example on how to fine tune mT5 model with Higgingface Transformers to solve multilingual task in 101 lanaguges. This notebook especially takes the problem of question generation in hindi lanagues

# In[1]:


import torch
print(torch.cuda.empty_cache())


# In[2]:


import gc


# In[ ]:





# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('only-english-mt5-dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


pd.set_option("expand_frame_repr", False) # print cols side by side as it's supposed to be


# In[5]:


train = pd.read_csv("only-english-mt5-dataset/train.csv")
# train_path = "../input/only-english-mt5-dataset/train.csv"
# val_path = "../input/only-english-mt5-dataset/valid.csv"

train.shape


# In[ ]:





# In[6]:


# train["comment"].head(10).apply(isEnglish)
# 


# In[7]:


# train['comment'].head(10)


# In[8]:


# train = processEnglish(train)


# In[9]:


# train = train[train.nonEng<0.35]


# In[10]:


# train[train.nonEng<0.35]


# In[11]:


# del train['nonEng']


# In[ ]:





# In[12]:




import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


# In[ ]:





# In[ ]:






# In[16]:


from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)


# In[17]:


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# In[18]:


import pytorch_lightning as pl


# We'll be pytorch-lightning library for training. Most of the below code is adapted from here https://github.com/huggingface/transformers/blob/master/examples/lightning_base.py
# 
# The trainer is generic and can be used for any text-2-text task. You'll just need to change the dataset. Rest of the code will stay unchanged for all the tasks.
# 
# This is the most intresting and powrfull thing about the text-2-text format. You can fine-tune the model on variety of NLP tasks by just formulating the problem in text-2-text setting. No need to change hyperparameters, learning rate, optimizer or loss function. Just plug in your dataset and you are ready to go!

# In[19]:


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
#         self.hparams = hparams
        self.save_hyperparameters(hparams)
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_name_or_path)

    def is_logger(self):
        return True

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

#     def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
#         if self.trainer.use_tpu:
#             xm.optimizer_step(optimizer)
#         else:
#             optimizer.step()
#         optimizer.zero_grad()
#         self.lr_scheduler.step()

    def optimizer_step(self,
                     epoch=None,
                     batch_idx=None,
                     optimizer=None,
                     optimizer_idx=None,
                     optimizer_closure=None,
                     on_tpu=None,
                     using_native_amp=None,
                     using_lbfgs=None):
        optimizer.step() # remove 'closure=optimizer_closure' here
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True,
                                num_workers=4)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="valid", args=self.hparams)
        print("val data set: ", len(val_dataset))
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


# In[20]:


logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
        def on_validation_end(self, trainer, pl_module):
            logger.info("***** Validation results *****")
            if pl_module.is_logger():
                  metrics = trainer.callback_metrics
                  # Log results
                  for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                      logger.info("{} = {}\n".format(key, str(metrics[key])))

        def on_test_end(self, trainer, pl_module):
            logger.info("***** Test results *****")

            if pl_module.is_logger():
                metrics = trainer.callback_metrics

                  # Log and save results to file
                output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
                with open(output_test_results_file, "w") as writer:
                    for key in sorted(metrics):
                          if key not in ["log", "progress_bar"]:
                            logger.info("{} = {}\n".format(key, str(metrics[key])))
                            writer.write("{} = {}\n".format(key, str(metrics[key])))


# Let's define the hyperparameters and other arguments. You can overide this dict for specific task as needed. While in most of cases you'll only need to change the data_dirand output_dir.
# 
# Here the batch size is 8 and gradient_accumulation_steps are 8 so the effective batch size is 64

# In[21]:


args_dict = dict(
    data_dir="", # path for data files
    output_dir="", # path to save the checkpoints
    model_name_or_path='google/mt5-base',
    tokenizer_name_or_path='google/mt5-base',
    max_seq_length=512,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=16,
    eval_batch_size=16,
    num_train_epochs=2,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)



class CommentDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=30):
        self.path = os.path.join(data_dir, type_path + '.csv')

        self.title = 'title'
        self.comment = 'comment'
        self.category = 'category'
        print("----1")
        self.data = pd.read_csv(self.path)
        print("---2--", len(self.data))
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):
        for idx in range(len(self.data)):
            input_text_1,input_text_2,output_text= self.data.loc[idx, self.title], self.data.loc[idx, self.category],self.data.loc[idx, self.comment]
   
            input_ = "Title: %s Category: %s" % (input_text_1,input_text_2)
            target = "%s " %(output_text)

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=45, pad_to_max_length=True, return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=45, pad_to_max_length=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)


# In[24]:


tokenizer = AutoTokenizer.from_pretrained('google/mt5-base')


# In[25]:


dataset = CommentDataset(tokenizer, 'only-english-mt5-dataset', 'train', 30)
print("Val dataset: ",len(dataset))


# In[26]:


data = dataset[20]
print(tokenizer.decode(data['source_ids']))
print(tokenizer.decode(data['target_ids']))


# In[27]:


print(torch.__version__)


# In[28]:


print(pl.__version__)


# In[29]:


args_dict.update({'data_dir': 'only-english-mt5-dataset', 'output_dir': 'working/result', 'num_train_epochs':10,'max_seq_length':30})
args = argparse.Namespace(**args_dict)
print(args_dict)


# In[30]:


checkpoint_callback = pl.callbacks.ModelCheckpoint(
    period =1,filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=1
)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()],
)


# In[31]:


def get_dataset(tokenizer, type_path, args):
    return CommentDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,  max_len=args.max_seq_length)


# In[34]:


print ("Initialize model", args)
model = T5FineTuner(args)
trainer = pl.Trainer(**train_params, accelerator="gpu")


# In[35]:


import torch
print(torch.cuda.device_count())
torch.cuda.device_count() 


# In[ ]:


# torch.cuda.memory_summary(device=None, abbreviated=False)


# In[44]:


# import torch
# torch.cuda.empty_cache()


# In[36]:


# !pip install GPUtil

# import torch
# from GPUtil import showUtilization as gpu_usage
# from numba import cuda

# def free_gpu_cache():
#     print("Initial GPU Usage")
#     gpu_usage()                             

#     torch.cuda.empty_cache()

#     cuda.select_device(0)
#     cuda.close()
#     cuda.select_device(0)

#     print("GPU Usage after emptying the cache")
#     gpu_usage()

# free_gpu_cache()                           


# In[39]:


print (" Training model")

trainer.fit(model)

print ("training finished")

print ("Saving model")

model.model.save_pretrained("/kaggle/working/result")

print ("Saved model")


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('ls result')


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


get_ipython().system('cp -r /kaggle/working/result/pytorch_model.bin /kaggle/working/')
get_ipython().system('cp -r /kaggle/working/result/config.json /kaggle/working/')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('rm -rf result')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


import os   
cwd = os.getcwd()   
print("Current working directory:") 
print(cwd)   


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


get_ipython().system('nvidia-smi')


# In[48]:


get_ipython().system('pip freeze')


# In[ ]:





# In[ ]:




