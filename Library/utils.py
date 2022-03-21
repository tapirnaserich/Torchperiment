import torch
import cv2
import torchvision.transforms as T
import numpy as np
import imutils
import csv
import pandas as pd
import json
import dill

def serialize_dill_to_disk(data, path):
   serialized = dill.dumps(data)
   dill_file = open(path, "wb")
   dill_file.write(dill.dumps(serialized))
   dill_file.close()

def deserialize_dill_from_disk(path):
   dill_read = open(path, "rb")
   bytes = dill.load(dill_read)
   bytes_deserialized = dill.loads(bytes)
   dill_read.close()
   return bytes_deserialized

def serialize_dict_to_disk(data, path):
   file = open(path, "w")
   json.dump(data, file, indent=4)

def deserialize_dict_from_disk(path):
   file = open(path, "r")
   return json.load(file)


def get_dict_list_from_df(df):
   column_names = df.columns.values
   all_data = []
   for index, row in df.iterrows():
      row_dict = {}
      for n in column_names:
         row_dict[n] = df[n][index]

      all_data.append(row_dict)

   return all_data if len(all_data) > 1 else all_data[0]

def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

def store_list_to_csv(data, path):
   file = open(path, 'w')

   for d in data:
      file.write(f'{d}\n')

   file.close()

def get_list_from_csv(path, fn=None):
   file = open(path, newline='')
   reader = csv.reader(file)
   data = list(reader)
   file.close()
   if not fn is None:
      data = list(map(fn, data))

   return data

def show_image(images, labels, mean, std, target_size):
   mean = torch.FloatTensor(mean)
   std = torch.FloatTensor(std)
   for i in range(images.shape[0]):
      image = images[i]
      label = labels[i]
      image = T.Lambda(lambda x: x.repeat(3,1,1))(image)
      image = image.permute(1,2,0)
      image = image * std + mean
      image = np.clip(image, 0, 1)

      out = np.array(image * 255,dtype= np.uint8)
      out = imutils.resize(out, width=target_size[0], height=target_size[1])

      cv2.imshow(f"{label}", out)
      cv2.waitKey()

