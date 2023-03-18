#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
from paddleocr import PPStructure,draw_structure_result,save_structure_res
from pdf2image import convert_from_path
poppler_path = r"C:\Users\samee\PycharmProjects\pdftocsv\poppler-23.01.0\Library\bin" # as an argument in convert_from_path.


# In[ ]:


images = convert_from_path('test.pdf', poppler_path=poppler_path)


# In[ ]:


get_ipython().system('mkdir pages')


# In[ ]:


for i in range(len(images)):
  images[i].save('pages/page'+str(i)+'.jpg', 'JPEG')


# In[ ]:


import os
import cv2
from paddleocr import PPStructure,draw_structure_result,save_structure_res

table_engine = PPStructure(show_log=True, image_orientation=True, lang='ch', 
                           det_model_dir=r'C:\Users\samee\PycharmProjects\pdftocsv\ch_PP-OCRv3_det_infer',
                          image_dir='output')

save_folder = 'output'
img_path = 'pages\page42.jpg'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder,os.path.basename(img_path).split('.')[0])

