#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
arr1 = arr.reshape(3,3)
print ('After reshaping having dimension 4x2:')
print (arr1)
print ('\n')


# In[36]:


import numpy as np
Assign_2 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
Assign_2 [Assign_2 % 2 == 1] = -1
print(Assign_2)


# In[37]:


x = np.array([21, 64, 86, 22, 74, 55, 81, 79, 90, 89])
y = np.array([21, 7, 3, 45, 10, 29, 55, 4, 37, 18])


# In[40]:


import numpy as np
x = np.array([21, 64, 86, 22, 74, 55, 81, 79, 90, 89])
y = np.array([21, 7, 3, 45, 10, 29, 55, 4, 37, 18])
print(np.where(x > y))
print(np.where(x == y))


# In[41]:


Assign_4= np.arange(100).reshape(5,-1)


# In[45]:


import numpy as np
Assign_4= np.arange(100).reshape(5,-1)
print("The original array: \n", Assign_4,"\n")
print("The extracted array:")
print(Assign_4 [:,:4])


# In[46]:


import numpy as np
Assign_5 = np.random.randint(30, 41, size = (10))
print(Assign_5)


# In[50]:


import numpy as np
A=np.array(((1,2,3),(4,5,6),(7,8,9)))
B = np.array(((7,8,10),(4,5,6),(1,2,3)))
print('MATRIX A')
print(A)
print('MATRIX B')
print(B)


# In[51]:


print('Addtion of matrix A and B')
C=np.add(A,B)
print(C)
print('Substraction of matrix A and B')
D=np.subtract(A,B)
print(D)


# In[52]:


print('sum of each elements of matrix A')
print(np.sum(A)) 
print('sum of each column of matrix B')
print(np.sum(B, axis=0))
print('sum of each row of matix C')
print(np.sum(C, axis=1))


# In[53]:


print('The product of two matrices A and B')
E=np.dot(A,B) 
print(E)


# In[55]:


print('sorting of matrix E')
print (np.sort(E))


# In[56]:


print('Transpose of matrix E')
print (np.transpose(E))


# In[ ]:




