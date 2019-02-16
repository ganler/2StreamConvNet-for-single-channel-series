import numpy as np

motion_out = np.loadtxt('motion_out.txt')
spatial_out = np.loadtxt('spatial_out.txt')
label_out = np.loadtxt('label_out.txt')
fusion = motion_out+spatial_out

arg_mx = np.argmax(fusion, axis=0)
print(f">>> acc: {(arg_mx == label_out).mean()}")