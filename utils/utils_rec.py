import numpy as np

def consistency_layer(img1,img2,mask):
    shape1=img1.shape;shape2=img2.shape
    img1=img1.squeeze();img2=img2.squeeze()
    orif1 = np.fft.fft2(img1)
    orifshift1 = np.fft.fftshift(orif1)
    
    orif2 = np.fft.fft2(img2)
    orifshift2 = np.fft.fftshift(orif2)
    
    orifshift2[mask>0]=orifshift1[mask>0]
    
    theishift = np.fft.ifftshift(orifshift2)
    iimg = np.fft.ifft2(theishift)
    iimg = np.abs(iimg)
    return iimg.reshape(shape1)