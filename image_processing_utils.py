import cv2
import numpy as np
import blend_modes
from builtins import input

def apply_processing(frame):
    #show_wait_destroy("captured frame", frame)
    frame = blend(frame)
    #frame = brightness(frame)
    frame = adjust_gamma(frame)
    frame = cv2.addWeighted(frame, 0.5, frame, 0.5, 0.0)
    frame = desaturate_blues(frame)
    # show_wait_destroy("gamma corrected", frame)
    # frame = increase_contrast(frame)
    # frame = cv2.add(frame,np.array([-10.0]))
    #show_wait_destroy("contrast frame", frame)

    return frame

def desaturate_blues(img):
    scale = 255
    percent = 0.5
    #percent = 0.25
    #percent = 0

    # separate b,g,r
    b,g,r = cv2.split(img)
    b = b.astype(np.float32)
    g = g.astype(np.float32)
    r = r.astype(np.float32)

    # convert to cmyk
    # see 
    # https://stackoverflow.com/questions/14088375/how-can-i-convert-rgb-to-cmyk-and-vice-versa-in-python/41220097
    # https://www.codeproject.com/Articles/4488/XCmyk-CMYK-to-RGB-Calculator-with-source-code
    c = 1 - r / scale
    m = 1 - g / scale
    y = 1 - b / scale
    k = cv2.min(cv2.min(c, m),y)
    c = scale * (c - k) / (1 - k)
    m = scale * (m - k) / (1 - k)
    y = scale * (y - k) / (1 - k)

    # desaturate C,M,Y
    c = cv2.multiply(c, percent)
    y = cv2.multiply(y, percent)
    m = cv2.multiply(m, percent)

    # convert back to bgr
    r = scale * (1.0 - c / scale) * (1.0 - k)
    g = scale * (1.0 - m / scale) * (1.0 - k)
    b = scale * (1.0 - y / scale) * (1.0 - k)
    r = r.clip(0,255).astype(np.uint8)
    g = g.clip(0,255).astype(np.uint8)
    b = b.clip(0,255).astype(np.uint8)
    img_desat = cv2.merge([b,g,r])

    # cv2.imshow('img', img)
    # cv2.imshow('img_desat', img_desat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_desat


def blend(img):

    a = img.astype(float)/255  
    b = img.astype(float)/255 # make float on range 0-1

    mask = a >= 0.5 # generate boolean mask of everywhere a > 0.5 
    ab = np.zeros_like(a) # generate an output container for the blended image 

    # now do the blending 
    ab[~mask] = (2*a*b)[~mask] # 2ab everywhere a<0.5
    ab[mask] = (1-2*(1-a)*(1-b))[mask] # else this

    return  (ab*255).astype(np.uint8)

def brightness(img):
    alpha = 0.9 # Simple contrast control
    beta = -10   # Simple brightness control
    new_image = np.zeros(img.shape, img.dtype)
    new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    #show_wait_destroy("Contrast and brightness adjustment", new_image)
    return new_image

def adjust_gamma(image, gamma=0.75):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def increase_contrast(img):
    #show_wait_destroy("img",img) 

    #-----Converting image to LAB Color model----------------------------------- 
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    # show_wait_destroy('l_channel', l)
    # show_wait_destroy('a_channel', a)
    # show_wait_destroy('b_channel', b)
    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5,5))
    cl = clahe.apply(l)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
    ca = clahe.apply(a)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))

    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    #show_wait_destroy('final', final)
    return final