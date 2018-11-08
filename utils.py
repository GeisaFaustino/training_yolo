import cv2
import matplotlib.pyplot as plt
import numpy as np


'''
Read image
'''
def read_image( image_path ):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, None, fx=0.3, fy=0.3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img



'''
Drawing boundingboxes according to predicted results. 
Note that only boundingboxes with confidence grather than 0.4 are drawn.
'''
def drawing_boundingboxes( image, results ):
    
    confidence_str = ''
    for each_result in results:
        tl = (each_result['topleft']['x'], each_result['topleft']['y'])
        br = (each_result['bottomright']['x'], each_result['bottomright']['y'])
        confidence = each_result['confidence']
        confidence_str = str( round(confidence, 3) )
        label = each_result['label'] + " " + confidence_str

        if confidence > 0.4:
            # add the box and label and display it
            image = cv2.rectangle(image, tl, br, (0, 255, 0), 7)
            image = cv2.putText(image, label, tl, cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)
                
    return image, confidence_str



'''
Rendering images and corresponding confidence 
'''  
def rendering_images(images_and_info):   

    for i in range( 0, len(images_and_info) ):
        image, confidence = drawing_boundingboxes( images_and_info[i][0], images_and_info[i][1] ) 
        plt.figure( figsize = (7,7) )
        plt.axis('off')
        plt.title( confidence )
        plt.imshow( image )
        plt.show()
        
        
'''
Rendering images and corresponding confidence in a grid
'''     
def rendering_images_grid(images_and_info):
    n = 6
    m = len(images_and_info)//n
    f, axes = plt.subplots( m, n, figsize = ( 32, 32 ) )
    for i in range(m):
        for j in range(n):
            indice = i * n + j 
            if indice < len(images_and_info):
                image, confidence = drawing_boundingboxes( images_and_info[indice][0], 
                                                           images_and_info[indice][1] ) 
                axes[i][j].imshow( image )
                axes[i][j].set_title(confidence)
                axes[i][j].axis( "off" )