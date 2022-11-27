# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 21:58:27 2022

@author: Steven Nuñez Murillo - B95614
"""

import numpy as np
import PIL
from random import sample
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import random

def load_image(filename, resize):
    img = np.array(PIL.Image.open(filename).resize(resize).convert('RGB')).astype(np.float32)
    return img

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1-p2)
    
    
def manhattan_distance(p1, p2):
    return np.sum(np.abs(p1-p2))
    
    
def nearest_centroid(point, centroids, distance):
    
    distances = []
    
    for i in centroids:
        
        distances.append(distance(point, i))
    
    idx_centroid = np.argmin(distances)
    
    return (idx_centroid, distances[idx_centroid])
    

def Average(lst):
    
    suma = 0
        
    for i in lst:
        
        suma += i[0]
        
    return suma / len(lst)

def sumDistances(lst):
    suma = 0
    
    for i in lst:
        suma += i[1]
        
    return suma

def parcial_Flatten(lst):
    return lst.reshape(lst.shape[0]**2, lst.shape[2])
    
    
def lloyd(data, k, iters, types, distance):
    
    #Inicialización de parámetros
    centroids =  []
    errors = []
     
    if distance == "manhattan_distance":
            distance = manhattan_distance
    else:
        distance = euclidean_distance
    
    #Sacando los indices a los cuales pertenecen mis centroides inciales, generados de manera aleatoria. 
    rand_center = sample(range(0,len(data)),k)
    
    #Arreglo de centroides inciales.
    for i in rand_center:
        centroids.append(data[i])
        
    
    while(iters > 0):
        
        print(iters)
        
        distance_per_centroid = 0
        points_centroids = []
        
        #Arreglo correspondiente a el arreglo de puntos de cada centroide
        for i in range(k):
            points_centroids.append([])
        temp_distances = []


        for point in data:
            idx_centroid, distanceh = nearest_centroid(point, centroids, distance)

            points_centroids[idx_centroid].append((point, distanceh))
            
        for i, center in enumerate(points_centroids):
            
            if types == "means":
                
                centroids[i] = Average(center)
                
                distance_per_centroid += sumDistances(center)
                
            else:
                
                
                #Alternativa con random
                
                rand_index = random.randint(0, len(center))
                suma = 0
                for point_term in center:
                    suma += distance(center[rand_index][0], point_term[0])
                
                current_error = sumDistances(center)
                if current_error > suma:
                    centroids[i] = center[rand_index][0]
                    distance_per_centroid += suma
                else:
                    distance_per_centroid += current_error
                
                
                #Alternatica con el error más bajo de los puntos
                """
                temp_distances = []
                print("Problema creando")
                for point in center:
                    suma = 0
                    for point_tem in center:
                        
                        #print("pase")
                        suma += distance(point_tem[0], point[0])
                        print("sume")

                        #print("si pase")
                        
                    temp_distances.append(suma)
                
                min_value = min(temp_distances)
                index = temp_distances.index(min_value)
                
                
                centroids[i] = center[index][0]
                
                distance_per_centroid += temp_distances[index]
                """
        
        errors = distance_per_centroid
        iters -= 1
            
    return centroids , errors


def merge(im1, im2):
    w = im1.size[0] + im2.size[0]
    h = max(im1.size[1], im2.size[1])
    im = Image.new("RGB", (w, h))

    im.paste(im1)
    im.paste(im2, (im1.size[0], 0))

    return im


def add_square_to(im, topleft, size, colour):
    draw = ImageDraw.Draw(im)
    draw.rectangle((topleft, (topleft[0] + size, topleft[1] + size)), fill=colour)
    return im


def main():
    images_used = ["Future0.jpg", "Future1.jpg", "Future2.jpg", "Future3.jpg"]
    
    types = ["mediods"]
    
    distances = ["euclidean", "manhattan"]

    num_pick = 1
    for photo in images_used:
        for tp in types:
            for dist in distances:
                                
                data = load_image(photo, (256, 256))

                data_array = parcial_Flatten(data)
                
                palettes = []
                errors = []
                
                for current_test in range(0,3):
                    
                    color_palette = lloyd(data_array, 5, 5, tp, dist)
                    palettes.append(color_palette[0])
                    errors.append(color_palette[1])
                
                
                min_index = np.argmin(errors)
                best_solution = palettes[min_index]
                                
                fondo = Image.new('RGB', (35, 256), (128, 128, 128))
                
                fotoi = Image.fromarray(np.uint8(data)).convert('RGB')
                                                      
                pos = 1
                
                
                for i in best_solution:
                    fondo = add_square_to(fondo, ((9, 20*pos)), 15, (int(i[0]), int(i[1]), int(i[2])))
                    pos+=1
                
                
                fondo = merge(fondo, fotoi)
                
                draw = ImageDraw.Draw(fondo)
                font = ImageFont.truetype("arial.ttf", 15)
                draw.text((20, 20), f"Error del clustering: {errors[min_index]}", font=font, fill="white")
                
                fondo.save(f'Final{num_pick}-{tp}-{dist}.jpg', quality=95)
                
        num_pick += 1


main()