import sys
from numpy import random
import numpy as np

def saveResults(r1,r2):
    np.savetxt("Tabular Form.txt", r1,fmt='%1.4f')
    np.savetxt("Function Approximation.txt", r1,fmt='%1.4f')



##Define policies used in the examples
world_4_3 = np.array([['r', 'r', 'r','1'], 
            ['u', 'b', 'u','-1'],
            ['u', 'l', 'l','l']])

world_10_10_b = np.array([
            ['r', 'r', 'r','r','r', 'r', 'r','r','r', '1'], 
            ['r', 'r', 'r','r','r', 'r', 'r','r','u', 'u'],
            ['r', 'r', 'r','r','r', 'r', 'r','u','u', 'u'],
            ['r', 'r', 'r','r','r', 'r', 'u','u','u', 'u'],
            ['r', 'r', 'r','r','r', 'u', 'u','u','u', 'u'],
            ['r', 'r', 'r','r','u', 'u', 'u','u','u', 'u'],
            ['r', 'r', 'r','u','u', 'u', 'u','u','u', 'u'],
            ['r', 'r', 'u','u','u', 'u', 'u','u','u', 'u'],
            ['r', 'u', 'u','u','u', 'u', 'u','u','u', 'u'],
            ['u', 'u', 'u','u','u', 'u', 'u','u','u', 'u'] ])

world_10_10_c = np.array([
            ['d', 'd', 'd','d','d', 'd', 'd','d','d', 'd'], 
            ['r', 'd', 'd','d','d', 'd', 'd','d','d', 'l'],
            ['r', 'r', 'd','d','d', 'd', 'd','d','l', 'l'],
            ['r', 'r', 'r','d','d', 'd', 'd','l','l', 'l'],
            ['r', 'r', 'r','r','1', 'l', 'l','l','l', 'l'],
            ['r', 'r', 'r','r','u', 'u', 'l','l','l', 'l'],
            ['r', 'r', 'r','u','u', 'u', 'u','l','l', 'l'],
            ['r', 'r', 'u','u','u', 'u', 'u','u','l', 'l'],
            ['r', 'u', 'u','u','u', 'u', 'u','u','u', 'l'],
            ['u', 'u', 'u','u','u', 'u', 'u','u','u', 'u'] ])
#define as global vars to be used with the tabular form
values = np.zeros_like(world_4_3, dtype = float)
count = np.zeros_like(world_4_3, dtype = int)

# utility func for getting next state
#take care world takes y,x not x,y
def nextState(x,y,world):
    y = (len(world)-1) - y
    #print("world in next state = ",world[y][x])
    if world[y][x] == 'u':
        policy = (x,y-1)
        per1 = (x+1,y)
        per2 = (x-1,y)
    elif world[y][x] == 'd':
        policy = (x,y+1)
        per1 = (x+1,y)
        per2 = (x-1,y)
    elif world[y][x] == 'r':
        policy = (x+1,y)
        per1 = (x,y+1)
        per2 = (x,y-1)
    elif world[y][x] == 'l':
        policy = (x-1,y)
        per1 = (x,y+1)
        per2 = (x,y-1)
    exit = False
    while(not exit):
        exit = True
        x = random.uniform(0,1)
        if x <= .8:
            final = policy
        elif x <=.9:
            final = per1
        else:
            final = per2
        if final[0] < 0 or final[0] > (len(world[0])-1):
            exit = False
            continue
        if final[1] < 0 or final[1] > (len(world)-1):
            exit = False
            continue
        if world[final[1]][final[0]] == 'b':
            exit = False
    return final[0], (len(world)-1) - final[1]

#tabular method
def directUtility(world, itr):
    global values
    global count
    values = np.zeros_like(world, dtype = float)
    count = np.ones_like(world, dtype = int)
    for i in range(itr):
        exit = False
        while(not exit):
            exit = True
            x = random.randint(0,len(world[0]))
            y = random.randint(0,len(world))
            if world[(len(world)-1) - y][x] == 'b':
                exit = False
        #print("starting x = ",x,"  starting y = ",y)
        directUtility_recursive(x,y,world)
        #print(values,count)
    return values/count
def directUtility_recursive(x,y,world):
    if world[(len(world)-1) - y][x] == '1':
        return 1.0
    if world[(len(world)-1) - y][x] == '-1':
        return -1.0
    #calculate next state 
    nextX, nextY = nextState(x,y,world)
    #print("next x = ",nextX,"  next y = ",nextY)
    #calc value through recurssion
    myValue = directUtility_recursive(nextX, nextY, world) - 0.04
    #accumlate
    values[(len(world)-1) - y][x]+= myValue
    count[(len(world)-1) - y][x]+= 1
    return myValue

#global variables to be used with function approximation
theta0 = .5
theta1 = .2
theta2 = .1
alpha = 0.001
#function approximation method
def funcApprox(world, itr):
    for i in range(itr):
        exit = False
        while(not exit):
            exit = True
            x = random.randint(0,len(world[0]))
            y = random.randint(0,len(world))
            if world[(len(world)-1) - y][x] == 'b':
                exit = False
        #print("starting x = ",x,"  starting y = ",y)
        funcApprox_recursive(x,y,world)
    final = np.zeros_like(world, dtype = float)
    global theta0
    global theta1
    global theta2
    for i in range(len(final)):
        for j in range(len(final[0])):
            if world[i][j] != '1' and world[i][j] != '-1' and world[i][j] != 'b':
                final[i][j] = theta0 + theta1 * j + theta2 * (len(world[0])-1-i)
            else:
                final[i][j] = 0
    return final
def funcApprox_recursive(x,y,world):
    if world[(len(world)-1) - y][x] == '1':
        return 1.0
    if world[(len(world)-1) - y][x] == '-1':
        return -1.0
    #calculate next state 
    nextX, nextY = nextState(x,y,world)
    #print("next x = ",nextX,"  next y = ",nextY)
    #calc value through recurssion
    myValue = directUtility_recursive(nextX, nextY, world) - 0.04
    #update thetas
    global theta0
    global theta1
    global theta2
    Us = theta0 + theta1 * x + theta2 * y
    theta0 = theta0 + alpha * (myValue - Us)
    theta1 = theta1 + alpha * (myValue - Us) * x
    theta2 = theta2 + alpha * (myValue - Us) * y
    return myValue



example =  str(sys.argv[1])
itr  = int(sys.argv[2])

if example == 'a':
    print("run example a with itr = ",itr)
    r1 = directUtility(world_4_3,itr)
    r2 = funcApprox(world_4_3,itr)
    saveResults(r1,r2)
elif example == 'b':
    print("run example b with itr = ",itr)
    r1 = directUtility(world_10_10_b,itr)
    r2 = funcApprox(world_10_10_b,itr)
    saveResults(r1,r2)
elif example == 'c':
    print("run example b with itr = ",itr)
    r1 = directUtility(world_10_10_c,itr)
    r2 = funcApprox(world_10_10_c,itr)
    saveResults(r1,r2)
else:
    print("invalid example")


