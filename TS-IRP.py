
############ Tabu Search Algorithm

import numpy as np
import os
import math
import xlwt
from xlwt import Workbook
import random
from scipy import spatial
import copy
import os, fnmatch
import xlwt
from xlwt import Workbook
import random
import csv
import time
import matplotlib.pyplot as plt


#reads a file and extracts information we need
def readFile(addrs):

    global total
    global sup
    global customers
    customer = []
    with open(addrs, 'r') as f:
        data = f.readlines()
        Lines = []
        for line in data:
            Line = line.strip('|').split()
            Lines.append(Line)
        # print(Lines)
        for i in range (0,3):
            total.append(Lines[0][i])


        for i in range (0,6):
            sup.append(Lines[1][i])
        
        for i in range(2,len(Lines)):
            cus = []
            for j in range (0,8):
                cus.append(Lines[i][j])
            customer.append(cus)       
        customers = np.array(customer)
        f.close



#calculate distance between 2 nodes
def calculate_dist(x1, x2):
        eudistance = spatial.distance.euclidean(x1, x2)    
        return(eudistance) 


#makes lists of different parts of data in order of use it in other functions
def set_value():

    global n    #number of customers + supplier
    global H    #number of discrete time instants of the planning time horizon
    global C    #transportation capacity
    global l    #number corresponding to the supplier
    global x_l  #x coordinate of the supplier
    global y_l  #y coordinate of the supplier
    global B_l  #starting level of the inventory at the supplier
    global r_l  #quantity of product made available at supplier at each discrete time instant of the planning time horizon
    global h_l  #unit inventory cost at the supplier
    global i_num    #retailer number
    global x_i  #x coordinate of the retailer i
    global y_i  #y coordinate of the retailer i
    global I_i  #starting level of the inventory at the retailer i
    global U_i  #maximum level of the inventory at the retailer I
    global L_i  #minimum level of the inventory at the retailer i
    global r_i  #quantity absorbed by the retailer i at each discrete time instant of the planning time horizon
    global h_i  #unit inventory cost at the retailer i
    global cij  #matrix of distances between every 2 nodes
    dictA = {}
    dictB = {}
    cij.clear()

    n = int(total[0])  
    H = int(total[1])
    C = int(total[2])

    l = int(sup[0])
    x_l = float(sup[1])
    y_l = float(sup[2])
    B_l = int(sup[3])
    r_l = int(sup[4])
    h_l = float(sup[5])

    for j in range (0,len(customers)):
        i_num.append(int(customers[j][0]))
        x_i.append(float(customers[j][1]))
        y_i.append(float(customers[j][2]))
        I_i.append( int(customers[j][3]))
        U_i.append(int(customers[j][4]))
        L_i.append(int(customers[j][5]))
        r_i.append(int(customers[j][6]))
        h_i.append(float(customers[j][7]))

    
    for a in range (0,n):
        if a == 0 :
            dictA[0]=(x_l,y_l)
        else:
            dictA[a]=(x_i[a-1],y_i[a-1])
        for b in range (0,n):
            if b == 0:
                dictB[0]=(x_l,y_l)
                cij[a,b]=int(round(calculate_dist(dictA[a],dictB[b])))
            else:
                dictB[b]=(x_i[b-1],y_i[b-1])
                cij[a,b]=int(round(calculate_dist(dictA[a],dictB[b])))
    # print(cij)
    

#calculates supplier inventory cost
def inventory_cost_sup(h,B):
    return (h*B)

#calculates customer inventory cost
def inventory_cost_customer(h,I):
    return (h*I)



def initial_solution_OU():

    global total_cost_OU
    global com1_visited
    global inv_per_t
    total_inv_cost_sup = 0
    total_inv_cost_cust = 0
    total_trans_cost = 0
    
    cost_C = 0
    cost_sup = 0

    com1_visited.clear()
    inv_per_t.clear()
    Inv_i = I_i.copy()
    B_l_ou = B_l

    required = []
    for j in range(0,len(Inv_i)): 
        required.append(0)
  

    for t in range (0,H):
        req_per_t = 0
        cust_per_t = []
        
        #calculate inventory cost
        total_inv_cost_sup += inventory_cost_sup(h_l,B_l_ou)
        for j in range(0,len(i_num)):
            total_inv_cost_cust += inventory_cost_customer(h_i[j],Inv_i[j])

        #find needed-amount for every customer
        B_l_ou += r_l
        for j in range (0,len(Inv_i)):
            Inv_i[j] = Inv_i[j] - r_i[j]
            required[j] = U_i[j] - Inv_i[j]
        inv_i2 = Inv_i.copy()
        inv_per_t.append(inv_i2)
        
        #find customers that may have stockout situation
        q_t = 0
        for j in range(0,len(Inv_i)):
            if Inv_i[j] < r_i[j]:
                if B_l_ou >= required[j] and C >= (req_per_t + required[j]):
                        req_per_t += required[j]
                        cust_per_t.append(j)
                        B_l_ou -= required[j]
                        Inv_i[j] += required[j]
                        q_t += required[j]
                else: 
                    return 0

        #find other customers that can be served in this iteration
        for d in range (0,len(Inv_i)):
            list1= []
            for j in range(0,len(Inv_i)):
                flag = False
                for k in range (0,len(cust_per_t)):
                    if i_num[j] == i_num[cust_per_t[k]]:
                        flag = True
                        break
                if flag == False :
                    if B_l_ou >= required[j] and C >= (req_per_t + required[j]):
                            list1.append(j)
                    else: continue
            if not list1:
                break
            else:
                random_customer = random.choice(list1)
                cust_per_t.append(random_customer)              
                req_per_t += required[random_customer]
                B_l_ou -= required[random_customer]
                Inv_i[random_customer] += required[random_customer]
                q_t += required[random_customer]
        
        #penalties for situations in which there is stockout at supplier or vehicle capacity is exceeded 
        cost_sup += max(0,(q_t - C))
        cost_C += max(0,-B_l_ou)  


        #which customer to visit first and find the route and calculate tranportation cost
        random_c1 = 0
        random_c2 = -1
        cust_to_visit = cust_per_t.copy()
        visited_per_t = []
        costA = 0 
        costB = 0
        for j in range(0,len(cust_per_t)):
            if random_c2 != -1:
                random_c1 = random_c2
            else:
                random_c1 = random.choice(cust_to_visit)
                visited_per_t.append(random_c1)
                cust_to_visit.remove(random_c1)
            if (j == 0):
                costA += cij[0,random_c1+1]
            elif (cust_to_visit == []):
                costA += cij[random_c1+1,0]
            else:
                random_c2 = random.choice(cust_to_visit)
                visited_per_t.append(random_c2)
                cust_to_visit.remove(random_c2) 
                costA += cij[random_c1+1,random_c2+1]

        for j in range(0,len(cust_per_t)):    
            if j == 0:
                costB += cij[0,cust_per_t[j]+1]
            elif j == len(cust_per_t)-1:
                costB += cij[cust_per_t[j]+1,0]
            else:
                costB += cij[cust_per_t[j]+1,cust_per_t[j+1]+1]

        total_trans_cost +=  min(costA,costB) 
        if costA < costB: com1_visited.append(visited_per_t)
        else : com1_visited.append(cust_per_t)

    total_inv_cost_sup += inventory_cost_sup(h_l,B_l_ou)
    for j in range(0,len(i_num)):
        total_inv_cost_cust += inventory_cost_customer(h_i[j],Inv_i[j])

    total_cost_OU = total_trans_cost + total_inv_cost_sup + total_inv_cost_cust + cost_C + cost_sup
    total_cost_OU = "{:.2f}".format(total_cost_OU)

    return total_cost_OU
 

def neighborhood_delete():

    global total_cost_nei1
    global inv_per_t_nei
    global C
    global visited
    global deleted
    global ran1
    global ran2

    total_inv_cost_sup = 0
    total_inv_cost_cust = 0
    total_trans_cost = 0
    cost_C = 0
    cost_sup = 0

    Inv_i = I_i.copy()
    B_l_ou = B_l

    inv_per_t_nei.clear()
    visited.clear()
    deleted.clear()

    visited = copy.deepcopy(com1_visited)

    required = []
    for j in range(0,len(Inv_i)): 
        required.append(0)

    #find a random customer that if its deleted there will be no stockout
    flag = True
    for a in range (0,3*len(visited)):
        ran1 = random.randint(0,len(visited)-1)
        for b in range (0,3):    
            ran2 = random.randint(0,len(visited[ran1])-1) 
            if inv_per_t[ran1][visited[ran1][ran2]] >= r_i[visited[ran1][ran2]]:
                cu_Del = visited[ran1][ran2]
                deleted.append(cu_Del)  
                visited[ran1].remove(cu_Del)
                flag = False
                break
        if flag == False:
            break
    
    #calculate costs and needed amount for every customer (neighbor solution)
    for q in range (0,len(visited)):
        listv = visited[q]

        total_inv_cost_sup += inventory_cost_sup(h_l,B_l_ou)
        for j in range(0,len(i_num)):
            total_inv_cost_cust += inventory_cost_customer(h_i[j],Inv_i[j])

        
        B_l_ou += r_l
        for j in range (0,len(Inv_i)):
            Inv_i[j] = (Inv_i[j] - r_i[j])
            required[j] = (U_i[j] - Inv_i[j])
        inv_i2 = Inv_i.copy()
        inv_per_t_nei.append(inv_i2)
 
        q_t = 0
        for x in range (0, len(listv)):
            B_l_ou -= required[listv[x]]
            Inv_i[listv[x]] += required[listv[x]]
            q_t += required[listv[x]]
            if x == 0:
                total_trans_cost += cij[0,listv[x]+1]
            elif x == len(listv)-1:
                total_trans_cost += cij[listv[x]+1,0]
            else:
                total_trans_cost += cij[listv[x]+1,listv[x+1]+1]
        cost_sup += max(0,(q_t - C))
        cost_C += max(0,-B_l_ou)
        

    total_inv_cost_sup += inventory_cost_sup(h_l,B_l_ou)
    for j in range(0,len(i_num)):
        total_inv_cost_cust += inventory_cost_customer(h_i[j],Inv_i[j])

    total_cost_nei1 = total_trans_cost + total_inv_cost_sup + total_inv_cost_cust + cost_sup + cost_C

    total_cost_nei1 = "{:.2f}".format(total_cost_nei1)
    return total_cost_nei1


def neighborhood_insert():

    global total_cost_nei2
    global inv_per_t_nei
    global C
    global visited
    global inserted
    global ran1
    global ran2


    total_inv_cost_sup = 0
    total_inv_cost_cust = 0
    total_trans_cost = 0
    cost_C = 0
    cost_sup = 0

    Inv_i = I_i.copy()
    B_l_ou = B_l

    inv_per_t_nei.clear()
    visited.clear()
    inserted.clear()

    visited = copy.deepcopy(com1_visited)

    required = []
    for j in range(0,len(Inv_i)): 
        required.append(0)
    
    #find a customer that have not been served in any iteration
    cu_ins = -1
    for cust in range(0,len(i_num)):
        counter = 0
        for v in visited:
            if cust in v : break
            else: counter +=1
        if counter == len(visited):
            cu_ins = cust


    #find a customer that can cause stockout situation
    if cu_ins == -1:
        f = False
        for r1 in range(0,len(inv_per_t)): 
            for r2 in range(0,len(inv_per_t[r1])):
                if inv_per_t[r1][r2] < 0:
                    cu_ins = r2
                    if r1 == 0 :
                        visited[r1].append(cu_ins)
                        inserted.append(cu_ins)
                        f = True
                        break
                    else:
                        visited[r1 - 1].append(cu_ins)
                        inserted.append(cu_ins)
                        f = True
                        break
            if f == True: break

    #find a random customer
    if cu_ins == -1:
        fa = False
        cu_ins = i_num.index(random.choice(i_num))
        for ele in visited:
            rand1 = random.randint(0,len(visited)-1)
            if cu_ins not in visited[rand1]:
                visited[rand1].append(cu_ins)
                inserted.append(cu_ins)
                fa = True
                break
        if fa == False:
            rand2 = random.randint(0,len(visited)-1)
            visited[rand2].append(cu_ins)
            inserted.append(cu_ins)


    #calculate costs and needed amount for every customer (neighbor solution)
    for q in range (0,len(visited)):
        listv = visited[q]

        total_inv_cost_sup += inventory_cost_sup(h_l,B_l_ou)
        for j in range(0,len(i_num)):
            total_inv_cost_cust += inventory_cost_customer(h_i[j],Inv_i[j])

        
        B_l_ou += r_l
        for j in range (0,len(Inv_i)):
            Inv_i[j] = (Inv_i[j] - r_i[j])
            required[j] = (U_i[j] - Inv_i[j])
        inv_i2 = Inv_i.copy()
        inv_per_t_nei.append(inv_i2)
        
        q_t = 0
        for x in range (0, len(listv)):
            B_l_ou -= required[listv[x]]
            Inv_i[listv[x]] += required[listv[x]]
            q_t += required[listv[x]]
            if x == 0:
                total_trans_cost += cij[0,listv[x]+1]
            elif x == len(listv)-1:
                total_trans_cost += cij[listv[x]+1,0]
            else:
                total_trans_cost += cij[listv[x]+1,listv[x+1]+1]
        cost_sup += max(0,(q_t - C))
        cost_C += max(0,-B_l_ou)
        

    total_inv_cost_sup += inventory_cost_sup(h_l,B_l_ou)
    for j in range(0,len(i_num)):
        total_inv_cost_cust += inventory_cost_customer(h_i[j],Inv_i[j])

    total_cost_nei2 = total_trans_cost + total_inv_cost_sup + total_inv_cost_cust + cost_sup + cost_C

    total_cost_nei2 = "{:.2f}".format(total_cost_nei2)
    return total_cost_nei2





def initial_solution_ML():

    global total_cost_ML
    global com1_visited
    global inv_per_t
    total_inv_cost_sup1 = 0
    total_inv_cost_cust1 = 0
    total_trans_cost1 = 0


    inv_per_t.clear()
    com1_visited.clear()
    B_l_ml = B_l
    Inv_i = I_i.copy()
    

    required = []
    for j in range(0,len(Inv_i)): 
        required.append(0)
  

    for t in range (0,H):
        req_per_t = 0
        cust_per_t = []

        total_inv_cost_sup1 += inventory_cost_sup(h_l,B_l_ml)
        for j in range(0,len(i_num)):
            total_inv_cost_cust1 += inventory_cost_customer(h_i[j],Inv_i[j])


        B_l_ml += r_l
        for j in range (0,len(Inv_i)):
            Inv_i[j] = (Inv_i[j] - r_i[j])
            required[j] = (U_i[j] - Inv_i[j])
        inv_i2 = Inv_i.copy()
        inv_per_t.append(inv_i2)


        for j in range(0,len(Inv_i)):
            if Inv_i[j] < r_i[j]:
                if B_l_ml >= required[j]:
                    if C >= (req_per_t + required[j]):
                        req_per_t += required[j]
                        cust_per_t.append(j)
                        B_l_ml -= required[j]
                        Inv_i[j] += required[j]
                    else: 
                        return 0
                else: 
                    return 0

        for d in range (0,len(Inv_i)):
            rand_num = 0
            while rand_num == 0:
                rand_num = random.uniform(0,1)
            list1= []
            for j in range(0,len(Inv_i)):
                flag = False
                for k in range (0,len(cust_per_t)):
                    if i_num[j] == i_num[cust_per_t[k]]:
                        flag = True
                        break
                if flag == False :
                    if B_l_ml >= round(required[j]*rand_num):
                        if C >= (req_per_t + round(required[j]*rand_num)):
                            list1.append(j)
                        else: continue
                    else: continue
            if not list1:
                break
            else:
                random_customer = random.choice(list1)
                cust_per_t.append(random_customer)              
                req_per_t += round(required[random_customer]*rand_num)
                B_l_ml -= round(required[random_customer]*rand_num)
                Inv_i[random_customer] += round(required[random_customer]*rand_num)
                        
        random_c1 = 0
        random_c2 = -1
        cust_to_visit = cust_per_t.copy()
        visited_per_t = []
        qua_per_t = []
        costA = 0 
        costB = 0
        for j in range(0,len(cust_per_t)):
            if random_c2 != -1:
                random_c1 = random_c2
            else:
                random_c1 = random.choice(cust_to_visit)
                visited_per_t.append(random_c1)
                qua_per_t.append(required[random_c1])
                cust_to_visit.remove(random_c1)
            if (j == 0):
                costA += cij[0,random_c1+1]
           
            elif (cust_to_visit == []):
                costA += cij[random_c1+1,0]

            else:
                random_c2 = random.choice(cust_to_visit)
                visited_per_t.append(random_c2)
                qua_per_t.append(required[random_c2])
                cust_to_visit.remove(random_c2) 
                costA += cij[random_c1+1,random_c2+1]
                

        for j in range(0,len(cust_per_t)):    
            if j == 0:
                costB += cij[0,cust_per_t[j]+1]
            elif j == len(cust_per_t)-1:
                costB += cij[cust_per_t[j]+1,0]
            else:
                costB += cij[cust_per_t[j]+1,cust_per_t[j+1]+1]

    
        total_trans_cost1 +=  min(costA,costB) 
        if costA < costB: com1_visited.append(visited_per_t)
        else : com1_visited.append(cust_per_t)

    total_inv_cost_sup1 += inventory_cost_sup(h_l,B_l_ml)
    for j in range(0,len(i_num)):
        total_inv_cost_cust1 += inventory_cost_customer(h_i[j],Inv_i[j])

    total_cost_ML = total_trans_cost1 + total_inv_cost_sup1 + total_inv_cost_cust1

    total_cost_ML = "{:.2f}".format(total_cost_ML)
    
    # print("initial solution cost: " , total_cost_ML)
    return total_cost_ML


def neighborhood_delete_ML():

    global total_cost_nei3
    global inv_per_t_nei
    global C
    global visited
    global deleted
    global ran1
    global ran2

    total_inv_cost_sup = 0
    total_inv_cost_cust = 0
    total_trans_cost = 0
    cost_C = 0
    cost_sup = 0

    Inv_i = I_i.copy()
    B_l_ou = B_l

    inv_per_t_nei.clear()
    visited.clear()
    deleted.clear()

    visited = copy.deepcopy(com1_visited)

    required = []
    for j in range(0,len(Inv_i)): 
        required.append(0)

 
    flag = True
    for a in range (0,3*len(visited)):
        ran1 = random.randint(0,len(visited)-1)
        for a in range (0,3):    
            ran2 = random.randint(0,len(visited[ran1])-1)
            if inv_per_t[ran1][visited[ran1][ran2]] >= r_i[visited[ran1][ran2]]:
                cu_Del = visited[ran1][ran2]
                deleted.append(cu_Del)  
                visited[ran1].remove(cu_Del)
                flag = False
                break
        if flag == False:
            break
    
    
    for q in range (0,len(visited)):
        listv = visited[q]

        total_inv_cost_sup += inventory_cost_sup(h_l,B_l_ou)
        for j in range(0,len(i_num)):
            total_inv_cost_cust += inventory_cost_customer(h_i[j],Inv_i[j])

        
        B_l_ou += r_l
        for j in range (0,len(Inv_i)):
            Inv_i[j] = (Inv_i[j] - r_i[j])
            required[j] = (U_i[j] - Inv_i[j])
        inv_i2 = Inv_i.copy()
        inv_per_t_nei.append(inv_i2)
        
        q_t = 0
        for x in range (0, len(listv)):
            rand_num = 0
            while rand_num == 0:
                rand_num = random.uniform(0,1)
            B_l_ou -= round(required[listv[x]])*rand_num
            Inv_i[listv[x]] += round(required[listv[x]])*rand_num
            q_t += round(required[listv[x]])*rand_num
            if x == 0:
                total_trans_cost += cij[0,listv[x]+1]
            elif x == len(listv)-1:
                total_trans_cost += cij[listv[x]+1,0]
            else:
                total_trans_cost += cij[listv[x]+1,listv[x+1]+1]
        cost_sup += max(0,(q_t - C))
        cost_C += max(0,-B_l_ou)
        

    total_inv_cost_sup += inventory_cost_sup(h_l,B_l_ou)
    for j in range(0,len(i_num)):
        total_inv_cost_cust += inventory_cost_customer(h_i[j],Inv_i[j])

    total_cost_nei3 = total_trans_cost + total_inv_cost_sup + total_inv_cost_cust + cost_sup + cost_C

    total_cost_nei3 = "{:.2f}".format(total_cost_nei3)
    return total_cost_nei3


def neighborhood_insert_ML():

    global total_cost_nei4
    global inv_per_t_nei
    global C
    global visited
    global inserted
    global ran1
    global ran2
    

    total_inv_cost_sup = 0
    total_inv_cost_cust = 0
    total_trans_cost = 0
    cost_C = 0
    cost_sup = 0

    Inv_i = I_i.copy()
    B_l_ou = B_l

    inv_per_t_nei.clear()
    visited.clear()
    inserted.clear()

    visited = copy.deepcopy(com1_visited)

    required = []
    for j in range(0,len(Inv_i)): 
        required.append(0)
    
    cu_ins = -1
    for cust in range(0,len(i_num)):
        counter = 0
        for v in visited:
            if cust in v : break
            else: counter +=1
        if counter == len(visited):
            cu_ins = cust


    if cu_ins == -1:
        f = False
        for r1 in range(0,len(inv_per_t)): 
            for r2 in range(0,len(inv_per_t[r1])):
                if inv_per_t[r1][r2] < 0:
                    cu_ins = r2
                    if r1 == 0 :
                        visited[r1].append(cu_ins)
                        inserted.append(cu_ins)
                        f = True
                        break
                    else:
                        visited[r1 - 1].append(cu_ins)
                        inserted.append(cu_ins)
                        f = True
                        break
            if f == True: break

    if cu_ins == -1:
        fa = False
        cu_ins = i_num.index(random.choice(i_num))
        for ele in visited:
            rand1 = random.randint(0,len(visited)-1)
            if cu_ins not in visited[rand1]:
                visited[rand1].append(cu_ins)
                inserted.append(cu_ins)
                fa = True
                break
        if fa == False:
            rand2 = random.randint(0,len(visited)-1)
            visited[rand2].append(cu_ins)
            inserted.append(cu_ins)


    
    for q in range (0,len(visited)):
        listv = visited[q]

        total_inv_cost_sup += inventory_cost_sup(h_l,B_l_ou)
        for j in range(0,len(i_num)):
            total_inv_cost_cust += inventory_cost_customer(h_i[j],Inv_i[j])

        
        B_l_ou += r_l
        for j in range (0,len(Inv_i)):
            Inv_i[j] = (Inv_i[j] - r_i[j])
            required[j] = (U_i[j] - Inv_i[j])
        inv_i2 = Inv_i.copy()
        inv_per_t_nei.append(inv_i2)
        
        q_t = 0
        for x in range (0, len(listv)):
            rand_num = 0
            while rand_num == 0:
                rand_num = random.uniform(0,1)
            B_l_ou -= required[listv[x]]*rand_num
            Inv_i[listv[x]] += required[listv[x]]*rand_num
            q_t += required[listv[x]]*rand_num
            if x == 0:
                total_trans_cost += cij[0,listv[x]+1]
            elif x == len(listv)-1:
                total_trans_cost += cij[listv[x]+1,0]
            else:
                total_trans_cost += cij[listv[x]+1,listv[x+1]+1]
        cost_sup += max(0,(q_t - C))
        cost_C += max(0,-B_l_ou)
        

    total_inv_cost_sup += inventory_cost_sup(h_l,B_l_ou)
    for j in range(0,len(i_num)):
        total_inv_cost_cust += inventory_cost_customer(h_i[j],Inv_i[j])

    total_cost_nei4 = total_trans_cost + total_inv_cost_sup + total_inv_cost_cust + cost_sup + cost_C

    total_cost_nei4 = "{:.2f}".format(total_cost_nei4)
    return total_cost_nei4




def TA_algorithm():

    global inv_per_t
    global com1_visited
    global deleted
    global inserted
    global nei_list


    T_list = [] #tabu list for neighbors which are created by delete a customer
    T_list_ins =[] #tabu list for neighbors which are created by insert a customer
    Vc = 0
    while Vc == 0:
        Vc = float(initial_solution_OU())
    best_cost = Vc  #best sulotion
    best_visited = []
    best_visited = copy.deepcopy(com1_visited)

    for k in range (0,100): 
        #remove first sulotion from tabu list if length of the list is equal to 7
        for d in range (0,800):
            if len(T_list) >= 7:
                T_list.pop(0)
            if len(T_list_ins) >= 7:
                T_list_ins.pop(0)

            #remove a solution from tabu list if it has remained for 5 iterations
            if T_list:
                for ele in T_list:
                    if ele[1] + 5 == d:
                        T_list.remove(ele)    
            if T_list_ins:
                for ele in T_list_ins:
                    if ele[1] + 5 == d:
                        T_list_ins.remove(ele)


            #check neighbors which are created by delete a customer. 
            #if the created solution is better than current solution and its not in the tabu list, solution will become our current solution
            Vn = float(neighborhood_delete())
            if not deleted: continue
            deleted.append(d)   
            del1 = copy.deepcopy(deleted)
            if (Vn < Vc) :
                flag = False
                for elem in T_list:
                    if del1[0] == elem[0]:
                        flag =True
                        break
            
                if flag == False:   
                    inv_per_t.clear()
                    inv_per_t = copy.deepcopy(inv_per_t_nei)
                    com1_visited.clear()
                    com1_visited = copy.deepcopy(visited)
                    nei_list.append(Vn)
                    Vc = Vn
                    T_list.append(del1)

            #check neighbors which are created by insert a customer. 
            #if the created solution is better than current solution and its not in the tabu list, solution will become our current solution
            Vn2 = float(neighborhood_insert())
            if not inserted: continue
            inserted.append(d) 
            ins1 = copy.deepcopy(inserted)
            if (Vn2 < Vc) :
                flag = False
                for elem in T_list_ins:
                    if ins1[0] == elem[0]:
                        flag =True
                        break

                if flag == False:   
                    inv_per_t.clear()
                    inv_per_t = copy.deepcopy(inv_per_t_nei)
                    com1_visited.clear()
                    com1_visited = copy.deepcopy(visited)
                    nei_list.append(Vn2)
                    Vc = Vn2
                    T_list_ins.append(ins1)

        #check if current solution is better than best solution or not
        if Vc < best_cost:
            best_cost = Vc
            best_visited.clear()
            best_visited = copy.deepcopy(com1_visited) 
    return best_cost



def polt():
    global nei_list

    plt.ylabel("cost")
    plt.plot(nei_list,label="accepted-neighbors",color='r')
    plt.show()

    

def writer(header, data, filename):
  with open (filename, "w", newline = "") as csvfile:
    movies = csv.writer(csvfile)
    movies.writerow(header)
    for x in data:
      movies.writerow(x)


filename = "C:\\Users\\Documents\\result.csv"
header = ("instance" , "n" , "best" , "worst" , "avg_result" , "avg_cpu_time")
dataa = []
d =  "C:\\Users\\Desktop\\data"
for path in os.listdir(d):
    n = 0 
    H = 0
    C = 0

    l = 0
    x_l = 0
    y_l = 0
    B_l = 0
    r_l = 0
    h_l = 0

    i_num = []
    x_i = []
    y_i = []
    I_i = []
    U_i = []
    L_i = []
    r_i = []
    h_i = []

    cij = {}

    total_cost_OU = 0
    total_cost_ML = 0
    total_cost_nei1 = 0
    total_cost_nei2 = 0
    total_cost_nei3 = 0
    total_cost_nei4 = 0


    total = []
    sup = []
    customers = [] 

    com1_visited = []
    com1_qua = []
    inv_per_t = []
    inv_per_t_nei = []

    quantity = []
    visited = []
    deleted = []
    inserted = []
    ran1 = -1
    ran2 = -1
    nei_list = []
   
    full_path = os.path.join(d, path)
    readFile(full_path)
    set_value()

    result = []
    cpu_time = []
    for z in range (0,5):
        start_time = time.time()
        result.append(TA_algorithm())
        cpu_time.append(float(" %s " % (time.time() - start_time)))


    avg_cpu_time = 0
    for z in range (0,len(cpu_time)):
        avg_cpu_time += cpu_time[z] 
    avg_cpu_time = avg_cpu_time/len(cpu_time)

    best = result[0]
    worst = result[0]
    avg_res = result[0]
    for z in range (1,len(result)):
        if result[z] < best:
            best = result[z]
        if result[z] > worst:
            worst = result[z]
        avg_res += result[z]
    avg_res = avg_res/len(result)


    da = []
    da.append(path)
    da.append(n-1)
    da.append(best)
    da.append(worst)
    da.append(avg_res)
    da.append(avg_cpu_time)
    dataa.append(da)
    print("File name: " , path)
    print("best:" , best)
    print("worst:" , worst)
    print("avg_res:" , avg_res)
    print("avg cpu time:" , avg_cpu_time)
    print("***************************")

    # polt()
    


# writer(header,dataa,filename)






    










    

   





   