
############ Simulated Annealing Algorithm

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


def calculate_dist(x1, x2):
        eudistance = spatial.distance.euclidean(x1, x2)    
        return(eudistance) 


def set_value():

    global n 
    global H 
    global C
    global l 
    global x_l 
    global y_l 
    global B_l 
    global r_l 
    global h_l 
    global i 
    global x_i 
    global y_i 
    global I_i
    global U_i 
    global L_i 
    global r_i 
    global h_i
    global cij
    
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
        i.append(int(customers[j][0]))
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
    

def inventory_cost_sup(h,B):
    return (h*B)

def inventory_cost_customer(h,I):
    return (h*I)


def initial_solution_OU():

    global customers_per_itr_with_inv_0
    global customers_per_itr_rand
    global visited_custs_per_itr
    global total_cost_OU
    global choices_per_t
    total_inv_cost_sup = 0
    total_inv_cost_cust = 0
    total_trans_cost = 0
    

    Inv_i = I_i.copy()
    B_l_ou = B_l

    required = []
    for j in range(0,len(Inv_i)): 
        required.append(0)
  

    for t in range (0,H):
        req_per_t = 0
        cust_per_t = []
        

        total_inv_cost_sup += inventory_cost_sup(h_l,B_l_ou)
        for j in range(0,len(i)):
            total_inv_cost_cust += inventory_cost_customer(h_i[j],Inv_i[j])


        B_l_ou += r_l
        for j in range (0,len(Inv_i)):
            Inv_i[j] = (Inv_i[j] - r_i[j])
            required[j] = (U_i[j] - Inv_i[j])

        cust_in_0 = []
        for j in range(0,len(Inv_i)):
            if Inv_i[j] < r_i[j]:
                if B_l_ou >= required[j]:
                    if C >= (req_per_t + required[j]):
                        req_per_t += required[j]
                        cust_per_t.append(j)
                        cust_in_0.append(j)
                        B_l_ou -= required[j]
                        Inv_i[j] += required[j]
                    else: 
                        # print("Solution Not Fouond")
                        return 0
                else: 
                    # print("Solution Not Fouond")
                    return 0

        customers_per_itr_with_inv_0.append(cust_in_0)
        rand_cust = []
        for d in range (0,len(Inv_i)):
            list1= []
            for j in range(0,len(Inv_i)):
                flag = False
                for k in range (0,len(cust_per_t)):
                    if i[j] == i[cust_per_t[k]]:
                        flag = True
                        break
                if flag == False :
                    if B_l_ou >= required[j]:
                        if C >= (req_per_t + required[j]):
                            list1.append(j)
                        else: continue
                    else: continue
            if d == 0 : 
                choices_per_t.append(list1)
            if not list1:
                break
            else:
                random_customer = random.choice(list1)
                cust_per_t.append(random_customer)  
                rand_cust.append(random_customer)            
                req_per_t += required[random_customer]
                B_l_ou -= required[random_customer]
                Inv_i[random_customer] += required[random_customer]
       
        customers_per_itr_rand.append(rand_cust) 
                      
        random_c1 = 0
        random_c2 = -1
        cust_to_visit = cust_per_t.copy()
        visited_per_t = []
        for j in range(0,len(cust_per_t)):
            if random_c2 != -1:
                random_c1 = random_c2
            else:
                random_c1 = random.choice(cust_to_visit)
                visited_per_t.append(random_c1)
                cust_to_visit.remove(random_c1)
            if (j == 0):
                # total_trans_cost += cij[0,random_c1+1]
                total_trans_cost += int(round(calculate_dist((x_l,y_l),(x_i[random_c1],y_i[random_c1]))))
              
            elif (cust_to_visit == []):
                # total_trans_cost += cij[random_c1+1,0]
                total_trans_cost += int(round(calculate_dist((x_i[random_c1],y_i[random_c1]),(x_l,y_l))))

            else:
                random_c2 = random.choice(cust_to_visit)
                visited_per_t.append(random_c2)
                cust_to_visit.remove(random_c2) 
                # total_trans_cost += cij[random_c1+1,random_c2+1]
                total_trans_cost += int(round(calculate_dist((x_i[random_c1],y_i[random_c1]),(x_i[random_c2],y_i[random_c2]))))
                

        visited_custs_per_itr.append(visited_per_t)

    total_inv_cost_sup += inventory_cost_sup(h_l,B_l_ou)
    for j in range(0,len(i)):
        total_inv_cost_cust += inventory_cost_customer(h_i[j],Inv_i[j])

    total_cost_OU = total_trans_cost + total_inv_cost_sup + total_inv_cost_cust

    total_cost_OU = "{:.2f}".format(total_cost_OU)

    return total_cost_OU
 


def initial_solutoin_ML():
    
    global total_cost_ML
    total_inv_cost_sup1 = 0
    total_inv_cost_cust1 = 0
    total_trans_cost1 = 0

    Inv_i = I_i.copy()
    B_l_ml = B_l

    required = []
    for j in range(0,len(Inv_i)): 
        required.append(0)
  

    for t in range (0,H):
        req_per_t = 0
        cust_per_t = []

        total_inv_cost_sup1 += inventory_cost_sup(h_l,B_l_ml)
        for j in range(0,len(i)):
            total_inv_cost_cust1 += inventory_cost_customer(h_i[j],Inv_i[j])


        B_l_ml += r_l
        for j in range (0,len(Inv_i)):
            Inv_i[j] = (Inv_i[j] - r_i[j])
            required[j] = (U_i[j] - Inv_i[j])


        for j in range(0,len(Inv_i)):
            if Inv_i[j] < r_i[j]:
                if B_l_ml >= required[j]:
                    if C >= (req_per_t + required[j]):
                        req_per_t += required[j]
                        cust_per_t.append(j)
                        B_l_ml -= required[j]
                        Inv_i[j] += required[j]
                    else: 
                        print("Solution Not Fouond")
                        break
                else: 
                    print("Solution Not Fouond")
                    break

        for d in range (0,len(Inv_i)):
            rand_num = 0
            while rand_num == 0:
                rand_num = random.uniform(0,1)
            list1= []
            for j in range(0,len(Inv_i)):
                flag = False
                for k in range (0,len(cust_per_t)):
                    if i[j] == i[cust_per_t[k]]:
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


        len_cust_per_t = len(cust_per_t)                   
        random_c1 = 0
        random_c2 = -1
        for j in range(0,len_cust_per_t):
            if random_c2 != -1:
                random_c1 = random_c2
            else:
                random_c1 = random.choice(cust_per_t)
                cust_per_t.remove(random_c1)

            if (j == 0):
                total_trans_cost1 += cij[0,random_c1+1]
              
            elif (cust_per_t == []):
                total_trans_cost1 += cij[random_c1+1,0]

            else:
                random_c2 = random.choice(cust_per_t)
                cust_per_t.remove(random_c2) 
                total_trans_cost1 += cij[random_c1+1,random_c2+1]

    
    total_inv_cost_sup1 += inventory_cost_sup(h_l,B_l_ml)
    for j in range(0,len(i)):
        total_inv_cost_cust1 += inventory_cost_customer(h_i[j],Inv_i[j])

    total_cost_ML = total_trans_cost1 + total_inv_cost_sup1 + total_inv_cost_cust1
   
    return "{:.2f}".format(total_cost_ML)


def random_neighbor():


    global customers_per_itr_rand2
    global total_cost_nei
    global customers_per_itr_rand1
    total_inv_cost_sup = 0
    total_inv_cost_cust = 0
    total_trans_cost = 0
    
    
    customers_per_itr_rand1.clear()
    customers_per_itr_rand1 = copy.deepcopy(customers_per_itr_rand)
    # print(customers_per_itr_rand1)
    Inv_i = I_i.copy()
    B_l_ou = B_l

    required = []
    for j in range(0,len(Inv_i)): 
        required.append(0)
  

    for t in range (0,H):
        req_per_t = 0
        cust_per_t = []
        

        total_inv_cost_sup += inventory_cost_sup(h_l,B_l_ou)
        for j in range(0,len(i)):
            total_inv_cost_cust += inventory_cost_customer(h_i[j],Inv_i[j])


        B_l_ou += r_l
        for j in range (0,len(Inv_i)):
            Inv_i[j] = (Inv_i[j] - r_i[j])
            required[j] = (U_i[j] - Inv_i[j])

        for j in range(0,len(Inv_i)):
            if Inv_i[j] < r_i[j]:
                if B_l_ou >= required[j]:
                    if C >= (req_per_t + required[j]):
                        req_per_t += required[j]
                        cust_per_t.append(j)
                        B_l_ou -= required[j]
                        Inv_i[j] += required[j]
                    else: 
                        # print("Solution Not Fouond")
                        return 0
                else: 
                    # print("Solution Not Fouond")
                    return 0

 
        rand_cust = []
        for d in range (0,len(Inv_i)):
            list1= []
            for j in range(0,len(Inv_i)):
                flag = False
                for k in range (0,len(cust_per_t)):
                    if i[j] == i[cust_per_t[k]]:
                        flag = True
                        break
                if flag == False :
                    if B_l_ou >= required[j]:
                        if C >= (req_per_t + required[j]):
                            list1.append(j)
                        else: continue
                    else: continue
            if d == 0 : 
                choices_per_t.append(list1)
        
            if not list1:
                break
            else:
                random_customer = customers_per_itr_rand1[t][0]
                if d== 0 :
                    while random_customer == customers_per_itr_rand1[t][0]:
                        random_customer = random.choice(list1)
                else: random_customer = random.choice(list1)
                cust_per_t.append(random_customer)  
                rand_cust.append(random_customer)            
                req_per_t += required[random_customer]
                B_l_ou -= required[random_customer]
                Inv_i[random_customer] += required[random_customer]
       
        customers_per_itr_rand1[t] = rand_cust 
                      
        random_c1 = 0
        random_c2 = -1
        cust_to_visit = cust_per_t.copy()
        for j in range(0,len(cust_per_t)):
            if random_c2 != -1:
                random_c1 = random_c2
            else:
                random_c1 = random.choice(cust_to_visit)
                cust_to_visit.remove(random_c1)
            if (j == 0):
                # total_trans_cost += cij[0,random_c1+1]
                total_trans_cost += int(round(calculate_dist((x_l,y_l),(x_i[random_c1],y_i[random_c1]))))
              
            elif (cust_to_visit == []):
                # total_trans_cost += cij[random_c1+1,0]
                total_trans_cost += int(round(calculate_dist((x_i[random_c1],y_i[random_c1]),(x_l,y_l))))

            else:
                random_c2 = random.choice(cust_to_visit)
                cust_to_visit.remove(random_c2) 
                # total_trans_cost += cij[random_c1+1,random_c2+1]
                total_trans_cost += int(round(calculate_dist((x_i[random_c1],y_i[random_c1]),(x_i[random_c2],y_i[random_c2]))))

    customers_per_itr_rand2.clear()
    customers_per_itr_rand2 = copy.deepcopy(customers_per_itr_rand1)
    total_inv_cost_sup += inventory_cost_sup(h_l,B_l_ou)
    for j in range(0,len(i)):
        total_inv_cost_cust += inventory_cost_customer(h_i[j],Inv_i[j])

    total_cost_nei = total_trans_cost + total_inv_cost_sup + total_inv_cost_cust

    # print(customers_per_itr_with_inv_0)
    # print(customers_per_itr_rand)
    # print(visited_custs_per_itr)
    total_cost_nei = "{:.2f}".format(total_cost_nei)
    # print( total_cost_nei)
    return total_cost_nei


def SA_algorithm():

    global customers_per_itr_rand
    global customers_per_itr_rand2

    initial_temp = 100
    current_temp = initial_temp
    final_temp = 1
    beta = 0.5
    Vc = 0
    while Vc == 0:
        Vc = float(initial_solution_OU())

    for k in range (0,initial_temp):
        if current_temp >= final_temp:
            for d in range (0,10):
                Vn = 0
                while Vn == 0:
                    Vn = float(random_neighbor())
                deltaE = Vn - Vc
                if deltaE <= 0 or random.uniform(0,1) < math.exp(-deltaE/current_temp) :
                    customers_per_itr_rand.clear()
                    customers_per_itr_rand =copy.deepcopy(customers_per_itr_rand2)
                    Vc = Vn
                else: continue
        else: break
        current_temp -= current_temp*beta
    # print(Vc)
    return Vc

    

def writer(header, data, filename):
  with open (filename, "w", newline = "") as csvfile:
    movies = csv.writer(csvfile)
    movies.writerow(header)
    for x in data:
      movies.writerow(x)








filename = "C:\\Users\\Documents\\Low.csv"
header = ("instance" , "n" , "best" , "worst" , "avg_result" , "avg_cpu_time")
dataa = []
# Instances_lowcost_H3
d = "C:\\Users\\Desktop\\data"
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

    i = []
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
    total_cost_nei = 0

    customers_per_itr_with_inv_0 = []
    customers_per_itr_rand = []
    visited_custs_per_itr = []
    choices_per_t = []
    customers_per_itr_rand2 = []
    customers_per_itr_rand1 = []
    total = []
    sup = []
    customers = [] 

   
    full_path = os.path.join(d, path)
    readFile(full_path)
    

    set_value()

    result = []
    cpu_time = []
    for z in range (0,5):
        start_time = time.time()
        result.append(SA_algorithm())
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
    print("best:" , best)
    print("worst:" , worst)
    print("avg_res:" , avg_res)
    print("avg cpu time:" , avg_cpu_time)
    


# writer(header,dataa,filename)






    










    

   





   