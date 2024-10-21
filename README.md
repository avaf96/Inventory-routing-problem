## Problem Description

The inventory-routing problem (IRP) integrates two well-studied problems, namely, inventory management and vehicle routing. We consider an inventory routing problem in discrete time where a supplier has to serve a set of customers over a multi period horizon. A capacity constraint for the inventory is given for each customer, and the service cannot cause any stockout situation. A single vehicle with a given capacity is available. The transportation cost is proportional to the distance traveled, whereas the inventory holding cost is proportional to the level of the inventory at the customers and at the supplier. The objective is the minimization of the sum of the inventory and transportation costs. 
The IRP is concerned with the repeated distribution of a single product from a single facility to a set of n customers (i.e., retailers) over a given planning horizon of length T. Customer i consumes the product at a given rate u_i (volume per day) and has the capability to maintain a local inventory of the product up to a maximum of C_i. The inventory at customer i is I_i at time 0. A single vehicle with capacity Q is available for distribution of the product. The objective is to minimize the distribution costs during the planning period without causing stockout at any of the customers.
So, three decisions have to be made:
1. When to serve a customer?
2. How much to deliver to a customer when it is served?
3. Which delivery routes to use?
It should be mentioned that IRP differs from traditional vehicle routing problems because it is based on customer’s usage instead of customers’ orders. Moreover, we consider two different replenishment policies that impose rules on the quantity that can be delivered in each delivery to a customer: the order-up-to-level (OU) and the maximum-level (ML) policies. In the OU policy each delivery must fill the inventory to its maximum capacity, effectively linking two of the decisions: once one decides to visit a customer, the quantity to be delivered is simply the difference between its maximum capacity and its current inventory level. In the ML policy, any quantity can be delivered as long as the maximum capacity is not exceeded. The ML policy clearly encompasses the OU one and is more flexible, but also more difficult to solve given the extra set of decision variables.

