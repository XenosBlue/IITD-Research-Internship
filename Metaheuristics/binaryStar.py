#EDIT: Leader positions over iterations


import random
import math
import copy
import sys
import os
import numpy as np
import matplotlib.pyplot as plt


#__________________________________________PARAMETERS

DIMENSIONS = 1000
LOWER_BOUND = -10.0
UPPER_BOUND = 10.0

PARTICLES = 100
ITERATIONS = 101
INERTIA = 0.4
COGNITION = 1
SOCIABILITY = 2
ANTISOCIAL = 2
PLOT = True
PLOT_FREQ = 10
BOUND = 5.12

max_iter      = ITERATIONS

line = "\n---------------------------------\n"


#__________________________________________FUNCTIONS

def SphereFunction( position ):

	fitnessScore = 0.0

	for i in range( len(position) ):

		xi = position[i]
		fitnessScore += ( xi * xi )

	return fitnessScore


def RastriginFunction( position ):

	fitnessScore = 0.0

	for i in range( len(position) ):
		
		xi = position[i]
		fitnessScore += ( xi * xi ) - ( 10 * np.cos( 2 * math.pi * xi ) ) + 10

	return fitnessScore


FITNESS       = RastriginFunction

#__________________________________________PARTICLE

class Particle:

	def __init__ ( self, fitness, dimensions, bounds, seed ):

		self.rnd            = random.Random( seed )
		self.dimensions     = dimensions
		self.position       = [ 0.0 for i in range( dimensions ) ]
		self.velocity       = [ 0.0 for i in range( dimensions ) ]
		self.best_part_pos  = [ 0.0 for i in range( dimensions ) ]

		minx = bounds[0]
		maxx = bounds[1]

		for i in range(dimensions):
			self.position[i]  = ((maxx - minx) * self.rnd.random() + minx)
			self.velocity[i]  = ((maxx - minx) * self.rnd.random() + minx)

		self.fitness              = fitness(self.position)
		self.best_part_pos        = copy.copy(self.position)
		self.best_part_fitnessVal = self.fitness

#________________________________________SWARM


class Swarm:

	def __init__( self, fitness, bounds ):

		self.w          =  INERTIA
		self.c1         =  COGNITION
		self.c2         =  SOCIABILITY
		self.c3         =  ANTISOCIAL
		self.population =  PARTICLES
		self.dimensions =  DIMENSIONS
		self.fitness    =  fitness
		self.rnd        =  random.Random(0)
		self.minx = bounds[0]
		self.maxx = bounds[1]

		self.swarmA = self.generate_particles( self.fitness, self.dimensions, bounds, self.population )
		self.best_swarm_posA        = [0.0 for i in range(self.dimensions)]
		self.best_swarm_fitnessValA = sys.float_info.max
		self.leaderA = 0

		self.swarmB = self.generate_particles( self.fitness, self.dimensions, bounds, self.population, spike = self.population )
		self.best_swarm_posB        = [0.0 for i in range(self.dimensions)]
		self.best_swarm_fitnessValB = sys.float_info.max
		self.leaderB = 0



	def generate_particles( self, fitness, dimensions, bounds, n ,spike = 0):
		return [ Particle( fitness, dimensions, bounds, i + spike ) for i in range( n ) ]


	def update_bestA( self ):
		for i in range(self.population):
			if self.swarmA[i].fitness < self.best_swarm_fitnessValA:
				self.best_swarm_fitnessValA = self.swarmA[i].fitness
				self.best_swarm_posA = copy.copy(self.swarmA[i].position)
		return

	def update_bestB( self ):
		for i in range(self.population):
			if self.swarmB[i].fitness < self.best_swarm_fitnessValB:
				self.best_swarm_fitnessValB = self.swarmB[i].fitness
				self.best_swarm_posB = copy.copy(self.swarmB[i].position)
		return






	def Run(self,max_iter):

		self.update_bestA()
		self.update_bestB()


		epoch = 0
		while epoch < max_iter:
			
			if epoch % PLOT_FREQ == 0 :
				if PLOT == True:
					x = np.linspace(-BOUND, BOUND, 1000)
					y = np.linspace(-BOUND, BOUND, 1000)
					X, Y = np.meshgrid(x, y)
					Z = self.fitness([X,Y])
					plt.contourf(X, Y, Z, 20, cmap='viridis')
					plt.colorbar();
					for particle in self.swarmA:
						x, y = particle.position
						plt.plot(x, y, marker="o", markersize=2, markeredgecolor="b", markerfacecolor="b")  ##32CD32
					for particle in self.swarmB:
						x, y = particle.position
						plt.plot(x, y, marker="o", markersize=2, markeredgecolor="r", markerfacecolor="r")
					x, y = self.swarmA[self.leaderA].position
					plt.plot(x, y, marker="o", markersize=5, markeredgecolor="y", markerfacecolor="b")
					x, y = self.swarmB[self.leaderB].position
					plt.plot(x, y, marker="o", markersize=5, markeredgecolor="y", markerfacecolor="r")
					plt.show()
				print("Epoch = " + str(epoch) + " best fitness = %.6f" % self.best_swarm_fitnessValA)
				print("Epoch = " + str(epoch) + " best fitness = %.6f" % self.best_swarm_fitnessValB)
			

			leaderA=0
			leaderB=0
			for i in range(self.population):
				for k in range(self.dimensions):
					r1 = self.rnd.random()
					r2 = self.rnd.random()
					r3 = self.rnd.random()
					#print('swarm',self.swarm)
		     
					self.swarmA[i].velocity[k] = (
										 self.w  * self.swarmA[i].velocity[k] +
										(self.c1 * r1 * (self.swarmA[i].best_part_pos[k] - self.swarmA[i].position[k])) + 
										(self.c2 * r2 * (self.best_swarm_posA[k]         - self.swarmA[i].position[k])) 
										#-(self.c3 * r3 * (self.best_swarm_posB[k]         - self.swarmA[i].position[k]))
										)
 
					r1 = self.rnd.random()
					r2 = self.rnd.random()
					r3 = self.rnd.random()

					self.swarmB[i].velocity[k] = (
										 self.w  * self.swarmB[i].velocity[k] +
										(self.c1 * r1 * (self.swarmB[i].best_part_pos[k] - self.swarmB[i].position[k])) + 
										(self.c2 * r2 * (self.best_swarm_posB[k]         - self.swarmB[i].position[k])) 
										#-(self.c3 * r3 * (self.best_swarm_posA[k]         - self.swarmB[i].position[k]))
										)
					if i == self.leaderA:
						self.swarmA[i].velocity[k] += self.c3*self.rnd.random() * (self.swarmB[self.leaderB].position[k]  - self.swarmA[i].position[k])
					if i == self.leaderB:
						self.swarmB[i].velocity[k] += self.c3*self.rnd.random() * (self.swarmA[self.leaderA].position[k]  - self.swarmB[i].position[k])
		        
			        # clipping
					if self.swarmA[i].velocity[k] < self.minx:
						self.swarmA[i].velocity[k] = self.minx
					elif self.swarmA[i].velocity[k] > self.maxx:
						self.swarmA[i].velocity[k] = self.maxx

					if self.swarmB[i].velocity[k] < self.minx:
						self.swarmB[i].velocity[k] = self.minx
					elif self.swarmB[i].velocity[k] > self.maxx:
						self.swarmB[i].velocity[k] = self.maxx
		 
		 
				# updtae pos
				for k in range(self.dimensions):
					self.swarmA[i].position[k] += self.swarmA[i].velocity[k]

				for k in range(self.dimensions):
					self.swarmB[i].position[k] += self.swarmB[i].velocity[k]
		   
		      	# update fitness
				self.swarmA[i].fitness = self.fitness(self.swarmA[i].position)
				self.swarmB[i].fitness = self.fitness(self.swarmB[i].position)

				if self.swarmA[i].fitness < self.swarmA[self.leaderA].fitness:
					leaderA = i
				if self.swarmB[i].fitness < self.swarmB[self.leaderB].fitness:
					leaderB = i
		 
		      	# update best
				if self.swarmA[i].fitness < self.swarmA[i].best_part_fitnessVal:
					self.swarmA[i].best_part_fitnessVal = self.swarmA[i].fitness
					self.swarmA[i].best_part_pos = copy.copy(self.swarmA[i].position)

				if self.swarmB[i].fitness < self.swarmB[i].best_part_fitnessVal:
					self.swarmB[i].best_part_fitnessVal = self.swarmB[i].fitness
					self.swarmB[i].best_part_pos = copy.copy(self.swarmB[i].position)
		 
		      	# update global best
				if self.swarmA[i].fitness < self.best_swarm_fitnessValA:
					self.best_swarm_fitnessValA = self.swarmA[i].fitness
					self.best_swarm_posA = copy.copy(self.swarmA[i].position)
				if self.swarmB[i].fitness < self.best_swarm_fitnessValB:
					self.best_swarm_fitnessValB = self.swarmB[i].fitness
					self.best_swarm_posB = copy.copy(self.swarmB[i].position)


				#@@@@@@@@@@@@@@@@@@@@@
				self.leaderA = leaderA
				self.leaderB = leaderB
				#@@@@@@@@@@@@@@@@@@@@@
			epoch += 1
		return self.best_swarm_posA, self.best_swarm_posB


#_________________________________________RUN






print("Begin particle swarm optimization on rastrigin function\n")
print("Minimizing SphereFunction with "+str(DIMENSIONS)+" dimensions")
print("Function has known min = 0.0 at (", end="")
'''
for i in range(DIMENSIONS-1):
  print("0, ", end="")
print("0)") 
'''
print("Setting num_particles = " + str(PARTICLES))
print("Setting max_iter    = "   + str(max_iter))
print("\nStarting PSO algorithm\n")
 
 
model = Swarm(FITNESS,[-BOUND, BOUND])
best_positionA, best_positionB = model.Run(max_iter)
 
print("\nPSO completed\n")

print(line)
print("Best solution found:")
#print(["%.6f"%best_positionA[k] for k in range(DIMENSIONS)])
fitnessVal = FITNESS(best_positionA)
print("fitness of best solution = %.6f" % fitnessVal)


#print(["%.6f"%best_positionB[k] for k in range(DIMENSIONS)])
fitnessVal = FITNESS(best_positionB)
print("fitness of best solution = %.6f" % fitnessVal)
print(line)
print("\nEnd particle swarm for RastriginFunction function\n")
 
