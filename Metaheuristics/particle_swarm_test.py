import random
import math
import copy
import sys
import os
import numpy as np
import matplotlib.pyplot as plt


#__________________________________________PARAMETERS

DIMENSIONS 		= 30
PARTICLES 		= 50
ITERATIONS 		= 500

INERTIA 		= 0.729844
COGNITION 		= 1.496180
SOCIABILITY 	= 1.496180

BOUND 			= 5.12


PLOT 			= False
PLOT_FREQ 		= 10

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
		fitnessScore += ( xi * xi ) - ( 10 * np.cos( 2 * np.pi * xi ) ) + 10

	return fitnessScore


FITNESS       =  RastriginFunction

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
		self.population =  PARTICLES
		self.dimensions =  DIMENSIONS
		self.fitness    =  fitness
		self.rnd        =  random.Random(0)

		self.swarm = None
		self.generate_particles( self.fitness, self.dimensions, bounds, self.population )
		self.best_swarm_pos = [0.0 for i in range(self.dimensions)]
		self.best_swarm_fitnessVal = sys.float_info.max
		self.minx = bounds[0]
		self.maxx = bounds[1]


	def generate_particles( self, fitness, dimensions, bounds, n ):
		self.swarm = [ Particle( fitness, dimensions, bounds, i ) for i in range( n ) ]
		return

	def update_best( self ):
		for i in range(self.population):
			if self.swarm[i].fitness < self.best_swarm_fitnessVal:
				self.best_swarm_fitnessVal = self.swarm[i].fitness
				self.best_swarm_pos = copy.copy(self.swarm[i].position)
		return






	def Run(self,max_iter):

		self.update_best()
		history = []
		epoch = 0
		while epoch < max_iter:

			
			if epoch % PLOT_FREQ == 0 and epoch > 1:
				if PLOT == True:
					x = np.linspace(-10, 10, 1000)
					y = np.linspace(-10, 10, 1000)
					X, Y = np.meshgrid(x, y)
					Z = self.fitness([X,Y])
					plt.contourf(X, Y, Z, 20, cmap='viridis')
					plt.colorbar();
					for particle in self.swarm:
						x, y = particle.position
						plt.plot(x, y, marker="o", markersize=2, markeredgecolor="#32CD32", markerfacecolor="#50C878")
					plt.show()
				print("Epoch = " + str(epoch) + " best fitness = %.3f" % self.best_swarm_fitnessVal)
				
			
			for i in range(self.population):
				for k in range(self.dimensions):
					r1 = self.rnd.random()
					r2 = self.rnd.random()
					#print('swarm',self.swarm)
		     
					self.swarm[i].velocity[k] = (
										 self.w  * self.swarm[i].velocity[k] +
										(self.c1 * r1 * (self.swarm[i].best_part_pos[k] - self.swarm[i].position[k])) + 
										(self.c2 * r2 * (self.best_swarm_pos[k]         - self.swarm[i].position[k]))
										) 
		        
			        # clipping
					if self.swarm[i].velocity[k] < self.minx:
						self.swarm[i].velocity[k] = self.minx
					elif self.swarm[i].velocity[k] > self.maxx:
						self.swarm[i].velocity[k] = self.maxx
		 
		 
					# updtae pos
				for k in range(self.dimensions):
					self.swarm[i].position[k] += self.swarm[i].velocity[k]
		   
		      	# update fitness
				self.swarm[i].fitness = self.fitness(self.swarm[i].position)
		 
		      	# update best
				if self.swarm[i].fitness < self.swarm[i].best_part_fitnessVal:
					self.swarm[i].best_part_fitnessVal = self.swarm[i].fitness
					self.swarm[i].best_part_pos = copy.copy(self.swarm[i].position)
		 
		      	# update global best
				if self.swarm[i].fitness < self.best_swarm_fitnessVal:
					self.best_swarm_fitnessVal = self.swarm[i].fitness
					self.best_swarm_pos = copy.copy(self.swarm[i].position)
			epoch += 1
			history.append(self.best_swarm_fitnessVal)
		fig = plt.figure(figsize=[7,5])
		ax = plt.subplot(111)
		l = ax.fill_between( np.linspace(0,len(history),len(history)), history)
		l.set_facecolors([[1.0,.0,.0,.7]])
		l.set_edgecolors([[.0, .0, .0, 1.0]])
		l.set_linewidths([1])
		ax.set_xlabel('Iterations')
		ax.set_ylabel('Fitness')
		ax.set_title('Particle Swarm')
		ax.grid('on')
		plt.show()
		return self.best_swarm_pos


#_________________________________________RUN





print()
print("Setting num_dimensions = " + str(DIMENSIONS))
print("Setting num_particles = " + str(PARTICLES))
print("Setting max_iter    = "   + str(max_iter))
print("\nStarting PSO algorithm\n")
 
 
model = Swarm(FITNESS,[-BOUND, BOUND])
best_position = model.Run(max_iter)
 
print("\nPSO completed\n")

print(line)
print("Best solution found:")
#print(["%.6f"%best_position[k] for k in range(DIMENSIONS)])
fitnessVal = FITNESS(best_position)
print("fitness of best solution = %.6f" % fitnessVal)
print(line)
print("\nEnd particle swarm for RastriginFunction function\n")
 
