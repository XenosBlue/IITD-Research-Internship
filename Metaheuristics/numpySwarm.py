import numpy as np 
import sys
import os
import copy
import math
import matplotlib.pyplot as plt
import numpy.ma as ma


#____________________________________________________PARAMETRS

DIMENSIONS = 30
PARTICLES = 50
BOUNDS = 5.12
ITERATIONS = 100000

INERTIA = 0.9
COGNITION = 2.1
SOCIABILITY = 2.1
ANTISOCIAL = 0

PLOT = False
PRINT_FREQ = 20

line = "\n-----------------------------------------\n"

DEBUG = False

#____________________________________________________FUNCTIONS


def RastriginFunction( swarm ):
	fitnessScore = np.reshape( np.sum( ( swarm * swarm ) - ( 10 * np.cos( 2 * np.pi * swarm ) ) + 10, axis=1 ), (-1,1) )
	return fitnessScore

def RastriginMesh( position ):

	fitnessScore = 0.0

	for i in range( len(position) ):
		
		xi = position[i]
		fitnessScore += ( xi * xi ) - ( 10 * np.cos( 2 * math.pi * xi ) ) + 10

	return fitnessScore

#____________________________________________________OPTIMIZE




def Optimize( Initialized ):

	swarm = Initialized
	fitnessScore = RastriginFunction( swarm )
	globalBest	= np.reshape(swarm[ np.argmin( fitnessScore ), : ], (1,DIMENSIONS) )
	localBest 	= np.copy( swarm )
	velocity 	= np.zeros( ( PARTICLES, DIMENSIONS ) )


	for epoch in range(ITERATIONS):


		if epoch % PRINT_FREQ == 0 :
			print('EPOCH ',epoch,'/',ITERATIONS,' Fitness Score : ',RastriginFunction(globalBest))
			if PLOT == True:
				x = np.linspace(-10, 10, 1000)
				y = np.linspace(-10, 10, 1000)
				X, Y = np.meshgrid(x, y)
				Z = RastriginMesh([X,Y])
				plt.contourf(X, Y, Z, 20, cmap='viridis')
				plt.colorbar();
				for i in range(PARTICLES):
					x, y = swarm[i]
					plt.plot(x, y, marker="o", markersize=2, markeredgecolor="#32CD32", markerfacecolor="#50C878")
				plt.show()
		
		#update velocity
		newVelocity = (  INERTIA * velocity
					+ COGNITION   *  np.random.rand( PARTICLES, DIMENSIONS ) * ( localBest  - swarm )
					+ SOCIABILITY *  np.random.rand( PARTICLES, DIMENSIONS ) * ( np.tile( globalBest, ( PARTICLES, 1 ) ) - swarm )
					)

		newVelocity = np.clip(newVelocity,-BOUNDS,BOUNDS)

		#update position
		swarm = swarm + newVelocity
		newfitnessScore = RastriginFunction( swarm )

		#update localBest
		update = np.tile(fitnessScore < newfitnessScore, (1,DIMENSIONS) )
		localBest = (localBest * update.astype(int)) + (swarm * np.invert(update).astype(int))

		#update globalBest
		globalFitnessScore = RastriginFunction(globalBest)
		if (globalFitnessScore > np.min(newfitnessScore)) == True:
			globalBest = np.reshape(swarm[ np.argmin( newfitnessScore ), : ],(1,DIMENSIONS))

		fitnessScore = np.copy(newfitnessScore)
		velocity = np.copy(newVelocity)

		
		
	return globalBest


#____________________________________________________RUN

print("DIMENSIONS = ",DIMENSIONS)
print("PARTICLES  = ",PARTICLES)
print("BOUNDS     = ",BOUNDS)
print('\n')

swarm			= np.random.rand(PARTICLES,DIMENSIONS)*20 -10
print("Swarm Initialized...")
fitnessScore	= RastriginFunction(swarm)
print("Fitness Calculated...")
print("\nBeginning Optimization...")
best			= Optimize(swarm)
print("\nOptimization Loop Complete\n")
print(line)
#print('best solution is : ',best)
print(' Fitness Score : ',RastriginFunction(best),line)


print(RastriginFunction(np.reshape([0,1],(1,2))))