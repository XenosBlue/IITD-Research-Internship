import numpy as np 
import sys
import os
import copy
import math
import matplotlib.pyplot as plt
import numpy.ma as ma


#____________________________________________________PARAMETRS

DIMENSIONS = 30
PARTICLES = 63
BOUNDS = 5.12
ITERATIONS = 30000

INERTIA 	= 0.8
COGNITION 	= 1.4
SOCIABILITY = 1.4
ALLEGIANCE  = 0.4
ANTISOCIAL 	= 0

DAMP = 1 #DIMENSIONS 

PLOT = False
PRINT_FREQ = 1000

line = "\n-----------------------------------------\n"

DEBUG = False

PARENT_O   = [ max(0,math.floor((i-1)/2)) for i in range(PARTICLES) ]
SIBLING_O  = [ min((max(0,i-1) * ((i+1)%2)) + ((i+1) * (i%2)),PARTICLES-1) for i in range(PARTICLES) ]

print(PARENT_O)
print(SIBLING_O)

print(len(PARENT_O))
print(len(SIBLING_O))

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
	lbFitnessScore = RastriginFunction(localBest)
	velocity 	= np.zeros( ( PARTICLES, DIMENSIONS ) )


	for epoch in range(ITERATIONS):

		#propagate root node
		'''
		for j in range(PARTICLES):
			lbFitnessScore = RastriginFunction(localBest)
			for i in range(PARTICLES-1,-1,-1):
				parent = max(0,math.floor((i-1)/2))		
				if lbFitnessScore[i,0] < lbFitnessScore[parent,0]:
					swarm[[i, parent]] = swarm[[parent, i]]
					localBest[[i, parent]] = localBest[[parent, i]]
		'''
		'''
		if epoch % PRINT_FREQ == 0 :
			print('EPOCH ',epoch,'/',ITERATIONS,' Fitness Score : ',(RastriginFunction(localBest)))
		'''
		lbFitnessScore = RastriginFunction(localBest)
		for i in range(PARTICLES-1,-1,-1):
			parent = min(max(0,math.floor((i-1)/2)),PARTICLES-1)		
			if lbFitnessScore[i,0] < lbFitnessScore[parent,0]:
				swarm[[i, parent]] = swarm[[parent, i]]
				localBest[[i, parent],:] = localBest[[parent, i],:]
				lbFitnessScore[[i, parent],:] = lbFitnessScore[[parent, i],:]


		if epoch % PRINT_FREQ == 0 :
			print('EPOCH ',epoch,'/',ITERATIONS,' Fitness Score : ',(RastriginFunction(localBest))[0])
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
					+ SOCIABILITY *  np.random.rand( PARTICLES, DIMENSIONS ) * ( swarm[PARENT_O ,:] - swarm )
					+ ALLEGIANCE  *  np.random.rand( PARTICLES, DIMENSIONS ) * ( swarm[SIBLING_O,:] - swarm )
					)


		newVelocity[0,:] /= DAMP

		newVelocity = np.clip(newVelocity,-BOUNDS,BOUNDS)

		#update position
		swarm = swarm + newVelocity
		newfitnessScore = RastriginFunction( swarm )

		#update localBest
		update = np.tile((lbFitnessScore < newfitnessScore), (1,DIMENSIONS)) 
		nlocalBest = (localBest * update.astype('int')) + (swarm * np.invert(update).astype('int'))


		#print(RastriginFunction(nlocalBest)[0,0],RastriginFunction(localBest)[0,0])
		if(RastriginFunction(nlocalBest)[0,0])>(RastriginFunction(localBest)[0,0]):
			print(fitnessScore)
			print(newfitnessScore)
			print(nlocalBest)
			print(localBest)
			print(RastriginFunction( nlocalBest ))
			print(update)
			print(update.astype('int'))
			print(np.invert(update).astype('int'))
			break

		#update globalBest
		'''
		globalFitnessScore = RastriginFunction(globalBest)
		if (globalFitnessScore > np.min(newfitnessScore)) == True:
			globalBest = np.reshape(swarm[ np.argmin( newfitnessScore ), : ],(1,DIMENSIONS))
		'''

		fitnessScore = np.copy(newfitnessScore)
		velocity = np.copy(newVelocity)
		localBest = np.copy(nlocalBest)

		
		
	return localBest


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



