import random
import copy


FEATURES = 100
POPULATION = 100



def generate_solution(feature = FEATURES):
	return [ random.randint(0, 1) for i in range(feature) ]

def generate_population(population=POPULATION):
	return [ generate_solution() for i in range(population) ]

def mutate( solution ):
	newSolution = copy.copy(solution)
	idx = random.randint( 0, len(solution)-1 )
	newSolution[idx] = random.randint(0, 1)
	return newSolution

def crossover( solutionA, solutionB ):
	solutionC = []
	for i in range(len(solutionA)):
		if random.randint(0, 1) == 1:
			solutionC.append( solutionA[i] )
		else:
			solutionC.append( solutionB[i] )
	return solutionC


def fitness(solution):
	score = 0
	for i in range(len(solution)):
		score += solution[i]
	return POPULATION/score

def evaluate(generation):
	eval = []
	for each in generation:
		eval.append(fitness(each))
	return eval

def Run(generation,iterations,elitism,mutations,randomness):
	

	for itr in range(iterations):

		NewGeneration = []
		generationfitness = evaluate(generation)

		zipped_lists = zip(generationfitness, generation)
		sorted_pairs = sorted(zipped_lists)
		tuples = zip(*sorted_pairs)
		generationfitness, generation = [ list(tuple) for tuple in  tuples]


		for i in range(elitism):
			NewGeneration.append(generation[i])

		for i in range(mutations):
			solution = random.choices(generation, weights=generationfitness, k=1)
			solution = mutate(solution[0])
			NewGeneration.append(solution)

		for i in range(randomness):
			solution = generate_solution()
			NewGeneration.append(solution)

		for i in range(len(generation) - len(NewGeneration)):
			solA, solB = random.choices(generation, weights=generationfitness, k=2)
			solution = crossover(solA, solB)
			NewGeneration.append(solution)

		generation = NewGeneration

		print('Generation {}  Best Solution Score : {}'.format(itr,fitness(generation[0])))

	return generation



generation = generate_population()

generation = Run(generation,10000,5,20,25)

print(fitness(generation[0]))
