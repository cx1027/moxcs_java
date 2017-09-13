package nxcs;

import static java.util.stream.Collectors.toCollection;

import java.awt.Point;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import com.rits.cloning.Cloner;

import nxcs.distance.DistanceCalculator;
import nxcs.testbed.DST;
import nxcs.testbed.EMaze;

/**
 * The main class of NXCS. This class stores the data of the current state of
 * the system, as well as the environment it is operating on. We opt to provide
 * a method for users to run a single iteration of the learning process,
 * allowing more fine grained control over inter-timestep actions such as
 * logging and stopping the process.
 *
 */
public class NXCS {
	/**
	 * The parameters of this system.
	 */
	private final NXCSParameters params;

	/**
	 * The Environment the system is acting on
	 */
	private final Environment env;

	/**
	 * The current population of this system
	 */
	private final List<Classifier> population;

	public List<Classifier> getPopulation() {
		return population;
	}

	/**
	 * The current timestamp in this system
	 */
	private int timestamp;

	/**
	 * The action performed in the previous timestep of this system
	 */
	private int previousAction;


	/**
	 * The reward received in the previous timestep of this system
	 */
	private ActionPareto Reward;

	/**
	 * The state this system was in in the previous timestep of this system
	 */
	private String previousState;

	private addVectorNList addVL;

	private ParetoCal pareto;

	private Cloner cloner;

	private static boolean flagga;
	private static boolean flag = false;
	private static int i = 1;

	/**
	 * Constructs an NXCS instance, operating on the given environment with the
	 * given parameters
	 * 
	 * @param _env
	 *            The environment this system is to operate on
	 * @param _params
	 *            The parameters this system is to use
	 */
	public NXCS(Environment _env, NXCSParameters _params) {
		// if (_env == null)
		// throw new IllegalArgumentException("Cannot operate on null
		// environment");
		// if (_params == null)
		// throw new IllegalArgumentException("Cannot operate with null
		// parameters");

		env = _env;
		params = _params;
		population = new ArrayList<Classifier>();
		timestamp = 0;
		this.cloner = new Cloner();
		this.addVL = new addVectorNList();

	}

	public NXCS() {
		this(null, null);
	}

	/**
	 * Prints the current population of this system to stdout
	 */
	public void printPopulation() {
		for (Classifier clas : population) {
			System.out.println(clas);
		}
	}

	/**
	 * Classifies the given state using the current knowledge of the system
	 * 
	 * @param curr
	 * @param timestamp2
	 * 
	 * @param currState
	 *            The state to classify
	 * @param prev
	 * @param prevState
	 * @return The class the system classifies the given state into
	 */
	public int classify(String currState, Point weight) {
		if (currState.length() != params.stateLength)
			throw new IllegalArgumentException(
					String.format("The given state (%s) is not of the correct length", currState));
		List<Classifier> matchSet = population.stream().filter(c -> stateMatches(c.condition, currState))
				.collect(Collectors.toList());
		List<Classifier> sortset = new ArrayList<Classifier>();
		for (int action = 0; action < params.numActions; action++) {
			final int act = action;
			List<Classifier> A = matchSet.stream().filter(b -> b.action == act).collect(Collectors.toList());
			if (A.size() == 0) {
				continue;
			}
			double avgExp = A.stream().mapToDouble(x -> x.experience).average().getAsDouble();
			List<Classifier> B = A.stream().filter(b -> b.experience >= avgExp).collect(Collectors.toList());
			if (B.size() == 0) {
				continue;
			}
			Collections.sort(B, new Comparator<Classifier>() {
				@Override
				public int compare(Classifier o1, Classifier o2) {
					return o1.error == o2.error ? 0 : (o1.error > o2.error ? 1 : -1);
				}
			});
			sortset.add(B.get(0));
		}

		// delete the cls which next state=prestate
//		sortset.removeIf(x -> x.conditionNext.equals(prevState));
		double[] predictions = generateWeightsPredictions(sortset, weight);
		return selectAction(predictions);
	}

	/**
	 * Classifies the given state using the current knowledge of the system
	 * 
	 * @param curr
	 * @param timestamp2
	 * 
	 * @param currState
	 *            The state to classify
	 * @param prev
	 * @param prevState
	 * @return The class the system classifies the given state into
	 */
	public int classify(String currState, String prevState) {
		if (currState.length() != params.stateLength)
			throw new IllegalArgumentException(
					String.format("The given state (%s) is not of the correct length", currState));
		List<Classifier> matchSet = population.stream().filter(c -> stateMatches(c.condition, currState))
				.collect(Collectors.toList());
		List<Classifier> sortset = new ArrayList<Classifier>();
		for (int action = 0; action < params.numActions; action++) {
			final int act = action;
			List<Classifier> A = matchSet.stream().filter(b -> b.action == act).collect(Collectors.toList());
			if (A.size() == 0) {
				continue;
			}
			Collections.sort(A, new Comparator<Classifier>() {
				@Override
				public int compare(Classifier o1, Classifier o2) {
					return o1.fitness == o2.fitness ? 0 : (o1.fitness > o2.fitness ? 1 : -1);
				}
			});
			sortset.add(A.get(A.size() - 1));
		}

		// delete the cls which next state=prestate
//		sortset.removeIf(x -> x.conditionNext.equals(prevState));
		double[] predictions = generatePredictions(sortset);
		return selectAction(predictions);
	}
	
	/**
	 * calculate hyper volume of current state
	 * 
	 * @param state
	 * 			the state to classifier
	 **/
	public double[] calHyper(String state) {
		HyperVolumn hypervolumn = new HyperVolumn();
		double[] hyper = { 0, 0, 0, 0 };
		List<Classifier> C = getMatchSet(state);
		//loop each action
		for (int action = 0; action < params.numActions; action++) {
			final int act = action;
			List<Classifier> A = C.stream().filter(b -> b.action == act).collect(Collectors.toList());

			Collections.sort(A, new Comparator<Classifier>() {
				@Override
				public int compare(Classifier o1, Classifier o2) {
					return o1.fitness == o2.fitness ? 0 : (o1.fitness > o2.fitness ? 1 : -1);
				}
			});
			if (A.size() == 0) {
				hyper[act] = 0;
			} else {
				double hyperP = hypervolumn.calcHyperVolumn(A.get(A.size() - 1).getV(), new Qvector(-10, -10));
				hyper[act] = hyperP;
			}
			// System.out.println(hyperP);
		}
		return hyper;
	}

	/* 
	 * Main Loop:
	 * run Iteration if determinal condition not meet. eg. finalStateCount=1000
	 * get current state
	 * generate [M] if not the first step, otherwise select an action randomly
	 * select action a
	 * get immediate reward
	 * get current state
	 * if not first step, generate and update [A]-1 and run GA
	 * a-1=a
	 * s-1=s
	 * t++
	 * */	
	public void runIteration(int finalStateCount, String previousState) {

		// System.out.println("privious state is above" + env.getState());
		int action = -1;
		/*if at s0 select random action, select best action with 50% oppotunity if not at the first step*/
		if (previousState != null) {
			/*form [M]*/
			List<Classifier> matchSet = generateMatchSet(previousState);
			/*select a*/
			if (XienceMath.randomInt(params.numActions) <= 1) {
				if (params.actionSelection.equals("maxN")) {
					double[] predictions = generatePredictions(matchSet);
					action = selectAction(predictions);
				}
				if (params.actionSelection.equals("maxH")) {
					double[] hyperP = calHyper(previousState);
					
					action = selectAction(hyperP);
				}
				if (params.actionSelection.equals("random")) {
					action = XienceMath.randomInt(params.numActions);
				}
			} else {
				action = XienceMath.randomInt(params.numActions);
			}
		} else {
			action = XienceMath.randomInt(params.numActions);
		}
		
		/*get immediate reward*/
		Reward = env.getReward(previousState, action);
		if (Reward.getAction() == 5) { /*???which means cant find F in 100, then reset in getReward()*/
			previousState = null;
		}
		/*get current state*/
		String curState = env.getState();
		
		/*if previous State!null, update [A]-1 and run ga*/
		if (previousState != null) {
			/*updateSet include P calculation*/
			List<Classifier> setA = updateSet(previousState, curState, action, Reward.getPareto());
			runGA(setA, previousState);
		}
        
		/*update a-1=a*/
		previousAction = action;
		/*update s-1=s*/
		previousState = curState;
		/*update timestamp*/
		timestamp = timestamp + 1;
		i++;
	}

	/**
	 * Generates [M].
	 * If number of cls in [M] less than thetaMNA, generates new classifiers with random
	 * actions and adds them to the match set. Reference: Page 7 'An Algorithmic
	 * Description of XCS'
	 * 
	 * @see NXCSParameters#thetaMNA
	 * @param state
	 *            the state to generate a match set for
	 * @return The set of classifiers that match the given state
	 */
	public List<Classifier> generateMatchSet(String state) {
		assert(state != null && state.length() == params.stateLength) : "Invalid state";
		List<Classifier> setM = new ArrayList<Classifier>();
		while (setM.size() == 0) {
			//covering
			setM = population.stream().filter(c -> stateMatches(c.condition, state)).collect(Collectors.toList());
			if (setM.size() < params.thetaMNA) {
				Classifier clas = generateCoveringClassifier(state, setM);
				insertIntoPopulation(clas);
//				if(flag == false){
				deleteFromPopulation();
//				}
				setM.clear();
			}
		}

		assert(setM.size() >= params.thetaMNA);
		// System.out.println("setM after coverubg:"+setM);
		return setM;

	}

	public List<Classifier> getMatchSet(String state) {
		return generateMatchSet(state);
	}

	public boolean actionMatches(int action, int a) {
		return action == a;
	}

	/**
	 * Deletes a random classifier in the population, with probability of being
	 * deleted proportional to the fitness of that classifier. Reference: Page
	 * 14 'An Algorithmic Description of XCS'
	 * death pressure (diversity), accuracy pressure, equal number of each niche
	 */
	private void deleteFromPopulation() {
		int numerositySum = population.stream().collect(Collectors.summingInt(c -> c.numerosity));
		if (numerositySum <= params.N) {
			return;
		}

		double averageFitness = population.stream().collect(Collectors.summingDouble(c -> c.fitness)) / numerositySum;
		double[] votes = population.stream()
				.mapToDouble(c -> c.deleteVote(averageFitness, params.thetaDel, params.delta)).toArray();
		double voteSum = Arrays.stream(votes).sum();
		//SUM of votes=1???where is d from
		//-->### this is a Java syntax, which d donates each element in a map(each vote in votes)
		votes = Arrays.stream(votes).map(d -> d / voteSum).toArray();
        //select one cl from population with possibility votes
		Classifier choice = XienceMath.choice(population, votes);
		if (choice.numerosity > 1) {
			choice.numerosity--;
		} else {
			population.remove(choice);
		}
	}

	/**
	 * Insert the given classifier into the population, checking first to see if
	 * any classifier already in the population is more general. If a more
	 * general classifier is found with the same action, that classifiers num is
	 * incremented. Else the given classifer is added to the population.
	 * Reference: Page 13 'An Algorithmic Description of XCS'
	 * 
	 * @param classifier
	 *            The classifier to add
	 */
	private void insertIntoPopulation(Classifier clas) {
		assert(clas != null) : "Cannot insert null classifier";
		Optional<Classifier> same = population.stream()
				.filter(c -> c.action == clas.action && c.condition.equals(clas.condition)).findFirst();
		if (same.isPresent()) {
			same.get().numerosity++;
		} else {
			population.add(clas);
		}
	}

	/**
	 * Generates a classifier with the given state as the condition and a random
	 * action not covered by the given set of classifiers Reference: Page 8 'An
	 * Algorithmic Description of XCS'
	 * 
	 * @param state
	 *            The state to use as the condition for the new classifier
	 * @param matchSet
	 *            The current covering classifiers
	 * @return The generated classifier
	 */
	private Classifier generateCoveringClassifier(String state, List<Classifier> matchSet) {
		assert(state != null && matchSet != null) : "Invalid parameters";
		assert(state.length() == params.stateLength) : "Invalid state length";

		Classifier clas = new Classifier(params, state);
		Set<Integer> usedActions = matchSet.stream().map(c -> c.action).distinct().collect(Collectors.toSet());
		Set<Integer> unusedActions = IntStream.range(0, params.numActions).filter(i -> !usedActions.contains(i)).boxed()
				.collect(Collectors.toSet());
		clas.action = unusedActions.iterator().next();
		clas.timestamp = timestamp;

		return clas;

	}


	/*get non-dominate V of current state
	 * 1)get V of highest cl in each action
	 * 2)collect the Vdots into NDdots
	 * 3)call pareto in NDdots to get non-dominated V
	 * */
	private List<ActionPareto> getParetoVVector(List<Classifier> setM) {
		assert(setM != null && setM.size() >= params.thetaMNA) : "Invalid match set";
		//???Hashmap 
		//-->### hashmap here where <Key, Value> where the key here is the Integer(action:left/right/up/down), and value is the ArrayList<Qvector>
		HashMap<Integer, ArrayList<Qvector>> Vdots = new HashMap<Integer, ArrayList<Qvector>>();
		ArrayList<ActionPareto> NDdots = new ArrayList<ActionPareto>();
	
		/*get V of highest cl in each action*/
		for (int act = 0; act < params.numActions; act++) {
			/*initailise???*/
			//-->### yes, seems just initialise Vdots by each action
			if (Vdots.get(act) == null) {
				ArrayList<Qvector> iniQL = new ArrayList<Qvector>();
				Vdots.put(act, iniQL);
			}

			/*filter the cls with action
			 * soring the cls in nitch by fitness
			 * then get the V of cl(in each action) with highest fitness
			 * then put V into Vdots
			 * */
			final int actIndex = act;
			List<Classifier> setAA = setM.stream().filter(c -> c.action == actIndex).collect(Collectors.toList());
			if (setAA.size() > 0) {
				try {
//					Collections.sort(setAA, (a, b) -> (int) ((a.fitness - b.fitness) * 10024));
					Collections.sort(setAA, new Comparator<Classifier>() {
						@Override
						public int compare(Classifier o1, Classifier o2) {
							return o1.fitness == o2.fitness ? 0 : (o1.fitness > o2.fitness ? 1 : -1);
						}
					});
				} catch (Exception e) {
					System.out.println(String.format("sorrrrrrrrrrrt"));
				}
				Vdots.get(actIndex).addAll(setAA.get(setAA.size() - 1).getV().stream().map(d -> d.clone())
						.collect(toCollection(ArrayList::new)));
			}
		}

		/*map Vdots (one action to multi vector) to NDdots(one action to one vector) ???*/
		//-->###Vdots is a hashmap, where the key is the Integer(action,left/right/up/down)
		//-->### loop Vdots of each action, each action has a list of Qvectors
		for (int index = 0; index < params.numActions; index++) {
			try {
				for (int i = 0; i < Vdots.get(index).size(); i++) {
					/*each vector related to an action*/
					NDdots.add(new ActionPareto(Vdots.get(index).get(i), index));
				}
			} catch (Exception e) {
				System.out.println("Exception!!!!!" + e);
				throw e;
			}
		}
		/*call pareto calculator to get non-dominated V*/
		pareto = new ParetoCal();
		List<ActionPareto> ParetoDotwithA = new ArrayList<ActionPareto>();
		ParetoDotwithA = pareto.getPareto(NDdots);

		return ParetoDotwithA;
	}

	/**
	 * Generates a normalized prediction array from the given match set, based
	 * on the softmax function.
	 * 
	 * @param setM
	 *            The match set to use consider for these predictions
	 * @return The prediction array calculated
	 */
	// TODO: ADD preState
	public double[] generatePredictions(List<Classifier> setM) {
		assert(setM != null && setM.size() >= params.thetaMNA) : "Invalid match set";
		double[] PA = new double[params.numActions];

		List<ActionPareto> NDV = new ArrayList<ActionPareto>();
		NDV = getParetoVVector(setM);
		// Sum the policy parameter for each action
		for (int i = 0; i < params.numActions; i++) {
			for (ActionPareto act : NDV) {
				if (act.getAction() == i) {
					PA[i] += 1;
				}
			}
		}
		return PA;
	}

	public double[] generateWeightsPredictions(List<Classifier> setM, Point weights) {
		assert(setM != null && setM.size() >= params.thetaMNA) : "Invalid match set";
		double[] PA = new double[params.numActions];
		int bestAction = 0;

		List<ActionPareto> NDV = new ArrayList<ActionPareto>();
		NDV = getParetoVVector(setM);

		int m = NDV.get(0).getAction();

		double maxScalar = NDV.get(0).getPareto().get(0) * weights.getX()
				+ NDV.get(0).getPareto().get(1) * weights.getY();

		for (ActionPareto act : NDV) {
			if (act.getPareto().get(0) * weights.getX() + act.getPareto().get(1) * weights.getY() > maxScalar) {
				m = act.getAction();
				maxScalar = act.getPareto().get(0) * weights.getX() + act.getPareto().get(1) * weights.getY();
			}

		}
		PA[m] = 1;
		return PA;
	}

	/**
	 * count number of non-dominated V for each action in [M]
	 * 
	 * @param setM
	 * @return
	 */
	private int[] getPreofClass(List<Classifier> setM) {
		List<ActionPareto> ParetoDot = new ArrayList<ActionPareto>();
		/*get non-dominated V for [M]*/
		ParetoDot = getParetoVVector(setM);

		/*count for each action*/
		int[] count = new int[params.numActions];

		for (ActionPareto ap : ParetoDot) {
			count[ap.getAction()] += 1;
		}
		return count;
	}

	// //Take the exponential of each value
	// double sum = 0;
	// for(int i = 0;i < predictions.length;i ++){
	// predictions[i] = XienceMath.clamp(predictions[i], -10, 10);
	// predictions[i] = Math.exp(predictions[i]);
	// sum += predictions[i];
	// }
	//
	// //Normalize
	// for(int i = 0;i < predictions.length;i ++){
	// predictions[i] /= sum;
	// }
	//
	// assert(predictions.length == params.numActions) : "Predictions are
	// missing?";
	// assert(Math.abs(Arrays.stream(predictions).sum() - 1) <= 0.0001) :
	// "Predictions not normalized";
	//
	// return predictions;

	/**
	 * Selects an action, stochastically, using the given predictions as
	 * probabilities for each action
	 * 
	 * @param predictions
	 *            The predictions to use to select the action
	 * @return The action selected
	 */
	// TODO:SELECT THE MAX NUMBER ACITON!!!!!!
	private int selectAction(double[] predictions) {
		return (int) XienceMath.choice(IntStream.range(0, params.numActions).boxed().toArray(), predictions);
	}

	/**
	 * Estimates the value for a state matched by the given match set
	 * 
	 * @param setM
	 *            The match set to estimate for
	 * @return The estimated maximum value of the state
	 */
	/*
	 * private double valueFunctionEstimation(List<Classifier> setM){ double[]
	 * predictions = generatePredictions(setM); Map<Integer, List<Classifier>>
	 * classifiersForActions =
	 * population.stream().collect(Collectors.groupingBy(c -> c.action)); double
	 * ret = 0; for(Map.Entry<Integer, List<Classifier>> entry :
	 * classifiersForActions.entrySet()){ double[] predictionForAction =
	 * entry.getValue().stream().mapToDouble(c -> c.prediction).toArray();
	 * double[] weights = entry.getValue().stream().mapToDouble(c ->
	 * c.fitness).toArray(); ret += predictions[entry.getKey()] *
	 * XienceMath.average(predictionForAction, weights); } return ret;
	 * 
	 * }
	 */
	// 氓娄鈥毭ε九損rediction盲赂锟矫︹�澛姑ワ拷藴茂录艗猫驴鈩⒚ぢ嘎モ�÷矫︹�⒙懊λ溌∶�擰'+R'氓戮鈥斆ニ喡癡'
	// 莽鈥灺睹ワ拷沤P=r+valueFunctionEstimation
	// 猫鈧捗ㄢ劉鈥榲alueFunctionEstimation氓鈥櫯抪rediction茅鈥∨捗╋拷垄PA莽拧鈥灻モ�β趁陈�
	private double valueFunctionEstimation(List<Classifier> setM) {
		double[] PA = generatePredictions(setM);
		double ret = 0;
		for (int i = 0; i < params.numActions; i++) {
			final int index = i;
			List<Classifier> setAA = setM.stream().filter(c -> c.action == index).collect(Collectors.toList());
			double fitnessSum = setAA.stream().mapToDouble(c -> c.fitness).sum();
			double predictionSum = setAA.stream().mapToDouble(c -> c.prediction * c.fitness).sum();

			if (fitnessSum != 0)
				ret += PA[i] * predictionSum / fitnessSum;
		}
		assert(!Double.isNaN(ret) && !Double.isInfinite(ret));
		return ret;
	}

	/**
	 * Updates the match set/action set of the previous state
	 * 
	 * @see NXCSParameters#gamma
	 * @see NXCSParameters#rho0
	 * @see NXCSParameters#e0
	 * @see NXCSParameters#nu
	 * @see NXCSParameters#alpha
	 * @see NXCSParameters#beta
	 * @see NXCSParameters#doActionSetSubsumption
	 * @see Classifier#averageSize
	 * @see Classifier#error
	 * @see Classifier#prediction
	 * @see Classifier#fitness
	 * @see Classifier#omega
	 * @param previousState
	 *            The previous state of the system
	 * @param currentState
	 *            The current state of the system
	 * @param preAction
	 *            The action performed in the previous state of the system
	 * @param preReward
	 *            The reward received from performing the given action in the
	 *            given previous state
	 * @return The action set of the previous state, with subsumption (possibly)
	 *         applied
	 * 
	 * previousState:s-1
	 * currentState:s
	 * preAction:a-1
	 * preReward=r
	 * 
	 * UPDATE
	 * 1)calcute P first
	 * P:P=r=V if current state is final state, P=r+Q'+R' if normal state
	 * P = reward + params.gamma * getV(generateMatchSet(currentState));
	 * 2)get pre-action set [A]-1 and set size
	 * 3)standard update
	 *   exp++
	 *   less experienced
	 * 
	 *   rich experienced
	 * 
	 * 
	 */
	private List<Classifier> updateSet(String previousState, String currentState, int preAction, Qvector preReward) {
		List<Classifier> previousMatchSet = generateMatchSet(previousState);
		//List<Classifier> curMatchSet = generateMatchSet(currentState);

		ArrayList<Qvector> P = new ArrayList<Qvector>();
		ArrayList<Qvector> V = new ArrayList<Qvector>();

        /*if final state observation P=V=r*/
		if (env.isEndOfProblem(currentState)) {
			P.add(preReward);
			V.add(preReward);
		} 
		/*if not final state,calculate observation P=V+r*/
		else {
			List<ActionPareto> VA = new ArrayList<ActionPareto>();
	
			/*get non-dominate V (with Action) for current state*/
			VA = getParetoVVector(generateMatchSet(currentState));
            /*add pareto vector to V, which is keep pareto value and remove action*/
			for (ActionPareto v : VA) {
				V.add(v.getPareto());
			}
            /*calculate observation P=V+r*/
			P = addVL.addVectorNList(V, preReward);
		}
        
		/*get pre-action set [A]-1 and set size*/
		List<Classifier> actionSet = previousMatchSet.stream().filter(cl -> cl.action == preAction)
				.collect(Collectors.toList());
		int setNumerosity = actionSet.stream().mapToInt(cl -> cl.numerosity).sum();


		/* Update parameters according to experience
                 * 1.clas.averageSize
				 * 2.Q
				 * 3.R
				 * 4.V=Q+R/V=r
				 * 5.clas.prediction: number of non-dominated V
				 * 6.clas.error:diff of V and P
				 * 7.fitness:no change
				 * 8.doActionSetSubsumption:no change
		 * */
		for (Classifier clas : actionSet) {
			/*update experience*/
			clas.experience++;
			/*update when less experienced*/
			if (clas.experience < 1. / params.beta) {
				clas.averageSize = clas.averageSize + (setNumerosity - clas.numerosity) / clas.experience;
				/*nextQ is current V without action, not pareto*/
				//???clone vectors in V into nextQ
				ArrayList<Qvector> nextQ = V.stream().map(d -> d.clone()).collect(toCollection(ArrayList::new));
		
				ParetoQvector paretoQ = new ParetoQvector();
                /*set V
				 *eop
				 * V=P, set V: select the vector close to P, and exp>1
				 *not eop
				 * Q:nondominated V of S
				 * R<-R+(r-R)/n
				 * V=Q+R, select the vector close to P, and exp>1
				 * accutally, it is same when experience is less or more
				 * 
				 * */
				if (env.isEndOfProblem(currentState)) {
					clas.setV(paretoQ.getPareto(nextQ), P);
				} else {
					clas.setQ(paretoQ.getPareto(nextQ));

					clas.setR(addVL.addVector(clas.getR(),
							addVL.divideVector(addVL.minusVector(preReward, clas.getR()), clas.experience)));

					clas.setV(addVL.addVectorNList(clas.getQ(), clas.getR()), P);

				}
				
				/*normal update for (delta update)
				 * clas.prediction: how many non-dominated V in current nitch 
				 * clas.error: diff of V and current V(calculated above)
				 * */
				clas.prediction = clas.prediction
						+ (getPredictionforupdate(previousState, preAction) - clas.prediction) / clas.experience;
				clas.error = clas.error + (params.disCalc.getDistance(P, clas.getV()) - clas.error) / clas.experience;
			} 
			/*update when more experienced*/
			 else {
				clas.averageSize = clas.averageSize + (setNumerosity - clas.numerosity) * params.beta;
				ArrayList<Qvector> nextQ = V.stream().map(d -> d.clone()).collect(toCollection(ArrayList::new));

				ParetoQvector paretoQ = new ParetoQvector();


				if (env.isEndOfProblem(currentState)) {
					clas.setV(paretoQ.getPareto(nextQ), P);
				} else {
					clas.setQ(paretoQ.getPareto(nextQ));

					clas.setR(addVL.addVector(clas.getR(),
							addVL.divideVector(addVL.minusVector(preReward, clas.getR()), clas.experience)));
					clas.setV(addVL.addVectorNList(clas.getQ(), clas.getR()), P);
				}

				clas.prediction = clas.prediction
						+ (getPredictionforupdate(previousState, preAction) - clas.prediction) * params.beta;

				clas.error = clas.error + (params.disCalc.getDistance(P, clas.getV()) - clas.error) * params.beta;

			}
		}

		/*Update Fitness:no change*/
		Map<Classifier, Double> kappa = actionSet.stream().collect(Collectors.toMap(cl -> cl,
				cl -> (cl.error < params.e0) ? 1 : params.alpha * Math.pow(cl.error / params.e0, -params.nu)));
		double accuracySum = kappa.entrySet().stream()
				.mapToDouble(entry -> entry.getValue() * entry.getKey().numerosity).sum();
		actionSet.forEach(cl -> cl.fitness += params.beta * (kappa.get(cl) * cl.numerosity / accuracySum - cl.fitness));

       /*doActionSetSubsumption*/
		if (params.doActionSetSubsumption) {
			return actionSetSubsumption(actionSet);
		}
		return actionSet;

	}

	/*non-dmoninated V number of a-1 in S-1*/
	private double getPredictionforupdate(String previousState, int preAction) {

		List<Classifier> previousMatchSet = generateMatchSet(previousState);
		int nP = 0;

		if (!env.isEndOfProblem(previousState)) {
			/*count number of non-dominated V for each action in S-1*/
			int[] NDVnum = getPreofClass(previousMatchSet);
			/*count number of non-dmoninated V of a in S-1*/
			nP = NDVnum[preAction];
		} else {
			nP = 1;
		}
		return nP;
	}

	/**
	 * Performs an action set subsumption, subsuming the action set into the
	 * most general of the classifiers. Reference: Page 15 'An Algorithmic
	 * Description of XCS'
	 * 
	 * @param setAA
	 *            The action set to subsume
	 * @return The updated action set
	 */
	private List<Classifier> actionSetSubsumption(List<Classifier> setA) {
		Classifier cl = setA.stream().reduce(null, (cl1, cl2) -> (!cl2.couldSubsume(params.thetaSub, params.e0)) ? cl1
				: (cl1 == null) ? cl2 : (cl1.isMoreGeneral(cl2) ? cl1 : cl2));

		if (cl != null) {
			List<Classifier> toRemove = new ArrayList<Classifier>();
			for (Classifier clas : setA) {
				if (cl.isMoreGeneral(clas)) {
					cl.numerosity = cl.numerosity + clas.numerosity;
					toRemove.add(clas);
				}
			}

			setA.removeAll(toRemove);
			population.removeAll(toRemove);
		}

		return setA;
	}

	/**
	 * Runs the genetic algorithm (assuming enough time has passed) in order to
	 * make new classifiers based on the ones currently in the action set
	 * Reference: Page 11 'An Algorithmic Description of XCS'
	 * 
	 * @see NXCSParameters#thetaGA
	 * @see NXCSParameters#mu
	 * @see NXCSParameters#chi
	 * @see NXCSParameters#doGASubsumption
	 * @param currentActionSet
	 *            The current action set in this timestep
	 * @param state
	 *            The current state from the environment
	 */
	private boolean runGA(List<Classifier> setA, String state) {
		assert(setA != null && state != null) : "Invalid parameters";
		// assert(setA.size() > 0) : "No action set";
		if (setA.size() == 0)
			return false;
		assert(state.length() == params.stateLength) : "Invalid state";
		if (timestamp - XienceMath.average(setA.stream().mapToDouble(cl -> cl.timestamp).toArray()) > params.thetaGA) {
			for (Classifier clas : setA) {
				clas.timestamp = timestamp;
			}

			double fitnessSum = setA.stream().mapToDouble(cl -> cl.fitness).sum();
			double[] p = setA.stream().mapToDouble(cl -> cl.fitness / fitnessSum).toArray();
			Classifier parent1 = XienceMath.choice(setA, p);
			Classifier parent2 = XienceMath.choice(setA, p);
			// Classifier child1 = parent1.deepcopy();
			// Classifier child2 = parent2.deepcopy();

			Classifier child1 = cloner.deepClone(parent1);
			child1.GLOBAL_ID++;
			child1.id = child1.GLOBAL_ID;
			Classifier child2 = cloner.deepClone(parent2);
			child2.GLOBAL_ID++;
			child2.id = child2.GLOBAL_ID;

			child1.numerosity = child2.numerosity = 1;
			child1.experience = child2.experience = 0;
			child1.initiateAfterCopied(params);
			child2.initiateAfterCopied(params);

			if (XienceMath.random() < params.crossoverRate) {
				crossover(child1, child2);
				child1.prediction = child2.prediction = (parent1.prediction + parent2.prediction) / 2;
				child1.error = child2.error = 0.25 * (parent1.error + parent2.error) / 2;
				child1.fitness = child2.fitness = 0.1 * (parent1.fitness + parent2.fitness) / 2;
			}

			Classifier[] children = new Classifier[] { child1, child2 };
			for (Classifier child : children) {
				child.mutate(state, params.mutationRate, params.numActions);

				if (params.doGASubsumption) {
					if (parent1.doesSubsume(child, params.thetaSub, params.e0)) {
						parent1.numerosity++;
					} else if (parent2.doesSubsume(child, params.thetaSub, params.e0)) {
						parent2.numerosity++;
					} else {
						insertIntoPopulation(child);
					}
				} else {
					insertIntoPopulation(child);
				}
				deleteFromPopulation();
			}
		}
		return flagga;
	}

	/**
	 * Checks whether the given condition matches the given state
	 * 
	 * @param condition
	 *            The condition to check
	 * @param state
	 *            The state to check against
	 * @return if condition[i] is '#' or state[i] for all i
	 */
	public boolean stateMatches(String condition, String state) {
		assert(condition != null && condition.length() == params.stateLength) : "Invalid condition";
		assert(state != null && state.length() == params.stateLength) : "Invalid state";
		return IntStream.range(0, condition.length())
				.allMatch(i -> condition.charAt(i) == '#' || condition.charAt(i) == state.charAt(i));
	}

	/**
	 * Performs a crossover between the two given conditions, updating both.
	 * Swaps a random number of bits between the two conditions.
	 * 
	 * @see NXCSParameters#chi
	 * @param child1
	 *            The first child to cross over
	 * @param child2
	 *            The second child to cross over
	 */
	private void crossover(Classifier child1, Classifier child2) {
		assert(child1 != null && child2 != null) : "Cannot crossover null child";
		int x = XienceMath.randomInt(params.stateLength);
		int y = XienceMath.randomInt(params.stateLength);
		if (x > y) {
			int tmp = x;
			x = y;
			y = tmp;
		}

		StringBuilder child1Build = new StringBuilder();
		StringBuilder child2Build = new StringBuilder();
		for (int i = 0; i < params.stateLength; i++) {
			if (i < x || i >= y) {
				child1Build.append(child1.condition.charAt(i));
				child2Build.append(child2.condition.charAt(i));
			} else {
				child1Build.append(child2.condition.charAt(i));
				child2Build.append(child1.condition.charAt(i));
			}
		}

		child1.condition = child1Build.toString();
		child2.condition = child2Build.toString();
	}
}
