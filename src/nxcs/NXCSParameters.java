package nxcs;

import java.util.ArrayList;

import nxcs.distance.IDistanceCalculator;

/**
 * The parameters of an NXCS system. These are much the same as the ones used in
 * XCS and thus the comments regarding their purpose are mostly taken from
 * Martin Butz's XCSJava. This class is designed to be mutable so as to allow creation of instances
 * which can be passed to multiple concurrent NXCS instances which may be running on different environments
 * with different parameters.
 *
 */
public class NXCSParameters {
	
	/**
	 * The number of bits in the state generated by this environment
	 */
	public int stateLength = 16;//maze 8 direction, 6 for other problem, boolearn multiplexer
	
	/**
	 * The number of actions the system can output
	 */
	public int numActions = 4;
	
	/**
	 * The minimum number of classifiers in the match set before covering occurs
	 */
	public int thetaMNA = 2;
	
	/**
	 * The initial value of the policy parameter in NXCS
	 */
	public double initialOmega = 0; //w in the gradient descent

	/**
	 * The initial prediction value when generating a new classifier (e.g in
	 * covering).
	 */
	public double initialPrediction = 0.01;//initial value for all classifiers 

	/**
	 * The initial prediction error value when generating a new classifier (e.g
	 * in covering).
	 */
	public double initialError = 0.01;//initial value for all classifiers, dont touch

	/**
	 * The initial prediction value when generating a new classifier (e.g in
	 * covering).
	 */
	public double initialFitness = 0.01;//initial value, dont touch

	/**
	 * The probability of using a don't care symbol in an allele when covering.
	 */
	public double pHash = 0;

	/**
	 * The discount rate in multi-step problems.
	 */
	public double gamma = 0.71;//how much the rewards are remembered

	/**
	 * The fall of rate in the fitness evaluation.
	 */
	public double alpha = 0.1;//learning rate for fitness

	/**
	 * The learning rate for updating fitness, prediction, prediction error, and
	 * action set size estimate in XCS's classifiers.
	 */
	public double beta = 0.2;//learning rate for prediction , dont change

	/**
	 * Specifies the exponent in the power function for the fitness evaluation.
	 */
	public double nu = 4;//learning rate =5 before

	/**
	 * Specifies the maximal number of micro-classifiers in the population.
	 */
	public int N = 5000;//maximum of the sum of all the numerosity 

	/**
	 * The error threshold under which the accuracy of a classifier is set to
	 * one.
	 */
	public double e0 = 1;//minimum error

	/**
	 * Specified the threshold over which the fitness of a classifier may be
	 * considered in its deletion probability.
	 */
	public int thetaDel = 20;//if the experience is below thetaDel, we dont trust the fitness if the classfier hasnt learn much

	/**
	 * The fraction of the mean fitness of the population below which the
	 * fitness of a classifier may be considered in its vote for deletion.
	 */
	public double delta = 0.1;//minimul fitness if already learned a lot, then kick them out

	/**
	 * The experience of a classifier required to be a subsumer.
	 */
	public int thetaSub = 20;//until they learn 20 steps, they cant subsumer
	
	/**
     * The threshold for the GA application in an action set.
     */
	public int thetaGA = 35;//avg timesteps involved GA

	/**
	 * The probability of applying crossover in an offspring classifier (chi in
	 * literature, pX in XCSJava).
	 */
	public double crossoverRate = 0.8;

	/**
	 * The probability of mutating one allele and the action in an offspring
	 * classifier (mu in literature, pM in XCSJava).
	 */
	public double mutationRate = 0.01;

	/**
	 * Specifies if GA subsumption should be executed.
	 */
	public boolean doGASubsumption = false;//wilson suggested, complex--false, simple problem-ture

	/**
	 * Specifies if action set subsumption should be executed.
	 */
	public boolean doActionSetSubsumption = false;//merge the classifiers to more general one
	
	/**
	 * The maximum reward possible from the Environment. Used for scaling learning
	 * rates by the reward scheme
	 */
	public double rho0 = 1000;
	

	public Qvector intR;
	
	public ArrayList<Qvector> intQ = new ArrayList<Qvector>();
	
	public ArrayList<Qvector> intV = new ArrayList<Qvector>();

//	public ArrayList<MinDistanceV> intVset = new ArrayList<MinDistanceV>();
	
	public IDistanceCalculator disCalc;
	public String actionSelection;
	
	public NXCSParameters(){
		this.intR = new Qvector(0,0);
		this.intQ = new ArrayList<Qvector>();
		this.intV = new ArrayList<Qvector>();

		intQ.add(new Qvector(-10, -10));
		intV.add(new Qvector(-10, -10));

		
	}
}
