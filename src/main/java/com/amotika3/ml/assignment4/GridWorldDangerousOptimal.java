package com.amotika3.ml.assignment4;

import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.Planner;

public class GridWorldDangerousOptimal {

	GridWorldDomain gwdg;
	OOSADomain domain;
	RewardFunction rf;
	GridWorldTerminalFunction tf;
	State initialState;
	HashableStateFactory hashingFactory;
	SimulatedEnvironment env;
	Map<Integer, List<Double>> runtimeMap = new HashMap<>();
	Map<Integer, List<Double>> rewardMap = new HashMap<>();
	Map<Integer, List<Double>> stepMap = new HashMap<>();
	
	protected static int[][] userMap = new int[][] { 
		{ 0, 0, 0, 0, 0},
		{ 0, 1, 1, 1, 0},
		{ 0, 0, 1, 0, 0},
		{ 0, 0, 1, 0, 1},
		{ 0, 0, 1, 0, 0}
	};
	
	public static void main(String[] args) {
		if (args.length < 1) {
			throw new RuntimeException("Input needs one args: probabilityOfSuccess");
		}
			
		double probabilityOfSuccess = Double.parseDouble(args[0]);
		
		GridWorldDangerousOptimal valueIteration = new GridWorldDangerousOptimal(probabilityOfSuccess);
		GridWorldDangerousOptimal policyIteration = new GridWorldDangerousOptimal(probabilityOfSuccess);
		GridWorldDangerousOptimal qLearning = new GridWorldDangerousOptimal(probabilityOfSuccess);
		String outputPath = "output/"; //directory to record results
		for (int i = 1; i < 20; i+=2) {
			valueIteration.runValueIteration(outputPath, i);
			policyIteration.runPolicyIteration(outputPath, i);

			
		}
		qLearning.runQLearningIteration(outputPath);
		valueIteration.printRuns("Value Iteration");
		policyIteration.printRuns("Policy Iteration");
		qLearning.printRuns("qLearning");
	}
	
	class SmallDangerousGridWorldRewardFunction implements RewardFunction {

		protected StateConditionTest gc;

		public SmallDangerousGridWorldRewardFunction(StateConditionTest gc) {
			this.gc = gc;
		}

		@Override
		public double reward(State s, Action a, State sprime) {
			if (gc.satisfies(sprime)) {
				int xPos = ((int) sprime.get("agent:x"));
				int yPos = ((int) sprime.get("agent:y"));
				
				if (xPos == 4 && yPos == 1) 
					return 50.;
				if (xPos == 2 && yPos == 1) 
					return -100.;
				if (xPos == 3 && yPos == 1) 
					return -100.;
				return 20.;
			}
			return -.01;
		}
	}
	
	public GridWorldDangerousOptimal(double probSucceed){
		gwdg = new GridWorldDomain(5, 5);
		gwdg.setMap(userMap);
		
		tf = new GridWorldTerminalFunction();
		tf.markAsTerminalPosition(4, 1); // Optimal (100)
		tf.markAsTerminalPosition(3, 1); // BAD (-100)
		tf.markAsTerminalPosition(2, 1); // BAD (-100)
		tf.markAsTerminalPosition(4, 4); // Suboptimal (20)
		
		rf = new SmallDangerousGridWorldRewardFunction(new TFGoalCondition(tf));

		gwdg.setTf(tf);
		gwdg.setRf(rf);
		gwdg.setProbSucceedTransitionDynamics(probSucceed);
		
		domain = gwdg.generateDomain();
		
		initialState = new GridWorldState(new GridAgent(0, 0));
		hashingFactory = new SimpleHashableStateFactory();

		env = new SimulatedEnvironment(domain, initialState);
	}
	
	public void visualize(String outputPath){
		Visualizer v = GridWorldVisualizer.getVisualizer(gwdg.getMap());
		new EpisodeSequenceVisualizer(v, domain, outputPath);
	}
	
	public void runValueIteration(String outputPath, int numIterations){
		List<Double> runtime = new ArrayList<>(); 
		List<Double> reward = new ArrayList<>();
		List<Double> steps = new ArrayList<>();
		
		Planner planner = null;
		Policy p = null;
		for (int i = 0; i < 100; i++) {
			long startTime = System.currentTimeMillis();
			hashingFactory = new SimpleHashableStateFactory();
			planner = new ValueIteration(domain, 0.90, hashingFactory, 0.1, numIterations);
			p = planner.planFromState(initialState);
			
			runtime.add(1.0 * System.currentTimeMillis() - startTime);
			
			Episode episode = PolicyUtils.rollout(p, initialState, domain.getModel());
			
			reward.add(this.calcRewardInEpisode(episode));
			steps.add(1.0 * episode.numActions());
		}
		
		this.runtimeMap.put(numIterations, runtime);
		this.rewardMap.put(numIterations, reward);
		this.stepMap.put(numIterations, steps);
		
		if (numIterations == 9)
			valueFunctionVisualization((ValueFunction)planner, p);
	}
	
	public void runPolicyIteration(String outputPath, int numIterations){
		List<Double> runtime = new ArrayList<>(); 
		List<Double> reward = new ArrayList<>();
		List<Double> steps = new ArrayList<>();
		
		Planner planner = null;
		Policy p = null;
		for (int i = 0; i < 100; i++) {
			long startTime = System.currentTimeMillis();
			hashingFactory = new SimpleHashableStateFactory();
			planner = new PolicyIteration(domain, 0.90, hashingFactory, 0.1, numIterations, numIterations);
			p = planner.planFromState(initialState);
			runtime.add(1.0 * System.currentTimeMillis() - startTime);
			
			Episode episode = PolicyUtils.rollout(p, initialState, domain.getModel());
			reward.add(this.calcRewardInEpisode(episode));
			steps.add(1.0 * episode.numTimeSteps());
		}
		
		this.runtimeMap.put(numIterations, runtime);
		this.rewardMap.put(numIterations, reward);
		this.stepMap.put(numIterations, steps);
		if (numIterations == 9) 
			valueFunctionVisualization((ValueFunction)planner, p);
	}
	
	public void runQLearning() {
		LearningAgentFactory qLearningFactory = new LearningAgentFactory() {

			public String getAgentName() {
				return "Q-learning";
			}

			public LearningAgent generateAgent() {
				return new QLearning(domain, 0.90, hashingFactory, 0.3, 0.1);
			}
		};

		//define learning environment
		SimulatedEnvironment env = new SimulatedEnvironment(domain, initialState);

		//define experiment
		LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(env,
				10, 100, qLearningFactory);

		
		
		exp.setUpPlottingConfiguration(500, 250, 2, 1000, 
				TrialMode.MOST_RECENT_AND_AVERAGE,
				PerformanceMetric.CUMULATIVE_STEPS_PER_EPISODE, 
				PerformanceMetric.AVERAGE_EPISODE_REWARD);


		//start experiment
		exp.startExperiment();
	}
	
	public void runQLearningIteration(String outputPath){
		List<Long> runtime = new ArrayList<>(); 
		List<Double> reward = new ArrayList<>();
		List<Integer> steps = new ArrayList<>();
		
		Planner planner = null;
		Policy p = null;
		Episode e = null;
//		for (int i = 0; i < 100; i++) {
		long startTime = System.currentTimeMillis();
		
		
		QLearning agent = new QLearning(domain, 0.90, hashingFactory, .3, .1);

		//run learning for 50 episodes
		for(int j = 0; j < 1000; j++){
			e = agent.runLearningEpisode(env);
			env.resetEnvironment();
		}
		
		//ini(rf, tf, 1);
		p = agent.planFromState(initialState);
		
		runtime.add(System.currentTimeMillis() - startTime);
		reward.add(this.calcRewardInEpisode(e));
		steps.add(e.numTimeSteps());
//		}
		
		valueFunctionVisualization((ValueFunction)agent, p);
	}
	
	public void printRuns(String name) {
		System.out.println("---------------------------------");
		System.out.println("---------------------------------");
		System.out.println("----------" + name + "-----------");
		
		System.out.println("----------Runtimes-----------");
		for (int key : this.runtimeMap.keySet()) {
			System.out.println(key + "," + this.getStats(this.runtimeMap.get(key)));
		}
		
		System.out.println("----------Rewards-----------");
		for (int key : this.rewardMap.keySet()) {
			System.out.println(key + "," + this.getStats(this.rewardMap.get(key)));
		}
		
		System.out.println("----------Steps-----------");
		for (int key : this.stepMap.keySet()) {
			System.out.println(key + "," + this.getStats(this.stepMap.get(key)));
		}
	}
	
	public String getStats(List<Double> vals) {
		double mean = vals.stream().mapToDouble(a -> (double) a).average().getAsDouble();
		
		double temp = vals.stream().mapToDouble(a -> (double) a).reduce((a,b) -> a + (b-mean) * (b-mean)).getAsDouble();
		
		double var = temp/(vals.size() - 1);
		
		return mean + ", " + Math.sqrt(var);
	}
	
	public double calcRewardInEpisode(Episode ea) {
		double myRewards = 0;

		//sum all rewards
		for (int i = 0; i<ea.rewardSequence.size(); i++) {
			myRewards += ea.rewardSequence.get(i);
		}
		return myRewards;
	}
	
	public void valueFunctionVisualization(ValueFunction valueFunction, Policy p){
		List<State> allStates = StateReachability.getReachableStates(initialState, 
				domain, hashingFactory);
		ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(
			allStates, 11, 11, valueFunction, p);
		gui.initGUI();
	}
}
